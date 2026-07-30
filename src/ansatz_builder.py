"""Generic, user-drawn quantum circuit ("ansatz") builder.

Lets a circuit be described as a plain JSON-serializable list of gate
placements instead of one of the 5 fixed architectures in src/qnn.py — the
backend for the interactive circuit designer (src/designer_app.py). config1
through config5 are each expressible as a spec here, but this module makes
no assumption about circuit structure (no fixed reps/entangler pattern).

Spec format — a list of gate dicts, applied in order:
    {
        "gate": "RX",            # see _FIXED_GATES / _PARAM_GATES / "PAULIROT" below
        "wires": [2],             # qubit indices, length must match the gate's arity
        "pauli_word": "XY",       # required only for "PAULIROT", e.g. "XY", "ZZZ", "YYYY"
        "param": {                # required iff the gate is parametric
            "init": "random",     # "random" | "custom"
            "value": None,        # required iff init == "custom"
            "frozen": False,      # if True, the angle is fixed and never trained
        },
    }
"""

from typing import Any

import pennylane as qml
import torch

from src.qnn import approx_equiv_measure

# name -> (pennylane operation, arity)
_FIXED_GATES: dict[str, tuple[Any, int]] = {
    "H": (qml.H, 1),
    "X": (qml.X, 1),
    "Y": (qml.Y, 1),
    "Z": (qml.Z, 1),
    "S": (qml.S, 1),
    "T": (qml.T, 1),
    "CNOT": (qml.CNOT, 2),
    "CZ": (qml.CZ, 2),
    "SWAP": (qml.SWAP, 2),
    "TOFFOLI": (qml.Toffoli, 3),
    "CSWAP": (qml.CSWAP, 3),
}

# name -> (pennylane operation, arity); called as op(angle, wires=wires)
_PARAM_GATES: dict[str, tuple[Any, int]] = {
    "RX": (qml.RX, 1),
    "RY": (qml.RY, 1),
    "RZ": (qml.RZ, 1),
    "ISINGXX": (qml.IsingXX, 2),
    "ISINGYY": (qml.IsingYY, 2),
    "ISINGZZ": (qml.IsingZZ, 2),
}

RANDOM_INIT_RANGE = (-0.1, 0.1)


def _gate_arity(gate_spec: dict[str, Any]) -> int:
    name = gate_spec["gate"]
    if name == "PAULIROT":
        return len(gate_spec["pauli_word"])
    if name in _PARAM_GATES:
        return _PARAM_GATES[name][1]
    if name in _FIXED_GATES:
        return _FIXED_GATES[name][1]
    raise ValueError(f"Unknown gate {name!r}")


def is_parametric(gate_spec: dict[str, Any]) -> bool:
    return gate_spec["gate"] == "PAULIROT" or gate_spec["gate"] in _PARAM_GATES


def validate_spec(spec: list[dict[str, Any]], num_qubits: int) -> None:
    if not spec:
        raise ValueError("Circuit spec is empty — add at least one gate.")

    for idx, gate_spec in enumerate(spec):
        name = gate_spec.get("gate")
        if name != "PAULIROT" and name not in _FIXED_GATES and name not in _PARAM_GATES:
            raise ValueError(f"Gate #{idx}: unknown gate {name!r}")

        if name == "PAULIROT":
            word = gate_spec.get("pauli_word", "")
            if not word or any(c not in "IXYZ" for c in word):
                raise ValueError(
                    f"Gate #{idx}: pauli_word must be a non-empty string of I/X/Y/Z, got {word!r}"
                )

        wires = gate_spec.get("wires", [])
        arity = _gate_arity(gate_spec)
        if len(wires) != arity:
            raise ValueError(
                f"Gate #{idx} ({name}): expected {arity} wire(s), got {len(wires)}"
            )
        if len(set(wires)) != len(wires):
            raise ValueError(f"Gate #{idx} ({name}): repeated wire in {wires}")
        if any(w < 0 or w >= num_qubits for w in wires):
            raise ValueError(
                f"Gate #{idx} ({name}): wires {wires} out of range for {num_qubits} qubits"
            )

        param = gate_spec.get("param")
        if is_parametric(gate_spec):
            if param is None:
                raise ValueError(
                    f"Gate #{idx} ({name}) is parametric but has no 'param'"
                )
            if param["init"] not in ("random", "custom"):
                raise ValueError(
                    f"Gate #{idx}: param.init must be 'random' or 'custom'"
                )
            if param["init"] == "custom" and param.get("value") is None:
                raise ValueError(f"Gate #{idx}: param.init='custom' requires a 'value'")
        elif param is not None:
            raise ValueError(
                f"Gate #{idx} ({name}) is not parametric but has a 'param'"
            )


def param_labels(spec: list[dict[str, Any]]) -> list[str]:
    """One label per TRAINABLE (non-frozen) parametric gate, in spec order —
    matches the ordering of the params tensor returned by build_qnn_from_spec.
    """
    labels = []
    for idx, gate_spec in enumerate(spec):
        param = gate_spec.get("param")
        if param is not None and not param["frozen"]:
            wires = "-".join(str(w) for w in gate_spec["wires"])
            labels.append(f"g{idx}_{gate_spec['gate']}_w{wires}")
    return labels


def resolve_spec(spec: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Returns a copy of spec where every parametric gate's random initial
    value has been drawn and fixed as an explicit "custom" value.

    Necessary for reproducibility: a frozen gate with init="random" must
    keep the *exact* value it was trained with when the circuit is rebuilt
    later (e.g. for inference from a checkpoint) — re-drawing it would
    silently change the circuit. Save this resolved spec, not the original.
    """
    resolved = []
    for gate_spec in spec:
        gate_spec = dict(gate_spec)
        param = gate_spec.get("param")
        if param is not None:
            value = (
                float(param["value"])
                if param["init"] == "custom"
                else float(torch.empty(1).uniform_(*RANDOM_INIT_RANGE).item())
            )
            gate_spec["param"] = {**param, "init": "custom", "value": value}
        resolved.append(gate_spec)
    return resolved


def build_qnn_from_spec(
    device: str, num_qubits: int, p_err: float, spec: list[dict[str, Any]]
) -> tuple[Any, torch.Tensor, list[dict[str, Any]]]:
    """Builds a QNN from a user-drawn circuit spec.

    Returns (qnn_forward, initial_params, resolved_spec):
      - initial_params holds ONLY the trainable parameters, in spec order
        (frozen gates get a fixed value baked directly into the circuit and
        never appear in this tensor).
      - resolved_spec is `spec` with every random init already drawn — save
        THIS (not the original spec) if the circuit needs to be rebuilt
        identically later (see resolve_spec).
    """
    validate_spec(spec, num_qubits)
    spec = resolve_spec(spec)

    param_slot: dict[int, int] = {}
    frozen_value: dict[int, float] = {}
    initial_values: list[float] = []
    for idx, gate_spec in enumerate(spec):
        param = gate_spec.get("param")
        if param is None:
            continue
        value = float(param["value"])
        if param["frozen"]:
            frozen_value[idx] = value
        else:
            param_slot[idx] = len(initial_values)
            initial_values.append(value)

    initial_params = torch.tensor(initial_values, dtype=torch.float64)

    dev = qml.device(device, wires=num_qubits, shots=None)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def qnn_base(
        embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor
    ) -> Any:
        qml.QubitUnitary(embedding_unitary, wires=range(num_qubits))

        for idx, gate_spec in enumerate(spec):
            name = gate_spec["gate"]
            wires = gate_spec["wires"]

            angle: torch.Tensor | None = None
            if idx in param_slot:
                angle = params[param_slot[idx]]
            elif idx in frozen_value:
                angle = torch.tensor(frozen_value[idx])

            if name == "PAULIROT":
                qml.PauliRot(angle, gate_spec["pauli_word"], wires=wires)
            elif name in _PARAM_GATES:
                op, _ = _PARAM_GATES[name]
                op(angle, wires=wires)
            else:
                op, _ = _FIXED_GATES[name]
                op(wires=wires)

            if p_err != 0:
                for w in wires:
                    qml.DepolarizingChannel(p_err, wires=w)

        approx_equiv_measure(torch.tensor(0.0), p_err, num_qubits)

        coeffs = [1.0 / num_qubits] * num_qubits
        observables = [qml.Z(i) for i in range(num_qubits)]
        H = qml.Hamiltonian(coeffs, observables)
        return qml.expval(H)

    def qnn_forward(
        embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor
    ) -> Any:
        return qnn_base(embedding_unitary, params, phi)

    return qnn_forward, initial_params, spec
