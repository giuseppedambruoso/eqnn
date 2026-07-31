"""Generic, user-drawn quantum circuit ("ansatz") builder.

Lets a circuit be described as a plain JSON-serializable list of gate
placements instead of one of the fixed architectures in src/qnn.py — the
backend for the interactive circuit designer (src/designer_app.py). This
module makes no assumption about circuit structure (no fixed reps/entangler
pattern). config1-config5 can be expanded into an equivalent spec via
architecture_to_spec() (the inner, untwirled rotation+entangler pattern —
whether the architecture additionally needs p4m twirling on top is exposed
separately via ARCHITECTURES[architecture]["twirled"], see qnn.py).
config6-config9 (the paper-style D4-equivariant/nonequivariant ansatzes) are
built via src.paper_ansatzes.paper_architecture_spec instead — see that
module.

Spec format — a list of gate dicts, applied in order:
    {
        "gate": "RX",            # see _FIXED_GATES / _PARAM_GATES / "PAULIROT" below
        "wires": [2],             # qubit indices, length must match the gate's arity
        "pauli_word": "XY",       # required only for "PAULIROT", e.g. "XY", "ZZZ", "YYYY"
        "param": {                # required iff the gate is parametric
            "init": "random",     # "random" | "custom"
            "value": None,        # required iff init == "custom"
            "frozen": False,      # if True, the angle is fixed and never trained
            "group": None,        # optional: gates sharing the same group string
                                   # share ONE trainable parameter (tied weights) —
                                   # needed by e.g. config6-9's paired row/column
                                   # rotations. Omit for an independent parameter.
        },
    }
"""

from typing import Any

import pennylane as qml
import torch

from src.data_encoding import embedding_unitary
from src.qnn import (
    ARCHITECTURES,
    FROZEN_ENTANGLER_ANGLE,
    apply_group_element,
    approx_equiv_measure,
)

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
READOUT_SCHEMES = ("sum_z", "x0_xhalf", "avg_x")


def _gate_arity(gate_spec: dict[str, Any]) -> int:
    name = gate_spec["gate"]
    if name == "PAULIROT":
        return len(gate_spec["pauli_word"])
    if name in _PARAM_GATES:
        return _PARAM_GATES[name][1]
    if name in _FIXED_GATES:
        return _FIXED_GATES[name][1]
    raise ValueError(f"Unknown gate {name!r}")


def is_parametric_gate(gate_name: str) -> bool:
    return gate_name == "PAULIROT" or gate_name in _PARAM_GATES


def is_parametric(gate_spec: dict[str, Any]) -> bool:
    return is_parametric_gate(gate_spec["gate"])


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
            group = param.get("group")
            if group is not None and not isinstance(group, str):
                raise ValueError(f"Gate #{idx}: param.group must be a string or None")
        elif param is not None:
            raise ValueError(
                f"Gate #{idx} ({name}) is not parametric but has a 'param'"
            )


def param_labels(spec: list[dict[str, Any]]) -> list[str]:
    """One label per TRAINABLE (non-frozen) parameter slot, in first-seen
    spec order — matches the ordering of the params tensor returned by
    build_qnn_from_spec. Gates sharing a "group" collapse to a single label
    (they share one trainable slot).
    """
    labels = []
    seen_groups: set[str] = set()
    for idx, gate_spec in enumerate(spec):
        param = gate_spec.get("param")
        if param is None or param["frozen"]:
            continue
        group = param.get("group")
        if group is not None:
            if group in seen_groups:
                continue
            seen_groups.add(group)
        wires = "-".join(str(w) for w in gate_spec["wires"])
        labels.append(f"g{idx}_{gate_spec['gate']}_w{wires}")
    return labels


def architecture_to_spec(
    architecture: str, num_qubits: int, reps: int
) -> list[dict[str, Any]]:
    """Expands config1-config5 into an equivalent gate-by-gate spec — the
    "inner" rotation+entangler pattern, e.g. as a starting point in the
    designer. Whether the architecture additionally needs p4m twirling on
    top of this spec is NOT encoded here: check
    ARCHITECTURES[architecture]["twirled"] and set build_qnn_from_spec's
    twirled= accordingly (the designer does this automatically when loading
    a preset).

    config6-config9 raise ValueError — use
    src.paper_ansatzes.paper_architecture_spec for those instead.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}")
    arch_spec = ARCHITECTURES[architecture]
    if arch_spec.get("kind", "uniform") != "uniform":
        raise ValueError(
            f"{architecture!r} isn't a uniform rotation+entangler architecture "
            "— use src.paper_ansatzes.paper_architecture_spec for config6-config9."
        )
    rotation_gate = arch_spec["rotation_gate"]
    entangler = arch_spec["entangler"]
    cross_edge_index = (num_qubits // 2) - 1

    spec: list[dict[str, Any]] = []
    for _ in range(reps):
        for i in range(num_qubits):
            spec.append(
                {
                    "gate": rotation_gate,
                    "wires": [i],
                    "param": {"init": "random", "value": None, "frozen": False},
                }
            )

        if entangler == "cnot":
            for i in range(num_qubits - 1):
                spec.append({"gate": "CNOT", "wires": [i, i + 1]})
        elif entangler == "frozen_rxy":
            for i in range(num_qubits - 1):
                spec.append(
                    {
                        "gate": "PAULIROT",
                        "wires": [i, i + 1],
                        "pauli_word": "XY",
                        "param": {
                            "init": "custom",
                            "value": FROZEN_ENTANGLER_ANGLE,
                            "frozen": True,
                        },
                    }
                )
        elif entangler == "frozen_ryy":
            for i in range(num_qubits - 1):
                if i == cross_edge_index:
                    wires = [i - 1, i, i + 1, i + 2]
                    spec.append(
                        {
                            "gate": "PAULIROT",
                            "wires": wires,
                            "pauli_word": "YYYY",
                            "param": {
                                "init": "custom",
                                "value": FROZEN_ENTANGLER_ANGLE,
                                "frozen": True,
                            },
                        }
                    )
                else:
                    spec.append(
                        {
                            "gate": "ISINGYY",
                            "wires": [i, i + 1],
                            "param": {
                                "init": "custom",
                                "value": FROZEN_ENTANGLER_ANGLE,
                                "frozen": True,
                            },
                        }
                    )
        else:
            raise ValueError(f"Unknown entangler {entangler!r}")
    return spec


def resolve_spec(spec: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Returns a copy of spec where every parametric gate's random initial
    value has been drawn and fixed as an explicit "custom" value.

    Necessary for reproducibility: a frozen gate with init="random" must
    keep the *exact* value it was trained with when the circuit is rebuilt
    later (e.g. for inference from a checkpoint) — re-drawing it would
    silently change the circuit. Save this resolved spec, not the original.

    Gates sharing a "group" are resolved together: the first occurrence
    draws/fixes the value, every later occurrence in the same group reuses
    it exactly (they are meant to be the same tied parameter).
    """
    resolved = []
    group_values: dict[str, dict[str, Any]] = {}
    for gate_spec in spec:
        gate_spec = dict(gate_spec)
        param = gate_spec.get("param")
        if param is not None:
            group = param.get("group")
            if group is not None and group in group_values:
                gate_spec["param"] = group_values[group]
            else:
                value = (
                    float(param["value"])
                    if param["init"] == "custom"
                    else float(torch.empty(1).uniform_(*RANDOM_INIT_RANGE).item())
                )
                resolved_param = {**param, "init": "custom", "value": value}
                gate_spec["param"] = resolved_param
                if group is not None:
                    group_values[group] = resolved_param
        resolved.append(gate_spec)
    return resolved


def build_qnn_from_spec(
    device: str,
    num_qubits: int,
    p_err: float,
    spec: list[dict[str, Any]],
    twirled: bool = False,
    readout: str = "sum_z",
) -> tuple[Any, torch.Tensor, list[dict[str, Any]]]:
    """Builds a QNN from a user-drawn circuit spec.

    twirled: wrap the spec in explicit p4m group-twirling (apply one of the
        8 p4m group elements before AND after the spec, averaged over all
        8) — the same mechanism that makes config2/config4/config5
        p4m-equivariant in src.qnn.create_qnn.
    readout: "sum_z" (default) measures the average of qml.Z over every
        qubit, preceded by the RZ+H noise-mixing layer used by
        config1-config5 (see src.qnn.approx_equiv_measure). "x0_xhalf"
        measures 0.5*(X_0 + X_{num_qubits//2}) with no mixing layer — the
        readout config6-config9 (src.paper_ansatzes) default to, to
        preserve their exact p4m-equivariance. "avg_x" measures the average
        of qml.X over every qubit (no mixing layer) — also p4m-invariant
        under the same row/column-flip + swap generators (X commutes with
        itself under an X-flip, and summing over all qubits is unaffected
        by permuting them via the row/column swap), so it's a valid
        alternative readout for config6-config9 too.

    Returns (qnn_forward, initial_params, resolved_spec):
      - initial_params holds ONLY the trainable parameters, in first-seen
        spec order (frozen gates get a fixed value baked directly into the
        circuit and never appear in this tensor; gates sharing a "group"
        collapse to a single shared trainable entry).
      - resolved_spec is `spec` with every random init already drawn — save
        THIS (not the original spec) if the circuit needs to be rebuilt
        identically later (see resolve_spec).
    """
    if readout not in READOUT_SCHEMES:
        raise ValueError(f"readout must be one of {READOUT_SCHEMES}, got {readout!r}")

    validate_spec(spec, num_qubits)
    spec = resolve_spec(spec)

    param_slot: dict[int, int] = {}
    frozen_value: dict[int, float] = {}
    group_slot: dict[str, int] = {}
    initial_values: list[float] = []
    for idx, gate_spec in enumerate(spec):
        param = gate_spec.get("param")
        if param is None:
            continue
        value = float(param["value"])
        if param["frozen"]:
            frozen_value[idx] = value
            continue
        group = param.get("group")
        if group is not None and group in group_slot:
            param_slot[idx] = group_slot[group]
            continue
        param_slot[idx] = len(initial_values)
        initial_values.append(value)
        if group is not None:
            group_slot[group] = param_slot[idx]

    initial_params = torch.tensor(initial_values, dtype=torch.float64)

    dev = qml.device(device, wires=num_qubits, shots=None)
    half = num_qubits // 2

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def qnn_base(
        embedding_unitary_matrix: torch.Tensor,
        params: torch.Tensor,
        phi: torch.Tensor,
        g_idx: int = 0,
    ) -> Any:
        qml.QubitUnitary(embedding_unitary_matrix, wires=range(num_qubits))

        if twirled:
            apply_group_element(g_idx, num_qubits)
            if p_err != 0:
                for w in range(num_qubits):
                    qml.DepolarizingChannel(p_err, wires=w)

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

        if twirled:
            apply_group_element(g_idx, num_qubits)
            if p_err != 0:
                for w in range(num_qubits):
                    qml.DepolarizingChannel(p_err, wires=w)

        if readout == "sum_z":
            approx_equiv_measure(torch.tensor(0.0), p_err, num_qubits)
            coeffs = [1.0 / num_qubits] * num_qubits
            observables = [qml.Z(i) for i in range(num_qubits)]
            H = qml.Hamiltonian(coeffs, observables)
        elif readout == "x0_xhalf":
            H = qml.Hamiltonian([0.5, 0.5], [qml.X(0), qml.X(half)])
        else:
            coeffs = [1.0 / num_qubits] * num_qubits
            observables = [qml.X(i) for i in range(num_qubits)]
            H = qml.Hamiltonian(coeffs, observables)
        return qml.expval(H)

    def qnn_forward(
        embedding_unitary_matrix: torch.Tensor, params: torch.Tensor, phi: torch.Tensor
    ) -> Any:
        if twirled:
            results = [
                qnn_base(embedding_unitary_matrix, params, phi, g) for g in range(8)
            ]
            return torch.stack(results).mean(dim=0)
        return qnn_base(embedding_unitary_matrix, params, phi, 0)

    # qml.draw()/qml.draw_mpl() need the actual QNode, not a function that
    # merely calls one — drawing qnn_forward directly silently truncates the
    # diagram after the first few operations (verified: everything from the
    # measurement's Hadamard layer onward goes missing). Expose it as
    # `qnn_forward.qnode` for callers that want a circuit diagram.
    qnn_forward.qnode = qnn_base  # type: ignore[attr-defined]

    return qnn_forward, initial_params, spec


def check_p4m_invariance(
    qnn_forward: Any,
    params: torch.Tensor,
    img_size: int,
    n_samples: int = 3,
    atol: float = 1e-6,
) -> tuple[bool, float]:
    """Numerically probes whether a built circuit is p4m-equivariant.

    Reuses the flip/transpose methodology validated in
    tests/test_equivariance.py: an equivariant circuit's output must be
    unchanged when the input image is flipped (either axis) or transposed —
    all p4m group elements. This is a finite-sample numerical check, not a
    proof, but it reliably distinguishes equivariant from non-equivariant
    architectures in this codebase: truly p4m-equivariant circuits (either
    via explicit group-twirling or generator-commuting design) show
    deviations at the ~1e-16 floating-point-noise level, while
    non-equivariant ones show deviations of at least ~1e-3 — a default
    atol of 1e-6 cleanly separates the two (an atol as loose as 1e-2, by
    contrast, was empirically observed to misclassify some non-equivariant
    circuits as invariant).

    Returns (is_invariant, max_observed_deviation).
    """
    torch.manual_seed(0)
    phi = torch.tensor(0.0)
    max_deviation = 0.0
    for _ in range(n_samples):
        img = torch.rand(img_size, img_size, dtype=torch.float64)
        img = img / torch.linalg.norm(img.reshape(-1))
        variants = [
            img,
            torch.flip(img, dims=[-1]),
            torch.flip(img, dims=[-2]),
            img.transpose(-1, -2),
        ]
        base = qnn_forward(embedding_unitary(variants[0]), params, phi)
        for variant in variants[1:]:
            out = qnn_forward(embedding_unitary(variant), params, phi)
            max_deviation = max(max_deviation, abs((out - base).item()))
    return max_deviation < atol, max_deviation
