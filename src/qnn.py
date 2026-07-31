import logging
import math
from typing import Any

import pennylane as qml
import torch

logger = logging.getLogger(__name__)

# --- Symmetries p4m (D4) ---


def V_x(num_qubits: int) -> None:
    half = num_qubits // 2
    for i in range(half):
        qml.X(wires=i)


def V_y(num_qubits: int) -> None:
    half = num_qubits // 2
    for i in range(half, num_qubits):
        qml.X(wires=i)


def apply_group_element(g_idx: int, num_qubits: int) -> None:
    """Applies one of the 8 p4m group elements for explicit twirling."""
    half = num_qubits // 2
    if g_idx == 0:
        pass  # Identity
    elif g_idx == 1:
        V_x(num_qubits)  # Reflection X
    elif g_idx == 2:
        V_y(num_qubits)  # Reflection Y
    elif g_idx == 3:  # Reflection XY (180 Rotation)
        V_x(num_qubits)
        V_y(num_qubits)
    elif g_idx == 4:  # Transpose (x-y swap)
        for i in range(half):
            qml.SWAP(wires=[i, i + half])
    elif g_idx == 5:  # 90 Rotation
        V_x(num_qubits)
        [qml.SWAP(wires=[i, i + half]) for i in range(half)]
    elif g_idx == 6:  # -90 Rotation
        V_y(num_qubits)
        [qml.SWAP(wires=[i, i + half]) for i in range(half)]
    elif g_idx == 7:  # Anti-diagonal reflection
        V_x(num_qubits)
        V_y(num_qubits)
        [qml.SWAP(wires=[i, i + half]) for i in range(half)]


def approx_equiv_measure(phi: torch.Tensor, p_err: float, num_qubits: int) -> None:
    for i in range(num_qubits):
        qml.RZ(phi, wires=i)
        if p_err != 0:
            qml.DepolarizingChannel(p_err, wires=i)
        qml.H(wires=i)
        if p_err != 0:
            qml.DepolarizingChannel(p_err, wires=i)


# --- QNode Factory ---

# Fixed (non-trainable) rotation angle for the frozen entanglers (RXY for
# config3/4, RYY/RYYYY for config5).
FROZEN_ENTANGLER_ANGLE = math.pi / 2

# The 9 supported architectures.
#
# "kind" selects how create_qnn builds the circuit:
#   - "uniform": the rotation+entangler+reps pattern below (config1-config5).
#   - "paper": the D4-matched block-schedule ansatzes from
#     src.paper_ansatzes (config6-config9) — a fixed 5-block schedule with a
#     small, tied ("group"-shared) set of trainable angles, unrelated to
#     "reps" (ignored for these architectures).
#
# "twirled" is the MECHANISM flag: it wraps the ansatz in explicit p4m
# group-twirling, averaged over the 8 group elements in qnn_forward.
# "is_equivariant" is the resulting PROPERTY: whether the built circuit is
# actually p4m-equivariant. They're decoupled because config6/config8
# achieve p4m-equivariance a different way — by construction, via
# generators that commute with the D4 group — without needing explicit
# twirling (see src.paper_ansatzes' module docstring).
#
# config3/4 use a frozen RXY entangler rather than CNOT: with a CNOT (or any
# entangler built only from I/X, e.g. RXX) the RX rotations get an *exactly*
# zero gradient — CNOT's Heisenberg conjugation maps X-type Paulis to X-type
# Paulis only, and RX(theta) is itself an I/X combination, so two operators
# built purely from I and X always commute, making the measured expectation
# value provably constant in theta (verified numerically: identical output
# to 1e-10 across a full sweep of theta). RXY breaks this for every qubit
# except the very first one in the chain (which only ever plays the "X" role
# in the wires=[i, i+1] convention below, so it still gets zero gradient).
ARCHITECTURES: dict[str, dict[str, Any]] = {
    "config1": {
        "kind": "uniform",
        "rotation_gate": "RY",
        "entangler": "cnot",
        "twirled": False,
        "is_equivariant": False,
    },
    "config2": {
        "kind": "uniform",
        "rotation_gate": "RY",
        "entangler": "cnot",
        "twirled": True,
        "is_equivariant": True,
    },
    "config3": {
        "kind": "uniform",
        "rotation_gate": "RX",
        "entangler": "frozen_rxy",
        "twirled": False,
        "is_equivariant": False,
    },
    "config4": {
        "kind": "uniform",
        "rotation_gate": "RX",
        "entangler": "frozen_rxy",
        "twirled": True,
        "is_equivariant": True,
    },
    "config5": {
        "kind": "uniform",
        "rotation_gate": "RX",
        "entangler": "frozen_ryy",
        "twirled": True,
        "is_equivariant": True,
    },
    "config6": {
        "kind": "paper",
        "paper_ansatz": "6",
        "symmetry": "equivariant",
        "twirled": False,
        "is_equivariant": True,
    },
    "config7": {
        "kind": "paper",
        "paper_ansatz": "6",
        "symmetry": "nonequivariant",
        "twirled": False,
        "is_equivariant": False,
    },
    "config8": {
        "kind": "paper",
        "paper_ansatz": "18",
        "symmetry": "equivariant",
        "twirled": False,
        "is_equivariant": True,
    },
    "config9": {
        "kind": "paper",
        "paper_ansatz": "18",
        "symmetry": "nonequivariant",
        "twirled": False,
        "is_equivariant": False,
    },
}


def frozen_rxy_cascade(num_qubits: int, p_err: float) -> None:
    """Cascade of fixed-angle XY rotations (PauliRot(pi/2, "XY")) over
    adjacent qubits — the entangler for config3/config4."""
    for i in range(num_qubits - 1):
        qml.PauliRot(FROZEN_ENTANGLER_ANGLE, "XY", wires=[i, i + 1])
        if p_err != 0:
            qml.DepolarizingChannel(p_err, wires=i)
            qml.DepolarizingChannel(p_err, wires=i + 1)


def frozen_ryy_cascade(num_qubits: int, cross_edge_index: int, p_err: float) -> None:
    """Cascade of fixed-angle RYY gates (IsingYY(pi/2)) over adjacent qubits.

    At the single step that would act on the two central qubits
    (i == cross_edge_index), it is replaced by one 4-qubit RYYYY
    (PauliRot(pi/2, "YYYY")) over the 4 central qubits instead.
    """
    for i in range(num_qubits - 1):
        if i == cross_edge_index:
            wires = [
                cross_edge_index - 1,
                cross_edge_index,
                cross_edge_index + 1,
                cross_edge_index + 2,
            ]
            qml.PauliRot(FROZEN_ENTANGLER_ANGLE, "YYYY", wires=wires)
            if p_err != 0:
                for w in wires:
                    qml.DepolarizingChannel(p_err, wires=w)
        else:
            qml.IsingYY(FROZEN_ENTANGLER_ANGLE, wires=[i, i + 1])
            if p_err != 0:
                qml.DepolarizingChannel(p_err, wires=i)
                qml.DepolarizingChannel(p_err, wires=i + 1)


def architecture_param_names(
    architecture: str, num_qubits: int, reps: int
) -> list[str]:
    """Names for the trainable-parameter tensor create_qnn's architecture
    needs — length matches what create_qnn(..., architecture) expects for
    its `params` argument. config1-config5 need num_qubits*reps
    independent rotation angles; config6-config9 have a fixed, tied
    parameter budget (6 or 18 total) and ignore `reps` entirely.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {sorted(ARCHITECTURES)}, got {architecture!r}"
        )
    spec = ARCHITECTURES[architecture]
    if spec["kind"] == "paper":
        # Local import: src.ansatz_builder imports ARCHITECTURES from this
        # module at top level, so importing it back here would be circular
        # if done at module scope.
        from src.ansatz_builder import param_labels
        from src.paper_ansatzes import paper_architecture_spec

        gate_spec = paper_architecture_spec(
            spec["paper_ansatz"], spec["symmetry"], num_qubits
        )
        return param_labels(gate_spec)
    return [f"rep{r}_q{i}" for r in range(reps) for i in range(num_qubits)]


def create_qnn(
    device: str,
    num_qubits: int,
    p_err: float,
    reps: int,
    architecture: str = "config1",
) -> Any:
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {sorted(ARCHITECTURES)}, got {architecture!r}"
        )
    spec = ARCHITECTURES[architecture]

    if spec["kind"] == "paper":
        # Local import — see architecture_param_names' comment above.
        from src.ansatz_builder import build_qnn_from_spec
        from src.paper_ansatzes import paper_architecture_spec

        gate_spec = paper_architecture_spec(
            spec["paper_ansatz"], spec["symmetry"], num_qubits
        )
        paper_qnn_forward, _, _ = build_qnn_from_spec(
            device, num_qubits, p_err, gate_spec, twirled=False, readout="x0_xhalf"
        )
        return paper_qnn_forward

    rotation_gate = spec["rotation_gate"]
    entangler = spec["entangler"]
    twirled = spec["twirled"]

    dev = qml.device(device, wires=num_qubits, shots=None)
    cross_edge_index = (num_qubits // 2) - 1

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def qnn_base(
        embedding_unitary: torch.Tensor,
        params: torch.Tensor,
        phi: torch.Tensor,
        g_idx: int = 0,
    ) -> Any:
        qml.QubitUnitary(embedding_unitary, wires=range(num_qubits))

        if twirled:
            apply_group_element(g_idx, num_qubits)
            if p_err != 0:
                for i in range(num_qubits):
                    qml.DepolarizingChannel(p_err, wires=i)

        for rep in range(reps):
            for i in range(num_qubits):
                if rotation_gate == "RX":
                    qml.RX(params[i + num_qubits * rep], wires=i)
                else:
                    qml.RY(params[i + num_qubits * rep], wires=i)
                if p_err != 0:
                    qml.DepolarizingChannel(p_err, wires=i)

            if entangler == "frozen_ryy":
                frozen_ryy_cascade(num_qubits, cross_edge_index, p_err)
                continue
            if entangler == "frozen_rxy":
                frozen_rxy_cascade(num_qubits, p_err)
                continue

            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                if p_err != 0:
                    qml.DepolarizingChannel(p_err, wires=i)
                    qml.DepolarizingChannel(p_err, wires=i + 1)

        if twirled:
            apply_group_element(g_idx, num_qubits)
            if p_err != 0:
                for i in range(num_qubits):
                    qml.DepolarizingChannel(p_err, wires=i)

        approx_equiv_measure(torch.tensor(0.0), p_err, num_qubits)

        coeffs = [1.0 / num_qubits] * num_qubits
        observables = [qml.Z(i) for i in range(num_qubits)]
        H = qml.Hamiltonian(coeffs, observables)
        return qml.expval(H)

    def qnn_forward(
        embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor
    ) -> Any:
        if twirled:
            results = [qnn_base(embedding_unitary, params, phi, g) for g in range(8)]
            return torch.stack(results).mean(dim=0)
        return qnn_base(embedding_unitary, params, phi, 0)

    # See src.ansatz_builder.build_qnn_from_spec's identical comment:
    # qml.draw()/qml.draw_mpl() need the actual QNode, not a wrapper function,
    # or the diagram silently truncates after the first few operations.
    qnn_forward.qnode = qnn_base  # type: ignore[attr-defined]

    return qnn_forward
