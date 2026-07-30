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

# Fixed (non-trainable) rotation angle for the frozen-RYY/RYYYY entangler
# used by config5.
FROZEN_ENTANGLER_ANGLE = math.pi / 2

# The 5 supported architectures. "twirled" wraps the ansatz in explicit p4m
# group-twirling, averaged over the 8 group elements in qnn_forward — that's
# what actually makes config2/4/5 p4m-equivariant (config1/3 are not).
ARCHITECTURES: dict[str, dict[str, Any]] = {
    "config1": {"rotation_gate": "RY", "entangler": "cnot", "twirled": False},
    "config2": {"rotation_gate": "RY", "entangler": "cnot", "twirled": True},
    "config3": {"rotation_gate": "RX", "entangler": "cnot", "twirled": False},
    "config4": {"rotation_gate": "RX", "entangler": "cnot", "twirled": True},
    "config5": {"rotation_gate": "RX", "entangler": "frozen_ryy", "twirled": True},
}


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
        g_idx: int,
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

    return qnn_forward
