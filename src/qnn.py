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


def compiled_cnot(c: int, t: int, p_err: float) -> None:
    """Compiles an exact CNOT gate using ONLY X, Y, and YY rotations."""
    qml.RY(-math.pi / 2, wires=t)
    qml.RX(math.pi / 2, wires=c)
    qml.RX(math.pi / 2, wires=t)
    qml.IsingYY(math.pi / 2, wires=[c, t])
    qml.RX(-math.pi / 2, wires=c)
    qml.RX(-math.pi / 2, wires=t)
    qml.RX(-math.pi / 2, wires=c)
    qml.RY(-math.pi / 2, wires=c)
    qml.RX(math.pi / 2, wires=c)
    qml.RX(-math.pi / 2, wires=t)
    qml.RY(-math.pi / 2, wires=t)
    qml.RX(math.pi / 2, wires=t)
    qml.RY(math.pi / 2, wires=t)

    if p_err != 0:
        qml.DepolarizingChannel(p_err, wires=c)
        qml.DepolarizingChannel(p_err, wires=t)


# --- QNode Factory ---


def create_qnn(
    device: str,
    num_qubits: int,
    p_err: float,
    reps: int,
    equivariance: bool = False,
    twirling: bool = False,
    remove_cross_edge: bool = False,
) -> Any:
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

        if equivariance and twirling:
            apply_group_element(g_idx, num_qubits)
            if p_err != 0:
                for i in range(num_qubits):
                    qml.DepolarizingChannel(p_err, wires=i)

        for rep in range(reps):
            for i in range(num_qubits):
                qml.RY(params[i + num_qubits * rep], wires=i)
                if p_err != 0:
                    qml.DepolarizingChannel(p_err, wires=i)

            for i in range(num_qubits - 1):
                if remove_cross_edge and i == cross_edge_index:
                    continue
                if equivariance and not twirling:
                    compiled_cnot(i, i + 1, p_err)
                else:
                    qml.CNOT(wires=[i, i + 1])
                    if p_err != 0:
                        qml.DepolarizingChannel(p_err, wires=i)
                        qml.DepolarizingChannel(p_err, wires=i + 1)

        if equivariance and twirling:
            apply_group_element(g_idx, num_qubits)
            if p_err != 0:
                for i in range(num_qubits):
                    qml.DepolarizingChannel(p_err, wires=i)
            phi = torch.tensor(0.0, requires_grad=False)
            approx_equiv_measure(phi, p_err, num_qubits)
        else:
            phi = torch.tensor(0.0, requires_grad=False)
            approx_equiv_measure(torch.tensor(0.0), p_err, num_qubits)

        coeffs = [1.0 / num_qubits] * num_qubits
        observables = [qml.Z(i) for i in range(num_qubits)]
        H = qml.Hamiltonian(coeffs, observables)
        return qml.expval(H)

    def qnn_forward(
        embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor
    ) -> Any:
        if equivariance and twirling:
            results = [qnn_base(embedding_unitary, params, phi, g) for g in range(8)]
            return torch.stack(results).mean(dim=0)
        return qnn_base(embedding_unitary, params, phi, 0)

    return qnn_forward
