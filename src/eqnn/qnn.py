# qnn.py
import logging
from typing import Any, Literal

import pennylane as qml
import torch

logger = logging.getLogger(__name__)


def scanning_gate(p: torch.Tensor, w1: int, w2: int, p_err: float) -> None:
    qml.RX(p[0], wires=w1)
    qml.RX(p[1], wires=w2)
    if p_err != 0:
        qml.DepolarizingChannel(p_err, w1)
        qml.DepolarizingChannel(p_err, w2)

    qml.IsingYY(p[2], wires=[w1, w2])
    if p_err != 0:
        qml.DepolarizingChannel(p_err, w1)
        qml.DepolarizingChannel(p_err, w2)

    qml.RX(p[3], wires=w1)
    qml.RX(p[4], wires=w2)
    if p_err != 0:
        qml.DepolarizingChannel(p_err, w1)
        qml.DepolarizingChannel(p_err, w2)

    qml.IsingYY(p[5], wires=[w1, w2])
    if p_err != 0:
        qml.DepolarizingChannel(p_err, w1)
        qml.DepolarizingChannel(p_err, w2)


def scanning_phase(scanning_params: torch.Tensor, p_err: float) -> None:
    num_pairs = 4
    for i in range(num_pairs):
        scanning_gate(scanning_params, 2 * i, 2 * i + 1, p_err)

    index = 0
    wire_pairs = [[0, 3], [1, 2], [4, 7], [5, 6]]
    for pair in wire_pairs:
        w1 = pair[0]
        w2 = pair[1]
        scanning_gate(scanning_params, w1, w2, p_err)
        index += 6


def U4(params: torch.Tensor, wires: list[int], p_err: float) -> None:
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RX(params[0], wires=wires[2])
    qml.RX(params[1], wires=wires[3])
    if p_err != 0:
        qml.DepolarizingChannel(p_err, wires[0])
        qml.DepolarizingChannel(p_err, wires[1])
        qml.DepolarizingChannel(p_err, wires[2])
        qml.DepolarizingChannel(p_err, wires[3])

    qml.PauliRot(params[2], "YYYY", wires=wires)
    if p_err != 0:
        qml.DepolarizingChannel(p_err, wires[0])
        qml.DepolarizingChannel(p_err, wires[1])
        qml.DepolarizingChannel(p_err, wires[2])
        qml.DepolarizingChannel(p_err, wires[3])


def equiv_ansatz(ansatz_params: torch.Tensor, p_err: float) -> None:
    combinations = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 6, 7], [2, 3, 4, 5]]
    index = 0
    for comb in combinations:
        U4(ansatz_params, comb, p_err)
        index += 3

    for i in range(0, 8, 2):
        qml.CZ([i, i + 1])

    U4(ansatz_params, [1, 3, 5, 7], p_err)

    for i in [1, 5]:
        qml.CZ([i, i + 2])


def approx_equiv_ansatz(ansatz_params: torch.Tensor, p_err: float) -> None:
    combinations = [[0, 1], [2, 3], [4, 5], [6, 7], [1, 2], [3, 4], [5, 6]]
    for comb in combinations:
        scanning_gate(ansatz_params, w1=comb[0], w2=comb[1], p_err=p_err)

    for i in range(0, 8, 2):
        qml.CZ([i, i + 1])

    index = 0
    combinations = [[1, 3], [5, 7]]
    for comb in combinations:
        scanning_gate(ansatz_params, w1=comb[0], w2=comb[1], p_err=p_err)
        index += 6

    combinations = [[1, 7], [3, 5]]
    for comb in combinations:
        scanning_gate(ansatz_params, w1=comb[0], w2=comb[1], p_err=p_err)

    for i in [1, 5]:
        qml.CZ([i, i + 2])


def V_x() -> None:
    for i in range(4):
        qml.X(wires=i)


def V_y() -> None:
    for i in range(4, 8):
        qml.X(wires=i)


def V_r() -> None:
    V_x()
    for i in range(4):
        qml.SWAP(wires=[i, i + 4])


def equivariantor() -> None:
    all_wires = list(range(8))
    mat_x = qml.matrix(V_x, wire_order=all_wires)()
    mat_y = qml.matrix(V_y, wire_order=all_wires)()
    mat_r = qml.matrix(V_r, wire_order=all_wires)()

    K0_mat = torch.tensor(mat_x, dtype=torch.complex64)
    K1_mat = torch.tensor(mat_y, dtype=torch.complex64)
    K2_mat = torch.tensor(mat_r, dtype=torch.complex64)

    coeff = torch.sqrt(torch.tensor(1 / 3, dtype=torch.complex64))

    K0 = coeff * K0_mat
    K1 = coeff * K1_mat
    K2 = coeff * K2_mat

    qml.QubitChannel([K0, K1, K2], wires=all_wires)


def approx_equiv_measure(phi: torch.Tensor, p_err: float) -> None:
    for i in [3, 7]:
        qml.RZ(phi, wires=i)
        if p_err != 0:
            qml.DepolarizingChannel(p_err, wires=i)
        qml.H(wires=i)
        if p_err != 0:
            qml.DepolarizingChannel(p_err, wires=i)


def create_qnn(
    device: str,
    non_equivariance: Literal[0, 1, 2],
    p_err: float,
) -> qml.QNode:
    device = qml.device(device, wires=8, shots=None)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def qnn(
        embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor
    ) -> Any:
        qml.QubitUnitary(embedding_unitary, wires=range(8))
        scanning_params = params[0:6]
        ansatz_params = params[0:3]
        if non_equivariance in [0, 1, 2]:
            scanning_phase(scanning_params, p_err)
            if non_equivariance == 0:
                equiv_ansatz(ansatz_params, p_err)
                phi = torch.tensor(0.0, requires_grad=False)
                approx_equiv_measure(phi, p_err)
            elif non_equivariance == 1:
                approx_equiv_ansatz(scanning_params, p_err)
                phi = torch.tensor(0.0, requires_grad=False)
                approx_equiv_measure(phi, p_err)
            elif non_equivariance == 2:
                approx_equiv_ansatz(scanning_params, p_err)
                approx_equiv_measure(phi, p_err)
        elif non_equivariance == 3:
            for i in range(8):
                qml.RY(params[i], wires=i)
                if p_err != 0:
                    qml.DepolarizingChannel(p_err, wires=i)
            for i in range(7):
                qml.CNOT(wires=[i, i + 1])
                if p_err != 0:
                    qml.DepolarizingChannel(p_err, wires=i)
                    qml.DepolarizingChannel(p_err, wires=i + 1)
        elif non_equivariance == 4:
            equivariantor()
            for i in range(8):
                qml.DepolarizingChannel(p_err, wires=i)
            for i in range(8):
                qml.RY(params[i], wires=i)
                qml.DepolarizingChannel(p_err, wires=i)
            for i in range(7):
                qml.CNOT(wires=[i, i + 1])
            for i in range(8):
                qml.DepolarizingChannel(p_err, wires=i)
            equivariantor()
            for i in range(8):
                qml.DepolarizingChannel(p_err, wires=i)
        else:
            raise ValueError("non_equivariance must be one among 0,1,2,3,4")
        return [qml.expval(qml.Z(3)), qml.expval(qml.Z(7))]

    return qnn
