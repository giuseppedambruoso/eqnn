# qnn.py
import logging
from typing import Any, Literal

import pennylane as qml
import torch

logger = logging.getLogger(__name__)


def scanning_gate(p: torch.Tensor, w1: int, w2: int) -> None:
    qml.RX(p[0], wires=w1)
    qml.RX(p[1], wires=w2)
    qml.IsingYY(p[2], wires=[w1, w2])
    qml.RX(p[3], wires=w1)
    qml.RX(p[4], wires=w2)
    qml.IsingYY(p[5], wires=[w1, w2])


def scanning_phase(scanning_params: torch.Tensor) -> None:
    num_pairs = 4
    for i in range(num_pairs):
        scanning_gate(scanning_params[6 * i : 6 * i + 6], 2 * i, 2 * i + 1)

    index = 0
    wire_pairs = [[0, 3], [1, 2], [4, 7], [5, 6]]
    for pair in wire_pairs:
        w1 = pair[0]
        w2 = pair[1]
        scanning_gate(scanning_params[index : index + 6], w1, w2)
        index += 6


def U4(params: torch.Tensor, wires: list[int]) -> None:
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RX(params[0], wires=wires[2])
    qml.RX(params[1], wires=wires[3])
    qml.PauliRot(params[2], "YYYY", wires=wires)


def equiv_ansatz(ansatz_params: torch.Tensor) -> None:
    combinations = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 6, 7], [2, 3, 4, 5]]
    index = 0
    for comb in combinations:
        U4(ansatz_params[index : index + 3], comb)
        index += 3

    for i in range(0, 8, 2):
        qml.CZ([i, i + 1])

    U4(ansatz_params[3:6], [1, 3, 5, 7])

    for i in [1, 5]:
        qml.CZ([i, i + 2])


def approx_equiv_ansatz(ansatz_params: torch.Tensor) -> None:
    combinations = [[0, 1], [2, 3], [4, 5], [6, 7]]
    index = 0
    for comb in combinations:
        scanning_gate(ansatz_params[index : index + 6], w1=comb[0], w2=comb[1])
        index += 6

    combinations = [[1, 2], [3, 4], [5, 6]]
    index = 0
    for comb in combinations:
        scanning_gate(ansatz_params[index : index + 6], w1=comb[0], w2=comb[1])
        index += 6

    for i in range(0, 8, 2):
        qml.CZ([i, i + 1])

    index = 0
    combinations = [[1, 3], [5, 7]]
    for comb in combinations:
        scanning_gate(ansatz_params[index : index + 6], w1=comb[0], w2=comb[1])
        index += 6

    index = 0
    combinations = [[1, 3], [5, 7]]
    for comb in combinations:
        scanning_gate(ansatz_params[index : index + 6], w1=comb[0], w2=comb[1])
        index += 6

    for i in [1, 5]:
        qml.CZ([i, i + 2])


def approx_equiv_measure(phi: torch.Tensor) -> None:
    for i in [3, 7]:
        qml.RX(phi, wires=i)
        qml.H(wires=i)


def create_qnn(
    embedding_unitary: torch.Tensor, device: str, non_equivariance: Literal[0, 1, 2]
) -> qml.QNode:
    device = qml.device(device, wires=8)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def qnn(params: torch.Tensor, phi: torch.Tensor) -> Any:
        qml.QubitUnitary(embedding_unitary, wires=range(8))
        scanning_params = params[0:24]
        ansatz_params = params[24:48]

        scanning_phase(scanning_params)
        if non_equivariance == 0:
            equiv_ansatz(ansatz_params)
            phi = torch.tensor(0.0, requires_grad=False)
            approx_equiv_measure(phi)
        elif non_equivariance == 1:
            approx_equiv_ansatz(ansatz_params)
            phi = torch.tensor(0.0, requires_grad=False)
            approx_equiv_measure(phi)
        elif non_equivariance == 2:
            approx_equiv_ansatz(ansatz_params)
            approx_equiv_measure(phi)
        return [qml.expval(qml.Z(3)), qml.expval(qml.Z(7))]

    return qnn
