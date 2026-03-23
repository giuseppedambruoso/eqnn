# qnn.py
import logging
import os
import time
from typing import Any, Literal

import pennylane as qml
import torch
from torch.nn import functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

# --- Sottogruppi e Porte di Base ---

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

    wire_pairs = [[0, 3], [1, 2], [4, 7], [5, 6]]
    for pair in wire_pairs:
        scanning_gate(scanning_params, pair[0], pair[1], p_err)


def U4(params: torch.Tensor, wires: list[int], p_err: float) -> None:
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RX(params[0], wires=wires[2])
    qml.RX(params[1], wires=wires[3])
    if p_err != 0:
        for w in wires: qml.DepolarizingChannel(p_err, wires=w)

    qml.PauliRot(params[2], "YYYY", wires=wires)
    if p_err != 0:
        for w in wires: qml.DepolarizingChannel(p_err, wires=w)


def equiv_ansatz(ansatz_params: torch.Tensor, p_err: float) -> None:
    combinations = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 6, 7], [2, 3, 4, 5], [1, 3, 5, 7], [0, 2, 4, 6]]
    for comb in combinations:
        U4(ansatz_params, comb, p_err)


def approx_equiv_ansatz(ansatz_params: torch.Tensor, p_err: float) -> None:
    combinations = [[0, 1], [2, 3], [4, 5], [6, 7], [1, 2], [3, 4], [5, 6], [7, 0]]
    for comb in combinations:
        scanning_gate(ansatz_params, w1=comb[0], w2=comb[1], p_err=p_err)

# --- Simmetrie p4m (D4) ---

def V_x() -> None:
    for i in range(4): qml.X(wires=i)

def V_y() -> None:
    for i in range(4, 8): qml.X(wires=i)

def apply_group_element(g_idx: int) -> None:
    """Applica uno degli 8 elementi del gruppo p4m per il twirling esatto."""
    if g_idx == 0: pass # Identità
    elif g_idx == 1: V_x() # Riflessione X
    elif g_idx == 2: V_y() # Riflessione Y
    elif g_idx == 3: # Riflessione XY (Rotazione 180)
        V_x(); V_y()
    elif g_idx == 4: # Trasposta (Scambio x-y)
        for i in range(4): qml.SWAP(wires=[i, i + 4])
    elif g_idx == 5: # Rotazione 90
        V_x(); [qml.SWAP(wires=[i, i + 4]) for i in range(4)]
    elif g_idx == 6: # Rotazione -90
        V_y(); [qml.SWAP(wires=[i, i + 4]) for i in range(4)]
    elif g_idx == 7: # Riflessione anti-diagonale
        V_x(); V_y(); [qml.SWAP(wires=[i, i + 4]) for i in range(4)]

def approx_equiv_measure(phi: torch.Tensor, p_err: float) -> None:
    for i in range(8):
        qml.RZ(phi, wires=i)
        if p_err != 0: qml.DepolarizingChannel(p_err, wires=i)
        qml.H(wires=i)
        if p_err != 0: qml.DepolarizingChannel(p_err, wires=i)

# --- QNode e Factory ---

def create_qnn(
    device: str,
    non_equivariance: Literal[0, 1, 2, 3, 4],
    p_err: float,
    reps: int,
) -> Any:
    dev = qml.device(device, wires=8, shots=None)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnn_base(
        embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor, g_idx: int
    ) -> Any:
        qml.QubitUnitary(embedding_unitary, wires=range(8))
        
        # Caso Equivariante (Twirling)
        if non_equivariance == 4:
            apply_group_element(g_idx)
            # Rumore iniziale se presente
            if p_err != 0: [qml.DepolarizingChannel(p_err, wires=i) for i in range(8)]
            # Circuito base (non_equiv=3)
            for rep in range(reps):
                for i in range(8):
                    qml.RY(params[i], wires=i)
                    if p_err != 0: qml.DepolarizingChannel(p_err, wires=i)
                for i in range(7):
                    qml.CNOT(wires=[i, i + 1])
                    if p_err != 0:
                        qml.DepolarizingChannel(p_err, wires=i)
                        qml.DepolarizingChannel(p_err, wires=i+1)
            # Twirling output
            apply_group_element(g_idx)
            if p_err != 0: [qml.DepolarizingChannel(p_err, wires=i) for i in range(8)]
            phi = torch.tensor(0.0, requires_grad=False)
            approx_equiv_measure(phi, p_err)

        # Caso Standard (non-equiv o simmetrie incorporate)
        elif non_equivariance in [0, 1, 2]:
            scanning_params = params[0:6]
            ansatz_params = params[0:3]
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
            for rep in range(reps):
                for i in range(8):
                    qml.RY(params[i], wires=i)
                    if p_err != 0: qml.DepolarizingChannel(p_err, wires=i)
                for i in range(7):
                    qml.CNOT(wires=[i, i + 1])
                    if p_err != 0:
                        qml.DepolarizingChannel(p_err, wires=i)
                        qml.DepolarizingChannel(p_err, wires=i+1)
            phi = torch.tensor(0.0, requires_grad=False)
            approx_equiv_measure(torch.tensor(0.0),p_err)

        # Misurazione Invariante
        coeffs = [1.0 / 8.0] * 8
        observables = [qml.Z(i) for i in range(8)]
        H = qml.Hamiltonian(coeffs, observables)
        return qml.expval(H)

    def qnn_twirled(embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor) -> Any:
        if non_equivariance == 4:
            # Twirling esatto su tutti gli 8 elementi
            results = [qnn_base(embedding_unitary, params, phi, g) for g in range(8)]
            return torch.stack(results).mean(dim=0)
        return qnn_base(embedding_unitary, params, phi, 0)

    return qnn_twirled
