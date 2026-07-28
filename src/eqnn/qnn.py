# qnn.py
import logging
import math
from typing import Any

import pennylane as qml
import torch

logger = logging.getLogger(__name__)

# --- Symmetries p4m (D4) ---

def V_x() -> None:
    for i in range(4): qml.X(wires=i)

def V_y() -> None:
    for i in range(4, 8): qml.X(wires=i)

def apply_group_element(g_idx: int) -> None:
    """Applies one of the 8 p4m group elements for explicit twirling."""
    if g_idx == 0: pass # Identity
    elif g_idx == 1: V_x() # Reflection X
    elif g_idx == 2: V_y() # Reflection Y
    elif g_idx == 3: # Reflection XY (180 Rotation)
        V_x(); V_y()
    elif g_idx == 4: # Transpose (x-y swap)
        for i in range(4): qml.SWAP(wires=[i, i + 4])
    elif g_idx == 5: # 90 Rotation
        V_x(); [qml.SWAP(wires=[i, i + 4]) for i in range(4)]
    elif g_idx == 6: # -90 Rotation
        V_y(); [qml.SWAP(wires=[i, i + 4]) for i in range(4)]
    elif g_idx == 7: # Anti-diagonal reflection
        V_x(); V_y(); [qml.SWAP(wires=[i, i + 4]) for i in range(4)]

def approx_equiv_measure(phi: torch.Tensor, p_err: float) -> None:
    for i in range(8):
        qml.RZ(phi, wires=i)
        if p_err != 0: qml.DepolarizingChannel(p_err, wires=i)
        qml.H(wires=i)
        if p_err != 0: qml.DepolarizingChannel(p_err, wires=i)

# --- QNode Factory ---

def create_qnn(
    device: str,
    p_err: float,
    reps: int,
    equivariance: bool = False,
    twirling: bool = False,
    remove_cross_edge: bool = False
) -> Any:
    dev = qml.device(device, wires=8, shots=None)

    # ---------------------------------------------------------
    # Equivariant = True, Twirling = False
    # (Reconstruction via X, Y, YY)
    # ---------------------------------------------------------
    if equivariance and not twirling:
        @qml.qnode(dev, interface="torch", diff_method="best")
        def edge_readout_j(embedding_unitary: torch.Tensor, params: torch.Tensor, j: int) -> Any:
            qml.QubitUnitary(embedding_unitary, wires=range(8))
            if p_err != 0: [qml.DepolarizingChannel(p_err, wires=i) for i in range(8)]
            
            # Apply L(t_j) = RY(t_j - pi/2)
            qml.RY(params[j] - math.pi/2, wires=j)
            if p_err != 0: qml.DepolarizingChannel(p_err, wires=j)
            
            # Apply M(t_{j+1}) = RX(pi/2) RY(pi/2) RX(-pi/2) RY(t_{j+1})
            qml.RY(params[j+1], wires=j+1)
            qml.RX(-math.pi/2, wires=j+1)
            qml.RY(math.pi/2, wires=j+1)
            qml.RX(math.pi/2, wires=j+1)
            if p_err != 0: qml.DepolarizingChannel(p_err, wires=j+1)
            
            # Apply fixed YY entangler E^{YY}_{j, j+1}
            qml.IsingYY(math.pi/2, wires=[j, j+1])
            if p_err != 0: 
                qml.DepolarizingChannel(p_err, wires=j)
                qml.DepolarizingChannel(p_err, wires=j+1)
                
            return qml.expval(qml.X(j))

        @qml.qnode(dev, interface="torch", diff_method="best")
        def node_readout_7(embedding_unitary: torch.Tensor, params: torch.Tensor) -> Any:
            qml.QubitUnitary(embedding_unitary, wires=range(8))
            if p_err != 0: [qml.DepolarizingChannel(p_err, wires=i) for i in range(8)]
            
            # Apply final one-qubit rotation R_Y(t_7)
            qml.RY(params[7], wires=7)
            if p_err != 0: qml.DepolarizingChannel(p_err, wires=7)
            
            return qml.expval(qml.X(7))
            
        def qnn_forward(embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor) -> Any:
            total = 0.0
            for j in range(7):
                # Skip edge j=3 (connecting qubit 3 and qubit 4) if requested
                if remove_cross_edge and j == 3:
                    continue
                total = total + edge_readout_j(embedding_unitary, params, j)
                
            total = total + node_readout_7(embedding_unitary, params)
            return total / 8.0 

        return qnn_forward

    # ---------------------------------------------------------
    # Explicit Twirling (Equivariance = True, Twirling = True) OR
    # Raw CNOT Staircase (Equivariance = False)
    # ---------------------------------------------------------
    else:
        @qml.qnode(dev, interface="torch", diff_method="best")
        def qnn_base(
            embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor, g_idx: int
        ) -> Any:
            qml.QubitUnitary(embedding_unitary, wires=range(8))
            
            # Optional Explicit Twirling Initial
            if equivariance and twirling:
                apply_group_element(g_idx)
                if p_err != 0: [qml.DepolarizingChannel(p_err, wires=i) for i in range(8)]
            
            # CNOT Staircase
            for rep in range(reps):
                for i in range(8):
                    qml.RY(params[i+8*rep], wires=i)
                    if p_err != 0: qml.DepolarizingChannel(p_err, wires=i)
                for i in range(7):
                    # Skip CNOT_{3,4} if requested
                    if remove_cross_edge and i == 3:
                        continue
                    qml.CNOT(wires=[i, i + 1])
                    if p_err != 0:
                        qml.DepolarizingChannel(p_err, wires=i)
                        qml.DepolarizingChannel(p_err, wires=i+1)

            # Optional Explicit Twirling Final
            if equivariance and twirling:
                apply_group_element(g_idx)
                if p_err != 0: [qml.DepolarizingChannel(p_err, wires=i) for i in range(8)]
                phi = torch.tensor(0.0, requires_grad=False)
                approx_equiv_measure(phi, p_err)
            else:
                # Raw output (equivariance = False)
                phi = torch.tensor(0.0, requires_grad=False)
                approx_equiv_measure(torch.tensor(0.0), p_err)

            # Invariant Measurement
            coeffs = [1.0 / 8.0] * 8
            observables = [qml.Z(i) for i in range(8)]
            H = qml.Hamiltonian(coeffs, observables)
            return qml.expval(H)

        def qnn_forward(embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor) -> Any:
            if equivariance and twirling:
                # Sequential 8-group elements averaging for explicit twirling
                results = []
                for g in range(8):
                    res = qnn_base(embedding_unitary, params, phi, g)
                    results.append(res)
                return torch.stack(results).mean(dim=0)
            else:
                # Single run for equivariance=False
                return qnn_base(embedding_unitary, params, phi, 0)
                
        return qnn_forward
