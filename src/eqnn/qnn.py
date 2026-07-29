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

def compiled_cnot(c: int, t: int, p_err: float) -> None:
    """
    Compiles an exact CNOT gate using ONLY X, Y, and YY rotations.
    Mathematically: CNOT = RY_t(pi/2) [RZ_c(-pi/2) RZ_t(-pi/2) E^{ZZ}] RY_t(-pi/2)
    """
    # 1. Target basis change BEFORE CZ
    qml.RY(-math.pi/2, wires=t)
    
    # 2. Synthesize E^{ZZ} from E^{YY}[cite: 9]
    qml.RX(math.pi/2, wires=c)
    qml.RX(math.pi/2, wires=t)
    
    # IsingYY(pi/2) applies exp(-i * pi/4 * Y_c Y_t)[cite: 9]
    qml.IsingYY(math.pi/2, wires=[c, t])
    
    qml.RX(-math.pi/2, wires=c)
    qml.RX(-math.pi/2, wires=t)
    
    # 3. Synthesize local RZ(-pi/2) rotations from RX and RY
    # RZ(-pi/2) = RX(pi/2) RY(-pi/2) RX(-pi/2) (applied top-to-bottom in PennyLane)[cite: 9]
    qml.RX(-math.pi/2, wires=c)
    qml.RY(-math.pi/2, wires=c)
    qml.RX(math.pi/2, wires=c)
    
    qml.RX(-math.pi/2, wires=t)
    qml.RY(-math.pi/2, wires=t)
    qml.RX(math.pi/2, wires=t)
    
    # 4. Target basis change AFTER CZ
    qml.RY(math.pi/2, wires=t)

    # Apply noise after the entire compiled block
    if p_err != 0:
        qml.DepolarizingChannel(p_err, wires=c)
        qml.DepolarizingChannel(p_err, wires=t)


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

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def qnn_base(
        embedding_unitary: torch.Tensor, params: torch.Tensor, phi: torch.Tensor, g_idx: int
    ) -> Any:
        qml.QubitUnitary(embedding_unitary, wires=range(8))
        
        # Optional Explicit Twirling Initial
        if equivariance and twirling:
            apply_group_element(g_idx)
            if p_err != 0: 
                for i in range(8): qml.DepolarizingChannel(p_err, wires=i)
        
        # CNOT Staircase (Raw or Compiled)
        for rep in range(reps):
            for i in range(8):
                qml.RY(params[i+8*rep], wires=i)
                if p_err != 0: qml.DepolarizingChannel(p_err, wires=i)
                
            for i in range(7):
                # Skip the cross edge if requested
                if remove_cross_edge and i == 3:
                    continue
                    
                # If Equivariance is True and Twirling is False, use the Compiled X/Y/YY CNOT
                if equivariance and not twirling:
                    compiled_cnot(i, i+1, p_err)
                else:
                    qml.CNOT(wires=[i, i+1])
                    if p_err != 0:
                        qml.DepolarizingChannel(p_err, wires=i)
                        qml.DepolarizingChannel(p_err, wires=i+1)

        # Optional Explicit Twirling Final
        if equivariance and twirling:
            apply_group_element(g_idx)
            if p_err != 0: 
                for i in range(8): qml.DepolarizingChannel(p_err, wires=i)
            phi = torch.tensor(0.0, requires_grad=False)
            approx_equiv_measure(phi, p_err)
        else:
            # Raw output 
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
            # Single run for equivariance=False or Compiled CNOTs
            return qnn_base(embedding_unitary, params, phi, 0)
            
    return qnn_forward
