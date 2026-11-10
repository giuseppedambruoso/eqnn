"""Independent checks of the dimensions computed exactly by eqnn.dla (results/dla.json).

1. DLA: the Lie closure of the same generators computed with PennyLane (qml.lie_closure, floating point).
2. Function classes: the number of independent features of each model estimated from the circuit simulator itself.
   For many random angle vectors of a deep circuit, the simulator gives the matrix Re(U^dag O U) (Heisenberg picture,
   eqnn.noise with the noise scale 0; for NonEquiv-Twirled its twirl). The dimension of their span is the numerical
   rank of the stacked matrices, read off a gap of many orders of magnitude in the singular values.

    python experiments/dla_checks.py > results/dla_checks.txt
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eqnn import noise  # noqa: E402
from eqnn.circuits import NUM_QUBITS, circuit_gates  # noqa: E402
from eqnn.models import d4_pixel_perms, twirl_matrix  # noqa: E402

LAYERS = 8  # deep enough to reach the whole span (48 independent angles)
SEED = 0


def pennylane_dla(equivariant: bool) -> int:
    import pennylane as qml
    from pennylane.pauli import PauliSentence, PauliWord
    col_rot, col_ent = ("X", "Y") if equivariant else ("Y", "Z")
    gens = []
    for a in range(4):  # X_a + X_a' (or X_a + Y_a')
        gens.append(PauliSentence({PauliWord({a: "X"}): 1.0, PauliWord({a + 4: col_rot}): 1.0}))
    for a, b in ((0, 1), (2, 3), (1, 2), (3, 0), (0, 2)):  # Y_a Y_b Y_a' Y_b' (or Y_a Y_b Z_a' Z_b')
        gens.append(PauliSentence({PauliWord({a: "Y", b: "Y", a + 4: col_ent, b + 4: col_ent}): 1.0}))
    return len(qml.lie_closure(gens, pauli=True))


def feature_rank(arch: str, n_samples: int, layers: int = LAYERS):
    """Numerical rank of {Re(U^dag O U)} over n_samples random angle vectors, and the singular values around it."""
    equivariant = arch in ("Equiv", "NoSym")
    terms = noise.READOUT_NOSYM if arch == "NoSym" else noise.READOUT_O
    prog = noise.build(circuit_gates(equivariant, layers), NUM_QUBITS, 0.0, 0, True, False, "heisenberg", terms)
    perms = d4_pixel_perms() if arch == "Twirled" else None
    iu = np.triu_indices(2 ** NUM_QUBITS)
    rng = torch.Generator().manual_seed(SEED)
    rows = []
    with torch.no_grad():
        for _ in range(n_samples):
            th = torch.empty(6 * layers, dtype=torch.float64).uniform_(-np.pi, np.pi, generator=rng)
            m = prog.observable(th).real
            if perms is not None:
                m = twirl_matrix(m, perms)
            rows.append(m.numpy()[iu])
    x = np.asarray(rows)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)  # singular values, descending (directly, not from the Gram matrix,
    # whose eigenvalues would carry a rounding error of order 1e-16 and so singular values of order 1e-8)
    rank = int((s > s[0] * 1e-10).sum())
    return rank, s


def outside_exact_space(n_samples: int = 60) -> float:
    """Largest relative norm of the component of simulated features Re(U^dag O U) of Equiv outside the feature space
    computed exactly by eqnn.dla (converted to dense matrices)."""
    from fractions import Fraction
    from eqnn import dla as D
    mats = {(0, 0): np.eye(2), (1, 0): np.array([[0, 1], [1, 0]]), (1, 1): np.array([[0, -1j], [1j, 0]]),
            (0, 1): np.diag([1.0, -1.0])}

    def dense(key):
        out = np.ones((1, 1), dtype=complex)
        for w in range(NUM_QUBITS):  # wire 0 is the most significant bit (the left factor)
            out = np.kron(out, mats[((key[0] >> w) & 1, (key[1] >> w) & 1)])
        return out

    obs = D.add(D.P({0: "X"}, Fraction(1, 2)), D.P({4: "X"}, Fraction(1, 2)))
    feats = [v for v in (D.real_part(m) for m in D.module_basis(D.eq, obs)) if v]
    iu = np.triu_indices(2 ** NUM_QUBITS)
    basis = np.array([sum(float(c) * dense(k) for k, c in v.items()).real[iu] for v in feats])
    q, _ = np.linalg.qr(basis.T)
    prog = noise.build(circuit_gates(True, LAYERS), NUM_QUBITS, 0.0, 0, True, False, "heisenberg", noise.READOUT_O)
    rng = torch.Generator().manual_seed(SEED)
    worst = 0.0
    with torch.no_grad():
        for _ in range(n_samples):
            th = torch.empty(6 * LAYERS, dtype=torch.float64).uniform_(-np.pi, np.pi, generator=rng)
            v = prog.observable(th).real.numpy()[iu]
            worst = max(worst, np.linalg.norm(v - q @ (q.T @ v)) / np.linalg.norm(v))
    return worst


def main():
    exact = json.load(open("results/dla.json"))
    print("== 0. the simulated observables lie in the exactly computed space (Equiv, L = 8, 60 random angle vectors)")
    print(f"  largest relative component outside the space: {outside_exact_space():.1e}", flush=True)
    print("\n== 1. dynamical Lie algebra: exact (eqnn.dla) vs PennyLane qml.lie_closure")
    for eq, name in ((True, "equivariant circuit"), (False, "non-equivariant circuit")):
        t0 = time.time()
        d = pennylane_dla(eq)
        ref = exact["dla"] if eq else exact["dla_nonequiv"]
        print(f"  {name}: PennyLane {d}, exact {ref} ({time.time() - t0:.0f} s)", flush=True)
    print(f"\n== 2. size of the function class: exact vs numerical rank of Re(U^dag O U) over random angles "
          f"(L = {LAYERS})")
    for arch in ("Equiv", "Twirled", "NoSym", "NonEquiv"):
        t0 = time.time()
        n = int(exact["dim"][arch] * 1.25) + 50
        rank, s = feature_rank(arch, n)
        k = exact["dim"][arch]
        print(f"  {arch}: numerical rank {rank} from {n} samples, exact {k}; singular values (relative) "
              f"s_{k} = {s[k - 1] / s[0]:.1e}, s_{k + 1} = {s[k] / s[0]:.1e} ({time.time() - t0:.0f} s)", flush=True)
    print("\n== 3. how many layers reach the whole function class (numerical rank of Re(U^dag O U), random angles)")
    for arch in ("Equiv", "Twirled", "NoSym", "NonEquiv"):
        n = int(exact["dim"][arch] * 1.25) + 50
        ranks = [feature_rank(arch, n, L)[0] for L in (1, 2, 4)]
        print(f"  {arch}: L = 1, 2, 4: {ranks[0]}, {ranks[1]}, {ranks[2]} (all circuits: {exact['dim'][arch]})",
              flush=True)


if __name__ == "__main__":
    main()
