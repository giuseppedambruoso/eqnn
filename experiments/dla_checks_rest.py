"""Remaining checks: (a) numerical rank with one angle per gate of the gate set (pairs tied, blocks independent);
(b) rank vs depth for NoSym and NonEquiv."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
from eqnn import noise
from eqnn.circuits import circuit_gates
from experiments.dla_checks import feature_rank
from eqnn.models import d4_pixel_perms, twirl_matrix
iu = np.triu_indices(256); perms = d4_pixel_perms()
exact = json.load(open("results/dla.json"))
POS = (0, 0, 1, 1, 2)
def rank_free(eq, twirl, terms, n, L=4):
    base = circuit_gates(eq, L)
    gates = [(w, 3 * (i // 5) + POS[i % 5]) for i, (w, k) in enumerate(base)]
    prog = noise.build(gates, 8, 0.0, 0, True, False, "heisenberg", terms)
    g = torch.Generator().manual_seed(1); rows = []
    with torch.no_grad():
        for _ in range(n):
            th = torch.empty(3 * len(base) // 5, dtype=torch.float64).uniform_(-np.pi, np.pi, generator=g)
            m = prog.observable(th).real
            if twirl: m = twirl_matrix(m, perms)
            rows.append(m.numpy()[iu])
    x = np.array(rows); x /= np.linalg.norm(x, axis=1, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    return int((s > s[0] * 1e-10).sum())
print("== (a) one angle per gate of the gate set (pairs tied, blocks independent), L = 4", flush=True)
for arch, eq, tw, terms in (("Twirled", False, True, noise.READOUT_O), ("NoSym", True, False, noise.READOUT_NOSYM),
                            ("NonEquiv", False, False, noise.READOUT_O)):
    t = time.time(); n = int(exact["dim"][arch] * 1.25) + 50
    print(f"  {arch}: {rank_free(eq, tw, terms, n)} (exact {exact['dim'][arch]}, {time.time()-t:.0f}s)", flush=True)
print("== (b) rank vs depth, shared angles", flush=True)
for arch in ("NoSym", "NonEquiv"):
    n = int(exact["dim"][arch] * 1.25) + 50
    print(f"  {arch}: L = 1, 2, 4: {[feature_rank(arch, n, L)[0] for L in (1, 2, 4)]} (exact {exact['dim'][arch]})", flush=True)
