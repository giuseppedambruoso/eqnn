"""Pilot runs used to design the synthetic task (paper: appendix "Design of the synthetic task";
results/synthetic_pilots/).

  v0        the first version: small shapes on a strong background noise (generator make_shapes_v0 below); no model
            learned it. The campaign (Equiv and NonEquiv, L in {1, 2, 4}, N in {10, 20, 40, 80}, seeds 1-5) was
            stopped after the first, longest runs: those in v0_small_noisy.jsonl.
  A ... M   variants of the final generator (eqnn.data.make_shapes) with large shapes and weak noise, compared in
            three rounds (field "pilot" of variants.jsonl). K is the task of the paper.

    python experiments/pilots.py variants   # appends to results/synthetic_pilots/variants.jsonl
    python experiments/pilots.py v0         # appends to results/synthetic_pilots/v0_small_noisy.jsonl
"""

import itertools
import json
import os
import sys
from multiprocessing import Pool

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eqnn.data as D  # noqa: E402

OUT = "results/synthetic_pilots"
VARIANTS = {  # generator parameters (eqnn.data.make_shapes); A_big_filled (round 1) and A_base are the same variant A
    "A_big_filled": dict(square_sides=(8, 14), cross_spans=(9, 15), cross_thick=(2, 4), hollow=False, noise=0.05),
    "A_base": dict(square_sides=(8, 14), cross_spans=(9, 15), cross_thick=(2, 4), hollow=False, noise=0.05),
    "B_big_hollow": dict(square_sides=(8, 14), cross_spans=(9, 15), cross_thick=(2, 4), hollow=True, noise=0.05),
    "C_medium": dict(square_sides=(5, 10), cross_spans=(7, 11), cross_thick=(1, 2), hollow=False, noise=0.05),
    "D_big_filled_nonoise": dict(square_sides=(8, 14), cross_spans=(9, 15), cross_thick=(2, 4), hollow=False, noise=0.0),
    "E_wide_sizes": dict(square_sides=(6, 14), cross_spans=(7, 15), cross_thick=(1, 4), hollow=False, noise=0.05),
    "F_noise10": dict(square_sides=(8, 14), cross_spans=(9, 15), cross_thick=(2, 4), hollow=False, noise=0.10),
    "G_thin_cross": dict(square_sides=(8, 14), cross_spans=(9, 15), cross_thick=(1, 2), hollow=False, noise=0.05),
    "H_small_big": dict(square_sides=(6, 11), cross_spans=(7, 13), cross_thick=(2, 3), hollow=False, noise=0.05),
    "I_noise10_wide": dict(square_sides=(6, 14), cross_spans=(7, 15), cross_thick=(1, 4), hollow=False, noise=0.10),
    "J_small_thin": dict(square_sides=(6, 11), cross_spans=(7, 13), cross_thick=(1, 2), hollow=False, noise=0.05),
    "K_thick1": dict(square_sides=(8, 14), cross_spans=(9, 15), cross_thick=(1, 1), hollow=False, noise=0.05),
    "M_mid_thin": dict(square_sides=(7, 12), cross_spans=(8, 14), cross_thick=(1, 2), hollow=False, noise=0.05),
}
# the three rounds: (pilot, variants, seeds, [(layers, N values)])
ROUNDS = [
    (1, ("A_big_filled", "B_big_hollow", "C_medium", "D_big_filled_nonoise"), (1, 2, 3), [(2, (40,))]),
    (2, ("A_base", "E_wide_sizes", "F_noise10", "G_thin_cross", "H_small_big", "I_noise10_wide"), (1, 2, 3, 4, 5),
     [(2, (20, 40))]),
    (3, ("G_thin_cross", "H_small_big", "J_small_thin", "K_thick1", "M_mid_thin"), (1, 2, 3, 4, 5),
     [(2, (40, 80, 160)), (4, (20, 40, 80))]),
]


def make_shapes_v0(n: int, seed: int, noise: float = 0.15):
    """First version of the synthetic task: a filled square of side 2-5 (class 0) or a plus-shaped cross with arms of
    1-3 pixels, one pixel thick (class 1), at a uniform position, on the background |N(0, 0.15^2)|."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 2
    labels = labels[torch.randperm(n, generator=g)]
    imgs = (torch.randn(n, D.SIDE, D.SIDE, generator=g, dtype=torch.float64) * noise).abs()
    for i in range(n):
        v = 0.5 + 0.5 * torch.rand(1, generator=g, dtype=torch.float64).item()
        if int(labels[i]) == 0:
            side = int(torch.randint(2, 6, (1,), generator=g))
            r, c = (int(t) for t in torch.randint(0, D.SIDE - side + 1, (2,), generator=g))
            imgs[i, r:r + side, c:c + side] = v
        else:
            arm = int(torch.randint(1, 4, (1,), generator=g))
            r, c = (int(t) for t in torch.randint(arm, D.SIDE - arm, (2,), generator=g))
            imgs[i, r - arm:r + arm + 1, c] = v
            imgs[i, r, c - arm:c + arm + 1] = v
    imgs = imgs.reshape(n, -1)
    return imgs / imgs.norm(dim=1, keepdim=True), labels


def _use(variant: str):
    """Points eqnn.data at the generator of a variant (each variant has its own cache of the splits)."""
    if variant == "v0":
        D._load_split = lambda name, train: make_shapes_v0(D.N_POOL + 10000 if train else 10000, 77 if train else 78)
    else:
        D.SHAPES = VARIANTS[variant]
    D.CACHE_DIR = f"data/cache_pilots/{variant}"


def _job(args):
    variant, arch, seed, N, L, pilot = args
    _use(variant)
    from eqnn.training import run_job
    r = run_job({"task": "shapes_cs", "arch": arch, "layers": L, "N": N, "seed": seed, "shortcut": "none"})
    r.pop("curve")
    if pilot:
        r.update({"variant": variant, "pilot": pilot})
    return r


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "variants"
    if which == "v0":
        jobs = [("v0", a, s, N, L, None) for a, L, N, s in
                itertools.product(("config6", "config7"), (1, 2, 4), (10, 20, 40, 80), (1, 2, 3, 4, 5))]
        out = f"{OUT}/v0_small_noisy.jsonl"
    else:
        jobs = [(v, a, s, N, L, pilot) for pilot, variants, seeds, grid in ROUNDS for L, Ns in grid
                for v, a, s, N in itertools.product(variants, ("config6", "config7"), seeds, Ns)]
        out = f"{OUT}/variants.jsonl"
    for v in sorted({j[0] for j in jobs}):  # build the cached splits once, before the workers start
        _use(v)
        D.load_split("shapes_cs", True), D.load_split("shapes_cs", False)
    jobs.sort(key=lambda j: -j[3] * j[4])  # longest first
    os.makedirs(OUT, exist_ok=True)
    with Pool(os.cpu_count()) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(_job, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r.get("variant", "v0"), r["arch"], r["layers"], r["N"], r["seed"],
                  f"train {r['train_acc']:.2f} test {r['clean_acc']:.2f}", flush=True)


if __name__ == "__main__":
    main()
