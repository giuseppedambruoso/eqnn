"""Trainability of the four models: variance of the gradient of the training loss over random angles.

For every task, model, depth L and initialization, the angles are drawn 100 times and the gradient of the training
loss (Eq. (loss) of the paper) is computed on the training images of the main setting (seed 1). The script reports,
for each configuration, the variance over the draws of each partial derivative, averaged over the angles, and the
mean squared norm of the gradient divided by the number of angles. The angles are drawn as at the start of training,
uniformly in [-0.1, 0.1].

    python experiments/gradient_variance.py results/gradient_variance.json
"""

import json
import os
import sys
from multiprocessing import Pool

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eqnn.data import build_data  # noqa: E402
from eqnn.models import build_model, model_loss  # noqa: E402

TASKS = {"mnist_bin": (40, "pose_rand"), "shapes_cs": (20, "none"), "galaxy_0_4": (40, "pose_rand"),
         "ery_1_2": (40, "pose_rand")}
MODELS = ("Equiv", "NonEquiv-Twirled", "NonEquiv", "Equiv-NoSym")
LAYERS = (1, 2, 4, 8, 16)
INITS = {"small": (-0.1, 0.1)}
DRAWS = 100


def one(cfg):
    task, arch, layers, init = cfg
    torch.set_num_threads(1)
    n, shortcut = TASKS[task]
    x, y, _, _ = build_data(task, n, 1, shortcut)
    forward, n_params = build_model(arch, layers)
    lo, hi = INITS[init]
    gen = torch.Generator().manual_seed(1000 * layers + 7)
    grads = []
    for _ in range(DRAWS):
        p = torch.empty(n_params, dtype=torch.float64).uniform_(lo, hi, generator=gen).requires_grad_()
        loss = model_loss(forward(x, p), y)
        (g,) = torch.autograd.grad(loss, p)
        grads.append(g)
    g = torch.stack(grads)
    return {"task": task, "arch": arch, "layers": layers, "init": init, "draws": DRAWS, "n_params": n_params,
            "var": float(g.var(dim=0, unbiased=True).mean()), "mean_sq": float((g ** 2).mean()),
            "var_per_angle": g.var(dim=0, unbiased=True).tolist()}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/gradient_variance.json"
    cfgs = [(t, a, L, i) for t in TASKS for a in MODELS for L in LAYERS for i in INITS]
    cfgs.sort(key=lambda c: -c[2] * (8 if c[1] == "NonEquiv-Twirled" else 1))  # longest first
    with Pool(int(os.environ.get("WORKERS", os.cpu_count()))) as pool:
        rows = pool.map(one, cfgs, chunksize=1)
    rows.sort(key=lambda r: (list(TASKS).index(r["task"]), r["init"], MODELS.index(r["arch"]), r["layers"]))
    json.dump(rows, open(out, "w"), indent=1)
    for r in rows:
        print(f"{r['task']:11s} {r['init']:8s} {r['arch']:17s} L={r['layers']:2d}  var={r['var']:.3e}")
