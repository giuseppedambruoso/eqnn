"""Distribution of the moduli of the gradient components of the four models at the initialization of training.

For every task and model, in the main setting (L = 4; N = 40, or 20 on the synthetic task), the angles are drawn 200
times uniformly in [-0.1, 0.1], as at the start of training, and the gradient of the training loss (Eq. (loss) of the
paper) is computed on the training images of seed 1. The script stores, for each configuration, the moduli of all the
partial derivatives (draws x angles) and the Euclidean norms of the gradients, for the histograms of the appendix.

    python experiments/gradient_histograms.py results/gradient_histograms.json
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
LAYERS = 4
INIT = (-0.1, 0.1)
DRAWS = 200


def one(cfg):
    task, arch = cfg
    torch.set_num_threads(1)
    n, shortcut = TASKS[task]
    x, y, _, _ = build_data(task, n, 1, shortcut)
    forward, n_params = build_model(arch, LAYERS)
    gen = torch.Generator().manual_seed(4007)
    grads = []
    for _ in range(DRAWS):
        p = torch.empty(n_params, dtype=torch.float64).uniform_(*INIT, generator=gen).requires_grad_()
        (g,) = torch.autograd.grad(model_loss(forward(x, p), y), p)
        grads.append(g)
    g = torch.stack(grads)
    return {"task": task, "arch": arch, "layers": LAYERS, "N": n, "init": list(INIT), "draws": DRAWS,
            "n_params": n_params, "abs_grad": g.abs().flatten().tolist(), "norm": g.norm(dim=1).tolist()}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/gradient_histograms.json"
    cfgs = [(t, a) for t in TASKS for a in MODELS]
    with Pool(int(os.environ.get("WORKERS", os.cpu_count()))) as pool:
        rows = pool.map(one, cfgs, chunksize=1)
    json.dump(rows, open(out, "w"))
    for r in rows:
        g = torch.tensor(r["abs_grad"])
        print(f"{r['task']:11s} {r['arch']:17s} median |dL/dtheta| = {g.median():.3e}  "
              f"mean |grad| = {torch.tensor(r['norm']).mean():.3e}")
