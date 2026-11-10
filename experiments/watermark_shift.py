"""Why removing the corner watermark lowers the accuracy of the invariant models on Galaxy10 (paper: Sec. "Shortcuts").

Hypothesis: the watermark carries a fixed share (20%) of the squared norm of every training image, so removing it
shifts the output of an invariant model by an amount that does not depend on the class; the shift moves test images
across the decision threshold f = 0, and more of them where the content of the images separates the classes less
sharply. The script reloads the angles of the stored runs trained with the corner watermark (main settings, five
seeds) and, on the test images with and without the watermark, measures
  shift_c        mean of f(clean) - f(cued) over the test images of class c (c = 0, 1)
  sep            mean of f(cued) on class 1 minus that on class 0 (how far apart the classes are)
  acc_cued, acc_clean   the accuracies with the threshold 0, as in the paper
  auc_cued, auc_clean   threshold-free separation of the classes (area under the ROC curve)
  acc_recal      the accuracy without the watermark after moving the threshold by the mean shift measured on the
                 TRAINING images (with and without their watermark; no test label is used)
If the hypothesis holds, the shift is about the same for the two classes, the AUC hardly changes when the watermark is
removed, and the recalibrated accuracy recovers the accuracy with the watermark.

    python experiments/watermark_shift.py            # writes results/watermark_shift.json and prints a table
"""

import json
import multiprocessing
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eqnn.data import build_data, load_task, random_d4  # noqa: E402
from eqnn.models import build_model  # noqa: E402
from eqnn.training import run_job  # noqa: E402

TASKS = {"galaxy_0_4": ("galaxy", "mark_e_rand", 40), "mnist_bin": ("mnist_pose", "mark_e_rand", 40),
         "shapes_cs": ("synthetic", "mark_e", 20)}
ARCHS = ("config6", "config10", "config7", "config6n")
LAYERS = 4


def auc(score, y):
    """Area under the ROC curve (Mann-Whitney), class 1 as positive."""
    s1, s0 = score[y == 1], score[y == 0]
    return float(((s1[:, None] > s0[None, :]).mean() + 0.5 * (s1[:, None] == s0[None, :]).mean()))


@torch.no_grad()
def out(model, params, x, chunk=500):
    return torch.cat([model(x[i:i + chunk], params) for i in range(0, x.shape[0], chunk)]).numpy()


def clean_train(task, N, seed, shortcut):
    """The training images before the watermark is added (same images and poses as build_data)."""
    xtr, ytr, _, _ = load_task(task, N, seed)
    if shortcut.endswith("_rand"):
        xtr = random_d4(xtr, torch.Generator().manual_seed(3000 + seed))
    return xtr


def _retrain(r):
    """The trained angles of a stored run that does not hold them (the first campaigns of Equiv and NonEquiv on
    MNIST and on the synthetic task): the run is repeated, and must reproduce the stored accuracies."""
    job = {k: r[k] for k in ("task", "arch", "layers", "N", "seed", "shortcut")}
    new = run_job(job)
    for k in ("train_acc", "cued_acc", "clean_acc"):
        assert abs(new[k] - r[k]) < 1e-9, (job, k, new[k], r[k])
    return new["params"]


def main():
    rows = []
    for task, (rdir, shortcut, N) in TASKS.items():
        runs = [json.loads(line) for line in open(f"results/{rdir}/runs.jsonl")]
        runs = [r for r in runs if r["shortcut"] == shortcut and r["layers"] == LAYERS and r["N"] == N]
        missing = [r for r in runs if "params" not in r]
        if missing:
            load_task(task, N, 0)  # writes the cached splits once, before the parallel runs read them
            with multiprocessing.Pool(min(len(missing), os.cpu_count() or 1)) as pool:
                for r, p in zip(missing, pool.map(_retrain, missing)):
                    r["params"] = p
        torch.set_num_threads(int(os.environ.get("EQNN_THREADS", "8")))
        for r in sorted(runs, key=lambda r: (ARCHS.index(r["arch"]), r["seed"])):
            x_tr, y_tr, tests, y_te = build_data(task, N, r["seed"], shortcut)
            x_tr0 = clean_train(task, N, r["seed"], shortcut)
            model, _ = build_model(r["arch"], LAYERS)
            p = torch.tensor(r["params"], dtype=torch.float64)
            y = y_te.numpy()
            fc, f0 = out(model, p, tests["cued"]), out(model, p, tests["clean"])
            assert abs((((fc > 0) == y).mean()) - r["cued_acc"]) < 1e-9, "stored run not reproduced"
            delta_train = float(np.mean(out(model, p, x_tr0) - out(model, p, x_tr)))  # f(clean) - f(cued), training
            row = {"task": task, "arch": r["arch"], "seed": r["seed"], "train_acc": r["train_acc"],
                   "acc_cued": float(((fc > 0) == y).mean()), "acc_clean": float(((f0 > 0) == y).mean()),
                   "acc_recal": float(((f0 - delta_train > 0) == y).mean()),
                   "auc_cued": auc(fc, y), "auc_clean": auc(f0, y),
                   "shift_0": float((f0 - fc)[y == 0].mean()), "shift_1": float((f0 - fc)[y == 1].mean()),
                   "shift_train": delta_train,
                   "sep_cued": float(fc[y == 1].mean() - fc[y == 0].mean()),
                   "sep_clean": float(f0[y == 1].mean() - f0[y == 0].mean()),
                   "std_cued": float(np.concatenate([fc[y == c] - fc[y == c].mean() for c in (0, 1)]).std()),
                   "frac_class1_clean": float((f0 > 0).mean())}
            rows.append(row)
            print(task, r["arch"], r["seed"], {k: round(v, 3) for k, v in row.items() if isinstance(v, float)},
                  flush=True)
    with open("results/watermark_shift.json", "w") as f:
        json.dump(rows, f, indent=1)

    keys = ("train_acc", "acc_cued", "acc_clean", "acc_recal", "auc_cued", "auc_clean", "shift_0", "shift_1",
            "shift_train", "sep_cued", "std_cued", "frac_class1_clean")
    print("\nmean over the seeds")
    print(f"{'task':11s} {'model':9s} " + " ".join(f"{k:>10s}" for k in keys))
    for task in TASKS:
        for a in ARCHS:
            sel = [r for r in rows if r["task"] == task and r["arch"] == a]
            print(f"{task:11s} {a:9s} " + " ".join(f"{np.mean([r[k] for r in sel]):10.3f}" for k in keys))


if __name__ == "__main__":
    main()
