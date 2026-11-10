"""Re-runs a sample of the stored runs with the code of this repository and compares every number of the new
record (accuracies, losses, learning curves, final angles, evaluations under noise) with the stored one.

    python experiments/reproduce_check.py            # the default sample below (about 20 minutes on 16 CPUs)
    python experiments/reproduce_check.py --quick    # without the run trained under noise (about 3 minutes)
"""

import json
import math
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE = [  # (results file, task, model id, layers, N, seed, shortcut, noise scale)
    ("mnist_pose", "mnist_bin", "config6", 2, 10, 1, "pose_rand", None),
    ("mnist_pose", "mnist_bin", "config7", 4, 40, 9, "pose_rand", None),
    ("mnist_pose", "mnist_bin", "config10", 1, 20, 2, "pose_rand", None),
    ("mnist_pose", "mnist_bin", "config6n", 2, 40, 12, "pose_rand", None),
    ("mnist_pose", "mnist_bin", "config7", 2, 10, 3, "mark_e_rand", None),
    ("mnist_pose", "mnist_bin", "config6n", 4, 10, 1, "gradient_rand", None),
    ("mnist_pose", "mnist_bin", "config10", 2, 10, 4, "invariant_rand", None),
    ("mnist_upright", "mnist_bin", "config6", 4, 20, 2, "none", None),
    ("synthetic", "shapes_cs", "config6", 4, 20, 13, "none", None),
    ("synthetic", "shapes_cs", "config7", 1, 80, 5, "none", None),
    ("synthetic", "shapes_cs", "config6", 2, 20, 1, "mark_e", None),
    ("noise_test_only", "shapes_cs", "config6n", 4, 20, 3, "none", None),
    ("noise_train_test", "shapes_cs", "config7", 4, 20, 2, "none", 1.0),
]


def stored(spec):
    d, task, arch, L, N, seed, sc, ns = spec
    for line in open(f"results/{d}/runs.jsonl"):
        r = json.loads(line)
        if (r["task"], r["arch"], r["layers"], r["N"], r["seed"], r["shortcut"], r.get("noise_scale")) == \
                (task, arch, L, N, seed, sc, ns):
            return r
    raise KeyError(spec)


def rerun(spec):
    from eqnn.training import run_job
    old = stored(spec)
    job = {k: old[k] for k in ("task", "arch", "layers", "N", "seed", "shortcut")}
    for k in ("noise_scale", "chip", "chip_spread"):
        if k in old:
            job[k] = old[k]
    if "noise_eval" in old:
        job["noise_eval"] = [float(s) for s in old["noise_eval"]]
    return spec, old, run_job(job)


IGNORED = ("seconds", "augment")  # run time; "augment" (always 0) is an option of the development code, removed here


def compare(a, b):
    """Largest absolute difference between the numbers of two records (over the keys of the stored record)."""
    if isinstance(a, dict):
        return max([compare(a[k], b[k]) if k in b else math.inf for k in a if k not in IGNORED] + [0.0])
    if isinstance(a, list):
        if len(a) != len(b):
            return math.inf
        return max([compare(x, y) for x, y in zip(a, b)] + [0.0])
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return abs(a - b)
    return 0.0 if a == b else math.inf


def main():
    sample = [s for s in SAMPLE if not ("--quick" in sys.argv and s[7] is not None)]
    worst = 0.0
    with Pool(min(len(sample), os.cpu_count())) as pool:
        for spec, old, new in pool.imap_unordered(rerun, sample):
            diff = compare(old, new)
            worst = max(worst, diff)
            print(f"{spec}: largest difference {diff:.3g}", flush=True)
    print("IDENTICAL" if worst == 0 else f"largest difference over the sample: {worst:.3g}")


if __name__ == "__main__":
    main()
