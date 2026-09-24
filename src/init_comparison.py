"""Preliminary comparison of the output-bias initialization, N = 80:

  default   w = 1, b = 0 (the model starts identical to the one without
            output bias) - what the campaign currently uses;
  datainit  w = 1/std, b = -mean/std of the untrained circuit's <O> over the
            training images, i.e. the output starts z-scored for every
            dataset. Same rule for all datasets, acts only on the invariant
            scalar output (equivariance untouched).

  standardize  FIXED (non-trainable) standardization z = (<O> - mean)/std with
            the same training-set statistics, followed by the trainable
            affine map w z + b starting at w = 1, b = 0: unlike datainit, w
            stays O(1), so Adam can still move it (and flip its sign).

Motivation: on PlanesNet <O> ~ 0.99 for BOTH classes (nearly uniform
images -> state close to |+>^n), so with the default init w would need to
grow to ~100 and training stays stuck at chance.

Usage (repository root):
    python -m src.init_comparison                  # run (resumable), then summarize
    python -m src.init_comparison --summary-only   # just print the table
Results: results_paper/init_comparison.jsonl (shared by scripts/sync_results.sh).
"""

import argparse
import itertools
import json
import os
import statistics
from multiprocessing import get_context

import torch

import src.paper_experiments as pe
from src.train import execute_batch, loss_function, validate

OUT = "results_paper/init_comparison.jsonl"
TASKS = ["planesnet", "ising", "satellite", "eurosat_hr", "eurosat_fi", "mnist45"]
ARCHS = ["config6", "config7"]
VARIANTS = ["standardize", "datainit", "default"]
SEEDS = [1, 2, 3, 4, 5]
N = 80


def run(args: tuple) -> dict:
    task, arch, variant, seed = args
    torch.set_num_threads(1)
    pe.random.seed(seed)
    pe.np.random.seed(seed)
    torch.manual_seed(seed)
    job = {"kind": "sweep", "task": task, "qubits": 8, "arch": arch, "N": N, "seed": seed}
    loaders = pe.loaders_for(task, N, seed, 8)
    qnn = pe._make_qnn(job)
    names = pe.architecture_param_names(arch, 8, 2, output_bias=True)
    params = pe.initial_parameters(names, torch.Generator().manual_seed(seed))
    if variant in ("datainit", "standardize"):
        raw_qnn = pe.create_qnn("default.qubit", 8, 2, arch, readout=pe.TASKS[task][0])
        with torch.no_grad():
            raw = torch.cat([raw_qnn(x, params[:-2]).reshape(-1) for x, _ in loaders[0]])
        mean, std = raw.mean().item(), max(raw.std().item(), 1e-6)
        if variant == "datainit":
            params[-2], params[-1] = 1.0 / std, -mean / std
        else:
            def qnn(x, p, raw_qnn=raw_qnn, mean=mean, std=std):
                return torch.tanh(p[-2] * (raw_qnn(x, p[:-2]) - mean) / std + p[-1])
    params.requires_grad_()
    opt = torch.optim.Adam([params], lr=pe.LEARNING_RATE, betas=(0.5, 0.999))
    dev = torch.device("cpu")
    for _ in range(pe.EPOCHS):
        for images, labels in loaders[0]:
            opt.zero_grad()
            batch = labels.shape[0]
            for st in range(0, batch, pe.MICRO_BATCH):
                y = labels[st : st + pe.MICRO_BATCH]
                pred = execute_batch(qnn, images[st : st + pe.MICRO_BATCH], dev, params).reshape(-1)
                (loss_function(pred, y) * (y.shape[0] / batch)).backward()
            opt.step()
    p = params.detach()
    return {"task": task, "arch": arch, "variant": variant, "seed": seed, "N": N,
            "train": validate(loaders[0], qnn, dev, p)[1],
            "val": validate(loaders[1], qnn, dev, p)[1],
            "aug": validate(loaders[2], qnn, dev, p)[1]}


def read(path: str) -> list[dict]:
    import glob

    files = [path] + glob.glob(os.path.join(os.path.dirname(path), "imported", "init_comparison.*.jsonl"))
    rows = {}
    for f in files:
        if os.path.exists(f):
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    rows[(r["task"], r["arch"], r["variant"], r["seed"])] = r
    return list(rows.values())


def summary(rows: list[dict]) -> str:
    def ms(v: list[float]) -> str:
        if not v:
            return "     -     "
        sem = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
        return f"{statistics.mean(v):.3f}+-{sem:.3f}"

    order = ("default", "datainit", "standardize")
    lines = [f"{'task':<11} {'arch':<9} | " + " | ".join(f"{v + ': test / rotated':<27}" for v in order) + " | n"]
    for task, arch in itertools.product(TASKS, ARCHS):
        cells, ns = [], []
        for variant in order:
            rs = [r for r in rows if (r["task"], r["arch"], r["variant"]) == (task, arch, variant)]
            ns.append(str(len(rs)))
            cells.append(f"{ms([r['val'] for r in rs])} / {ms([r['aug'] for r in rs])}")
        label = pe.ARCH_LABELS[arch]
        lines.append(f"{task:<11} {label:<9} | " + " | ".join(f"{c:<27}" for c in cells) + f" | {'/'.join(ns)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if not args.summary_only:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        done = {(r["task"], r["arch"], r["variant"], r["seed"]) for r in read(OUT)}
        jobs = [j for j in itertools.product(TASKS, ARCHS, VARIANTS, SEEDS) if j not in done]
        workers = args.workers or pe.default_workers()
        print(f"{len(jobs)} runs to do with {workers} workers", flush=True)
        with get_context("spawn").Pool(workers, maxtasksperchild=1) as pool, open(OUT, "a") as f:
            for k, r in enumerate(pool.imap_unordered(run, jobs), 1):
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"[{k}/{len(jobs)}] {r['task']} {r['arch']} {r['variant']} seed={r['seed']} "
                      f"val={r['val']:.3f} aug={r['aug']:.3f}", flush=True)
    print(summary(read(OUT)))


if __name__ == "__main__":
    main()
