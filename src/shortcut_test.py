"""Preliminary test: does equivariance protect against an orientation
shortcut?

A corner watermark is added to the TRAINING images, at a class-dependent
position: top-left for class 0, top-right for class 1. It predicts the
label perfectly, but its position is only meaningful relative to the image
frame: all four corners lie in one D4 orbit, so for an exactly invariant
model (Equiv) the watermark position carries no information, whereas a
non-equivariant model (NonEquiv) can learn it as a shortcut.

Test sets (same images, N per set):
  shortcut      watermark placed as in training (shortcut still valid)
  transformed   the watermarked images under a random D4 element (the
                watermark lands in a random corner: shortcut broken)
  clean         no watermark (what was learned about the content itself)

The watermark is a k x k block (k = side / 8) set to twice the image's
maximum absolute amplitude, followed by L2 renormalization.

Usage (repository root):
    python -m src.shortcut_test [--qubits 8] [--N 160] [--workers K]
    python -m src.shortcut_test --summary-only
Results: results_paper/shortcut_test.jsonl (shared by scripts/sync_results.sh).
"""

import argparse
import itertools
import json
import os
import statistics
from multiprocessing import get_context

import torch
from torch.utils.data import DataLoader, TensorDataset

import src.paper_experiments as pe
from src.train import validate
from src.watermark import add_watermark, loader_tensors as _tensors, random_d4

OUT = "results_paper/shortcut_test.jsonl"
TASKS = ["mnist45", "satellite", "ising", "eurosat_fi", "galaxy_round_edgeon",
         "galaxy_round_spiral", "resisc_airport_harbor"]
ARCHS = ["config6", "config7"]
SEEDS = [1, 2, 3, 4, 5]


def run(args: tuple) -> dict:
    task, arch, seed, qubits, N = args
    torch.set_num_threads(1)
    pe.random.seed(seed)
    pe.np.random.seed(seed)
    torch.manual_seed(seed)
    job = {"kind": "shortcut", "task": task, "qubits": qubits, "arch": arch, "N": N, "seed": seed}
    base = pe.loaders_for(task, N, seed, qubits)
    x_tr, y_tr = _tensors(base[0])
    x_te, y_te = _tensors(base[1])
    batch = max(1, N // 10)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(TensorDataset(add_watermark(x_tr, y_tr), y_tr), batch_size=batch,
                       shuffle=True, generator=g)
    x_short = add_watermark(x_te, y_te)
    tests = {
        "shortcut": x_short,
        "transformed": random_d4(x_short, torch.Generator().manual_seed(1000 + seed)),
        "clean": x_te,
    }
    params0 = pe._init_params(job)
    raw = pe._raw_qnn(job)
    stats = pe._output_stats(raw, train, params0)
    model = pe._model(raw, stats)
    params = pe._train(model, train, params0, job)
    dev = torch.device("cpu")
    result = {**job, "train_acc": validate(train, model, dev, params)[1]}
    for name, x in tests.items():
        loader = DataLoader(TensorDataset(x, y_te), batch_size=batch)
        result[f"{name}_acc"] = validate(loader, model, dev, params)[1]
    return result


def read(path: str) -> list[dict]:
    import glob

    files = [path] + glob.glob(os.path.join(os.path.dirname(path), "imported", "shortcut_test.*.jsonl"))
    rows = {}
    for f in files:
        if os.path.exists(f):
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    rows[(r["task"], r["arch"], r["seed"], r["qubits"], r["N"])] = r
    return list(rows.values())


def summary(rows: list[dict]) -> str:
    def ms(v: list[float]) -> str:
        sem = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
        return f"{statistics.mean(v):.3f}+-{sem:.3f}"

    cols = ("train_acc", "shortcut_acc", "transformed_acc", "clean_acc")
    lines = [f"{'task':<22} {'q':>2} {'N':>4} {'arch':<9} | " + " ".join(f"{c[:-4]:<12}" for c in cols) + " | n"]
    keys = sorted({(r["task"], r["qubits"], r["N"]) for r in rows}, key=lambda k: (TASKS.index(k[0]) if k[0] in TASKS else 99, k[1], k[2]))
    for task, q, N in keys:
        for arch in ARCHS:
            rs = [r for r in rows if (r["task"], r["qubits"], r["N"], r["arch"]) == (task, q, N, arch)]
            if rs:
                lines.append(f"{task:<22} {q:>2} {N:>4} {pe.ARCH_LABELS[arch]:<9} | "
                             + " ".join(f"{ms([r[c] for r in rs]):<12}" for c in cols) + f" | {len(rs)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=8, choices=(8, 10, 12))
    ap.add_argument("--N", type=int, default=160)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if not args.summary_only:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        for task in args.tasks:
            pe.loaders_for(task, 40, 1, args.qubits)  # download / cache once
        done = {(r["task"], r["arch"], r["seed"], r["qubits"], r["N"]) for r in read(OUT)}
        jobs = [(t, a, s, args.qubits, args.N) for t, a, s in itertools.product(args.tasks, ARCHS, SEEDS)
                if (t, a, s, args.qubits, args.N) not in done]
        workers = args.workers or pe.default_workers(pe.GB_PER_WORKER_BY_QUBITS[args.qubits])
        print(f"{len(jobs)} runs to do with {workers} workers", flush=True)
        with get_context("spawn").Pool(workers, maxtasksperchild=1) as pool, open(OUT, "a") as f:
            for k, r in enumerate(pool.imap_unordered(run, jobs), 1):
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"[{k}/{len(jobs)}] {r['task']} {r['arch']} seed={r['seed']} train={r['train_acc']:.3f} "
                      f"shortcut={r['shortcut_acc']:.3f} transformed={r['transformed_acc']:.3f} "
                      f"clean={r['clean_acc']:.3f}", flush=True)
    print(summary(read(OUT)))


if __name__ == "__main__":
    main()
