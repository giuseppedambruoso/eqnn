"""Preliminary screening of candidate datasets (astronomical and
remote-sensing): can the six-parameter models learn them at all?

Equiv (config6) and NonEquiv (config7), standardized output (the campaign
model), N = 80, 5 seeds, noiseless; accuracy on the original and on the
p4m-transformed test images.

Usage (repository root):
    python -m src.dataset_screening [--qubits 8] [--workers K]
    python -m src.dataset_screening --summary-only
Results: results_paper/dataset_screening.jsonl (shared by scripts/sync_results.sh).
"""

import argparse
import itertools
import json
import os
import statistics
from multiprocessing import get_context

import torch

import src.paper_experiments as pe

OUT = "results_paper/dataset_screening.jsonl"
TASKS = ["galaxy_round_spiral", "galaxy_round_edgeon", "resisc_airport_harbor", "resisc_farmland"]
ARCHS = ["config6", "config7"]
SEEDS = [1, 2, 3, 4, 5]
N = 80


def run(args: tuple) -> dict:
    task, arch, seed, qubits = args
    torch.set_num_threads(1)
    pe.random.seed(seed)
    pe.np.random.seed(seed)
    torch.manual_seed(seed)
    job = {"kind": "screening", "task": task, "qubits": qubits, "arch": arch, "N": N, "seed": seed}
    loaders = pe.loaders_for(task, N, seed, qubits)
    params0 = pe._init_params(job)
    raw = pe._raw_qnn(job)
    stats = pe._output_stats(raw, loaders[0], params0)
    model = pe._model(raw, stats)
    params = pe._train(model, loaders[0], params0, job)
    return {**job, **pe._evaluate(model, loaders, params),
            "output_mean": stats[0], "output_std": stats[1]}


def read(path: str) -> list[dict]:
    import glob

    files = [path] + glob.glob(os.path.join(os.path.dirname(path), "imported", "dataset_screening.*.jsonl"))
    rows = {}
    for f in files:
        if os.path.exists(f):
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    rows[(r["task"], r["arch"], r["seed"], r["qubits"])] = r
    return list(rows.values())


def summary(rows: list[dict]) -> str:
    def ms(v: list[float]) -> str:
        if not v:
            return "     -     "
        sem = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
        return f"{statistics.mean(v):.3f}+-{sem:.3f}"

    lines = [f"{'task':<22} {'q':>2} {'arch':<9} | {'test':<12} {'rotated':<12} {'train':<12} | n"]
    for task, q, arch in itertools.product(TASKS, sorted({r['qubits'] for r in rows}), ARCHS):
        rs = [r for r in rows if (r["task"], r["qubits"], r["arch"]) == (task, q, arch)]
        if rs:
            lines.append(f"{task:<22} {q:>2} {pe.ARCH_LABELS[arch]:<9} | {ms([r['val_acc'] for r in rs]):<12} "
                         f"{ms([r['val_aug_acc'] for r in rs]):<12} {ms([r['train_acc'] for r in rs]):<12} | {len(rs)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", default=[8], choices=(8, 10, 12))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if not args.summary_only:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        for task in TASKS:  # download + cache the datasets once, serially
            for q in args.qubits:
                pe.loaders_for(task, 40, 1, q)
        done = {(r["task"], r["arch"], r["seed"], r["qubits"]) for r in read(OUT)}
        jobs = [(t, a, s, q) for t, a, s, q in itertools.product(TASKS, ARCHS, SEEDS, args.qubits)
                if (t, a, s, q) not in done]
        jobs.sort(key=lambda j: -j[3])  # 12-qubit (slower) runs first
        workers = args.workers or pe.default_workers(
            max(pe.GB_PER_WORKER_BY_QUBITS[q] for q in args.qubits))
        print(f"{len(jobs)} runs to do with {workers} workers", flush=True)
        with get_context("spawn").Pool(workers, maxtasksperchild=1) as pool, open(OUT, "a") as f:
            for k, r in enumerate(pool.imap_unordered(run, jobs), 1):
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"[{k}/{len(jobs)}] {r['task']} {r['qubits']}q {r['arch']} seed={r['seed']} "
                      f"val={r['val_acc']:.3f} aug={r['val_aug_acc']:.3f}", flush=True)
    print(summary(read(OUT)))


if __name__ == "__main__":
    main()
