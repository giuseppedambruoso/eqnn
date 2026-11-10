"""Runs a grid of jobs (eqnn.training) in parallel on one machine and appends one JSON line per run to
<out>/runs.jsonl. Runs already in the file are skipped, so an interrupted campaign can simply be restarted.

One worker process per CPU (one thread each, pinned to its CPU), fed from a single queue sorted from the heaviest job
to the lightest. Example (the main MNIST setting of the paper):

    python -m eqnn.runner --out results/mnist_pose --tasks mnist_bin --shortcuts pose_rand \
        --models Equiv NonEquiv NonEquiv-Twirled Equiv-NoSym --layers 4 --n-values 40 --seeds 1 2 3

The exact commands of every campaign of the paper are in experiments/run_all.sh.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"  # one thread per worker process

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402

from eqnn import training as T  # noqa: E402
from eqnn.data import SHORTCUTS, TASKS  # noqa: E402
from eqnn.models import MODELS, NAMES, model_id  # noqa: E402

POLL_SECONDS = 5


def build_jobs(tasks, models, layers, n_values, seeds, shortcuts, noise_scales=(None,), chip=0, chip_spread=True,
               noise_eval=None) -> list[dict]:
    jobs = []
    for task, arch, L, N, seed, sc, ns in itertools.product(tasks, models, layers, n_values, seeds, shortcuts,
                                                           noise_scales):
        job = {"task": task, "arch": model_id(arch), "layers": L, "N": N, "seed": seed, "shortcut": sc}
        if ns is not None:
            job.update({"noise_scale": ns, "chip": chip, "chip_spread": chip_spread})
        if noise_eval:
            job.update({"noise_eval": list(noise_eval), "chip": chip, "chip_spread": chip_spread})
        jobs.append(job)
    return jobs


def _worker(job_queue, result_queue, cpu, progress_dir):
    os.sched_setaffinity(0, {cpu})
    while True:
        job = job_queue.get()
        if job is None:
            return
        try:
            result_queue.put(("ok", T.run_job(job, progress_dir)))
        except Exception as e:  # noqa: BLE001
            result_queue.put(("error", (job, repr(e))))


def done_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                keys.add(T.job_key(json.loads(line)))
    return keys


def run(jobs: list[dict], cpus: list[int], out: str) -> None:
    os.makedirs(out, exist_ok=True)
    job_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    for j in jobs:
        job_queue.put(j)
    progress = os.path.join(out, "progress")
    workers = [mp.Process(target=_worker, args=(job_queue, result_queue, c, progress), daemon=True) for c in cpus]
    for p in workers:
        p.start()

    def cleanup(*_a):
        for p in workers:
            if p.is_alive():
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    done, failed = 0, 0
    with open(os.path.join(out, "runs.jsonl"), "a") as f:
        while done + failed < len(jobs):
            try:
                status, r = result_queue.get(timeout=POLL_SECONDS)
            except Exception:  # noqa: BLE001
                continue
            if status == "error":
                failed += 1
                print(f"JOB FAILED: {r[0]} -> {r[1]}", flush=True)
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            done += 1
            print(f"[{done}/{len(jobs)}] {r['task']} {MODELS[r['arch']]['name']} L={r['layers']} N={r['N']} "
                  f"seed={r['seed']} {r['shortcut']} noise={r.get('noise_scale')} train={r['train_acc']:.3f} "
                  f"test={r['clean_acc']:.3f} ({r['seconds']}s)", flush=True)
    for _ in workers:
        job_queue.put(None)
    print("done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results directory (runs.jsonl is appended)")
    ap.add_argument("--tasks", nargs="+", required=True, choices=list(TASKS))
    ap.add_argument("--models", nargs="+", required=True, choices=list(NAMES) + list(MODELS))
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--n-values", nargs="+", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--shortcuts", nargs="+", required=True, choices=list(SHORTCUTS))
    ap.add_argument("--noise-scale", nargs="+", type=float, default=[None],
                    help="train and test under realistic noise, every error rate times this factor")
    ap.add_argument("--noise-eval", nargs="+", type=float, default=None,
                    help="train without noise, then evaluate the final angles under noise at these scales")
    ap.add_argument("--chip", type=int, default=0, help="seed of the per-qubit noise parameters")
    ap.add_argument("--identical-qubits", action="store_true", help="noise without qubit-to-qubit dispersion")
    ap.add_argument("--workers", type=int, default=0, help="default: one per CPU")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cpus = sorted(os.sched_getaffinity(0))
    if args.workers:
        cpus = cpus[:args.workers]
    grid = build_jobs(args.tasks, args.models, args.layers, args.n_values, args.seeds, args.shortcuts,
                      args.noise_scale, args.chip, not args.identical_qubits, args.noise_eval)
    done = done_keys(os.path.join(args.out, "runs.jsonl"))
    jobs = [j for j in grid if T.job_key(j) not in done]
    jobs.sort(key=lambda j: (-T.estimated_cost(j), j["seed"]))  # longest first
    print(f"{len(jobs)} of {len(grid)} jobs to run, {len(cpus)} workers", flush=True)
    if args.dry_run or not jobs:
        return
    run(jobs, cpus, args.out)


if __name__ == "__main__":
    main()
