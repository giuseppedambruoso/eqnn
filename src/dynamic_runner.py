"""Elastic-worker runner for src.paper_experiments jobs: grows the worker
pool while RAM is plentiful and stops replenishing workers (letting them
exit after their current job) when it gets tight, instead of a fixed
--workers count. Meant for cases like L=8 where per-job memory varies a
lot (twirled/N=640 uses far more than non-twirled/small-N), so a single
static worker count is either too slow or risks OOM.

Same resumability as src.paper_experiments.main(): jobs already present
in the output file (here or imported) are skipped.

Usage:
    python -m src.dynamic_runner --qubits 8 --layers 8 --kinds sweep \
        --tasks mnist45 satellite ising eurosat_fi galaxy_round_edgeon \
        galaxy_round_spiral resisc_airport_harbor
"""

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time

from src.paper_experiments import (
    ALL_TASKS,
    TASKS,
    build_jobs,
    imported_paths,
    job_key,
    lpt_order,
    out_path,
    run_job,
)

MIN_WORKERS = 2
GROW_MARGIN_GB = 4.0   # spawn another worker while (available - this) stays above FLOOR
FLOOR_GB = 6.0         # never spawn (and start shedding) below this available RAM
POLL_SECONDS = 5
# A freshly spawned worker takes a while (loading the dataset, then ramping
# up through its first few epochs) to reach its steady-state memory. Growing
# on every poll tick reacts to a reading that doesn't yet reflect workers
# spawned moments ago, which overshoots. Only grow once per cooldown so each
# new worker's footprint has time to show up in MemAvailable before the next
# growth decision.
SPAWN_COOLDOWN_SECONDS = 45


def available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024**2
    return 0.0


def _worker_loop(job_queue: mp.Queue, result_queue: mp.Queue, stop_event) -> None:
    while not stop_event.is_set():
        try:
            job = job_queue.get(timeout=1.0)
        except Exception:
            continue
        if job is None:
            return
        try:
            records = run_job(job)
            result_queue.put(("ok", records))
        except Exception as e:  # noqa: BLE001
            result_queue.put(("error", (job, repr(e))))


def run(jobs: list[dict], out_file: str, max_workers: int) -> None:
    job_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    for j in jobs:
        job_queue.put(j)

    workers: dict[int, mp.Process] = {}
    stop_events: dict[int, "mp.synchronize.Event"] = {}
    next_id = 0
    done = 0
    total = len(jobs)

    def spawn() -> None:
        nonlocal next_id
        ev = mp.Event()
        p = mp.Process(target=_worker_loop, args=(job_queue, result_queue, ev), daemon=True)
        p.start()
        workers[next_id] = p
        stop_events[next_id] = ev
        next_id += 1

    def cleanup(*_a) -> None:
        for ev in stop_events.values():
            ev.set()
        for p in list(workers.values()):
            if p.is_alive():
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    for _ in range(MIN_WORKERS):
        spawn()
    last_spawn = time.time()

    with open(out_file, "a") as f:
        last_report = 0.0
        while done < total:
            avail = available_gb()
            alive = {i: p for i, p in workers.items() if p.is_alive()}
            n_alive = len(alive)

            can_grow = (avail - GROW_MARGIN_GB > FLOOR_GB and n_alive < max_workers
                        and time.time() - last_spawn > SPAWN_COOLDOWN_SECONDS)
            if can_grow:
                spawn()
                last_spawn = time.time()
            elif avail < FLOOR_GB and n_alive > MIN_WORKERS:
                # ask exactly one worker to stop after its current job (it
                # will see job_queue empty of a poison pill and just exit
                # once it times out getting a job — instead we push one
                # None sentinel so exactly one worker retires)
                job_queue.put(None)

            # drain any finished workers bookkeeping
            for i in list(workers):
                if not workers[i].is_alive():
                    del workers[i]
                    del stop_events[i]

            try:
                status, payload = result_queue.get(timeout=POLL_SECONDS)
            except Exception:
                if time.time() - last_report > 30:
                    print(f"[{done}/{total}] worker={len(workers)} avail={avail:.1f}GB "
                          f"(waiting)", flush=True)
                    last_report = time.time()
                continue

            if status == "error":
                job, err = payload
                print(f"JOB FAILED: {job} -> {err}", flush=True)
                total -= 1
                continue

            for r in payload:
                f.write(json.dumps(r) + "\n")
            f.flush()
            done += 1
            r0 = payload[0]
            print(f"[{done}/{total}] worker={len(workers)} avail={avail:.1f}GB  "
                  f"{r0['task']} {r0['arch']} N={r0['N']} seed={r0['seed']} "
                  f"train={r0.get('train_acc', float('nan')):.3f} "
                  f"val={r0['val_acc']:.3f} ({r0['seconds']}s)", flush=True)

    for _ in range(len(workers)):
        job_queue.put(None)
    for p in workers.values():
        p.join(timeout=30)
    print("done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--kinds", nargs="+", default=["sweep"])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--n-values", nargs="+", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=None)
    args = ap.parse_args()

    from src.paper_experiments import N_VALUES, SEEDS

    seeds = args.seeds or list(SEEDS)
    n_values = args.n_values or list(N_VALUES)
    max_workers = args.max_workers or max(MIN_WORKERS, (os.cpu_count() or 4) - 2)

    for t in args.tasks:
        if args.qubits > ALL_TASKS[t][1]:
            raise ValueError(f"{t} has too few pixels for {args.qubits} qubits")

    out_file = out_path(args.qubits, args.layers, watermark=False)
    done_keys = set()
    for path in [out_file] + imported_paths(args.qubits, args.layers, watermark=False):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] in ("sweep", "train_noise", "watermark"):
                    done_keys.add(job_key(r))

    jobs = lpt_order(build_jobs(args.tasks, args.kinds, args.qubits, seeds, n_values, args.layers))
    jobs = [j for j in jobs if job_key(j) not in done_keys]
    print(f"{len(jobs)} jobs remaining, max_workers={max_workers}, "
          f"grow while avail-{GROW_MARGIN_GB}GB > {FLOOR_GB}GB floor", flush=True)
    if not jobs:
        return

    os.makedirs("results_paper", exist_ok=True)
    run(jobs, out_file, max_workers)


if __name__ == "__main__":
    main()
