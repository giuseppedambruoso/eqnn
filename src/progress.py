"""Live view of the experiment campaign: overall progress plus, for every
worker process, the job it is training and its current epoch.

Usage (from the repository root):
    python -m src.progress            # refresh every 5 s, Ctrl+C to quit
    python -m src.progress --once     # print a single snapshot
    python -m src.progress --qubits 12 --interval 10
"""

import argparse
import glob
import json
import os
import subprocess
import time

# Deliberately self-contained (no torch / pennylane imports, which take
# over a minute to load on a machine saturated by the campaign workers).
# Must mirror src.paper_experiments.
PROGRESS_DIR = "results_paper/progress"
ARCH_LABELS = {"config6": "Equiv", "config7": "NonEquiv", "config10": "NonEquiv-Twirled"}
TASK_MAX_QUBITS = {"mnist45": 8, "satellite": 8, "planesnet": 8,
                   "ising": 12, "eurosat_hr": 12, "eurosat_fi": 12}
N_SEEDS = 6
SWEEP_JOBS_PER_TASK = 3 * 5 * N_SEEDS  # archs x N values x seeds
NOISE_JOBS_PER_TASK = 3 * (1 + 10 * 3)  # archs x (p=0 + 10 p values x 3 realizations)
POINTS_PER_TASK = 3 * 5


def out_path(qubits: int) -> str:
    return f"results_paper/campaign_{qubits}q.jsonl"


def result_files(path: str) -> list[str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    imported = glob.glob(os.path.join(os.path.dirname(path), "imported", f"{stem}.*.jsonl"))
    return [p for p in [path] + sorted(imported) if os.path.exists(p)]


def read_records(path: str) -> list[dict]:
    """All records of the tasks still in the study (e.g. the dropped
    MNIST 3 vs 4 is ignored)."""
    records = []
    for file in result_files(path):
        with open(file) as f:
            records += [r for r in map(json.loads, f) if r["task"] in TASK_MAX_QUBITS]
    return records


def job_key(r: dict) -> tuple:
    return (r["kind"], r["task"], r["qubits"], r["arch"], r["N"], r["seed"],
            r.get("noise_p", 0.0), r.get("noise_seed"))


def count_points(path: str) -> tuple[int, int]:
    """(plotted points = (task, arch, N) with >= 1 finished seed, completed
    rounds = min seeds over all points), as in src.plot_campaign."""
    seeds: dict[tuple, set] = {}
    for r in read_records(path):
        if r["kind"] == "sweep":
            seeds.setdefault((r["task"], r["arch"], r["N"]), set()).add(r["seed"])
    return len(seeds), seeds


BAR_WIDTH = 24


def _worker_pids() -> dict[int, int]:
    """{pid: seconds since start} of every live multiprocessing worker."""
    out = subprocess.run(["ps", "-eo", "pid=,etimes=,args="], capture_output=True, text=True).stdout
    pids = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and "multiprocessing.spawn" in parts[2]:
            pids[int(parts[0])] = int(parts[1])
    return pids


def _bar(done: int, total: int) -> str:
    filled = int(BAR_WIDTH * done / total) if total else 0
    return "[" + "#" * filled + "." * (BAR_WIDTH - filled) + "]"


def snapshot(qubits: int) -> str:
    lines = []
    tasks = [t for t, q in TASK_MAX_QUBITS.items() if q >= qubits]
    total_jobs = len(tasks) * (SWEEP_JOBS_PER_TASK + NOISE_JOBS_PER_TASK)
    path = out_path(qubits)
    done = 0
    if os.path.exists(path):
        records = read_records(path)
        done = len({job_key(r) for r in records if r["kind"] in ("sweep", "train_noise")})
        n_imported = len(result_files(path)) - 1
        n_points, seeds = count_points(path)
    else:
        n_points, seeds, n_imported = 0, {}, 0
    total_points = len(tasks) * POINTS_PER_TASK
    rounds = min(len(v) for v in seeds.values()) if n_points == total_points else 0
    lines.append(time.strftime("%H:%M:%S") + f"  campaign {qubits} qubits"
                 + (f"  (incl. results imported from {n_imported} other machine(s))" if n_imported else ""))
    lines.append(f"jobs completed : {done}/{total_jobs} {_bar(done, total_jobs)} "
                 f"{100 * done / total_jobs:.1f}%")
    lines.append(f"plot points    : {n_points}/{total_points} (noiseless accuracy grid)")
    lines.append(f"rounds (seeds) : {rounds}/{N_SEEDS} complete, round {min(rounds + 1, N_SEEDS)} in progress")
    lines.append("")

    pids = _worker_pids()
    reports = {}
    for p in glob.glob(os.path.join(PROGRESS_DIR, "*.json")):
        pid = int(os.path.basename(p).split(".")[0])
        if pid in pids:
            try:
                with open(p) as f:
                    reports[pid] = json.load(f)
            except (OSError, ValueError):
                pass
    lines.append(f"running workers: {len(pids)}")
    now = time.time()
    rows = []
    for pid, age in pids.items():
        r = reports.get(pid)
        if r is None or r.get("qubits") != qubits:
            rows.append(("~", f"pid {pid:<7} (started before progress reporting) "
                              f"running for {age / 60:.1f} min"))
            continue
        d, tot = r["epochs_done"], r["epochs"]
        eta = r["elapsed"] / d * (tot - d) if d else float("nan")
        extra = f" p={r['noise_p']:.2f}" if r["kind"] == "train_noise" else ""
        stale = now - r["updated"]
        state = "evaluating" if d >= tot else f"~{eta / 60:4.1f} min left"
        rows.append((r["task"] + r["arch"] + str(r["N"]),
                     f"{r['kind']:<11} {r['task']:<10} {ARCH_LABELS[r['arch']]:<16} "
                     f"N={r['N']:<4} seed={r['seed']}{extra:<8} "
                     f"epoch {d:>2}/{tot} {_bar(d, tot)} {state}"
                     + (f"  (no update for {stale:.0f}s)" if stale > 300 else "")))
    lines += [text for _, text in sorted(rows)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=8, choices=(8, 10, 12))
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if args.once:
        print(snapshot(args.qubits))
        return
    try:
        while True:
            text = snapshot(args.qubits)
            print("\033[2J\033[H" + text + "\n\n(Ctrl+C to quit)", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
