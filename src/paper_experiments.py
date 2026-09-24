"""Experiment campaign for the paper: Equiv (config6) vs. NonEquiv
(config7) vs. NonEquiv-Twirled (config10) on every task, at a chosen
number of qubits (8, 10 or 12 -> 16x16, 32x32 or 64x64 images).

Job kinds, all appended as JSON lines to one results file (jobs already
present are skipped, so an interrupted campaign can simply be resumed):

  sweep       noiseless training for N in N_VALUES x SEEDS; records clean
              and p4m-augmented test accuracy. The (N=NOISE_N,
              seed=NOISE_PARAM_SEED) model is also evaluated under
              test-time noise, producing "test_noise" records.
  train_noise training AND evaluation under Monte Carlo depolarizing noise
              (N=NOISE_N, fixed parameter seed, several noise realizations).

No training-set augmentation is used. Every model uses the same
classical post-processing (see MODEL_OPTIONS).

Usage:
    python -m src.paper_experiments --qubits 8 --workers 13 \
        [--tasks mnist45 satellite ...] [--rounds 1 2 ...] [--kinds sweep train_noise]
"""

import argparse
import itertools
import json
import os
import random
import time
from multiprocessing import get_context

import numpy as np
import torch

from src.data_loading import (
    load_aero_data_full,
    load_eurosat_data_full,
    load_ising_data_full,
    load_mnist_data_full,
    load_planesnet_data_full,
)
from src.qnn import architecture_param_names, create_qnn, initial_parameters
from src.train import execute_batch, loss_function, validate

ARCHS = ("config6", "config7", "config10")
ARCH_LABELS = {"config6": "Equiv", "config7": "NonEquiv", "config10": "NonEquiv-Twirled"}
N_VALUES = (40, 80, 160, 320, 640)
SEEDS = (1, 2, 3, 4, 5, 6)
NOISE_N = 80
NOISE_PARAM_SEED = 1
NOISE_SEEDS = (101, 102, 103)
NOISE_P_TEST = tuple(round(0.01 * k, 2) for k in range(21))
NOISE_P_TRAIN = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2)
EPOCHS = 70
MICRO_BATCH = 32
LEARNING_RATE = 0.05

# Single model configuration used for every task (chosen once, for all
# datasets, from the preliminary comparison — see the paper's Methods).
MODEL_OPTIONS = {"output_bias": True, "center": False, "standardize": True}
# standardize: the measured <O> is standardized with FIXED statistics,
# z = (<O> - mean) / std, computed once per training run on the training
# images with the initial (untrained) circuit, and fed to the trainable
# affine output map tanh(w z + b) (w = 1, b = 0 at start). Keeps w of order
# one for every dataset, so the output map can be learned (and its sign
# flipped); acts only on the invariant scalar output. Chosen from
# src.init_comparison (preliminary comparison, N = 80, 5 seeds).

# task -> (readout, max supported qubits given the native resolution)
TASKS = {
    # MNIST 3 vs 4 was dropped from the study (user decision, 2026-09-24).
    "mnist45": ("x0_xhalf", 8),  # 28x28
    "satellite": ("avg_x", 8),  # planes 20x20, ships cropped to 20x20
    # PlanesNet and EuroSAT Highway vs River were dropped: at chance level for
    # every model and output initialization in the preliminary comparison.
    "ising": ("avg_x", 12),  # generated natively at any size
    "eurosat_fi": ("avg_x", 12),  # 64x64
}


def readout_for(task: str) -> str:
    """MNIST: X on the two pooled-to qubits; every other task: mean X."""
    return "x0_xhalf" if task.startswith("mnist") else "avg_x"


def img_size_for(num_qubits: int) -> int:
    return 2 ** (num_qubits // 2)


def loaders_for(task: str, N: int, seed: int, num_qubits: int):
    b = max(1, N // 10)
    size = img_size_for(num_qubits)
    center = MODEL_OPTIONS["center"]
    if task in ("mnist34", "mnist45"):
        c1, c2 = (3, 4) if task == "mnist34" else (4, 5)
        return load_mnist_data_full(b, N, 0, size, "data", seed, False, "none", c1, c2, center=center)
    if task == "satellite":
        return load_aero_data_full(b, N, 0, size, seed, False, "none",
                                   crop_to_plane_scale=True, center=center)
    if task == "planesnet":
        return load_planesnet_data_full(b, N, 0, size, seed, False, "none", center=center)
    if task == "ising":
        return load_ising_data_full(b, N, 0, size, "data", seed, False, "none", center=center)
    if task in ("eurosat_hr", "eurosat_fi"):
        sub = "highway_river" if task == "eurosat_hr" else "forest_industrial"
        return load_eurosat_data_full(b, N, 0, size, "data", seed, False, "none", sub, center=center)
    raise ValueError(task)


def _make_qnn(job: dict, noise_p: float = 0.0, noise_seed: int = 0):
    return create_qnn("default.qubit", job["qubits"], 2, job["arch"],
                      readout=readout_for(job["task"]), noise_p=noise_p,
                      noise_seed=noise_seed, output_bias=MODEL_OPTIONS["output_bias"])


def _raw_qnn(job: dict, noise_p: float = 0.0, noise_seed: int = 0):
    """The circuit alone: returns the measured <O> (no output map)."""
    return create_qnn("default.qubit", job["qubits"], 2, job["arch"],
                      readout=readout_for(job["task"]), noise_p=noise_p, noise_seed=noise_seed)


def _init_params(job: dict) -> torch.Tensor:
    names = architecture_param_names(job["arch"], job["qubits"], 2,
                                     output_bias=MODEL_OPTIONS["output_bias"])
    return initial_parameters(names, torch.Generator().manual_seed(job["seed"]))


def _output_stats(raw_qnn, train_loader, params: torch.Tensor) -> tuple[float, float]:
    """Mean and std of <O> over the training images, initial circuit."""
    with torch.no_grad():
        values = torch.cat([raw_qnn(x, params[:-2]).reshape(-1) for x, _ in train_loader])
    return values.mean().item(), max(values.std().item(), 1e-6)


def _model(raw_qnn, stats: tuple[float, float]):
    """Full classifier: tanh(w (<O> - mean) / std + b), (w, b) = params[-2:]."""
    mean, std = stats

    def forward(encoded: torch.Tensor, params: torch.Tensor):
        return torch.tanh(params[-2] * (raw_qnn(encoded, params[:-2]) - mean) / std + params[-1])

    return forward


def _train(qnn, train_loader, params: torch.Tensor, job: dict) -> torch.Tensor:
    params = params.clone().requires_grad_()
    opt = torch.optim.Adam([params], lr=LEARNING_RATE, betas=(0.5, 0.999))
    dev = torch.device("cpu")
    t_start = time.time()
    for epoch in range(EPOCHS):
        _write_progress(job, epoch, t_start)
        for images, labels in train_loader:
            # Micro-batching: the batch-mean BCE gradient is accumulated over
            # chunks of MICRO_BATCH images, each weighted by its share of the
            # batch — the same gradient (up to float rounding) as one
            # full-batch backward pass, with a fraction of the autograd
            # memory (which otherwise dominates for twirled/large circuits).
            opt.zero_grad()
            batch = labels.shape[0]
            for start in range(0, batch, MICRO_BATCH):
                x = images[start : start + MICRO_BATCH]
                y = labels[start : start + MICRO_BATCH]
                pred = execute_batch(qnn, x, dev, params).reshape(-1)
                loss = loss_function(pred, y) * (y.shape[0] / batch)
                loss.backward()
            opt.step()
    _write_progress(job, EPOCHS, t_start)
    return params.detach()


PROGRESS_DIR = "results_paper/progress"


def _write_progress(job: dict, epochs_done: int, t_start: float) -> None:
    """One small JSON file per worker process, overwritten after every
    epoch: which job it is running and how many epochs are done (read by
    src.progress). Purely informational — failures are ignored."""
    try:
        os.makedirs(PROGRESS_DIR, exist_ok=True)
        path = os.path.join(PROGRESS_DIR, f"{os.getpid()}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({**job, "epochs_done": epochs_done, "epochs": EPOCHS,
                       "elapsed": round(time.time() - t_start, 1),
                       "updated": time.time()}, f)
        os.replace(tmp, path)
    except OSError:
        pass


def _evaluate(qnn, loaders, params) -> dict:
    dev = torch.device("cpu")
    return {
        "train_acc": validate(loaders[0], qnn, dev, params)[1],
        "val_acc": validate(loaders[1], qnn, dev, params)[1],
        "val_aug_acc": validate(loaders[2], qnn, dev, params)[1],
    }


def run_job(job: dict) -> list[dict]:
    torch.set_num_threads(1)
    t0 = time.time()
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    loaders = loaders_for(job["task"], job["N"], job["seed"], job["qubits"])
    records = []
    params0 = _init_params(job)
    if job["kind"] == "sweep":
        raw = _raw_qnn(job)
        stats = _output_stats(raw, loaders[0], params0)
        qnn = _model(raw, stats)
        params = _train(qnn, loaders[0], params0, job)
        records.append({**job, "noise_p": 0.0, "noise_seed": None, **_evaluate(qnn, loaders, params)})
        if (job["N"], job["seed"]) == (NOISE_N, NOISE_PARAM_SEED):
            for p, ns in itertools.product(NOISE_P_TEST, NOISE_SEEDS):
                if p == 0.0 and ns != NOISE_SEEDS[0]:
                    continue
                # noiselessly-trained model (same fixed output statistics)
                noisy = _model(_raw_qnn(job, p, ns), stats)
                records.append({**job, "kind": "test_noise", "noise_p": p, "noise_seed": ns,
                                **_evaluate(noisy, loaders, params)})
    elif job["kind"] == "train_noise":
        raw = _raw_qnn(job, job["noise_p"], job["noise_seed"])
        stats = _output_stats(raw, loaders[0], params0)
        qnn = _model(raw, stats)
        params = _train(qnn, loaders[0], params0, job)
        records.append({**job, **_evaluate(qnn, loaders, params)})
    else:
        raise ValueError(job["kind"])
    for r in records:
        r["output_mean"], r["output_std"] = stats
    for r in records:
        r["seconds"] = round(time.time() - t0, 1)
        r["model_options"] = MODEL_OPTIONS
    return records


def job_key(job: dict) -> tuple:
    return (job["kind"], job["task"], job["qubits"], job["arch"], job["N"], job["seed"],
            job.get("noise_p", 0.0), job.get("noise_seed"))


def build_jobs(tasks, kinds, qubits: int, seeds=SEEDS, n_values=N_VALUES) -> list[dict]:
    jobs = []
    for task in tasks:
        if qubits > TASKS[task][1]:
            raise ValueError(f"{task} has too few pixels for {qubits} qubits")
        base = {"task": task, "qubits": qubits}
        if "sweep" in kinds:
            for arch, N, seed in itertools.product(ARCHS, n_values, seeds):
                jobs.append({**base, "kind": "sweep", "arch": arch, "N": N, "seed": seed})
        if "train_noise" in kinds:
            for arch, p, ns in itertools.product(ARCHS, NOISE_P_TRAIN, NOISE_SEEDS):
                if p == 0.0 and ns != NOISE_SEEDS[0]:
                    continue
                jobs.append({**base, "kind": "train_noise", "arch": arch, "N": NOISE_N,
                             "seed": NOISE_PARAM_SEED, "noise_p": p, "noise_seed": ns})
    return jobs


# Relative cost of one training epoch per image, measured: 12 qubits is
# ~3.4x slower than 8; the twirled model runs 8 circuits per image.
QUBIT_COST = {8: 1.0, 10: 1.8, 12: 3.4}


def estimated_cost(job: dict) -> float:
    cost = job["N"] * QUBIT_COST[job["qubits"]] * (8 if job["arch"] == "config10" else 1)
    if job["kind"] == "sweep" and (job["N"], job["seed"]) == (NOISE_N, NOISE_PARAM_SEED):
        cost *= 1.6  # + 61 test-time-noise evaluations of the trained model
    return cost + 5.0  # fixed per-job overhead (data loading)


def lpt_order(jobs: list[dict]) -> list[dict]:
    """Longest-processing-time-first: on a pool of identical workers this
    keeps the final stragglers short, minimizing total wall-clock time."""
    return sorted(jobs, key=estimated_cost, reverse=True)


def job_round(job: dict) -> int:
    """Round r (1..len(SEEDS)) a job belongs to: noiseless sweeps by their
    parameter seed; noisy-training jobs by their noise realization
    (NOISE_SEEDS[k] -> round k+1, the p = 0 reference run -> round 1).
    After round 1 every panel has a complete (single-seed) picture."""
    if job["kind"] == "sweep":
        return SEEDS.index(job["seed"]) + 1
    if job["noise_p"] == 0.0:
        return 1
    return NOISE_SEEDS.index(job["noise_seed"]) + 1


def rounds_in_lpt_order(jobs: list[dict]) -> list[tuple[int, list[dict]]]:
    """Jobs grouped into rounds (run strictly one after the other), each
    round internally in LPT order."""
    rounds: dict[int, list[dict]] = {}
    for j in jobs:
        rounds.setdefault(job_round(j), []).append(j)
    return [(r, lpt_order(rounds[r])) for r in sorted(rounds)]


# v2: campaign with the standardized output (the first campaign, with the
# default output init, stays in campaign_<q>q.jsonl and is not used).
RESULTS_STEM = "campaign_v2"


def out_path(qubits: int) -> str:
    return f"results_paper/{RESULTS_STEM}_{qubits}q.jsonl"


def imported_paths(qubits: int) -> list[str]:
    """Results produced on OTHER machines, brought here by src.sync_results."""
    import glob

    return sorted(glob.glob(f"results_paper/imported/{RESULTS_STEM}_{qubits}q.*.jsonl"))


def default_workers(gb_per_worker: float = 0.7) -> int:
    """All CPUs but two, capped by the RAM (~0.65 GB per worker at 8 qubits)."""
    cpus = os.cpu_count() or 2
    try:
        with open("/proc/meminfo") as f:
            mem_kb = next(int(line.split()[1]) for line in f if line.startswith("MemTotal"))
        by_mem = int((mem_kb / 1024**2 - 1.5) / gb_per_worker)
    except (OSError, StopIteration):
        by_mem = cpus
    return max(1, min(cpus - 2, by_mem))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", choices=(8, 10, 12), default=[8])
    ap.add_argument("--workers", type=int, default=None,
                    help="default: all CPUs but two, capped by available RAM")
    ap.add_argument("--rounds", type=int, nargs="+", default=None,
                    help="only run these rounds (default: all)")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--kinds", nargs="+", default=["sweep", "train_noise"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--n-values", nargs="+", type=int, default=list(N_VALUES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs("results_paper", exist_ok=True)
    jobs = []
    for q in args.qubits:
        tasks = [t for t in (args.tasks or TASKS) if TASKS[t][1] >= q]
        done = set()
        for path in [out_path(q)] + imported_paths(q):
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["kind"] in ("sweep", "train_noise"):
                        done.add(job_key(r))
        jobs += [j for j in build_jobs(tasks, args.kinds, q, args.seeds, args.n_values)
                 if job_key(j) not in done]
    rounds = rounds_in_lpt_order(jobs)
    if args.rounds:
        rounds = [(r, js) for r, js in rounds if r in args.rounds]
        jobs = [j for _, js in rounds for j in js]
    if args.workers is None:
        args.workers = default_workers()
    print(f"using {args.workers} worker processes", flush=True)
    total = sum(estimated_cost(j) for j in jobs)
    print(f"{len(jobs)} jobs to run in rounds "
          f"{[(r, len(js)) for r, js in rounds]}, estimated cost {total:.0f} units", flush=True)
    if args.dry_run:
        return
    for task, q in sorted({(j["task"], j["qubits"]) for j in jobs}):
        loaders_for(task, 40, 1, q)  # warm dataset caches once, serially

    ctx = get_context("spawn")
    files = {q: open(out_path(q), "a") for q in args.qubits}
    try:
        with ctx.Pool(args.workers, maxtasksperchild=10) as pool:
            for rnd, round_jobs in rounds:
                print(f"=== round {rnd}: {len(round_jobs)} jobs ===", flush=True)
                # imap_unordered only returns once the whole round is done,
                # so round r+1 never starts before round r has finished.
                for k, records in enumerate(pool.imap_unordered(run_job, round_jobs), 1):
                    f = files[records[0]["qubits"]]
                    for r in records:
                        f.write(json.dumps(r) + "\n")
                    f.flush()
                    r0 = records[0]
                    print(f"[round {rnd} {k}/{len(round_jobs)}] {r0['kind']} {r0['task']} "
                          f"{r0['qubits']}q {r0['arch']} N={r0['N']} seed={r0['seed']} "
                          f"p={r0.get('noise_p')} val={r0['val_acc']:.3f} "
                          f"aug={r0['val_aug_acc']:.3f} ({r0['seconds']}s)", flush=True)
                print(f"=== round {rnd} complete ===", flush=True)
    finally:
        for f in files.values():
            f.close()


if __name__ == "__main__":
    main()
