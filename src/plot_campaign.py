"""Grid of noiseless test-accuracy-vs-N plots from a campaign results file:
one panel per task, six lines per panel (Equiv / NonEquiv /
NonEquiv-Twirled, each on the original and on the p4m-transformed test
set). The campaign runs in rounds, one parameter seed per round: each
(task, architecture, N) point shows the mean over the seeds finished so
far (+- SEM once there are at least two).

Usage:
    python -m src.plot_campaign results_paper/campaign_v2_8q.jsonl out.pdf
Prints "<plotted points> <completed rounds>" (used to detect updates).
"""

import collections
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Self-contained on purpose (no torch / pennylane imports: they take over a
# minute to load on a machine saturated by the campaign). Mirrors
# src.paper_experiments.
ARCHS = ("config6", "config7", "config10")
ARCH_LABELS = {"config6": "Equiv", "config7": "NonEquiv", "config10": "NonEquiv-Twirled"}
N_VALUES = (40, 80, 160, 320, 640)
SEEDS = (1, 2, 3, 4, 5, 6)

TASK_TITLES = {
    "mnist45": "MNIST 4 vs 5",
    "satellite": "SATELLITE (ship vs plane)",
    "ising": "Ising (ordered vs disordered)",
    "eurosat_fi": "EuroSAT Forest vs Industrial",
    "galaxy_round_edgeon": "Galaxy10 round vs edge-on",
    "galaxy_round_spiral": "Galaxy10 round vs spiral",
    "resisc_airport_harbor": "RESISC45 airport vs harbor",
}
COLORS = {"config6": "#1f5fa8", "config7": "#b8202e", "config10": "#d9a21b"}


def mean_sem(values: list[float]) -> tuple[float, float]:
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var / len(values))


def result_files(path: str) -> list[str]:
    """This machine's results plus those imported from other machines
    (src.sync_results): results_paper/imported/<name>.<machine>.jsonl."""
    import glob

    stem = os.path.splitext(os.path.basename(path))[0]
    imported = glob.glob(os.path.join(os.path.dirname(path), "imported", f"{stem}.*.jsonl"))
    return [p for p in [path] + sorted(imported) if os.path.exists(p)]


TASK_MAX_QUBITS = {"mnist45": 8, "satellite": 8, "ising": 12, "eurosat_fi": 12,
                   "galaxy_round_edgeon": 12, "galaxy_round_spiral": 12,
                   "resisc_airport_harbor": 12}

# (kind, [(record field, line style, marker, label)]) per study
STANDARD = ("sweep", [("val_acc", "-", "o", "test"), ("val_aug_acc", "--", "s", "rotated test")])
WATERMARK = ("watermark", [("shortcut_acc", "-", "o", "watermarked test"),
                           ("transformed_acc", "--", "s", "watermark transformed"),
                           ("clean_acc", ":", "^", "no watermark")])


def study_of(path: str) -> tuple:
    return WATERMARK if "_wm" in os.path.basename(path) else STANDARD


def tasks_with_data(path: str) -> list[str]:
    """Tasks for this qubit count that appear in the results (the standard
    grid also works while only some tasks have been run)."""
    kind, _ = study_of(path)
    present = set()
    for file in result_files(path):
        with open(file) as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == kind:
                    present.add(r["task"])
    return [t for t in tasks_for(qubits_of(path)) if t in present] or tasks_for(qubits_of(path))


def qubits_of(path: str) -> int:
    import re

    match = re.search(r"_(\d+)q", os.path.basename(path))
    return int(match.group(1)) if match else 8


def tasks_for(qubits: int) -> list[str]:
    return [t for t in TASK_TITLES if TASK_MAX_QUBITS[t] >= qubits]


def load_points(path: str) -> tuple[dict, int]:
    """Returns ({(task, arch): {N: ([(mean, sem) per plotted field], n_seeds)}},
    completed rounds), where a round r is complete once every
    (task, arch, N) point has at least r seeds."""
    kind, fields = study_of(path)
    runs = collections.defaultdict(dict)
    for file in result_files(path):
        with open(file) as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == kind:
                    runs[(r["task"], r["arch"], r["N"])][r["seed"]] = r
    points = collections.defaultdict(dict)
    for (task, arch, N), by_seed in runs.items():
        rs = list(by_seed.values())
        points[(task, arch)][N] = ([mean_sem([r[f] for r in rs]) for f, *_ in fields], len(rs))
    all_keys = [(t, a, n) for t in tasks_with_data(path) for a in ARCHS for n in N_VALUES]
    rounds = min(len(runs.get(k, {})) for k in all_keys)
    return points, rounds


def plot(points: dict, rounds: int, out: str, path: str) -> int:
    qubits = qubits_of(path)
    _, fields = study_of(path)
    tasks = tasks_with_data(path)
    ncols = 3  # one extra panel slot holds the legend
    nrows = math.ceil((len(tasks) + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    n_points = 0
    for ax, task in zip(axes.flat, tasks):
        for arch in ARCHS:
            pts = points.get((task, arch), {})
            Ns = sorted(pts)
            n_points += len(Ns)
            for k, (_, style, marker, label) in enumerate(fields):
                if not Ns:
                    continue
                ax.errorbar(Ns, [pts[N][0][k][0] for N in Ns], yerr=[pts[N][0][k][1] for N in Ns],
                            color=COLORS[arch], ls=style, marker=marker, ms=4, capsize=2, lw=1.4)
        ax.set_title(TASK_TITLES[task], fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES])
        ax.set_xlim(32, 800)
        ax.set_ylim(0.3, 1.02)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("N (training-set size)")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(tasks):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=COLORS[a], ls=ls, marker=m, label=f"{ARCH_LABELS[a]} ({lab})")
               for a in ARCHS for _, ls, m, lab in fields]
    list(axes.flat)[-1].legend(handles=handles, loc="center", fontsize=8, frameon=False)
    total = len(tasks) * len(ARCHS) * len(N_VALUES)
    status = (f"all {len(SEEDS)} seeds complete" if rounds >= len(SEEDS)
              else f"round {rounds + 1}/{len(SEEDS)} in progress, {rounds} seed(s) complete everywhere")
    what = ("Watermark shortcut study (watermark in training)" if fields is WATERMARK[1]
            else "Noiseless test accuracy vs N")
    fig.suptitle(f"{what} ({qubits} qubits) - mean over finished seeds ($\\pm$ SEM from 2 seeds on)"
                 f" - {n_points}/{total} points - {status}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)
    return n_points


if __name__ == "__main__":
    pts, rnd = load_points(sys.argv[1])
    print(plot(pts, rnd, sys.argv[2], sys.argv[1]), rnd)
