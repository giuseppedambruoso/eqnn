"""Grid of noiseless test-accuracy-vs-N plots from a campaign results file:
one panel per task, six lines per panel (Equiv / NonEquiv /
NonEquiv-Twirled, each on the original and on the p4m-transformed test
set). The campaign runs in rounds, one parameter seed per round: each
(task, architecture, N) point shows the mean over the seeds finished so
far (+- SEM once there are at least two).

Usage:
    python -m src.plot_campaign results_paper/campaign_8q.jsonl out.pdf
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
    "planesnet": "PlanesNet (plane vs no plane)",
    "ising": "Ising (ordered vs disordered)",
    "eurosat_hr": "EuroSAT Highway vs River",
    "eurosat_fi": "EuroSAT Forest vs Industrial",
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


def load_points(path: str) -> tuple[dict, int]:
    """Returns ({(task, arch): {N: ((clean_mean, clean_sem), (aug_mean,
    aug_sem), n_seeds)}}, completed rounds), where a round r is complete
    once every (task, arch, N) point has at least r seeds."""
    runs = collections.defaultdict(dict)
    for file in result_files(path):
        with open(file) as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "sweep":
                    runs[(r["task"], r["arch"], r["N"])][r["seed"]] = r
    points = collections.defaultdict(dict)
    for (task, arch, N), by_seed in runs.items():
        rs = list(by_seed.values())
        points[(task, arch)][N] = (
            mean_sem([r["val_acc"] for r in rs]),
            mean_sem([r["val_aug_acc"] for r in rs]),
            len(rs),
        )
    all_keys = [(t, a, n) for t in TASK_TITLES for a in ARCHS for n in N_VALUES]
    rounds = min(len(runs.get(k, {})) for k in all_keys)
    return points, rounds


def plot(points: dict, rounds: int, out: str, qubits: int = 8) -> int:
    tasks = [t for t in TASK_TITLES]
    ncols = 4
    nrows = math.ceil(len(tasks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    n_points = 0
    for ax, task in zip(axes.flat, tasks):
        for arch in ARCHS:
            pts = points.get((task, arch), {})
            Ns = sorted(pts)
            n_points += len(Ns)
            if not Ns:
                continue
            for k, (style, marker, suffix) in enumerate(
                [("-", "o", "test"), ("--", "s", "rotated test")]
            ):
                ys = [pts[N][k][0] for N in Ns]
                es = [pts[N][k][1] for N in Ns]
                ax.errorbar(Ns, ys, yerr=es, color=COLORS[arch], ls=style, marker=marker,
                            ms=4, capsize=2, lw=1.4,
                            label=f"{ARCH_LABELS[arch]} ({suffix})")
        ax.set_title(TASK_TITLES[task], fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES])
        ax.set_xlim(32, 800)
        ax.set_ylim(0.3, 1.0)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("N (training-set size)")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(tasks):]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], color=COLORS[a], ls=ls, marker=m, label=f"{ARCH_LABELS[a]} ({s})")
        for a in ARCHS
        for ls, m, s in [("-", "o", "test"), ("--", "s", "rotated test")]
    ]
    legend_ax = list(axes.flat)[-1]
    legend_ax.legend(handles=handles, loc="center", fontsize=9, frameon=False)
    total = len(TASK_TITLES) * len(ARCHS) * len(N_VALUES)
    status = (f"all {len(SEEDS)} seeds complete" if rounds >= len(SEEDS)
              else f"round {rounds + 1}/{len(SEEDS)} in progress, {rounds} seed(s) complete everywhere")
    fig.suptitle(
        f"Noiseless test accuracy vs N ({qubits} qubits) - mean over finished seeds "
        f"($\\pm$ SEM from 2 seeds on) - {n_points}/{total} points - {status}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)
    return n_points


if __name__ == "__main__":
    pts, rnd = load_points(sys.argv[1])
    print(plot(pts, rnd, sys.argv[2]), rnd)
