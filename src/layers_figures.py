"""Generalization-vs-parameters figure: for L = 1, 2, 4, 6, 8 stacked
circuit layers (6*L trainable angles), the train/val_aug accuracy gap of
Equiv (config6), NonEquiv (config7) and NonEquiv-Twirled (config10) on
every task, at fixed N. A growing gap for NonEquiv as L increases (while
Equiv/Twirled, invariant by construction, stay flat) is the overfitting
signature this script is meant to surface — see CAMPAIGN.md and
scripts/run_layers_study.sh for how the data is produced.

Usage:
    python -m src.layers_figures results_paper/gap_vs_layers.pdf
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.plot_campaign import ARCH_LABELS, ARCHS, COLORS, SEEDS, TASK_TITLES, result_files  # noqa: E402

LAYERS = (1, 2, 4, 8)  # L=6 was skipped for this study
QUBITS = 8
TASKS = ("mnist45", "satellite", "ising", "eurosat_fi",
          "galaxy_round_edgeon", "galaxy_round_spiral", "resisc_airport_harbor")
N_FOCUS = 40  # smallest N: most parameters relative to data, clearest overfitting signal


def path_for(layers: int) -> str:
    stem = "campaign_v2" if layers == 1 else f"campaign_v2_L{layers}"
    return f"results_paper/{stem}_{QUBITS}q.jsonl"


def mean_sem(values: list[float]) -> tuple[float, float]:
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var / len(values))


def load(layers: int) -> list[dict]:
    records = []
    for path in result_files(path_for(layers)):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "sweep" and r["N"] == N_FOCUS:
                    records.append(r)
    return records


def gap_by_layers(task: str, arch: str) -> tuple[list[int], list[float], list[float]]:
    """(layers present, mean train-val_aug gap, SEM) across available seeds."""
    xs, means, sems = [], [], []
    for layers in LAYERS:
        records = [r for r in load(layers) if r["task"] == task and r["arch"] == arch]
        gaps = [r["train_acc"] - r["val_aug_acc"] for r in records]
        if not gaps:
            continue
        m, s = mean_sem(gaps)
        xs.append(layers)
        means.append(m)
        sems.append(s)
    return xs, means, sems


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "results_paper/gap_vs_layers.pdf"
    ncols = 4
    nrows = math.ceil(len(TASKS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharey=True)
    for ax, task in zip(axes.flat, TASKS):
        for arch in ARCHS:
            xs, means, sems = gap_by_layers(task, arch)
            if not xs:
                continue
            ax.errorbar(xs, means, yerr=sems, marker="o", ms=4, color=COLORS[arch],
                        label=ARCH_LABELS[arch])
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        ax.set_title(TASK_TITLES.get(task, task), fontsize=9)
        ax.set_xlabel("layers (6L params)")
        ax.set_xticks(list(LAYERS))
    for ax in list(axes.flat)[len(TASKS):]:
        ax.axis("off")
    axes.flat[0].set_ylabel(f"train_acc - val_aug_acc (N={N_FOCUS})")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Generalization gap vs. circuit depth (N={N_FOCUS})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
