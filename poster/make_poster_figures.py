"""Generates the poster's result figures: one polished, publication-quality
image per (class pair, augment_train) panel, plus a combined 3x3 grid.

Reads results_def/summary.csv directly (already deduplicated by
collect_results.py, one row per run) — a pure read, unlike
src.plot_wandb_results's local-fetch path, which also PRUNES duplicate
run directories on disk as a side effect. That pruning targets outputs/
and multirun/ (ephemeral, gitignored) and must never be pointed at
results_def/ (git-tracked) — doing so once already deleted ~390 tracked
files here (recovered via `git checkout -- results_def/`, since nothing
had been staged). This script only changes presentation, not results.

Satellite has no augment_train="online" runs (the sweep only covered
none/once) — per explicit instruction, the "once" series is duplicated
under "online" as a stated approximation (annotated with a dagger on the
figure) rather than left blank.

Usage:
    poetry run python3 poster/make_poster_figures.py
"""

import copy
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(_PROJECT_ROOT))
from src.plot_wandb_results import _normalize_augment_train  # noqa: E402
SUMMARY_CSV = _PROJECT_ROOT / "results_def" / "summary.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_VALUES = [40, 80, 160, 320, 640]
ARCHITECTURES = ["config6", "config7"]
ARCH_LABELS = {"config6": "Equiv", "config7": "NonEquiv"}
ARCH_COLORS = {"config6": "#2166AC", "config7": "#B2182B"}
METRIC_LABELS = {"val": "validation accuracy", "val_aug": "validation accuracy (D4-augmented)"}
METRIC_LINESTYLES = {"val": "-", "val_aug": "--"}
METRIC_MARKERS = {"val": "o", "val_aug": "s"}

ROWS = [
    # (class1, class2, dataset, readout, row_title)
    (0, 1, "satellite", "avg_x", "Satellite: ship (0) vs plane (1)"),
    (3, 4, "mnist", "x0_xhalf", "MNIST: digit 3 vs 4"),
    (4, 5, "mnist", "x0_xhalf", "MNIST: digit 4 vs 5"),
]
AUGMENT_MODES = ["none", "online", "once"]

plt.rcParams.update(
    {
        "font.size": 13,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 1.1,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.4,
        "lines.markersize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / len(values) ** 0.5


def _load_grouped():
    """{(architecture, augment_train, readout, dataset, class1, class2): {N: {"val": [...], "val_aug": [...]}}}"""
    grouped: dict = {}
    with open(SUMMARY_CSV) as f:
        for row in csv.DictReader(f):
            if row["architecture"] not in ARCHITECTURES:
                continue
            if not row["N"] or int(row["N"]) not in N_VALUES:
                continue
            if not row["val_accuracy"] or not row["val_aug_accuracy"]:
                continue
            augment_train = _normalize_augment_train(row["augment_train"]) or "none"
            key = (
                row["architecture"],
                augment_train,
                row["readout"],
                row["dataset"],
                int(row["class1"]),
                int(row["class2"]),
            )
            bucket = grouped.setdefault(key, {}).setdefault(
                int(row["N"]), {"val": [], "val_aug": []}
            )
            bucket["val"].append(float(row["val_accuracy"]))
            bucket["val_aug"].append(float(row["val_aug_accuracy"]))
    return grouped


def _synthesize_satellite_online(grouped):
    """Copies every satellite augment_train="once" group's data onto the
    corresponding "online" key — see module docstring."""
    augmented = dict(grouped)
    for key, by_n in list(grouped.items()):
        architecture, augment_train, readout, dataset, class1, class2 = key
        if augment_train != "once" or dataset != "satellite":
            continue
        online_key = (architecture, "online", readout, dataset, class1, class2)
        augmented[online_key] = copy.deepcopy(by_n)
    return augmented


def _plot_panel(ax, grouped, architectures, augment_train, readout, dataset, class1, class2):
    for architecture in architectures:
        key = (architecture, augment_train, readout, dataset, class1, class2)
        if key not in grouped or not grouped[key]:
            continue
        points = sorted(grouped[key].items())
        Ns = [p[0] for p in points]
        for metric in ("val", "val_aug"):
            means = [sum(p[1][metric]) / len(p[1][metric]) for p in points]
            sems = [_sem(p[1][metric]) for p in points]
            ax.errorbar(
                Ns,
                means,
                yerr=sems,
                label=f"{ARCH_LABELS[architecture]} — {METRIC_LABELS[metric]}",
                color=ARCH_COLORS[architecture],
                linestyle=METRIC_LINESTYLES[metric],
                marker=METRIC_MARKERS[metric],
                capsize=3,
                elinewidth=1.3,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_ylim(0.30, 1.0)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.axhline(0.5, color="0.75", linewidth=1.0, linestyle=":", zorder=0)
    ax.set_xlabel("N (training set size)")
    ax.set_ylabel("accuracy")


def main():
    grouped = _load_grouped()
    grouped = _synthesize_satellite_online(grouped)

    # --- individual panels ---
    for class1, class2, dataset, readout, row_title in ROWS:
        for augment_train in AUGMENT_MODES:
            is_approximated = dataset == "satellite" and augment_train == "online"
            fig, ax = plt.subplots(figsize=(6.4, 6.3 if is_approximated else 6.0))
            _plot_panel(ax, grouped, ARCHITECTURES, augment_train, readout, dataset, class1, class2)
            title = f"{row_title}\naugment_train={augment_train}, readout={readout}"
            ax.set_title(title, fontsize=13)
            if is_approximated:
                fig.text(
                    0.5,
                    0.965 if is_approximated else 1.0,
                    "†approximated from augment_train=once (not separately trained)",
                    ha="center",
                    fontsize=10,
                    style="italic",
                    color="0.35",
                )
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=1,
                frameon=False,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94) if is_approximated else None)
            fname = f"{dataset}_{class1}v{class2}_{augment_train}.png"
            fig.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {OUTPUT_DIR / fname}")

    # --- combined 3x3 grid ---
    fig, axes = plt.subplots(3, 3, figsize=(19, 15), squeeze=False)
    for row, (class1, class2, dataset, readout, row_title) in enumerate(ROWS):
        for col, augment_train in enumerate(AUGMENT_MODES):
            ax = axes[row][col]
            _plot_panel(ax, grouped, ARCHITECTURES, augment_train, readout, dataset, class1, class2)
            subtitle = f"augment_train={augment_train}"
            if dataset == "satellite" and augment_train == "online":
                subtitle += "†"
            ax.set_title(subtitle, fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{row_title}\n(readout={readout})\naccuracy", fontsize=12)
            else:
                ax.set_ylabel("")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=14,
    )
    fig.suptitle(
        "Equivariant vs. non-equivariant quantum classifier: accuracy vs. training-set size N",
        fontsize=18,
        y=0.995,
    )
    fig.text(
        0.5,
        0.965,
        "† satellite augment_train=online approximated from augment_train=once (not separately trained)",
        ha="center",
        fontsize=11,
        style="italic",
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.955))
    combined_path = OUTPUT_DIR / "combined_grid.png"
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)
    print(f"wrote {combined_path}")


if __name__ == "__main__":
    main()
