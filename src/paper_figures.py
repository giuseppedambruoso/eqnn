"""Publication figures from the campaign results (local + imported files).

    python -m src.paper_figures <out_dir> [--qubits 8]

Writes:
  accuracy_vs_N.pdf   noiseless test accuracy vs N, one panel per task
  noise.pdf           accuracy vs depolarizing probability (N = 80):
                      noise in training and test / in test only
  dataset_samples.png examples of every task as seen by the model
  summary.json        the aggregated numbers used in the text
"""

import argparse
import collections
import json
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.plot_campaign import ARCH_LABELS, ARCHS, COLORS, N_VALUES, result_files  # noqa: E402

TASKS = ("mnist45", "satellite", "ising", "eurosat_fi")
TITLES = {
    "mnist45": "MNIST 4 vs 5",
    "satellite": "SATELLITE",
    "ising": "Ising",
    "eurosat_fi": "EuroSAT Forest vs Industrial",
}
STYLES = (("-", "o", "original test set"), ("--", "s", "p4m-transformed test set"))

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8})


def load(path: str) -> list[dict]:
    rows = {}
    for file in result_files(path):
        with open(file) as f:
            for line in f:
                r = json.loads(line)
                key = (r["kind"], r["task"], r["arch"], r["N"], r["seed"],
                       r.get("noise_p", 0.0), r.get("noise_seed"))
                rows[key] = r
    return list(rows.values())


def mean_sem(values: list[float]) -> tuple[float, float]:
    m = statistics.mean(values)
    return m, (statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0)


def aggregate(records: list[dict], kind: str, x: str) -> dict:
    """{(task, arch): {x_value: ((clean_m, clean_s), (rot_m, rot_s), n)}}"""
    groups = collections.defaultdict(list)
    for r in records:
        if r["kind"] == kind:
            groups[(r["task"], r["arch"], r[x])].append(r)
    out = collections.defaultdict(dict)
    for (task, arch, xv), rs in groups.items():
        out[(task, arch)][xv] = (mean_sem([r["val_acc"] for r in rs]),
                                 mean_sem([r["val_aug_acc"] for r in rs]), len(rs))
    return out


def legend_handles():
    return [plt.Line2D([], [], color=COLORS[a], ls=ls, marker=m, ms=4,
                       label=f"{ARCH_LABELS[a]}, {s}")
            for a in ARCHS for ls, m, s in STYLES]


def draw(ax, series: dict, xs_log: bool) -> None:
    for arch in ARCHS:
        pts = series.get(arch, {})
        xs = sorted(pts)
        for k, (ls, marker, _) in enumerate(STYLES):
            ax.errorbar(xs, [pts[x][k][0] for x in xs], yerr=[pts[x][k][1] for x in xs],
                        color=COLORS[arch], ls=ls, marker=marker, ms=3.5, lw=1.2, capsize=2)
    if xs_log:
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES])
    ax.set_ylim(0.3, 1.02)
    ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.grid(alpha=0.3)


def fig_accuracy(records: list[dict], out: str) -> dict:
    agg = aggregate(records, "sweep", "N")
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
    for ax, task in zip(axes, TASKS):
        draw(ax, {a: agg.get((task, a), {}) for a in ARCHS}, xs_log=True)
        ax.set_title(TITLES[task])
        ax.set_xlabel("training-set size $N$")
    axes[0].set_ylabel("test accuracy")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {f"{t}|{a}": {str(n): v for n, v in agg[(t, a)].items()} for (t, a) in agg}


def fig_noise(records: list[dict], out: str) -> dict:
    both = aggregate(records, "train_noise", "noise_p")
    test = aggregate(records, "test_noise", "noise_p")
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharey=True, sharex=True)
    for col, task in enumerate(TASKS):
        for row, (agg, label) in enumerate([(both, "noise in training and test"),
                                            (test, "noise in test only")]):
            ax = axes[row, col]
            draw(ax, {a: agg.get((task, a), {}) for a in ARCHS}, xs_log=False)
            if row == 0:
                ax.set_title(TITLES[task])
            else:
                ax.set_xlabel("depolarizing probability $p$")
            if col == 0:
                ax.set_ylabel(f"test accuracy\n({label})")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {kind: {f"{t}|{a}": {str(p): v for p, v in agg[(t, a)].items()} for (t, a) in agg}
            for kind, agg in (("train_and_test", both), ("test_only", test))}


def fig_samples(out: str, n_per_class: int = 5) -> None:
    import torch

    from src.paper_experiments import loaders_for

    class_names = {
        "mnist45": ("digit 4", "digit 5"),
        "satellite": ("ship", "plane"),
        "ising": ("disordered", "ordered"),
        "eurosat_fi": ("Forest", "Industrial"),
    }
    fig, axes = plt.subplots(len(TASKS), 2 * n_per_class, figsize=(8.4, 4.0))
    for row, task in enumerate(TASKS):
        # the first test images of each class, as encoded (16x16 amplitudes)
        loader = loaders_for(task, 80, 1, 8)[1]
        xs = torch.cat([x for x, _ in loader])
        ys = torch.cat([y for _, y in loader])
        for cls in (0, 1):
            picks = xs[ys == cls][:n_per_class]
            for k, state in enumerate(picks):
                ax = axes[row, cls * n_per_class + k]
                ax.imshow(state.reshape(16, 16).numpy(), cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                if k == 0:
                    ax.set_title(class_names[task][cls], fontsize=8, loc="left")
        axes[row, 0].set_ylabel(TITLES[task].replace(" Forest vs Industrial", ""), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--no-samples", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    records = [r for r in load(f"results_paper/campaign_v2_{args.qubits}q.jsonl")
               if r["task"] in TASKS]
    summary = {
        "accuracy_vs_N": fig_accuracy(records, os.path.join(args.out_dir, "accuracy_vs_N.pdf")),
        "noise": fig_noise(records, os.path.join(args.out_dir, "noise.pdf")),
    }
    sweeps = [r for r in records if r["kind"] == "sweep"]
    summary["stuck_runs"] = collections.Counter(
        f"{r['task']}|{r['arch']}" for r in sweeps
        if abs(r["val_acc"] - 0.5) < 0.013 and abs(r["train_acc"] - 0.5) < 0.013)
    summary["n_sweep_runs"] = len(sweeps)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if not args.no_samples:
        fig_samples(os.path.join(args.out_dir, "dataset_samples.png"))


if __name__ == "__main__":
    main()
