"""Publication figures from the campaign results (local + imported files).

    python -m src.paper_figures <out_dir> [--qubits 8]

Writes:
  accuracy_vs_N.pdf   noiseless test accuracy vs N, one panel per task
  watermark.pdf       watermark study: accuracy vs N on the three test sets
  gaps.pdf            invariance, shortcut and generalization gaps
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

TASKS = ("mnist45", "satellite", "ising", "eurosat_fi",
         "galaxy_round_edgeon", "galaxy_round_spiral", "resisc_airport_harbor")
TITLES = {
    "mnist45": "MNIST 4 vs 5",
    "satellite": "SATELLITE",
    "ising": "Ising",
    "eurosat_fi": "EuroSAT Forest vs Industrial",
    "galaxy_round_edgeon": "Galaxy10 round vs edge-on",
    "galaxy_round_spiral": "Galaxy10 round vs spiral",
    "resisc_airport_harbor": "RESISC45 airport vs harbor",
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
    ncols = 4
    nrows = math.ceil((len(TASKS) + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.9 * nrows), sharey=True, squeeze=False)
    for ax, task in zip(axes.flat, TASKS):
        draw(ax, {a: agg.get((task, a), {}) for a in ARCHS}, xs_log=True)
        ax.set_title(TITLES[task])
        ax.set_xlabel("training-set size $N$")
    for row in axes:
        row[0].set_ylabel("test accuracy")
    for ax in list(axes.flat)[len(TASKS):]:
        ax.axis("off")
    list(axes.flat)[-1].legend(handles=legend_handles(), loc="center", frameon=False)
    fig.tight_layout()
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


SHORT = {"mnist45": "MNIST", "satellite": "SATELLITE", "ising": "Ising", "eurosat_fi": "EuroSAT",
         "galaxy_round_edgeon": "Galaxy10\n(edge-on)", "galaxy_round_spiral": "Galaxy10\n(spiral)",
         "resisc_airport_harbor": "RESISC45"}


def fig_samples(out: str, n_per_class: int = 5) -> None:
    import torch

    from src.paper_experiments import loaders_for

    class_names = {
        "mnist45": ("digit 4", "digit 5"),
        "satellite": ("ship", "plane"),
        "ising": ("disordered", "ordered"),
        "eurosat_fi": ("Forest", "Industrial"),
        "galaxy_round_edgeon": ("round", "edge-on"),
        "galaxy_round_spiral": ("round", "spiral"),
        "resisc_airport_harbor": ("airport", "harbor"),
    }
    fig, axes = plt.subplots(len(TASKS), 2 * n_per_class, figsize=(8.4, 1.0 * len(TASKS)))
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
        axes[row, 0].set_ylabel(SHORT[task], fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)



WM_TASKS = ("mnist45", "satellite", "eurosat_fi", "galaxy_round_edgeon",
            "galaxy_round_spiral", "resisc_airport_harbor")
WM_STYLES = (("shortcut_acc", "-", "o", "watermarked test"),
             ("transformed_acc", "--", "s", "watermark transformed"),
             ("clean_acc", ":", "^", "no watermark"))
SHORT_LABEL = {"mnist45": "MNIST", "satellite": "SATELLITE", "ising": "Ising",
               "eurosat_fi": "EuroSAT", "galaxy_round_edgeon": "Gal. edge-on",
               "galaxy_round_spiral": "Gal. spiral", "resisc_airport_harbor": "RESISC45"}


def fig_watermark(records: list[dict], out: str) -> dict:
    """Watermark study: accuracy vs N on the three test sets, per task."""
    groups = collections.defaultdict(list)
    for r in records:
        if r["kind"] == "watermark":
            groups[(r["task"], r["arch"], r["N"])].append(r)
    agg = {k: [mean_sem([r[f] for r in rs]) for f, *_ in WM_STYLES] for k, rs in groups.items()}
    ncols = 3
    nrows = math.ceil(len(WM_TASKS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.9 * nrows + 0.8), sharey=True, squeeze=False)
    for ax, task in zip(axes.flat, WM_TASKS):
        for arch in ARCHS:
            Ns = sorted(N for (t, a, N) in agg if t == task and a == arch)
            for k, (_, ls, marker, _) in enumerate(WM_STYLES):
                ax.errorbar(Ns, [agg[(task, arch, N)][k][0] for N in Ns],
                            yerr=[agg[(task, arch, N)][k][1] for N in Ns], color=COLORS[arch],
                            ls=ls, marker=marker, ms=3.5, lw=1.2, capsize=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES])
        ax.set_ylim(0.2, 1.02)
        ax.axhline(0.5, color="gray", lw=0.7, ls=":")
        ax.grid(alpha=0.3)
        ax.set_title(TITLES[task])
        ax.set_xlabel("training-set size $N$")
    for row in axes:
        row[0].set_ylabel("test accuracy")
    handles = [plt.Line2D([], [], color=COLORS[a], ls=ls, marker=m, ms=4, label=f"{ARCH_LABELS[a]}, {lab}")
               for a in ARCHS for _, ls, m, lab in WM_STYLES]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {f"{t}|{a}|{N}": v for (t, a, N), v in agg.items()}


def _bars(ax, tasks: tuple, values: dict, ylabel: str) -> None:
    """Grouped bars: one group per task, one bar per architecture (mean +- SEM)."""
    width = 0.27
    for k, arch in enumerate(ARCHS):
        xs = [i + (k - 1) * width for i in range(len(tasks))]
        ms = [values.get((t, arch), (float("nan"), 0.0)) for t in tasks]
        ax.bar(xs, [m for m, _ in ms], width, yerr=[e for _, e in ms], color=COLORS[arch],
               capsize=2, label=ARCH_LABELS[arch])
    ax.set_xticks(range(len(tasks)), [SHORT_LABEL[t] for t in tasks], rotation=35, ha="right")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")


def fig_gaps(records: list[dict], records_wm: list[dict], out: str) -> dict:
    """(a) invariance gap (original - transformed test accuracy) without the
    watermark; (b) shortcut gap (watermarked - transformed-watermark test
    accuracy); both averaged over all N and seeds (mean +- SEM over runs);
    (c) generalization gap (train - test accuracy) vs N without the
    watermark, averaged over the tasks (mean +- SEM over tasks)."""
    inv, short = collections.defaultdict(list), collections.defaultdict(list)
    gen = collections.defaultdict(list)
    for r in records:
        if r["kind"] == "sweep":
            inv[(r["task"], r["arch"])].append(r["val_acc"] - r["val_aug_acc"])
            gen[(r["task"], r["arch"], r["N"])].append(r["train_acc"] - r["val_acc"])
    for r in records_wm:
        if r["kind"] == "watermark":
            short[(r["task"], r["arch"])].append(r["shortcut_acc"] - r["transformed_acc"])
    inv_ms = {k: mean_sem(v) for k, v in inv.items()}
    short_ms = {k: mean_sem(v) for k, v in short.items()}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.15, 1, 0.9]})
    _bars(axes[0], TASKS, inv_ms, "original $-$ transformed test acc.")
    axes[0].set_title("(a) invariance gap, no watermark")
    _bars(axes[1], WM_TASKS, short_ms, "watermarked $-$ transformed test acc.")
    axes[1].set_title("(b) shortcut gap, watermark in training")
    gen_ms = {}
    for arch in ARCHS:
        pts = [mean_sem([statistics.mean(gen[(t, arch, N)]) for t in TASKS if gen.get((t, arch, N))])
               for N in N_VALUES]
        gen_ms[arch] = dict(zip(map(str, N_VALUES), pts))
        axes[2].errorbar(N_VALUES, [m for m, _ in pts], yerr=[e for _, e in pts], color=COLORS[arch],
                         marker="o", ms=3.5, lw=1.2, capsize=2, label=ARCH_LABELS[arch])
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(N_VALUES, [str(n) for n in N_VALUES])
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_xlabel("training-set size $N$")
    axes[2].set_ylabel("train $-$ test accuracy")
    axes[2].set_title("(c) train $-$ test, no watermark")
    axes[2].grid(alpha=0.3)
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"invariance_gap": {f"{t}|{a}": v for (t, a), v in inv_ms.items()},
            "shortcut_gap": {f"{t}|{a}": v for (t, a), v in short_ms.items()},
            "generalization_gap": gen_ms}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--noise", action="store_true", help="also the noise figure")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    records = [r for r in load(f"results_paper/campaign_v2_{args.qubits}q.jsonl")
               if r["task"] in TASKS]
    summary = {
        "accuracy_vs_N": fig_accuracy(records, os.path.join(args.out_dir, "accuracy_vs_N.pdf")),
    }
    records_wm = [r for r in load(f"results_paper/campaign_v2_wm_{args.qubits}q.jsonl")
                  if r["task"] in WM_TASKS]
    if records_wm:
        summary["watermark"] = fig_watermark(records_wm, os.path.join(args.out_dir, "watermark.pdf"))
        summary["gaps"] = fig_gaps(records, records_wm, os.path.join(args.out_dir, "gaps.pdf"))
    if args.noise:
        summary["noise"] = fig_noise(records, os.path.join(args.out_dir, "noise.pdf"))
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
