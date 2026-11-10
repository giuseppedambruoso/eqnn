"""Every data figure and table of the paper, from the runs in results/ (into paper/figures and paper/tables).

    python analysis/paper_figures.py              # all of them (about one minute)
    python analysis/paper_figures.py fig_cues     # one of them (the name of a function below)

Data (one JSON line per run, written by eqnn.runner; see the README for the fields):
  results/mnist_pose         MNIST 4 vs 5 in random pose, with and without the cues
  results/mnist_upright      MNIST 4 vs 5, upright images (control)
  results/synthetic          synthetic cross vs square, with and without the cues
  results/noise_train_test   trained and tested under realistic noise
  results/noise_test_only    trained without noise, the final angles evaluated under noise
  results/synthetic_pilots   pilot variants of the synthetic task
  results/dla.json           dimensions of the DLAs and of the function classes (eqnn.dla)

Every comparison between models uses the seeds shared by all the compared series, so that it is paired: the same
training images, poses, initial angles and order of the mini-batches for every model.
"""

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import (COMPARISON_1, COMPARISON_2, CUE_TASKS, CUES, LAYERS, MAIN, RESULT_DIR, MODELS, NATURAL, NOISE_SCALES,  # noqa: E402
                             RESULTS, TASK_SHORT, TASKS, UPRIGHT, cells, common, fmt, fmt_p, model_handles, ms,
                             paired, panel, plt, row_labels, runs, save, style, task_cells, write_table)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TASK_CLASSES = {"mnist_bin": ("4", "5"), "shapes_cs": ("square", "cross"), "galaxy_0_4": ("face-on", "edge-on"),
                "ery_1_2": ("elongated", "other")}
TRAIN_TEST = (("train_acc", 0.95, None), ("clean_acc", 0.35, "////"))
# paired comparisons of the paper: (first, second); the differences are first - second (test accuracy) and
# second - first (gap)
MAIN_LAYERS = tuple(int(v) for v in os.environ.get("EQNN_MAIN_LAYERS", "4").split(","))  # depths of the tables
PAIRS = [("config6", "config7"), ("config10", "config7"), ("config6", "config10"), ("config6", "config6n"),
         ("config6n", "config7")]


def _bars(ax, task, archs, w=2.4, gap_y=1.03, fs=7.8):
    """Training (solid) and test (hatched) accuracy of the models in the main setting, with the single seeds."""
    c = task_cells(task)
    L, N = MAIN[task]
    seeds = common(*[c[(L, N, a)] for a in archs])
    rng = np.random.default_rng(0)
    for j, arch in enumerate(archs):
        col = MODELS[arch][2]
        rs = [c[(L, N, arch)][s] for s in seeds]
        for k, (key, alpha, hatch) in enumerate(TRAIN_TEST):
            v = [r[key] for r in rs]
            m, e = ms(v)
            x = j * w + k * 1.05
            ax.bar(x, m, 0.95, color=col, alpha=alpha, hatch=hatch, edgecolor=col, lw=0.7, yerr=e, capsize=1.8,
                   error_kw={"lw": 0.7})
            ax.scatter(x + rng.uniform(-0.25, 0.25, len(v)), v, s=3.5, color="k", alpha=0.4, zorder=3, lw=0)
        g = ms([r["gap"] for r in rs])[0]
        ax.text(j * w + 0.52, gap_y, f"gap\n{g:.2f}", ha="center", va="bottom", fontsize=fs, color=col,
                fontweight="bold")
    ax.set_xticks([j * w + 0.52 for j in range(len(archs))])
    ax.set_xticklabels([MODELS[a][1] for a in archs])
    ax.set_ylim(0.5, 1.13)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("accuracy")
    style(ax)


def _paired_panel(ax, task, pairs, key, sign, ylabel, rng, labels=None):
    """Per-seed differences of the pairs of models in the main setting: bars (mean and standard error) and dots (the
    seeds)."""
    c = task_cells(task)
    L, N = MAIN[task]
    seeds = common(*[c[(L, N, a)] for p in pairs for a in p])
    for k, (a, b) in enumerate(pairs):
        m, e, _, _, _, d = paired(c[(L, N, a)], c[(L, N, b)], key, sign, seeds)
        col = MODELS[a][2]
        ax.bar(k, m, 0.6, color=col, alpha=0.85, edgecolor=col, lw=0.6, yerr=e, capsize=2, error_kw={"lw": 0.7})
        ax.scatter(k + rng.uniform(-0.16, 0.16, len(d)), d, s=4.5, color="k", alpha=0.5, zorder=3, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(labels or [f"{MODELS[a][1]}\nvs {MODELS[b][1]}" for a, b in pairs], fontsize=7.8)
    ax.set_xlim(-0.6, len(pairs) - 0.4)
    ax.set_ylabel(ylabel)
    style(ax)


# ================================================================ figures of the main text
def fig_tasks():
    """Training images of the tasks and the three cues."""
    from eqnn.data import build_data
    nt = len(TASKS)
    fig = plt.figure(figsize=(7.0, 0.3 + 0.88 * nt))
    gs = fig.add_gridspec(nt, 11, width_ratios=[1] * 4 + [0.3] + [1] * 6, wspace=0.06, hspace=0.34,
                          left=0.035, right=1.0, top=1 - 0.3 / (0.3 + 0.88 * nt), bottom=0.0)
    axes = np.empty((nt, 10), dtype=object)
    for row, task in enumerate(TASKS):
        names = TASK_CLASSES[task]
        x, y, _, _ = build_data(task, 40, 1, NATURAL[task])
        idx = [int(i) for c in (0, 1) for i in (y == c).nonzero()[:2, 0]]
        images = [(x[i], names[int(y[i])]) for i in idx]
        for sc, _ in CUES.get(task, ()):
            xc, yc, _, _ = build_data(task, 20, 1, sc)
            images += [(xc[int((yc == c).nonzero()[0, 0])], names[c]) for c in (0, 1)]
        for k, (img, name) in enumerate(images):
            ax = fig.add_subplot(gs[row, k if k < 4 else k + 1])
            ax.imshow(img.reshape(16, 16).numpy(), cmap="gray_r", vmin=0)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            ax.set_title(name, fontsize=7.8, pad=1.5)
            axes[row, k] = ax
    for text, a, b in (("no cue", 0, 3), ("corner watermark", 4, 5), ("brightness gradient", 6, 7),
                       ("center/corner mark", 8, 9)):
        x0, x1 = axes[0, a].get_position().x0, axes[0, b].get_position().x1
        y = axes[0, a].get_position().y1 + 0.22 / (0.3 + 0.88 * len(TASKS))
        fig.text((x0 + x1) / 2, y + 0.015, text, ha="center", va="bottom", fontsize=8.7)
        fig.add_artist(Line2D([x0 + 0.005, x1 - 0.005], [y, y], color="0.3", lw=0.6, transform=fig.transFigure))
    row_labels(fig, axes[:, 0], [TASKS[t].replace(", ", ",\n") for t in TASKS], x=0.0)
    save(fig, "fig_tasks.pdf")


def _vs_n(ax, task, archs, key, L=4, upright=False):
    """key (test accuracy or gap) as a function of N at depth L, on the seeds shared by the models (and by the upright
    control, if shown)."""
    c = task_cells(task)
    up = cells(runs("mnist_upright", "none")) if upright else {}
    for arch in archs:
        _, _, col, mk = MODELS[arch]
        ns = sorted(n for (lay, n, a) in c if lay == L and a == arch)
        v = [ms([c[(L, n, arch)][s][key] for s in common(*[c[(L, n, a)] for a in archs], up.get((L, n, "config7")))])
             for n in ns]
        ax.errorbar(ns, [a for a, _ in v], yerr=[b for _, b in v], color=col, marker=mk, ms=3.5, lw=1.2, capsize=1.2)
    if upright:
        lab, col, mk, ls = UPRIGHT
        ns = sorted(n for (lay, n, a) in up if lay == L and a == "config7")
        v = [ms([up[(L, n, "config7")][s][key] for s in common(*[c[(L, n, a)] for a in archs],
                                                               up[(L, n, "config7")])]) for n in ns]
        ax.errorbar(ns, [a for a, _ in v], yerr=[b for _, b in v], color=col, ls=ls, marker=mk, ms=3.5, lw=1.2,
                    capsize=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([10, 20, 40, 80]), ax.set_xticklabels(["10", "20", "40", "80"])
    ax.set_xlim(10 / 1.3, 80 * 1.3)
    ax.minorticks_off()
    ax.set_xlabel("training images $N$")
    if key == "gap":
        ax.axhline(0, color="k", lw=0.5)
    style(ax, "both")


def fig_main():
    """The four models on the four tasks in the main settings (L = 4; N = 40, or 20 on the synthetic task; 15 seeds):
    training and test accuracy with the generalization gap (top), and per-seed differences in test accuracy between
    pairs of models (bottom)."""
    archs = list(MODELS)
    pairs = [("config6", "config7"), ("config10", "config7"), ("config6", "config10"), ("config6", "config6n")]
    labels = ["Equiv $-$ NonEquiv", "Twirled $-$ NonEquiv", "Equiv $-$ Twirled", "Equiv $-$ NoSym"]
    nt = len(TASKS)
    fig, axes = plt.subplots(2, nt, figsize=(7.0, 4.4), gridspec_kw={"height_ratios": [1.15, 1]})
    rng = np.random.default_rng(1)
    for j, task in enumerate(TASKS):
        ax = axes[0, j]
        _bars(ax, task, archs, w=2.35, gap_y=1.02, fs=7.4)
        ax.set_xticklabels([MODELS[a][1] for a in archs], rotation=35, ha="right", rotation_mode="anchor",
                           fontsize=7.8)
        L, N = MAIN[task]
        ax.set_title(f"{TASK_SHORT[task]}\n($L={L}$, $N={N}$)")
        if j:
            ax.set_ylabel("")
        _paired_panel(axes[1, j], task, pairs, "clean_acc", 1, "difference in test accuracy" if j == 0 else "", rng,
                      labels=labels)
        plt.setp(axes[1, j].get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor", fontsize=7.8)
        panel(axes[0, j], "abcd"[j], x=-0.2 if j else -0.33, y=1.13)
        panel(axes[1, j], "efgh"[j], x=-0.2 if j else -0.33, y=1.04)
    ylo = min(ax.get_ylim()[0] for ax in axes[1])
    yhi = max(ax.get_ylim()[1] for ax in axes[1])
    for ax in axes[1]:
        ax.set_ylim(ylo, yhi)
    hs = model_handles(archs) + [Patch(facecolor="0.35", edgecolor="0.35", label="training"),
                                 Patch(facecolor="0.35", alpha=0.35, hatch="////", edgecolor="0.35", label="test")]
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False, columnspacing=1.0, handlelength=1.6,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.0, w_pad=0.6)
    save(fig, "fig_main.pdf")


def fig_scaling():
    """Dependence on the number of training images and on the depth: test accuracy and gap as functions of N (L = 4;
    on MNIST also NonEquiv on upright digits), and gap as a function of L (N of the main setting)."""
    archs = list(MODELS)
    nt = len(TASKS)
    fig, axes = plt.subplots(3, nt, figsize=(7.0, 4.9))
    for j, task in enumerate(TASKS):
        _vs_n(axes[0, j], task, archs, "clean_acc")
        _vs_n(axes[1, j], task, archs, "gap", upright=task == "mnist_bin")
        axes[0, j].set_xlabel("")
        axes[0, j].set_title(f"{TASK_SHORT[task]}\n($L=4$)")
        c = task_cells(task)
        _, N = MAIN[task]
        ax = axes[2, j]
        layers = LAYERS[task]
        for arch in archs:
            _, _, col, mk = MODELS[arch]
            v = [ms([c[(L, N, arch)][s]["gap"] for s in common(*[c[(L, N, a)] for a in archs])]) for L in layers]
            ax.errorbar(layers, [a for a, _ in v], yerr=[b for _, b in v], color=col, marker=mk, ms=3.5, lw=1.2,
                        capsize=1.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(layers)), ax.set_xticklabels([str(L) for L in layers])
        ax.minorticks_off()
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("layers $L$")
        ax.set_title(f"$N={N}$")
        style(ax, "both")
        for i in range(3):
            panel(axes[i, j], "abcdefghijkl"[nt * i + j], x=-0.2 if j else -0.36, y=1.02)
    axes[0, 0].set_ylabel("test accuracy")
    axes[1, 0].set_ylabel("generalization gap")
    axes[2, 0].set_ylabel("generalization gap")
    lab, col, mk, ls = UPRIGHT
    hs = model_handles(archs) + [Line2D([], [], color=col, ls=ls, marker=mk, ms=3.5, label=lab)]
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.8, w_pad=0.5)
    save(fig, "fig_scaling.pdf")


def fig_grid(task):
    """Comparison 1 over the whole grid: accuracies and gap as functions of N, for every depth."""
    c = task_cells(task)
    up = cells(runs("mnist_upright", "none")) if task == "mnist_bin" else {}
    layers = LAYERS[task]
    fig, axes = plt.subplots(2, len(layers), figsize=(7.0, 3.6), sharex=True, sharey="row")

    def sel(L, n):  # seeds shared by the models (and by the upright control, if it is shown) at (L, n)
        return common(*[c.get((L, n, a)) for a in COMPARISON_1], up.get((L, n, "config7")))

    for j, L in enumerate(layers):
        for arch in COMPARISON_1:
            _, _, col, mk = MODELS[arch]
            ns = sorted(n for (lay, n, a) in c if lay == L and a == arch)
            for key, ls, mfc in (("train_acc", ":", col), ("clean_acc", "-", "white")):
                v = [ms([c[(L, n, arch)][s][key] for s in sel(L, n)]) for n in ns]
                axes[0, j].errorbar(ns, [a for a, _ in v], yerr=[b for _, b in v], color=col, ls=ls, marker=mk,
                                    mfc=mfc, ms=3.5, lw=1.1, capsize=1.2)
            v = [ms([c[(L, n, arch)][s]["gap"] for s in sel(L, n)]) for n in ns]
            axes[1, j].errorbar(ns, [a for a, _ in v], yerr=[b for _, b in v], color=col, marker=mk, ms=3.5, lw=1.2,
                                capsize=1.2)
        if any(lay == L for (lay, n, a) in up):
            lab, col, mk, ls = UPRIGHT
            ns = sorted(n for (lay, n, a) in up if lay == L and a == "config7")
            v = [ms([up[(L, n, "config7")][s]["gap"] for s in sel(L, n)]) for n in ns]
            axes[1, j].errorbar(ns, [a for a, _ in v], yerr=[b for _, b in v], color=col, ls=ls, marker=mk, ms=3.5,
                                lw=1.2, capsize=1.2)
        axes[0, j].set_title(f"$L={L}$ ({6 * L} angles)")
        axes[1, j].axhline(0, color="k", lw=0.5)
        axes[1, j].set_xlabel("training images $N$")
        for ax in axes[:, j]:
            ax.set_xscale("log", base=2)
            ax.set_xticks([10, 20, 40, 80]), ax.set_xticklabels(["10", "20", "40", "80"])
            ax.set_xlim(10 / 1.3, 80 * 1.3)
            ax.minorticks_off()
            style(ax, "both")
    panel(axes[0, 0], "a", x=-0.42, y=1.0)
    panel(axes[1, 0], "b", x=-0.42, y=1.0)
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("generalization gap")
    hs = model_handles(COMPARISON_1) + [
        Line2D([], [], color="0.3", ls=":", marker="o", mfc="0.3", ms=3.5, label="training"),
        Line2D([], [], color="0.3", ls="-", marker="o", mfc="white", ms=3.5, label="test")]
    if up:
        lab, col, mk, ls = UPRIGHT
        hs.append(Line2D([], [], color=col, ls=ls, marker=mk, ms=3.5, label=lab))
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, f"fig_grid_{RESULT_DIR[task].split('_')[0]}.pdf")


TESTS = (("cued_acc", "with the cue of the true class (as in training)", 1.0, None),
         ("cued_t_acc", "moved by a random element of $D_4$", 0.6, "...."),
         ("clean_acc", "without the cue (the true rule)", 0.3, "////"))


def cue_stats(task, L, N=None):
    """{(cue, model id): {test set: (mean, se)}, "n", "train"} over the shared seeds, at one N or over N in
    {10, 20, 40, 80} (N = None)."""
    name = RESULT_DIR[task]
    out = {}
    for sc, _ in CUES[task]:
        rs = [r for r in runs(name, sc) if r["layers"] == L and (N is None or r["N"] == N)]
        keys = set.intersection(*[{(r["N"], r["seed"]) for r in rs if r["arch"] == a} for a in MODELS])
        for arch in MODELS:
            v = [r for r in rs if r["arch"] == arch and (r["N"], r["seed"]) in keys]
            out[(sc, arch)] = {k: ms([r[k] for r in v]) for k, *_ in TESTS} | {
                "n": len(v), "train": ms([r["train_acc"] for r in v])}
    return out


def fig_cues():
    """Test accuracy of the models trained with a cue in every training image, in the main settings (five seeds)."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharey=True)
    axes = axes.ravel()
    gw = 1.3 * len(MODELS) + 0.5
    for ax, task, letter in zip(axes, CUE_TASKS, "abcd"):
        st = cue_stats(task, *MAIN[task])
        for i, (sc, _) in enumerate(CUES[task]):
            for j, arch in enumerate(MODELS):
                col = MODELS[arch][2]
                for k, (key, _, alpha, hatch) in enumerate(TESTS):
                    m, e = st[(sc, arch)][key]
                    ax.bar(i * gw + j * 1.3 + (k - 1) * 0.4, m, 0.38, color=col, alpha=alpha, hatch=hatch,
                           edgecolor=col, lw=0.5, yerr=e, capsize=1, error_kw={"lw": 0.5})
        ax.set_xticks([i * gw + 1.3 * (len(MODELS) - 1) / 2 for i in range(3)])
        ax.set_xticklabels([n.replace(" ", "\n", 1) for _, n in CUES[task]])
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_ylim(0.4, 1.02)
        ax.set_title(f"{TASK_SHORT[task]} ($L={MAIN[task][0]}$, $N={MAIN[task][1]}$)")
        panel(ax, letter, x=-0.08)
        style(ax)
    axes[0].set_ylabel("test accuracy")
    axes[2].set_ylabel("test accuracy")
    hs = [Patch(facecolor="0.35", edgecolor="0.35", alpha=a, hatch=h, label=f"test images {t}") for _, t, a, h in TESTS]
    fig.legend(handles=hs, loc="lower left", ncol=1, frameon=False, bbox_to_anchor=(0.05, -0.07))
    fig.legend(handles=[Patch(facecolor=MODELS[a][2], label=MODELS[a][1]) for a in MODELS], loc="lower right",
               ncol=2, frameon=False, bbox_to_anchor=(0.97, -0.045))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_cues.pdf")


def _complete(rows, by_scale=True):
    """The models whose runs are complete: as many runs per (task, noise scale) as Equiv."""
    def count(a):
        return collections.Counter((r["task"], r.get("noise_scale") if by_scale else 0) for r in rows
                                   if r["arch"] == a)
    return [a for a in MODELS if count(a) == count("config6")]


def fig_noise():
    """Test accuracy and gap under realistic noise: trained on the noisy device (solid) or trained without noise and
    only run on it (dashed), on the seeds shared by the two scenarios."""
    both, test_only = runs("noise_train_test"), runs("noise_test_only")
    shown = [a for a in _complete(test_only, False) if a in _complete(both)]
    fig, axes = plt.subplots(2, len(CUE_TASKS), figsize=(7.0, 4.0), sharex=True)
    x = range(len(NOISE_SCALES))
    for j, task in enumerate(CUE_TASKS):
        seeds = set(common(*[{r["seed"] for r in test_only if r["task"] == task and r["arch"] == a} for a in shown],
                           *[{r["seed"] for r in both if r["task"] == task and r["arch"] == a
                              and r["noise_scale"] == s} for a in shown for s in NOISE_SCALES]))
        for arch in shown:
            _, _, col, mk = MODELS[arch]
            t = [r for r in test_only if r["task"] == task and r["arch"] == arch and r["seed"] in seeds]
            b = [r for r in both if r["task"] == task and r["arch"] == arch and r["seed"] in seeds]
            ev = lambda r, s, k: r["noise_eval"][f"{s:g}"][k]  # noqa: E731
            series = {
                "both": ([ms([r["clean_acc"] for r in b if r["noise_scale"] == s]) for s in NOISE_SCALES],
                         [ms([r["gap"] for r in b if r["noise_scale"] == s]) for s in NOISE_SCALES]),
                "test": ([ms([ev(r, s, "clean_acc") for r in t]) for s in NOISE_SCALES],
                         [ms([ev(r, s, "train_acc") - ev(r, s, "clean_acc") for r in t]) for s in NOISE_SCALES]),
            }
            for scen, ls, mfc, lw in (("both", "-", col, 1.4), ("test", "--", "white", 0.9)):
                for i in (0, 1):
                    v = series[scen][i]
                    axes[i, j].errorbar(x, [a for a, _ in v], yerr=[e for _, e in v], color=col, ls=ls, marker=mk,
                                        mfc=mfc, ms=3.5, lw=lw, capsize=1.2)
        axes[0, j].set_title(TASK_SHORT[task])
        axes[1, j].axhline(0, color="k", lw=0.5)
        axes[1, j].set_xlabel("noise scale $s$")
        for ax in axes[:, j]:
            ax.set_xticks(list(x)), ax.set_xticklabels([f"{s:g}" for s in NOISE_SCALES])
            ax.axvline(NOISE_SCALES.index(1.0), color="0.6", lw=0.7, ls=":")
            style(ax, "both")
    nt = len(CUE_TASKS)
    for i in range(2):
        for j in range(nt):
            panel(axes[i, j], "abcdefgh"[nt * i + j], x=-0.18)
    axes[0, 0].set_ylabel("test accuracy")
    axes[1, 0].set_ylabel("generalization gap")
    hs = model_handles(shown) + [
        Line2D([], [], color="0.3", ls="-", marker="o", mfc="0.3", ms=3.5, label="trained and tested under noise"),
        Line2D([], [], color="0.3", ls="--", marker="o", mfc="white", ms=3.5, lw=0.9,
               label="trained without noise, tested under noise")]
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_noise.pdf")


# ================================================================ figures of the appendices
GALAXY_PILOT_CLASSES = ((0, "face-on disk,\nno spiral arms"), (4, "edge-on disk,\nrounded bulge"),
                        (1, "smooth,\ncompletely round"), (2, "smooth,\nin-between round"))


def fig_galaxy_classes(per_class=5):
    """The four classes of Galaxy10 that enter the pilot runs (class 0 against 4 is the task of the paper): the first
    images of each class in the data set, as they are (69 x 69, color) and as the circuits see them (luminance,
    16 x 16 with antialiasing, the preprocessing of eqnn.data)."""
    import h5py
    from eqnn.data import DATA_ROOT, _to_tensor_16
    with h5py.File(os.path.join(DATA_ROOT, "galaxy10", "Galaxy10.h5"), "r") as f:
        labels = f["ans"][:]
        images = {c: f["images"][np.flatnonzero(labels == c)[:per_class]] for c, _ in GALAXY_PILOT_CLASSES}
    nr, nc = len(GALAXY_PILOT_CLASSES), 2 * per_class
    fig = plt.figure(figsize=(7.0, 0.3 + 0.66 * nr))
    gs = fig.add_gridspec(nr, nc + 1, width_ratios=[1] * per_class + [0.25] + [1] * per_class, wspace=0.06,
                          hspace=0.12, left=0.15, right=1.0, top=1 - 0.3 / (0.3 + 0.66 * nr), bottom=0.0)
    axes = np.empty((nr, nc), dtype=object)
    for row, (c, name) in enumerate(GALAXY_PILOT_CLASSES):
        small = _to_tensor_16(images[c]).reshape(-1, 16, 16).numpy()
        for k in range(per_class):
            for half, (img, kw) in enumerate(((images[c][k], {}), (small[k], {"cmap": "gray_r", "vmin": 0}))):
                ax = fig.add_subplot(gs[row, k if half == 0 else per_class + 1 + k])
                ax.imshow(img, interpolation="nearest", **kw)
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4)
                axes[row, k + half * per_class] = ax
        pos = axes[row, 0].get_position()
        fig.text(pos.x0 - 0.01, (pos.y0 + pos.y1) / 2, f"class {c}\n{name}", ha="right", va="center", fontsize=8.2)
    for text, a, b in (("original images ($69\\times69$, color)", 0, per_class - 1),
                       ("as seen by the circuits ($16\\times16$, luminance)", per_class, nc - 1)):
        x0, x1 = axes[0, a].get_position().x0, axes[0, b].get_position().x1
        y = axes[0, a].get_position().y1 + 0.1 / (0.3 + 0.66 * nr)
        fig.text((x0 + x1) / 2, y + 0.015, text, ha="center", va="bottom", fontsize=8.7)
        fig.add_artist(Line2D([x0 + 0.005, x1 - 0.005], [y, y], color="0.3", lw=0.6, transform=fig.transFigure))
    save(fig, "fig_galaxy_classes.pdf")


def fig_curves():
    """Learning curves of the four models in the main settings."""
    fig, axes = plt.subplots(len(TASKS), 4, figsize=(7.0, 1.3 * len(TASKS)), sharex=True, sharey=True)
    for i, task in enumerate(TASKS):
        c = task_cells(task)
        L, N = MAIN[task]
        seeds = common(*[c[(L, N, a)] for a in MODELS])
        for j, arch in enumerate(MODELS):
            lab, _, col, mk = MODELS[arch]
            rs = [c[(L, N, arch)][s] for s in seeds]
            ax = axes[i, j]
            ep = np.array(rs[0]["curve"]["epoch"])
            for key, ls, mfc, name in (("train_acc", "-", col, "training"), ("clean_acc", "--", "white", "test")):
                Y = np.array([r["curve"][key] for r in rs])
                m, e = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y))
                ax.fill_between(ep, m - e, m + e, color=col, alpha=0.15, lw=0)
                ax.plot(ep, m, ls=ls, color=col, marker=mk, mfc=mfc, ms=3, lw=1.2, label=name)
            if i == 0:
                ax.set_title(lab)
            ax.set_xlim(-2, 75)
            style(ax, "both")
            if i == len(TASKS) - 1:
                ax.set_xlabel("epoch")
            if j == 0:
                ax.set_ylabel("accuracy")
    axes[0, 0].set_ylim(0.45, 1.02)
    fig.legend(handles=[Line2D([], [], color="0.3", ls="-", marker="o", mfc="0.3", ms=3, label="training"),
                        Line2D([], [], color="0.3", ls="--", marker="o", mfc="white", ms=3, label="test")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.05, 0.035, 1, 1))
    row_labels(fig, axes[:, 0], [TASKS[t].replace(", ", ",\n") for t in TASKS])
    save(fig, "fig_curves.pdf")


def fig_noise_training():
    """Training and test accuracy and gap of the models trained and tested under noise (five seeds)."""
    both = runs("noise_train_test")
    shown = _complete(both)
    fig, axes = plt.subplots(2, len(CUE_TASKS), figsize=(7.0, 3.8), sharex=True)
    x = range(len(NOISE_SCALES))
    for j, task in enumerate(CUE_TASKS):
        seeds = {s: common(*[{r["seed"] for r in both if r["task"] == task and r["arch"] == a
                              and r["noise_scale"] == s} for a in shown]) for s in NOISE_SCALES}
        for arch in shown:
            _, _, col, mk = MODELS[arch]
            d = {s: [r for r in both if r["task"] == task and r["arch"] == arch and r["noise_scale"] == s
                     and r["seed"] in seeds[s]] for s in NOISE_SCALES}
            for key, ls, mfc in (("train_acc", ":", col), ("clean_acc", "-", "white")):
                v = [ms([r[key] for r in d[s]]) for s in NOISE_SCALES]
                axes[0, j].errorbar(x, [a for a, _ in v], yerr=[b for _, b in v], color=col, ls=ls, marker=mk,
                                    mfc=mfc, ms=3.5, lw=1.1, capsize=1.2)
            v = [ms([r["gap"] for r in d[s]]) for s in NOISE_SCALES]
            axes[1, j].errorbar(x, [a for a, _ in v], yerr=[b for _, b in v], color=col, marker=mk, ms=3.5, lw=1.2,
                                capsize=1.2)
        axes[0, j].set_title(TASK_SHORT[task])
        axes[1, j].axhline(0, color="k", lw=0.5)
        axes[1, j].set_xlabel("noise scale $s$")
        for ax in axes[:, j]:
            ax.set_xticks(list(x)), ax.set_xticklabels([f"{s:g}" for s in NOISE_SCALES])
            ax.axvline(NOISE_SCALES.index(1.0), color="0.6", lw=0.7, ls=":")
            style(ax, "both")
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("generalization gap")
    fig.legend(handles=model_handles(shown) + [
        Line2D([], [], color="0.3", ls=":", marker="o", mfc="0.3", ms=3.5, label="training"),
        Line2D([], [], color="0.3", ls="-", marker="o", mfc="white", ms=3.5, label="test")],
        loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_noise_training.pdf")


def fig_gradients():
    """Gradients of the training loss at the initialization used in training (angles uniform in [-0.1, 0.1]): variance
    over 100 draws as a function of the depth (results/gradient_variance.json), and histograms of the moduli of the
    partial derivatives over 200 draws in the main settings (results/gradient_histograms.json)."""
    rows = json.load(open(os.path.join(RESULTS, "gradient_variance.json")))
    hist = json.load(open(os.path.join(RESULTS, "gradient_histograms.json")))
    names = {"Equiv": "config6", "NonEquiv-Twirled": "config10", "NonEquiv": "config7", "Equiv-NoSym": "config6n"}
    fig, axes = plt.subplots(2, len(TASKS), figsize=(7.0, 4.2))
    bins = np.linspace(-5, 1, 49)
    for j, task in enumerate(TASKS):
        ax = axes[0, j]
        for name, arch in names.items():
            r = sorted((x for x in rows if x["task"] == task and x["arch"] == name), key=lambda x: x["layers"])
            _, _, col, mk = MODELS[arch]
            ax.plot([x["layers"] for x in r], [x["var"] for x in r], color=col, marker=mk, ms=3.5, lw=1.2)
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16]), ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.minorticks_off()
        style(ax, "both")
        ax.set_title(TASK_SHORT[task])
        ax.set_xlabel("layers $L$")
        ax = axes[1, j]
        for name, arch in names.items():
            (r,) = [x for x in hist if x["task"] == task and x["arch"] == name]
            g = np.log10(np.maximum(np.asarray(r["abs_grad"]), 1e-12))
            ax.hist(g, bins=bins, histtype="step", color=MODELS[arch][2], lw=1.1, density=True)
            ax.axvline(np.median(g), color=MODELS[arch][2], lw=0.7, ls=":")
        L, N = MAIN[task]
        ax.set_title(f"$L={L}$, $N={N}$")
        ax.set_xlabel(r"$\log_{10}|\partial\mathcal{L}/\partial\theta_k|$")
        ax.set_xlim(bins[0], bins[-1])
        style(ax, "both")
        for i in range(2):
            panel(axes[i, j], "abcdefgh"[4 * i + j], x=-0.3 if j else -0.4, y=1.0)
    axes[0, 0].set_ylabel("gradient variance")
    axes[1, 0].set_ylabel("density")
    fig.legend(handles=model_handles(list(MODELS)), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.0)
    save(fig, "fig_gradients.pdf")


# ================================================================ tables
def tab_paired():
    """Paired comparisons in the main settings (15 seeds): difference in test accuracy and reduction of the gap."""
    lines = []
    for task, (_, N) in MAIN.items():
        c = task_cells(task)
        for L in MAIN_LAYERS:
            seeds = common(*[c[(L, N, a)] for a in MODELS])
            for k, (a, b) in enumerate(PAIRS):
                te = paired(c[(L, N, a)], c[(L, N, b)], "clean_acc", 1, seeds)
                gp = paired(c[(L, N, a)], c[(L, N, b)], "gap", -1, seeds)
                lines.append(" & ".join([
                    TASK_SHORT[task] if k == 0 and L == MAIN_LAYERS[0] else ""] + (
                    [f"{L}" if k == 0 else ""] if len(MAIN_LAYERS) > 1 else []) + [
                    f"{MODELS[a][1]} vs {MODELS[b][1]}",
                    fmt(te[0], te[1], True), f"{te[2]}", fmt_p(te[4]),
                    fmt(gp[0], gp[1], True), f"{gp[2]}", fmt_p(gp[4])]) + r" \\")
            lines.append(r"\midrule")
    write_table("tab_paired.tex", lines[:-1])


def tab_factorial():
    """Effect of making the model invariant in the main settings (15 seeds): per seed, the mean of Equiv and Twirled
    minus the mean of NoSym and NonEquiv, on the test accuracy and on the gap (sign: reduction)."""
    from scipy.stats import wilcoxon
    lines = []
    for task, (_, N) in MAIN.items():
        c = task_cells(task)
        for L in MAIN_LAYERS:
            seeds = common(*[c[(L, N, a)] for a in MODELS])
            row = [TASK_SHORT[task] if L == MAIN_LAYERS[0] else ""] + ([f"{L}"] if len(MAIN_LAYERS) > 1 else [])
            for which in ("inv",):
                for key, sign in (("clean_acc", 1), ("gap", -1)):
                    v = {a: np.array([c[(L, N, a)][s][key] for s in seeds]) for a in MODELS}
                    if which == "inv":
                        x = (v["config6"] + v["config10"] - v["config6n"] - v["config7"]) / 2
                    else:
                        x = (v["config6"] + v["config6n"] - v["config10"] - v["config7"]) / 2
                    x = sign * x
                    row += [fmt(*ms(x), True), fmt_p(wilcoxon(x).pvalue)]
            lines.append(" & ".join(row) + r" \\")
    write_table("tab_factorial.tex", lines)


def tab_main():
    """The main settings (15 seeds): training accuracy, test accuracy and gap of the four models on every task."""
    lines = []
    for task, (_, N) in MAIN.items():
        c = task_cells(task)
        for L in MAIN_LAYERS:
            seeds = common(*[c[(L, N, a)] for a in MODELS])
            row = [TASK_SHORT[task] if L == MAIN_LAYERS[0] else "", str(N) if L == MAIN_LAYERS[0] else ""] + (
                [str(L)] if len(MAIN_LAYERS) > 1 else [])
            for a in MODELS:
                v = [c[(L, N, a)][s] for s in seeds]
                row += [fmt(*ms([r["train_acc"] for r in v])), fmt(*ms([r["clean_acc"] for r in v])),
                        fmt(*ms([r["gap"] for r in v]))]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    write_table("tab_main.tex", lines[:-1])


def tab_full(task):
    """Every (L, N) of a task: training accuracy, test accuracy and gap of the four models."""
    c = task_cells(task)
    lines = []
    for L in sorted({lay for (lay, n, a) in c}):
        for N in sorted({n for (lay, n, a) in c if lay == L}):
            series = [c.get((L, N, a)) for a in MODELS]
            seeds = common(*series)
            row = [str(L), str(N), str(len(seeds))]
            for s in series:
                v = [s[sd] for sd in seeds]
                row += [fmt(*ms([r["train_acc"] for r in v])), fmt(*ms([r["clean_acc"] for r in v])),
                        fmt(*ms([r["gap"] for r in v]))]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    write_table(f"tab_full_{RESULT_DIR[task].split('_')[0]}.tex", lines[:-1])


def tab_cues():
    """Shortcuts at L = 2 and 4, averaged over N in {10, 20, 40, 80} (the supplementary tables of the repository)."""
    lines = []
    for task in CUE_TASKS:
        for L in (2, 4):
            st = cue_stats(task, L)
            for i, (sc, cname) in enumerate(CUES[task]):
                for j, arch in enumerate(MODELS):
                    s = st[(sc, arch)]
                    lines.append(" & ".join([
                        TASK_SHORT[task] if i == j == 0 and L == 2 else "", f"{L}" if i == j == 0 else "",
                        cname if j == 0 else "", MODELS[arch][1], fmt(*s["train"])]
                        + [fmt(*s[k]) for k, *_ in TESTS]) + r" \\")
            lines.append(r"\midrule")
    write_table("tab_cues.tex", lines[:-1])


def tab_noise():
    """Noise in training and test: test accuracy and gap at every noise scale (five seeds)."""
    both = runs("noise_train_test")
    present = _complete(both)
    lines = []
    for task in CUE_TASKS:
        for s in NOISE_SCALES:
            by = {a: [r for r in both if r["task"] == task and r["arch"] == a and r["noise_scale"] == s]
                  for a in present}
            seeds = set(common(*[{r["seed"] for r in v} for v in by.values()]))
            by = {a: {r["seed"]: r for r in v if r["seed"] in seeds} for a, v in by.items()}
            row = [TASK_SHORT[task] if s == 0 else "", f"{s:g}"]
            for a in MODELS:
                row += ([fmt(*ms([r["clean_acc"] for r in by[a].values()])),
                         fmt(*ms([r["gap"] for r in by[a].values()]))] if a in by else ["--", "--"])
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    write_table("tab_noise.tex", lines[:-1])


def tab_noise_test():
    """Noise in the test only: test accuracy and gap of the models trained without noise (15 seeds)."""
    rows = runs("noise_test_only")
    present = _complete(rows, False)
    lines = []
    for task in CUE_TASKS:
        for s in NOISE_SCALES:
            key = f"{s:g}"
            by = {a: {r["seed"]: r["noise_eval"][key] for r in rows if r["task"] == task and r["arch"] == a}
                  for a in present}
            seeds = common(*by.values())
            row = [TASK_SHORT[task] if s == 0 else "", key]
            for a in MODELS:
                row += ([fmt(*ms([by[a][k]["clean_acc"] for k in seeds])),
                         fmt(*ms([by[a][k]["train_acc"] - by[a][k]["clean_acc"] for k in seeds]))]
                        if a in by else ["--", "--"])
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    write_table("tab_noise_test.tex", lines[:-1])


def tab_symbreak():
    """Breaking of the symmetry of the equivariant circuit by the noise: mean |m(x) - m(gx)| / mean |m(x)| over 200
    MNIST test images in random pose, ten draws of the angles uniform in [-0.5, 0.5], L = 4, chip 0."""
    import torch
    from eqnn.data import apply_d4, build_data
    from eqnn.models import build_model
    x = build_data("mnist_bin", 40, 1, "pose_rand")[2]["clean"][:200]
    elems = (7, 1, 4, 2)  # transposition, rotation by 90 degrees, reflection, rotation by 180 degrees
    lines = []
    for spread, lab in ((False, "identical"), (True, "dispersion")):
        for sc in (0.25, 1.0, 4.0):
            model, n = build_model("config6", 4, noise_scale=sc, chip_seed=0, chip_spread=spread)
            vals = {g: [] for g in elems}
            for k in range(10):
                th = torch.empty(n, dtype=torch.float64).uniform_(-0.5, 0.5, generator=torch.Generator().manual_seed(k))
                with torch.no_grad():
                    m = model(x, th)
                    for g in elems:
                        mg = model(apply_d4(x.reshape(-1, 16, 16), g).reshape(x.shape[0], -1), th)
                        vals[g].append(float((m - mg).abs().mean() / m.abs().mean()))
            lines.append(" & ".join([lab if sc == 0.25 else "", f"{sc:g}"]
                                    + [f"{100 * np.mean(vals[g]):.1f}\\%" for g in elems]) + r" \\")
    write_table("tab_symbreak.tex", lines)


def tab_dla():
    d = json.load(open(os.path.join(RESULTS, "dla.json")))
    lines = []
    for k in ("Equiv", "Twirled", "NonEquiv", "NoSym"):
        dla = d["dla"] if k in ("Equiv", "NoSym") else d["dla_nonequiv"]
        shared = d["dim"]["Equiv"] if k == "Equiv" else d["pair"][f"Equiv&{k}"]
        lines.append(" & ".join(str(v) for v in (f"\\{k}", dla, d["observables"][k], d["dim"][k], d["inv"][k],
                                                  shared)) + r" \\")
    write_table("tab_dla.tex", lines)


VARIANTS = {"A_big_filled": "A", "A_base": "A", "B_big_hollow": "B", "C_medium": "C", "D_big_filled_nonoise": "D",
            "E_wide_sizes": "E", "F_noise10": "F", "G_thin_cross": "G", "H_small_big": "H", "I_noise10_wide": "I",
            "J_small_thin": "J", "K_thick1": "K", "M_mid_thin": "M"}


PILOT_PAIRS = {
    "galaxy": {"galaxy_0_4": "face-on disk vs edge-on disk", "galaxy_1_0": "round vs face-on disk",
               "galaxy_1_4": "round vs edge-on disk", "galaxy_2_4": "in-between round vs edge-on disk",
               "galaxy_1_2": "round vs in-between round"},
    "erythrocytes": {"ery_1_2": "elongated vs other", "ery_0_1": "circular vs elongated",
                     "ery_0_2": "circular vs other"}}


def tab_pilots(name):
    """Pilot runs used to choose the task of a real data set: test accuracy (gap) of Equiv and NonEquiv, N = 40,
    three seeds."""
    d = collections.defaultdict(dict)
    for line in open(os.path.join(RESULTS, f"{name}_pilots", "runs.jsonl")):
        r = json.loads(line)
        d[(r["task"], r["layers"], r["arch"])][r["seed"]] = r
    lines = []
    for t, label in PILOT_PAIRS[name].items():
        row = [label]
        for L in (2, 4):
            for arch in ("config6", "config7"):
                rs = list(d[(t, L, arch)].values())
                row.append(f"${np.mean([r['clean_acc'] for r in rs]):.2f}$ "
                           f"(${np.mean([r['train_acc'] - r['clean_acc'] for r in rs]):+.2f}$)")
        lines.append(" & ".join(row) + r" \\")
    write_table(f"tab_{name}_pilots.tex", lines)


def tab_variants():
    """Pilot variants of the synthetic task: test accuracy (gap) of Equiv and NonEquiv (the first run of every
    seed, in the order of the pilot rounds)."""
    d = collections.defaultdict(dict)
    for line in open(os.path.join(RESULTS, "synthetic_pilots", "variants.jsonl")):
        r = json.loads(line)
        d[(VARIANTS[r["variant"]], r["layers"], r["N"], r["arch"])].setdefault(r["seed"], r)
    lines = []
    for v in sorted({k[0] for k in d}):
        row = [v]
        for L, N in ((2, 40), (4, 20)):
            for arch in ("config6", "config7"):
                rs = list(d.get((v, L, N, arch), {}).values())
                row.append(f"${np.mean([r['clean_acc'] for r in rs]):.2f}$ "
                           f"(${np.mean([r['train_acc'] - r['clean_acc'] for r in rs]):+.2f}$)" if rs else "--")
        row.append(str(len(d.get((v, 2, 40, "config6"), {}))))
        lines.append(" & ".join(row) + r" \\")
    write_table("tab_variants.tex", lines)


# the figures and tables of the paper; the full grid, the shortcuts averaged over N, the noise tables and the pilot runs
# are in supplementary/ (analysis/supplementary.py)
ALL = ["fig_tasks", "fig_main", "fig_scaling", "fig_cues", "fig_noise", "fig_curves", "fig_gradients", "fig_galaxy_classes",
       "tab_paired", "tab_factorial", "tab_dla", "tab_symbreak"]

if __name__ == "__main__":
    for call in sys.argv[1:] or ALL:
        name, _, arg = call.partition("(")
        args = [a.strip("'\" )") for a in arg.split(",")] if arg else []
        globals()[name](*args)
        print(call, flush=True)
