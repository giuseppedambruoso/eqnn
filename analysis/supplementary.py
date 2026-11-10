"""Supplementary tables and figures: the results that the paper summarizes but does not report in full, written to
supplementary/ (README.md with every table in Markdown, figures/ with the figures of the whole grid).

    python analysis/supplementary.py

Contents: every (L, N) of the four tasks, the main settings with two and four layers, the experiments on shortcuts at
every N, the two noise scenarios at every noise scale, the pilot runs used to choose the tasks, and the failures to
train.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "supplementary")
os.environ["EQNN_PAPER_DIR"] = OUT  # figures/ and tables/ of the functions of paper_figures.py go to supplementary/
os.environ["EQNN_MAIN_LAYERS"] = "2,4"
sys.path.insert(0, ROOT)

from analysis import paper_figures as P  # noqa: E402
from analysis.common import CUE_TASKS, CUES, MAIN, MODELS, RESULT_DIR, TASK_SHORT, TASKS, fmt, ms, runs, \
    task_cells  # noqa: E402

M = [MODELS[a][1] for a in MODELS]


def md_cell(c: str) -> str:
    """A LaTeX table cell as Markdown text."""
    c = c.strip()
    c = re.sub(r"\\times10\^\{(-?\d+)\}", r"e\1", c)
    c = c.replace("$", "").replace("\\%", "%").replace("\\", "")
    return c


def latex_to_md(path: str, header: list[str]) -> str:
    rows = []
    for line in open(path):
        line = line.split("%")[0] if "\\%" not in line else line
        line = line.strip()
        if not line or line.startswith("\\midrule"):
            continue
        rows.append([md_cell(c) for c in line.rstrip("\\").rstrip().rstrip("\\").split("&")])
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def failures() -> str:
    """Runs whose training accuracy stays at or below 0.6, per task, model and depth (all N, all seeds)."""
    lines = ["| Task | L | " + " | ".join(M) + " |", "|" + "---|" * (2 + len(M))]
    for task in TASKS:
        c = task_cells(task)
        for L in sorted({k[0] for k in c}):
            row = [TASK_SHORT[task], str(L)]
            for a in MODELS:
                rs = [r for (lay, n, arch), d in c.items() if lay == L and arch == a for r in d.values()]
                row.append(f"{sum(r['train_acc'] <= 0.6 for r in rs)}/{len(rs)}")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def upright() -> str:
    """NonEquiv and Equiv trained on upright MNIST digits, tested on upright digits and on digits in random pose."""
    from analysis.common import cells, common
    pose, up = task_cells("mnist_bin"), cells(runs("mnist_upright", "none"))
    lines = ["| L | N | seeds | NonEquiv train | NonEquiv test (upright) | NonEquiv gap | NonEquiv test (random pose) "
             "| Equiv test (upright) | NonEquiv gap, trained in random pose |", "|" + "---|" * 9]
    for L in (2, 4):
        for N in (10, 20, 40, 80):
            seeds = common(up[(L, N, "config7")], up[(L, N, "config6")], pose[(L, N, "config7")])
            u = [up[(L, N, "config7")][s] for s in seeds]
            e = [up[(L, N, "config6")][s] for s in seeds]
            p = [pose[(L, N, "config7")][s] for s in seeds]
            lines.append("| " + " | ".join([str(L), str(N), str(len(seeds))] + [md_cell(fmt(*ms(v))) for v in (
                [r["train_acc"] for r in u], [r["clean_acc"] for r in u], [r["gap"] for r in u],
                [r["clean_t_acc"] for r in u], [r["clean_acc"] for r in e], [r["gap"] for r in p])]) + " |")
    return "\n".join(lines)


def cues_main() -> str:
    """Shortcuts in the main settings (L = 4, N of the main setting, five seeds), as in the figure of the paper."""
    lines = ["| Task | Cue | Model | train | with cue | cue moved | no cue |", "|---|---|---|---|---|---|---|"]
    for task in CUE_TASKS:
        st = P.cue_stats(task, *MAIN[task])
        for sc, name in CUES[task]:
            for a in MODELS:
                s = st[(sc, a)]
                lines.append("| " + " | ".join([TASK_SHORT[task], name, MODELS[a][1], md_cell(fmt(*s["train"]))]
                                               + [md_cell(fmt(*s[k])) for k, *_ in P.TESTS]) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    tdir = os.path.join(OUT, "tables")
    for call in ["tab_main", "tab_paired", "tab_factorial", "tab_cues", "tab_noise", "tab_noise_test",
                 "tab_variants", "tab_pilots('galaxy')", "tab_pilots('erythrocytes')"] + \
            [f"tab_full('{t}')" for t in TASKS] + [f"fig_grid('{t}')" for t in TASKS] + ["fig_noise_training"]:
        name, _, arg = call.partition("(")
        getattr(P, name)(*[a.strip("'\" )") for a in arg.split(",")] if arg else [])
        print(call, flush=True)

    tam = ["train", "test", "gap"]
    full_header = ["L", "N", "seeds"] + [f"{m} {k}" for m in M for k in tam]
    sec = []
    sec.append("## 1. The main settings with two and four layers\n\nTraining accuracy, test accuracy and "
               "generalization gap (mean over 15 seeds, standard error on the last digit in brackets). The paper "
               "reports L = 4.\n\n" + latex_to_md(os.path.join(tdir, "tab_main.tex"),
                                                   ["Task", "N", "L"] + [f"{m} {k}" for m in M for k in tam]))
    sec.append("### Paired comparisons\n\nFor the pair A vs B: difference in test accuracy (A minus B) and "
               "reduction of the gap (gap of B minus gap of A); k: seeds with a positive difference (of 15); p: "
               "two-sided Wilcoxon signed-rank test.\n\n" + latex_to_md(
                   os.path.join(tdir, "tab_paired.tex"),
                   ["Task", "L", "Pair", "Δ test", "k", "p", "Δ gap", "k", "p"]))
    sec.append("### Effect of the invariance\n\nPer seed, the mean of the invariant models minus the mean of the "
               "non-invariant ones (positive: higher test accuracy, smaller gap).\n\n" + latex_to_md(
                   os.path.join(tdir, "tab_factorial.tex"), ["Task", "L", "test", "p", "gap", "p"]))
    files = {"mnist_bin": "mnist", "shapes_cs": "synthetic", "galaxy_0_4": "galaxy", "ery_1_2": "erythrocytes"}
    grid = ["## 2. Every number of layers and of training images\n\nMean over the seeds shared by the four models "
            "(five, or 15 in the main settings). Figures: `figures/fig_grid_<task>.pdf`."]
    for t, f in files.items():
        grid.append(f"### {TASKS[t]}\n\n" + latex_to_md(os.path.join(tdir, f"tab_full_{f}.tex"), full_header))
    sec.append("\n\n".join(grid))
    sec.append("### Failures to train\n\nRuns whose training accuracy stays at or below 0.6 (over all N and "
               "seeds).\n\n" + failures())
    sec.append("### Upright control (MNIST)\n\nModels trained and tested on upright digits; the column \"random "
               "pose\" tests the same trained NonEquiv on the test digits moved by a random element of D4.\n\n"
               + upright())
    sec.append("## 3. Shortcuts\n\n### In the main settings (as in the paper; five seeds)\n\n" + cues_main()
               + "\n\n### Averaged over N ∈ {10, 20, 40, 80} (five seeds each)\n\n" + latex_to_md(
                   os.path.join(tdir, "tab_cues.tex"),
                   ["Task", "L", "Cue", "Model", "train", "with cue", "cue moved", "no cue"]))
    nh = ["Task", "s"] + [f"{m} {k}" for m in M for k in ("test", "gap")]
    sec.append("## 4. Hardware noise\n\n### Trained and tested under noise (L = 4, five seeds)\n\n"
               + latex_to_md(os.path.join(tdir, "tab_noise.tex"), nh)
               + "\n\nTraining accuracies: `figures/fig_noise_training.pdf`.\n\n"
               "### Trained without noise, tested under noise (L = 4, 15 seeds)\n\n"
               + latex_to_md(os.path.join(tdir, "tab_noise_test.tex"), nh))
    ph = ["Classes", "Equiv L=2", "NonEquiv L=2", "Equiv L=4", "NonEquiv L=4"]
    sec.append("## 5. Pilot runs used to choose the tasks\n\nTest accuracy and, in brackets, generalization gap of "
               "Equiv and NonEquiv.\n\n### Variants of the synthetic task (L = 2, N = 40 and L = 4, N = 20)\n\n"
               "A: squares 8–14, crosses 9–15, thickness 2–4, noise 0.05; B: A with hollow squares; C: squares 5–10, "
               "crosses 7–11, thickness 1–2; D: A without noise; E: squares 6–14, crosses 7–15, thickness 1–4; F: A "
               "with noise 0.10; G: A with thickness 1–2; H: squares 6–11, crosses 7–13, thickness 2–3; I: E with "
               "noise 0.10; J: H with thickness 1–2; K: A with thickness 1 (the task of the paper); M: squares 7–12, "
               "crosses 8–14, thickness 1–2. A first version with squares of side 2–5 and crosses with arms of 1–3 "
               "pixels on a background noise of 0.15 was learned by no circuit (test accuracy 0.46–0.57).\n\n"
               + latex_to_md(os.path.join(tdir, "tab_variants.tex"),
                             ["Variant", "Equiv L=2,N=40", "NonEquiv L=2,N=40", "Equiv L=4,N=20", "NonEquiv L=4,N=20",
                              "seeds"])
               + "\n\n### Pairs of classes of Galaxy10 (N = 40, three seeds)\n\n"
               + latex_to_md(os.path.join(tdir, "tab_galaxy_pilots.tex"), ph)
               + "\n\n### Pairs of classes of erythrocytesIDB (N = 40, three seeds)\n\n"
               + latex_to_md(os.path.join(tdir, "tab_erythrocytes_pilots.tex"), ph))
    head = ("# Supplementary results\n\nThe results that the paper summarizes but does not report in full. Every "
            "number is computed from the runs in `results/` by `analysis/supplementary.py`; the LaTeX version of "
            "every table is in `tables/`, the figures of the whole grid in `figures/`. Models: Equiv (equivariant "
            "circuit, invariant observable), Twirled (NonEquiv-Twirled), NonEquiv, NoSym (Equiv-NoSym). Accuracies "
            "are fractions; the generalization gap is the training minus the test accuracy.\n")
    open(os.path.join(OUT, "README.md"), "w").write(head + "\n" + "\n\n".join(sec) + "\n")
    print(os.path.join(OUT, "README.md"))
