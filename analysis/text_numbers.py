"""Every number quoted in the text of the paper that is not read directly from a table, computed from results/,
in the order in which it appears (main text, then appendices).

    python analysis/text_numbers.py > paper/numbers.txt
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import (CUE_TASKS, CUES, LAYERS, MAIN, MODELS, NOISE_SCALES, RESULTS, TASK_SHORT, TASKS,  # noqa: E402
                             cells, common, ms, paired, runs, task_cells)

name = {a: MODELS[a][1] for a in MODELS}
EQ, TW, NE, NS = "config6", "config10", "config7", "config6n"
INV, NONINV = (EQ, TW), (NE, NS)


def main_cells(task):
    c = task_cells(task)
    L, N = MAIN[task]
    seeds = common(*[c[(L, N, a)] for a in MODELS])
    return {a: [c[(L, N, a)][s] for s in seeds] for a in MODELS}, c, L, N


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main_settings():
    print("== Main settings (L = 4; N = 40, 20 on the synthetic task; 15 seeds): train / test / gap")
    gi, gn, ratios = [], [], []
    for task in TASKS:
        v, *_ = main_cells(task)
        print(f"  {TASK_SHORT[task]}: " + "  ".join(
            f"{name[a]} {mean(v[a], 'train_acc'):.2f}/{mean(v[a], 'clean_acc'):.2f}/{mean(v[a], 'gap'):+.2f}"
            for a in MODELS))
        gi += [mean(v[a], "gap") for a in INV]
        gn += [mean(v[a], "gap") for a in NONINV]
        ratios += [mean(v[NE], "gap") / mean(v[a], "gap") for a in INV]
    print(f"  gaps of the invariant models {min(gi):.2f}-{max(gi):.2f}, of the non-invariant ones "
          f"{min(gn):.2f}-{max(gn):.2f}; gap of NonEquiv / gap of an invariant model {min(ratios):.1f}-{max(ratios):.1f}")
    for a, b in ((EQ, NE), (EQ, NS), (NS, NE), (EQ, TW)):
        out = []
        for task in TASKS:
            v, *_ = main_cells(task)
            d = [x["clean_acc"] - y["clean_acc"] for x, y in zip(v[a], v[b])]
            g = [x["gap"] - y["gap"] for x, y in zip(v[a], v[b])]
            out.append(f"{TASK_SHORT[task]} {np.mean(d):+.2f} ({sum(x > 0 for x in d)}/15), gap {np.mean(g):+.2f}")
        print(f"  test accuracy {name[a]} - {name[b]}: " + "; ".join(out))


def upright():
    print("\n== Upright MNIST (main setting): NonEquiv on upright digits")
    pose, up = task_cells("mnist_bin"), cells(runs("mnist_upright", "none"))
    L, N = MAIN["mnist_bin"]
    seeds = common(up[(L, N, NE)], up[(L, N, EQ)], pose[(L, N, NE)], pose[(L, N, EQ)])
    u = [up[(L, N, NE)][s] for s in seeds]
    p = [pose[(L, N, NE)][s] for s in seeds]
    e = [pose[(L, N, EQ)][s] for s in seeds]
    print(f"  {len(seeds)} seeds: gap upright {mean(u, 'gap'):.2f} vs random pose {mean(p, 'gap'):.2f}; "
          f"test upright {mean(u, 'clean_acc'):.2f} vs Equiv {mean(e, 'clean_acc'):.2f}; "
          f"trained upright, tested on rotated digits {mean(u, 'clean_t_acc'):.2f}")


def cues():
    print("\n== Shortcuts in the main settings (five seeds): train / with cue / cue moved / no cue")
    for task in CUE_TASKS:
        L, N = MAIN[task]
        for sc, label in CUES[task]:
            rs = [r for r in runs({"mnist_bin": "mnist_pose", "shapes_cs": "synthetic", "galaxy_0_4": "galaxy",
                                   "ery_1_2": "erythrocytes"}[task], sc) if r["layers"] == L and r["N"] == N]
            print(f"  {TASK_SHORT[task]}, {label}: " + "  ".join(
                f"{name[a]} " + "/".join(f"{mean([r for r in rs if r['arch'] == a], k):.2f}"
                                         for k in ("train_acc", "cued_acc", "cued_t_acc", "clean_acc"))
                for a in MODELS))


def noise():
    both, test = runs("noise_train_test"), runs("noise_test_only")
    print("\n== Noise in training and test (five seeds): test accuracy / gap at every scale")
    for task in TASKS:
        for a in MODELS:
            print(f"  {TASK_SHORT[task]} {name[a]}: " + "  ".join(
                f"s={s:g} {mean([r for r in both if r['task'] == task and r['arch'] == a and r['noise_scale'] == s], 'clean_acc'):.2f}"
                f"/{mean([r for r in both if r['task'] == task and r['arch'] == a and r['noise_scale'] == s], 'gap'):+.2f}"
                for s in NOISE_SCALES))
    print("  Equiv - NonEquiv in test accuracy: " + "; ".join(
        f"{TASK_SHORT[t]} " + " ".join(
            f"{mean([r for r in both if r['task'] == t and r['arch'] == EQ and r['noise_scale'] == s], 'clean_acc') - mean([r for r in both if r['task'] == t and r['arch'] == NE and r['noise_scale'] == s], 'clean_acc'):+.2f}"
            for s in NOISE_SCALES) for t in TASKS))
    print("\n== Noise in the test only (15 seeds): test accuracy at s = 0, 1, 2, 4")
    for task in TASKS:
        print(f"  {TASK_SHORT[task]}: " + "  ".join(
            f"{name[a]} " + "/".join(f"{np.mean([r['noise_eval'][s]['clean_acc'] for r in test if r['task'] == task and r['arch'] == a]):.2f}"
                                     for s in ("0", "1", "2", "4")) for a in MODELS))


def failures():
    print("\n== Failures to train (training accuracy <= 0.6)")
    for task in TASKS:
        c = task_cells(task)
        L, N = MAIN[task]
        main = {name[a]: sum(r["train_acc"] <= 0.6 for r in c[(L, N, a)].values()) for a in MODELS}
        deep = {name[a]: f"{sum(r['train_acc'] <= 0.6 for (l, n, b), d in c.items() if l == max(LAYERS[task]) and b == a for r in d.values())}"
                         f"/{sum(len(d) for (l, n, b), d in c.items() if l == max(LAYERS[task]) and b == a)}" for a in MODELS}
        print(f"  {TASK_SHORT[task]}: main setting {main}; L={max(LAYERS[task])} {deep}")


def dla():
    d = json.load(open(os.path.join(RESULTS, "dla.json")))
    print(f"\n== DLA and function classes (results/dla.json): DLA {d['dla']} / {d['dla_nonequiv']}; features {d['dim']}; "
          f"invariant {d['inv']}; shared {d['pair']}; all quadratic features {d['total']}, invariant {d['total_inv']}")


def gradients():
    rows = json.load(open(os.path.join(RESULTS, "gradient_histograms.json")))
    print("\n== Gradients at initialization, main settings (200 draws): median |dL/dtheta_k|, and median of Equiv / NoSym")
    arch = {"Equiv": EQ, "NonEquiv-Twirled": TW, "NonEquiv": NE, "Equiv-NoSym": NS}
    for task in TASKS:
        med = {name[arch[r["arch"]]]: float(np.median(r["abs_grad"])) for r in rows if r["task"] == task}
        print(f"  {TASK_SHORT[task]}: " + "  ".join(f"{k} {v:.3f}" for k, v in med.items())
              + f"  Equiv/NoSym {med['Equiv'] / med['NoSym']:.1f}")


def scaling():
    print("\n== Dependence on N (L = 4) and on L (N of the main setting): gaps")
    for task in TASKS:
        c = task_cells(task)
        L, N = MAIN[task]
        for a in MODELS:
            byn = [mean([c[(L, n, a)][s] for s in common(*[c[(L, n, b)] for b in MODELS])], "gap") for n in (10, 20, 40, 80)]
            byl = [mean([c[(l, N, a)][s] for s in common(*[c[(l, N, b)] for b in MODELS])], "gap") for l in LAYERS[task]]
            print(f"  {TASK_SHORT[task]} {name[a]}: N=10..80 " + " ".join(f"{g:+.2f}" for g in byn)
                  + f" | L={LAYERS[task]} " + " ".join(f"{g:+.2f}" for g in byl))
    c = task_cells("mnist_bin")
    s80 = common(*[c[(4, 80, a)] for a in MODELS])
    print(f"  MNIST: test accuracy of NonEquiv at N=80 {mean([c[(4, 80, NE)][s] for s in s80], 'clean_acc'):.2f}, "
          f"of Equiv at N=40 {mean(main_cells('mnist_bin')[0][EQ], 'clean_acc'):.2f}")


def curves():
    print("\n== Learning curves (main settings): test / train accuracy at epochs 7, 14, 21, 35, 70")
    for task in TASKS:
        v, *_ = main_cells(task)
        for a in MODELS:
            ep = v[a][0]["curve"]["epoch"]
            sel = [ep.index(e) for e in (7, 14, 21, 35, 70)]
            Y = np.array([r["curve"]["clean_acc"] for r in v[a]]).mean(0)
            T = np.array([r["curve"]["train_acc"] for r in v[a]]).mean(0)
            print(f"  {TASK_SHORT[task]} {name[a]}: test " + " ".join(f"{Y[i]:.2f}" for i in sel)
                  + " | train " + " ".join(f"{T[i]:.2f}" for i in sel))


def pilots_v0():
    rows = [json.loads(line) for line in open(os.path.join(RESULTS, "synthetic_pilots", "v0_small_noisy.jsonl"))]
    acc = [r["clean_acc"] for r in rows if r["N"] in (40, 80)]
    print(f"\n== First version of the synthetic task: {len(rows)} runs; test accuracy at N=40, 80: "
          f"{min(acc):.2f}-{max(acc):.2f}")


if __name__ == "__main__":
    main_settings()
    upright()
    cues()
    noise()
    failures()
    dla()
    gradients()
    scaling()
    curves()
    pilots_v0()
