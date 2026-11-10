"""Schematic figures of the paper:
  encoding_p4m_diagram.pdf  the amplitude encoding and the action of D4 on a 4 x 4 image (text typeset by LaTeX)
  fig_design.pdf            the four models (circuit equivariant or not, model invariant or not) and the comparisons
                            (text typeset by LaTeX)
  fig_function_classes.pdf  the function classes of the four models (from results/dla.json, written by eqnn.dla)
  fig_noise_model.pdf       where the realistic noise (eqnn.noise) acts on one block of the circuit
  fig_twirled.pdf           how NonEquiv-Twirled computes its output (text typeset by LaTeX)

    python analysis/schematics.py [out_dir]     (default: paper/figures)

The figures with text typeset by LaTeX, in the font of the paper (Latin Modern), need pdflatex or tectonic.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

with plt.rc_context():  # analysis.common sets the style of the data figures on import: only its font is used
    from analysis.common import use_latex_font  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper/figures"
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
NEQ, TW, NOISE, IDLE, READ = "#D55E00", "#009E73", "#F0E442", "#E69F00", "#56B4E9"
EQ, NS = "#0072B2", "#CC79A7"
GNAMES = ["$e$", "$r$", "$r^2$", "$r^3$", "$t$", "$tr$", "$tr^2$", "$tr^3$"]

# Figures whose text is typeset by LaTeX (pgf backend of matplotlib), with the font and the notation of the paper.
LATEX_RC = {"pgf.rcfonts": False, "pgf.texsystem": "pdflatex", "font.family": "serif", "font.size": 9,
            "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}\usepackage{bm}"}


def save_latex(fig, path):
    """Saves fig as a PDF in which every text is typeset by LaTeX. With pdflatex on the PATH matplotlib compiles the
    figure itself; otherwise the pgf picture is compiled with tectonic. Every text of these figures is a single line
    placed by its anchor (rotated texts with rotation_mode="anchor"), so LaTeX positions it and the text metrics,
    which matplotlib would ask to LaTeX, are not needed: without pdflatex they are replaced by a rough estimate."""
    if shutil.which("pdflatex"):
        fig.savefig(path, backend="pgf")
        return
    if not shutil.which("tectonic"):
        raise RuntimeError("pdflatex or tectonic is needed to typeset the text of the figures")
    import matplotlib.backends.backend_pgf as pgf

    def rough_metrics(self, s, prop, ismath):
        size = prop.get_size_in_points() * self.dpi / 72
        return 0.5 * size * len(s), 0.7 * size, 0.2 * size

    w, h = fig.get_size_inches()
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(pgf.RendererPgf, "get_text_width_height_descent", rough_metrics), \
            mock.patch.object(pgf, "_get_image_inclusion_command", lambda: r"\includegraphics"):
        fig.savefig(os.path.join(tmp, "figure.pgf"), backend="pgf")
        with open(os.path.join(tmp, "figure.tex"), "w") as f:
            f.write("\n".join([pgf._DOCUMENTCLASS, rf"\usepackage[papersize={{{w}in,{h}in}},margin=0in]{{geometry}}",
                               r"\usepackage{pgf}", pgf._get_preamble(), r"\pagestyle{empty}",
                               r"\begin{document}", r"\centering", r"\input{figure.pgf}", r"\end{document}", ""]))
        subprocess.run(["tectonic", "-X", "compile", "figure.tex"], cwd=tmp, check=True, capture_output=True)
        shutil.copyfile(os.path.join(tmp, "figure.pdf"), path)


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=kw.get("color", "k"),
                                 shrinkA=0, shrinkB=0))


def _meter(ax, x, y, s=0.11):
    """A measurement symbol centered at (x, y), in axes units."""
    from matplotlib.patches import Arc
    ax.add_patch(Rectangle((x - s, y - 0.75 * s), 2 * s, 1.5 * s, fc="white", ec="k", lw=0.7, zorder=4))
    ax.add_patch(Arc((x, y - 0.45 * s), 1.4 * s, 1.4 * s, theta1=0, theta2=180, lw=0.7, zorder=5))
    ax.plot([x, x + 0.6 * s], [y - 0.45 * s, y + 0.5 * s], color="k", lw=0.7, zorder=5)


def fig_twirled(out):
    """How NonEquiv-Twirled computes its output: the eight transformed copies of the image go through the same
    NonEquiv circuit V with the same angles, and the outputs are averaged. Text typeset by LaTeX, in the font and the
    notation of the paper (R_X(g)x, f_tw, f_neq, V)."""
    from eqnn.data import apply_d4, build_data
    x, y, _, _ = build_data("mnist_bin", 40, 1, "none")
    img = x[int((y == 0).nonzero()[0, 0])].reshape(16, 16)
    names = ["$e$", "$r$", "$r^2$", "$r^3$", "$t$", "$tr$", "$tr^2$", "$tr^3$"]
    with plt.rc_context(LATEX_RC):
        W, H = 7.0, 2.05  # inches; the axes are in inches
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes((0, 0, 1, 1))
        ym = 0.98  # vertical center of the pipeline

        def picture(a, x0, y0, s):
            ax.imshow(a, cmap="gray_r", extent=(x0, x0 + s, y0, y0 + s), aspect="auto", interpolation="nearest")
            ax.add_patch(Rectangle((x0, y0), s, s, fc="none", ec="0.45", lw=0.5))

        # the image
        s0 = 0.7
        picture(img.numpy(), 0.04, ym - s0 / 2, s0)
        ax.text(0.04 + s0 / 2, ym - s0 / 2 - 0.07, "image $x$", ha="center", va="top")
        arrow(ax, (0.82, ym), (1.02, ym))
        # the eight transformed copies, 2 rows x 4 columns
        s1, dx, x1 = 0.33, 0.41, 1.1
        for g in range(8):
            r, c = divmod(g, 4)
            x0, y0 = x1 + dx * c, ym + 0.11 - r * 0.6
            picture(apply_d4(img, g).numpy(), x0, y0, s1)
            ax.text(x0 + s1 / 2, y0 - 0.035, names[g], ha="center", va="top", fontsize=8.7)
        ax.text(x1 + 1.5 * dx + s1 / 2, ym + 0.53, r"copies $R_{\mathcal X}(g)x$, $g\in D_4$", ha="center", va="bottom")
        arrow(ax, (2.69, ym), (2.89, ym))
        # the circuit: encoding, NonEquiv unitary V, measurement of O = (X_0 + X_4)/2
        cx0, cx1 = 3.0, 4.42
        wires = [ym + 0.49 - 0.14 * k for k in range(8)]
        for k, yw in enumerate(wires):
            ax.plot([cx0 + 0.14, cx1 - 0.08], [yw, yw], color="k", lw=0.6, zorder=1)
            ax.text(cx0 + 0.09, yw, f"${k}$", ha="right", va="center", fontsize=8)
        top, bot = wires[0] + 0.09, wires[-1] - 0.09
        ax.add_patch(FancyBboxPatch((cx0 + 0.22, bot), 0.3, top - bot, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc="white", ec="k", lw=0.8, zorder=3))
        ax.text(cx0 + 0.37, ym, "feature encoding", ha="center", va="center", rotation=90,
                rotation_mode="anchor", fontsize=8.7, zorder=4)
        ax.add_patch(FancyBboxPatch((cx0 + 0.64, bot), 0.44, top - bot, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc="#fbe3d6", ec=NEQ, lw=1.0, zorder=3))
        ax.text(cx0 + 0.86, ym, r"$V(\bm{\theta})$", ha="center", va="center", fontsize=10, zorder=4)
        for k in (0, 4):
            _meter(ax, cx1 - 0.19, wires[k], s=0.09)
        ax.add_patch(FancyBboxPatch((cx0 - 0.06, bot - 0.06), cx1 - cx0 + 0.08, top - bot + 0.12,
                                    boxstyle="round,pad=0,rounding_size=0.06", fc="none", ec=NEQ, lw=1.1))
        ax.text((cx0 + cx1) / 2, top + 0.22, r"\textsc{NonEquiv} circuit,", ha="center", va="bottom", color=NEQ)
        ax.text((cx0 + cx1) / 2, top + 0.07, r"same angles $\bm{\theta}$ for every copy", ha="center", va="bottom",
                color=NEQ, fontsize=8.7)
        arrow(ax, (4.5, ym), (4.7, ym))
        # the average
        bx0, bx1 = 4.8, 6.96
        ax.add_patch(FancyBboxPatch((bx0, ym - 0.3), bx1 - bx0, 0.6, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="#e3f3ec", ec=TW, lw=1.1))
        ax.text((bx0 + bx1) / 2, ym, r"$f_{\mathrm{tw}}(x)=\dfrac{1}{8}\displaystyle\sum_{g\in D_4}"
                r"f_{\mathrm{neq}}\bigl(R_{\mathcal X}(g)x\bigr)$", ha="center", va="center")
        ax.text((bx0 + bx1) / 2, ym + 0.38, "mean of the eight outputs", ha="center", va="bottom", fontsize=8.7)
        ax.text((bx0 + bx1) / 2, ym - 0.38,
                r"invariant: $f_{\mathrm{tw}}\bigl(R_{\mathcal X}(h)x\bigr)=f_{\mathrm{tw}}(x)$",
                ha="center", va="top", fontsize=8.7, color=TW)
        ax.text((bx0 + bx1) / 2, ym - 0.58, "eight circuit evaluations per image", ha="center", va="top", fontsize=8.7,
                color=TW)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
        save_latex(fig, f"{out}/fig_twirled.pdf")
        plt.close(fig)


def fig_noise_model(out):
    """One fine block of the circuit (bit positions j=0, k=1: qubits 0, 1, 4, 5) followed by the readout, with the noise of
    eqnn.noise drawn where it acts."""
    use_latex_font()
    fig = plt.figure(figsize=(7.0, 3.5))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-4, 100); ax.set_ylim(-11, 37); ax.axis("off")
    ys = {q: 31 - 4 * q for q in range(8)}
    for q, yq in ys.items():
        ax.plot([2, 82], [yq, yq], color="k", lw=0.8, zorder=0)
        ax.text(0.5, yq, f"{q}", ha="right", va="center")

    def box(x, y, label, fc, w, fs=8.2):
        ax.add_patch(Rectangle((x, y - 1.3), w, 2.6, fc=fc, ec="k", lw=0.8, zorder=2))
        ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=fs, zorder=3)

    # single-qubit rotations on 0, 4 (eta_1) and 1, 5 (eta_2), simultaneous, each followed by its noise
    for q, th in ((0, 1), (4, 1), (1, 2), (5, 2)):
        box(5, ys[q], f"$\\mathrm{{R}}(\\eta_{th})$", "#fde7c8", 7)
        box(12.6, ys[q], "$\\mathcal{N}$", NOISE, 3.6)
    # four-qubit rotation on 0, 1, 4, 5: two boxes joined by a vertical line (it does not act on 2, 3)
    x4 = 19
    ax.plot([x4 + 4, x4 + 4], [ys[5], ys[0]], color="k", lw=1.6, zorder=1)
    for lo, hi in ((0, 1), (4, 5)):
        ax.add_patch(Rectangle((x4, ys[hi] - 1.3), 8, ys[lo] - ys[hi] + 2.6, fc="#fde7c8", ec="k", lw=0.8, zorder=2))
    ax.text(x4 + 4, (ys[0] + ys[1]) / 2, "$\\mathrm{R}_{PPPP}(\\eta_3)$", ha="center", va="center", fontsize=7.8, zorder=3)
    for q in (0, 1, 4, 5):
        box(x4 + 8.6, ys[q], "$\\mathcal{N}$", NOISE, 3.6)
    # qubits 2, 3, 6, 7 wait during the whole block
    for q in (2, 3, 6, 7):
        box(5, ys[q], "$\\mathcal{R}(t_{\\rm wait})$", IDLE, 26.2, fs=8)
    # the rest of the circuit
    ax.add_patch(Rectangle((35, ys[7] - 2), 20, ys[0] - ys[7] + 4, fc="white", ec="0.6", lw=0.8, ls="--", zorder=2))
    ax.text(45, (ys[3] + ys[4]) / 2, "the other blocks\nand layers,\nwith the same\nnoise", ha="center",
            va="center", fontsize=8.2, zorder=3)
    # readout of X_0 and X_4
    for q in (0, 4):
        box(58, ys[q], "$\\mathcal{R}(t_{\\rm wait})$", IDLE, 12, fs=7.8)
        box(72, ys[q], "$X$", READ, 4.5)
    ax.annotate("", xy=(84, (ys[0] + ys[4]) / 2), xytext=(82.5, ys[0]), arrowprops={"arrowstyle": "-", "lw": 0.6})
    ax.annotate("", xy=(84, (ys[0] + ys[4]) / 2), xytext=(82.5, ys[4]), arrowprops={"arrowstyle": "-", "lw": 0.6})
    ax.text(84.5, (ys[0] + ys[4]) / 2, "$f=\\frac{1}{2}(\\tilde X_0+\\tilde X_4)$", va="center", fontsize=8.7)
    legend = [(NOISE, "$\\mathcal{N}$: after every gate, on each of its qubits: depolarizing (gate error), then"
                      " thermal relaxation ($T_1$, $T_2$) for the gate duration"),
              (IDLE, "$\\mathcal{R}$: thermal relaxation of a qubit that waits (for the other qubits of a gate, or for"
                     " the readout)"),
              (READ, "readout of $X$ with asymmetric assignment errors: $\\langle\\tilde X\\rangle=(1-p_{10}-p_{01})"
                     "\\langle X\\rangle+p_{01}-p_{10}$")]
    for i, (col, text) in enumerate(legend):
        yl = -3 - 3.2 * i
        ax.add_patch(Rectangle((2, yl - 1), 3, 2, fc=col, ec="k", lw=0.6))
        ax.text(6.5, yl, text, va="center", fontsize=8)
    ax.text(2, 35.3, "One block ($j=0$, $k=1$) with realistic noise: single-qubit channels only "
            "($\\mathrm{R}=\\mathrm{R}_X$ or $\\mathrm{R}_Y$, $P=Y$ or $Z$)", fontsize=8.7)
    fig.savefig(f"{out}/fig_noise_model.pdf")
    plt.close(fig)


def fig_encoding(out):
    """The symmetry-aware amplitude encoding on a 4 x 4 image: pixel (i, j) is the amplitude of |i>_r |j>_c; the
    reflections t_x (rows reversed: about the horizontal axis) and t_y (columns reversed: about the vertical axis) and
    the anticlockwise rotation r generate D4. Text typeset by LaTeX, in the font of the paper."""
    import math
    with plt.rc_context(LATEX_RC):
        u = 0.27  # inches per cell: the cells are a little larger than their kets
        (xa, xb), (ya, yb) = (-3.45, 2.45), (-2.45, 3.4)
        fig = plt.figure(figsize=((xb - xa) * u, (yb - ya) * u))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(xa, xb); ax.set_ylim(ya, yb); ax.set_aspect("equal"); ax.axis("off")
        fills = ("#fff4dc", "#e3efe0")
        for i in range(4):
            for j in range(4):
                x, y = -2 + j, 1 - i
                ax.add_patch(Rectangle((x, y), 1, 1, fc=fills[(i + j) % 2], ec="#b9cdb4", lw=0.6, zorder=1))
                ax.text(x + 0.5, y + 0.5, rf"$|{i}{j}\rangle$", ha="center", va="center", fontsize=9, color="0.3",
                        zorder=3)
        # the two reflection axes
        for a, b in (((-2.35, 0), (2.35, 0)), ((0, -2.35), (0, 2.35))):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>", mutation_scale=6, lw=0.8, color="0.25",
                                         shrinkA=0, shrinkB=0, zorder=2))
        TY, TX, RC = "#009E73", "#E69F00", "#7B3FA0"
        flip = {"arrowstyle": "<|-|>", "mutation_scale": 6, "lw": 1.1, "shrinkA": 0, "shrinkB": 0}
        # t_y: reflection about the vertical axis, a flip across it
        ax.add_patch(FancyArrowPatch((-0.5, 2.55), (0.5, 2.55), color=TY, connectionstyle="arc3,rad=-0.45", **flip))
        ax.text(0, 2.9, "$t_y$", ha="center", va="bottom", fontsize=9, color=TY)
        # t_x: reflection about the horizontal axis, a flip across it
        ax.add_patch(FancyArrowPatch((-2.55, 0.5), (-2.55, -0.5), color=TX, connectionstyle="arc3,rad=0.45", **flip))
        ax.text(-2.92, 0, "$t_x$", ha="right", va="center", fontsize=9, color=TX)
        # r: anticlockwise rotation by 90 degrees about the center
        p1, p2 = [(3.0 * math.cos(math.radians(t)), 3.0 * math.sin(math.radians(t))) for t in (118, 152)]
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=7, lw=1.1, color=RC,
                                     connectionstyle="arc3,rad=0.15", shrinkA=0, shrinkB=0))
        ax.text(-2.55, 2.55, "$r$", ha="center", va="center", fontsize=9, color=RC)
        save_latex(fig, f"{out}/encoding_p4m_diagram.pdf")
        plt.close(fig)


def fig_design(out):
    """The four models: a circuit (equivariant or not) and a model (invariant or not), with the two comparisons. Text
    typeset by LaTeX, in the font and the notation of the paper (Eq. models: f_eq, f_ns, f_tw, f_neq; circuits U, V)."""
    with plt.rc_context(LATEX_RC):
        fig = plt.figure(figsize=(4.9, 2.95))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
        cells = {  # (row, column): (name, color, output, comparisons)
            (0, 0): (r"\textsc{Equiv}", EQ, r"$f_{\mathrm{eq}}(x)=\langle\psi(x)|U^\dagger O\,U|\psi(x)\rangle$", "12"),
            (0, 1): (r"\textsc{Equiv-NoSym}", NS,
                     r"$f_{\mathrm{ns}}(x)=\langle\psi(x)|U^\dagger O_{\mathrm{ns}}\,U|\psi(x)\rangle$", "2"),
            (1, 0): (r"\textsc{NonEquiv-Twirled}", TW,
                     r"$\textstyle f_{\mathrm{tw}}(x)=\frac{1}{8}\sum_{g}f_{\mathrm{neq}}(R_{\mathcal X}(g)x)$",
                     "1"),
            (1, 1): (r"\textsc{NonEquiv}", NEQ,
                     r"$f_{\mathrm{neq}}(x)=\langle\psi(x)|V^\dagger O\,V|\psi(x)\rangle$", "1"),
        }
        x0, y0, w, h = 29, 8, 34, 18.5
        tag = {"1": "#444444", "2": "#888888"}
        for (r, c), (name, col, formula, comps) in cells.items():
            x, y = x0 + c * (w + 2.5), y0 + (1 - r) * (h + 2.5)
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=col, alpha=0.1,
                                        ec="none"))
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec=col,
                                        lw=1.3))
            ax.text(x + w / 2, y + h - 4.3, name, ha="center", va="center", fontsize=9.5, color=col)
            ax.text(x + w / 2, y + h / 2 - 1.6, formula, ha="center", va="center", fontsize=8.2)
            for k, cmp in enumerate(comps):
                cx, cy = x + w - 2.6 - 4.6 * k, y + 2.8
                ax.add_patch(plt.Circle((cx, cy), 1.9, fc=tag[cmp], ec="none"))
                ax.text(cx, cy, cmp, ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
        for c, text in enumerate(("invariant model", "non-invariant model")):
            ax.text(x0 + c * (w + 2.5) + w / 2, y0 + 2 * h + 2.5 + 2.2, text, ha="center", va="bottom", fontsize=8.5)
        for r, (text, circuit) in enumerate((("equivariant circuit", r"$U(\bm{\theta})$"),
                                             ("non-equivariant circuit", r"$V(\bm{\theta})$"))):
            yc = y0 + (1 - r) * (h + 2.5) + h / 2
            ax.text(x0 - 1.5, yc + 2.0, text, ha="right", va="center", fontsize=8.2)
            ax.text(x0 - 1.5, yc - 2.6, circuit, ha="right", va="center", fontsize=8.2)
        # the definitions, under the four models (shifted left, so that the line fits in the figure)
        ax.text(59.5, 2.8, r"$V=C_{\mathrm c}\,U C_{\mathrm c}^\dagger,\quad O=\tfrac{1}{2}(X_0+X_4),\quad "
                r"O_{\mathrm{ns}}=(X_0+Z_0+Z_4)/(1+\sqrt{2})$", ha="center", va="center", fontsize=8.2, color="0.2")
        for k, (cmp, text) in enumerate((("1", "comparison 1"), ("2", "comparison 2"))):
            cx, cy = 3, 56.5 - 4.6 * k
            ax.add_patch(plt.Circle((cx, cy), 1.9, fc=tag[cmp], ec="none"))
            ax.text(cx, cy, cmp, ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
            ax.text(cx + 3, cy, text, ha="left", va="center", fontsize=8.2)
        save_latex(fig, f"{out}/fig_design.pdf")
        plt.close(fig)


def fig_function_classes(out):
    """Function classes: the quadratic features x^T M x that each model combines, split into the D4-invariant ones
    (left) and the others (right), with the part shared with Equiv (results/dla.json, written by eqnn.dla)."""
    use_latex_font()
    d = json.load(open("results/dla.json"))
    dim, inv, pair = d["dim"], d["inv"], d["pair"]
    fig, bx = plt.subplots(figsize=(3.45, 2.2))
    rows = [("Equiv", EQ), ("Twirled", TW), ("NonEquiv", NEQ), ("NoSym", NS)]
    for i, (k, col) in enumerate(rows):
        sh = dim["Equiv"] if k == "Equiv" else pair[f"Equiv&{k}"]
        iv, nv = inv[k], dim[k] - inv[k]
        bx.barh(i, -sh, color=col, height=0.62, hatch="////", edgecolor="white", lw=0)
        if iv > sh:
            bx.barh(i, -(iv - sh), left=-sh, color=col, height=0.62, lw=0)
        if nv:
            bx.barh(i, nv, color=col, height=0.62, alpha=0.45, lw=0)
        note = f"{iv}" + (f"\n({sh} shared)" if 0 < sh < iv else "")
        bx.text(-iv - 30, i, note, ha="right", va="center", fontsize=7.5 if note.isdigit() else 7)
        bx.text(nv + 30, i, f"{nv}", ha="left", va="center", fontsize=7.5)
    bx.axvline(0, color="k", lw=0.7)
    bx.set_yticks(range(len(rows)), [f"{k}\n{dim[k]} in all" for k, _ in rows], fontsize=7.5)
    bx.tick_params(axis="y", length=0)
    bx.invert_yaxis()
    bx.set_xlim(-1250, 1550)
    bx.set_xticks([-600, -300, 0, 300, 600, 900, 1200])
    bx.set_xticklabels(["600", "300", "0", "300", "600", "900", "1200"], fontsize=7.5)
    bx.set_xlabel(r"independent quadratic features $x^{\top}Mx$", fontsize=8.2)
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    bx.grid(axis="x", alpha=0.25, lw=0.5)
    bx.text(-20, -0.62, r"$\leftarrow$ invariant", ha="right", va="bottom", fontsize=7.5)
    bx.text(20, -0.62, r"not invariant $\rightarrow$", ha="left", va="bottom", fontsize=7.5)
    leg = [Rectangle((0, 0), 1, 1, fc="0.45", hatch="////", ec="white", lw=0),
           Rectangle((0, 0), 1, 1, fc="0.45", lw=0), Rectangle((0, 0), 1, 1, fc="0.45", alpha=0.45, lw=0)]
    bx.legend(leg, ["shared with Equiv", "other invariant", "not invariant"], loc="upper right", fontsize=7,
              frameon=False, handlelength=1.5, bbox_to_anchor=(1.02, 1.0))
    fig.savefig(f"{out}/fig_function_classes.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_encoding(OUT)
    fig_design(OUT)
    fig_function_classes(OUT)
    fig_noise_model(OUT)
    fig_twirled(OUT)
