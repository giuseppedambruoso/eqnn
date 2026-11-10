"""Shared helpers of the analysis scripts: loading the runs, statistics, formatting and the style of the figures."""

import collections
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PAPER = os.environ.get("EQNN_PAPER_DIR", os.path.join(ROOT, "paper"))  # where figures/ and tables/ are written
FIGURES = os.path.join(PAPER, "figures")
TABLES = os.path.join(PAPER, "tables")

# model id -> (name in the paper, short label, color (Okabe-Ito), marker); the invariant models first
MODELS = {
    "config6": ("Equiv", "Equiv", "#0072B2", "o"),
    "config10": ("NonEquiv-Twirled", "Twirled", "#009E73", "^"),
    "config7": ("NonEquiv", "NonEquiv", "#D55E00", "s"),
    "config6n": ("Equiv-NoSym", "NoSym", "#CC79A7", "D"),
}
COMPARISON_1 = ("config6", "config10", "config7")  # how the invariance is obtained
COMPARISON_2 = ("config6", "config6n")  # the same circuit, with and without an invariant readout
UPRIGHT = ("NonEquiv, upright images", "#E69F00", "v", ":")
TASKS = {"mnist_bin": "MNIST, random pose", "shapes_cs": "synthetic", "galaxy_0_4": "Galaxy10, random pose",
         "ery_1_2": "red blood cells, random pose"}
CUE_TASKS = ("mnist_bin", "shapes_cs", "galaxy_0_4", "ery_1_2")  # the tasks of the experiments on shortcuts and on hardware noise
TASK_SHORT = {"mnist_bin": "MNIST", "shapes_cs": "Synthetic", "galaxy_0_4": "Galaxy10", "ery_1_2": "Blood cells"}
RESULT_DIR = {"mnist_bin": "mnist_pose", "shapes_cs": "synthetic", "galaxy_0_4": "galaxy", "ery_1_2": "erythrocytes"}
LAYERS = {"mnist_bin": (1, 2, 4, 8), "shapes_cs": (1, 2, 4), "galaxy_0_4": (1, 2, 4, 8), "ery_1_2": (1, 2, 4, 8)}
# main settings: L = 4 and N = 40 (MNIST, Galaxy10, red blood cells) or 20 (synthetic)
MAIN = {"mnist_bin": (4, 40), "shapes_cs": (4, 20), "galaxy_0_4": (4, 40), "ery_1_2": (4, 40)}
NATURAL = {"mnist_bin": "pose_rand", "shapes_cs": "none", "galaxy_0_4": "pose_rand", "ery_1_2": "pose_rand"}  # the images without a cue
CUES = {"mnist_bin": (("mark_e_rand", "corner watermark"), ("gradient_rand", "brightness gradient"),
                      ("invariant_rand", "center/corner mark")),
        "shapes_cs": (("mark_e", "corner watermark"), ("gradient", "brightness gradient"),
                      ("invariant", "center/corner mark")),
        "galaxy_0_4": (("mark_e_rand", "corner watermark"), ("gradient_rand", "brightness gradient"),
                       ("invariant_rand", "center/corner mark")),
        "ery_1_2": (("mark_e_rand", "corner watermark"), ("gradient_rand", "brightness gradient"),
                    ("invariant_rand", "center/corner mark"))}
NOISE_SCALES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

FONT_DIR = os.path.join(ROOT, "data", "fonts")  # Latin Modern files extracted from the TeX bundle (data/ is not tracked)
LATIN_MODERN = ("lmroman10-regular", "lmroman10-italic", "lmroman10-bold", "lmroman10-bolditalic",
                "lmromancaps10-regular")


def _otf_to_ttf(src, dst):
    """Converts an OpenType font with PostScript (CFF) outlines into a TrueType font (the recipe otf2ttf of
    fontTools): matplotlib embeds fonts in the PDF as TrueType (pdf.fonttype 42), and an embedded CFF font would make
    the PDF malformed."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable
    font = TTFont(src)
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyphs[name] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, glyphs
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    for name, g in glyphs.items():
        if hasattr(g, "xMin"):
            font["hmtx"][name] = (font["hmtx"][name][0], g.xMin)
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    for k in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
              "maxSizeOfInstructions"):
        setattr(maxp, k, 0)
    maxp.maxComponentElements = max(len(getattr(g, "components", [])) for g in glyphs.values())
    maxp.compile(font)
    post = font["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    try:
        post.compile(font)
    except OverflowError:
        post.formatType = 3
    font.sfntVersion = "\000\001\000\000"
    from fontTools.misc.timeTools import timestampFromString
    font["head"].created = font["head"].modified = timestampFromString("Mon Nov 30 00:00:00 2009")
    font.save(dst)


def use_latex_font():
    """The font of the paper in every figure: text in Latin Modern Roman, the font of the LaTeX text, and mathematics
    in Computer Modern, from which Latin Modern is derived. The OpenType files of Latin Modern are taken from the TeX
    bundle of tectonic (or from an installation of the Debian/Ubuntu package fonts-lmodern) and converted once into
    TrueType files in data/fonts. Model names in small caps, as in the text, can be set with
    fontfamily="Latin Modern Roman Caps"."""
    import shutil
    import subprocess
    from matplotlib import font_manager
    os.makedirs(FONT_DIR, exist_ok=True)
    for name in LATIN_MODERN:
        ttf = os.path.join(FONT_DIR, f"{name}.ttf")
        if not os.path.exists(ttf):
            otf = os.path.join(FONT_DIR, f"{name}.otf")
            system = f"/usr/share/texmf/fonts/opentype/public/lm/{name}.otf"
            if os.path.exists(system):
                shutil.copyfile(system, otf)
            else:
                with open(otf, "wb") as f:
                    subprocess.run(["tectonic", "-X", "bundle", "cat", f"{name}.otf"], stdout=f,
                                   stderr=subprocess.DEVNULL, check=True)
            _otf_to_ttf(otf, ttf)
            os.remove(otf)
        font_manager.fontManager.addfont(ttf)
    # \partial is missing from the Computer Modern table of mathtext, which would take it from STIX: it is the
    # character 0x40 of cmmi10
    from matplotlib import _mathtext_data
    _mathtext_data.latex_to_bakoma.setdefault(r"\partial", ("cmmi10", 0x40))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True})


plt.rcParams.update({
    "font.size": 8.7, "axes.titlesize": 8.7, "axes.labelsize": 8.7, "legend.fontsize": 8.2, "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8, "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.3,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "hatch.linewidth": 0.6, "legend.handlelength": 1.8,
})
use_latex_font()


def runs(name: str, shortcut: str | None = None) -> list[dict]:
    """The runs of results/<name>/runs.jsonl (optionally of one shortcut), with the generalization gap."""
    out = []
    for line in open(os.path.join(RESULTS, name, "runs.jsonl")):
        r = json.loads(line)
        if shortcut is None or r["shortcut"] == shortcut:
            r["gap"] = r["train_acc"] - r["clean_acc"]
            out.append(r)
    return out


def cells(rows) -> dict:
    """(layers, N, model id) -> {seed: run}."""
    d = collections.defaultdict(dict)
    for r in rows:
        d[(r["layers"], r["N"], r["arch"])][r["seed"]] = r
    return d


def task_cells(task: str) -> dict:
    """The noiseless runs of a task without a cue."""
    return cells(runs(RESULT_DIR[task], NATURAL[task]))


def common(*series) -> list[int]:
    """Seeds shared by every series of a comparison (every comparison of the paper is paired)."""
    sets = [set(s) for s in series if s]
    return sorted(set.intersection(*sets)) if sets else []


def ms(v):
    """Mean and standard error."""
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def paired(a: dict, b: dict, key: str, sign: int = 1, seeds=None):
    """Paired difference sign * (a - b) over the seeds: (mean, standard error, number of positive differences,
    number of seeds, two-sided Wilcoxon signed-rank p value, the differences)."""
    seeds = seeds if seeds is not None else common(a, b)
    d = np.array([sign * (a[s][key] - b[s][key]) for s in seeds])
    p = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else float("nan")
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d), p, d


def fmt(m, e, plus=False):
    """Mean with the standard error on the last digit in brackets, e.g. 0.90(1)."""
    return f"${m:+.2f}({round(e * 100):d})$" if plus else f"${m:.2f}({round(e * 100):d})$"


def fmt_p(p):
    if p != p:
        return "--"
    if p >= 0.01:
        return f"${p:.2f}$"
    mant, exp = f"{p:.0e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def style(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel(ax, letter, x=-0.12, y=1.02):
    """Bold panel letter (Springer style) at the top left of an axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")


def row_labels(fig, axes, labels, x=0.004):
    """The task of every row of panels, written once, in bold, at the left of the row (call after tight_layout)."""
    for ax, lab in zip(axes, labels):
        pos = ax.get_position()
        fig.text(x, (pos.y0 + pos.y1) / 2, lab, rotation=90, ha="left", va="center", fontsize=8.7, fontweight="bold")


def model_handles(archs, **kw):
    return [Line2D([], [], color=MODELS[a][2], marker=MODELS[a][3], ls="-", ms=3.5, label=MODELS[a][1], **kw)
            for a in archs]


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name))
    plt.close(fig)


def write_table(name, lines):
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, name), "w") as f:
        f.write("\n".join(lines) + "\n")
