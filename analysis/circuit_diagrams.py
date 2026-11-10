"""Figure of the circuits in the paper: one layer of the Equiv and NonEquiv circuits side by side, with one legend, vector PDF. Gates that share the same trainable
angle are filled with the same color (six angles: three for the four 'fine' blocks, three for the 'coarse' block).

    python analysis/circuit_diagrams.py [out_dir]    # writes circuits.pdf (default: paper/figures)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from pennylane.drawer.drawable_layers import drawable_layers  # noqa: E402
from pennylane.drawer.mpldrawer import MPLDrawer  # noqa: E402

from eqnn.circuits import layer_gates  # noqa: E402

with plt.rc_context():  # analysis.common sets the style of the data figures on import: only its font is used
    from analysis.common import use_latex_font  # noqa: E402

N = 8
# the figure spans the text width at about half its size: FS = 17 prints at about 8.5 pt, close to the 9 pt of the
# captions
FS = 17
# Okabe-Ito colors ordered by luminance (about 0.87, 0.76, 0.65, 0.55, 0.46, 0.29), so that the six angles are also
# told apart in grayscale; dark fills get white labels
COLORS = ["#F2E86D", "#8DD0F2", "#E69F00", "#CC79A7", "#009E73", "#0072B2"]
LABEL_COLORS = ["black", "black", "black", "black", "white", "white"]
# the angles of layer l (paper: Eq. circuit_layer): alpha_l for the four fine blocks, beta_l for the coarse block
LEGEND = [r"$\alpha_{l,1}$", r"$\alpha_{l,2}$", r"$\alpha_{l,3}$", r"$\beta_{l,1}$", r"$\beta_{l,2}$", r"$\beta_{l,3}$"]


def build_ops(equivariant: bool):
    ops, colors = [qml.QubitUnitary(np.eye(2 ** N), wires=range(N))], [None]
    for word, k in layer_gates(equivariant):  # k: index of the angle in the layer, 0..5
        w = list(word)
        if len(w) > 1:
            ops.append(qml.PauliRot(0.1, "".join(word[q] for q in w), wires=w))
        else:
            ops.append({"X": qml.RX, "Y": qml.RY}[word[w[0]]](0.1, wires=w[0]))
        colors.append(COLORS[k])
    return ops, colors


def draw_circuit(fig, rect, equivariant: bool, title: str, measured=(0, 4)):
    """Draws one layer of a circuit in the axes `rect` (left, bottom, width, height) of `fig`."""
    ops, colors = build_ops(equivariant)
    wire_map = {w: i for i, w in enumerate(range(N))}
    layers = drawable_layers(ops, wire_map=wire_map)
    color_of = {id(op): c for op, c in zip(ops, colors)}
    drawer = MPLDrawer(n_layers=len(layers) + 1, wire_map=wire_map, wire_options={"linewidth": 1.2, "color": "black"},
                       figsize=(7.2, 3.3))
    # MPLDrawer draws on a figure of its own: its axes are moved into `fig`, so that the two circuits share a figure
    # and a legend
    own, src = drawer.fig, drawer.ax
    ax = fig.add_axes(rect, xlim=src.get_xlim(), ylim=src.get_ylim(), xticks=[], yticks=[])
    ax.axis("off")
    for line in src.lines:
        ax.add_line(plt.Line2D(*line.get_data(), zorder=line.get_zorder(), linewidth=line.get_linewidth(),
                               color=line.get_color()))
    drawer._ax, drawer._fig = ax, fig
    plt.close(own)
    drawer.label([str(w) for w in range(N)], text_options={"fontsize": FS})
    for l, layer in enumerate(layers):
        for op in layer:
            if isinstance(op, qml.QubitUnitary):
                text = "feature encoding"  # the amplitude encoding |psi(x)> (paper: Eq. encoding), as in fig_twirled
            elif isinstance(op, qml.PauliRot):
                text = r"$\mathrm{R}_{" + op.hyperparameters["pauli_word"] + "}$"  # R_P = exp(-i eta P/2), upright R
            else:
                text = r"$\mathrm{R}_" + op.name[1] + "$"  # R_X, R_Y (upright R: R is the representation)
            opts = {"zorder": 3, "linewidth": 1.4}
            opts["facecolor"] = color_of[id(op)] or "white"
            tcol = LABEL_COLORS[COLORS.index(color_of[id(op)])] if color_of[id(op)] else "black"
            drawer.box_gate(l, list(op.wires), text, box_options=opts, text_options={"fontsize": FS, "color": tcol},
                             extra_width=0.25 if len(op.wires) == 1 else 0)  # room for the labels at FS
    for w in measured:
        drawer.measure(len(layers), w, box_options={"facecolor": "white", "zorder": 3})
    ax.set_title(title, fontsize=FS + 1, pad=8)


def draw(fname: str):
    """The two circuits side by side, with one legend of the six angles of a layer."""
    qml.drawer.use_style("black_white")
    use_latex_font()
    fig = plt.figure(figsize=(14.4, 4.8))
    draw_circuit(fig, (0.0, 0.08, 0.49, 0.92), True, "equivariant circuit (Equiv, Equiv-NoSym)")
    draw_circuit(fig, (0.51, 0.08, 0.49, 0.92), False, "non-equivariant circuit (NonEquiv, NonEquiv-Twirled)")
    handles = [Patch(facecolor=c, edgecolor="black", label=lab) for c, lab in zip(COLORS, LEGEND)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=FS, frameon=False, bbox_to_anchor=(0.5, -0.03),
               columnspacing=3.0)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "paper/figures"
    os.makedirs(out, exist_ok=True)
    draw(f"{out}/circuits.pdf")
    print("ok")
