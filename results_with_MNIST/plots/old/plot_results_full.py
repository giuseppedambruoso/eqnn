import math
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1. Reading and Parsing Data ---
file_path = "test_accuracies_DONT_DELETE.txt"
data = []

# Regex pattern
pattern = re.compile(
    r"Seed: (\d+), Sample size: (\d+), Non equivariance: ([\d\.]+), Noise: ([\d\.]+), Test Accuracy: ([\d\.]+) \((?:at epoch|Epoch:) (\d+)\)"
)

try:
    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ne_val = int(float(match.group(3)))
                data.append(
                    {
                        "seed": int(match.group(1)),
                        "sample_size": int(match.group(2)),
                        "non_equivariance": ne_val,
                        "p_err": float(match.group(4)),
                        "test_accuracy": float(match.group(5)),
                    }
                )
except FileNotFoundError:
    # Creating dummy data for demonstration if file doesn't exist
    print(f"Warning: {file_path} not found. Generating dummy data for visualization.")
    np.random.seed(42)
    sample_sizes_demo = [100, 200, 400, 800, 1600]
    p_errs_demo = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ne_types = [0, 1, 2, 3, 4]
    for _ in range(500):
        data.append(
            {
                "seed": np.random.randint(0, 100),
                "sample_size": np.random.choice(sample_sizes_demo),
                "non_equivariance": np.random.choice(ne_types),
                "p_err": np.random.choice(p_errs_demo),
                "test_accuracy": np.random.uniform(0.5, 1.0),
            }
        )

df = pd.DataFrame(data)

# --- 2. Aggregation ---
agg = (
    df.groupby(["sample_size", "non_equivariance", "p_err"])["test_accuracy"]
    .agg(["mean", "std"])
    .reset_index()
)

# ==============================================================================
# --- 3. APS / Physical Review Style Settings ---
# ==============================================================================
plt.rcParams.update(
    {
        # Fonts: STIX is very close to Times/LaTeX standard in physics
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        # Axes and Ticks (Inward ticks, ticks on all sides)
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 1.0,  # Slightly thicker box
        "axes.grid": False,  # Grid is usually off or very subtle in PRL
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,  # Ticks on top
        "ytick.right": True,  # Ticks on right
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Legend
        "legend.fontsize": 8,
        "legend.frameon": False,  # Often cleaner without box, or verify thin box
        # Lines and Markers
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    }
)

# --- 4. Define Styles ---
# Using high-contrast colors suitable for academic printing
styles = {
    0: {"color": "black", "marker": "o", "linestyle": "-", "label": r"Equiv"},
    1: {
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "--",
        "label": r"ApproxEquiv$_1$",
    },  # Blue
    2: {
        "color": "#D55E00",
        "marker": "^",
        "linestyle": "-.",
        "label": r"ApproxEquiv$_2$",
    },  # Vermillion
    3: {
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": ":",
        "label": r"NonEquiv",
    },  # Reddish Purple
    4: {
        "color": "#009E73",
        "marker": "x",
        "linestyle": "-",
        "label": r"NonEquiv$^+$",
    },  # Bluish Green
}

draw_order = [3, 2, 1, 0, 4]
sample_sizes = sorted(agg["sample_size"].unique())


# Helper to format axes
def format_prl_axis(ax):
    # Ensure limits are tight but visible
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)


# ==============================================================================
# --- 5. Grid 1: Test Accuracy vs Noise (Grouped by N) ---
# ==============================================================================
fig, axs = plt.subplots(
    2, 2, figsize=(7, 6), sharex=True, sharey=True
)  # Width ~7 inches (double column)
axs = axs.flatten()

for i, N in enumerate(sample_sizes):
    if i >= len(axs):
        break
    ax = axs[i]
    subset_N = agg[agg["sample_size"] == N]

    if subset_N.empty:
        continue

    for ne in draw_order:
        if ne not in styles:
            continue
        sub = subset_N[subset_N["non_equivariance"] == ne].sort_values("p_err")
        if sub.empty:
            continue

        x = sub["p_err"]
        y = sub["mean"]
        yerr = sub["std"].fillna(0)
        s = styles[ne]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=s["label"],
            color=s["color"],
            marker=s["marker"],
            linestyle=s["linestyle"],
            capsize=2,  # Error bars with caps are very common in physics
            elinewidth=0.8,
            markerfacecolor="none",  # Hollow markers are often used to see overlap
        )

    ax.text(
        0.05, 0.05, f"$N = {N}$", transform=ax.transAxes, fontsize=10, fontweight="bold"
    )
    format_prl_axis(ax)

    # Legend only in the first plot or optimized position
    if i == 0:
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 0.15),
            frameon=False,
            fontsize=8,
            ncol=1,
        )

# Formatting
fig.text(0.5, 0.01, r"Noise Probability ($p_{\mathrm{err}}$)", ha="center", fontsize=10)
fig.text(0.00, 0.5, "Test Accuracy", va="center", rotation="vertical", fontsize=10)
plt.tight_layout(rect=[0.02, 0.03, 1, 0.98])
plt.savefig("acc_vs_p_by_N.pdf", dpi=300)


# ==============================================================================
# --- 6. Grid 2: Test Accuracy vs N (Fixed Non-Equivariance) ---
# ==============================================================================
unique_ne = sorted(agg["non_equivariance"].unique())
unique_p = sorted(agg["p_err"].unique())

num_plots = len(unique_ne)
cols = 2
rows = math.ceil(num_plots / cols)

fig2, axs2 = plt.subplots(rows, cols, figsize=(7, 2.5 * rows), sharey=True, sharex=True)
axs2 = axs2.flatten()

# Sequential colormap for noise
colors = cm.viridis(np.linspace(0, 0.9, len(unique_p)))

for i, ne in enumerate(unique_ne):
    if i >= len(axs2):
        break
    ax = axs2[i]
    subset_ne = agg[agg["non_equivariance"] == ne]

    for j, p in enumerate(unique_p):
        sub = subset_ne[subset_ne["p_err"] == p].sort_values("sample_size")
        if sub.empty:
            continue

        x = sub["sample_size"]
        y = sub["mean"]
        yerr = sub["std"].fillna(0)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=colors[j],
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            label=f"$p_{{err}}={p}$" if i == 0 else "",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(sample_sizes)

    title_label = styles.get(ne, {}).get("label", f"NE {ne}")
    ax.text(0.5, 0.9, title_label, transform=ax.transAxes, ha="center", fontsize=9)
    format_prl_axis(ax)

# Remove empty axes
for j in range(i + 1, len(axs2)):
    fig2.delaxes(axs2[j])

fig2.text(0.5, 0.01, r"Sample Size $N$", ha="center", fontsize=10)
fig2.text(0.00, 0.5, "Test Accuracy", va="center", rotation="vertical", fontsize=10)

# Legend (Outside or inside one plot)
handles, labels = axs2[0].get_legend_handles_labels()
fig2.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(unique_p) // 2,
    frameon=False,
    fontsize=8,
)

plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])
plt.savefig("acc_vs_N_by_equiv.pdf", dpi=300)


# ==============================================================================
# --- 7. New Grid: Test Accuracy vs N (Fixed Noise) ---
# ==============================================================================
num_plots_3 = len(unique_p)
cols_3 = 3
rows_3 = math.ceil(num_plots_3 / cols_3)

fig3, axs3 = plt.subplots(
    rows_3, cols_3, figsize=(7, 2.2 * rows_3), sharey=True, sharex=True
)
axs3 = axs3.flatten()

for i, p in enumerate(unique_p):
    ax = axs3[i]
    subset_p = agg[agg["p_err"] == p]

    for ne in draw_order:
        if ne not in styles:
            continue
        sub = subset_p[subset_p["non_equivariance"] == ne].sort_values("sample_size")

        if sub.empty:
            continue

        x = sub["sample_size"]
        y = sub["mean"]
        yerr = sub["std"].fillna(0)
        s = styles[ne]

        ax.plot(
            x,
            y,
            label=s["label"] if i == 0 else "",
            color=s["color"],
            marker=s["marker"],
            linestyle=s["linestyle"],
            linewidth=1.2,
            markersize=4,
        )
        # Optional: Light fill for confidence, though error bars are more strictly physics
        # ax.fill_between(x, y - yerr, y + yerr, color=s["color"], alpha=0.1, linewidth=0)

    ax.text(
        0.95, 0.05, f"$p_{{err}} = {p}$", transform=ax.transAxes, ha="right", fontsize=9
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sample_sizes)
    # Only label bottom row
    if i >= len(unique_p) - cols_3:
        ax.set_xticklabels(sample_sizes)
    else:
        ax.set_xticklabels([])

    format_prl_axis(ax)

# Remove empty subplots
for j in range(i + 1, len(axs3)):
    fig3.delaxes(axs3[j])

fig3.text(0.5, 0.01, r"Sample Size $N$", ha="center", fontsize=10)
fig3.text(0.00, 0.5, "Test Accuracy", va="center", rotation="vertical", fontsize=10)

# Legend at top
handles, labels = axs3[0].get_legend_handles_labels()
fig3.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=5,
    frameon=False,
    fontsize=8,
)

plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])
plt.savefig("acc_vs_N_by_p.pdf", dpi=300)
plt.show()

