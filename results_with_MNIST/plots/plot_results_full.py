import math
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1. Reading and Parsing Data ---
file_path = "summary_DONT_DELETE.txt"
data = []

pattern = re.compile(
    r"Seed: (\d+), Sample size: (\d+), Non equivariance: (\d+), Noise: ([\d\.]+), Test Accuracy: ([\d\.]+) \(at epoch (\d+)\)"
)

with open(file_path, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            data.append(
                {
                    "seed": int(match.group(1)),
                    "sample_size": int(match.group(2)),
                    "non_equivariance": int(match.group(3)),
                    "p_err": float(match.group(4)),
                    "test_accuracy": float(match.group(5)),
                }
            )

df = pd.DataFrame(data)

# --- 2. Aggregation ---
agg = (
    df.groupby(["sample_size", "non_equivariance", "p_err"])["test_accuracy"]
    .agg(["mean", "std"])
    .reset_index()
)

# --- 3. Plotting Settings ---
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# --- 4. Define Styles ---
styles = {
    0: {"color": "black", "marker": "o", "linestyle": "-", "label": r"Equiv"},
    1: {"color": "#1F77B4", "marker": "s", "linestyle": "--", "label": r"ApproxEquiv1"},
    2: {"color": "#D62728", "marker": "^", "linestyle": "-.", "label": r"ApproxEquiv2"},
    3: {"color": "#DAA520", "marker": "D", "linestyle": ":", "label": r"NonEquiv"},
    4: {"color": "purple", "marker": "x", "linestyle": "-", "label": r"NonEquiv+"},
}

draw_order = [3, 2, 1, 0]
sample_sizes = sorted(agg["sample_size"].unique())

# ==============================================================================
# --- 5. Grid 1: Test Accuracy vs Noise (Grouped by N) ---
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axs = axs.flatten()

for i, N in enumerate(sample_sizes):
    if i >= len(axs):
        break
    ax = axs[i]
    subset_N = agg[agg["sample_size"] == N]

    if subset_N.empty:
        ax.text(0.5, 0.5, f"No Data for N={N}", ha="center", va="center")
        continue

    for ne in draw_order:
        sub = subset_N[subset_N["non_equivariance"] == ne].sort_values("p_err")
        if sub.empty:
            continue

        x = sub["p_err"]
        y = sub["mean"]
        yerr = sub["std"].fillna(0)
        s = styles.get(ne, styles[3])

        ax.fill_between(x, y - yerr, y + yerr, color=s["color"], alpha=0.2, linewidth=0)
        ax.plot(
            x,
            y,
            label=s["label"],
            color=s["color"],
            marker=s["marker"],
            linestyle=s["linestyle"],
        )

    ax.set_title(f"Sample Size $N = {N}$")
    ax.set_ylim(0.4, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="gray")

fig.text(0.5, 0.02, r"Noise Probability ($p_{err}$)", ha="center", fontsize=14)
fig.text(0.02, 0.5, "Test Accuracy", va="center", rotation="vertical", fontsize=14)
plt.suptitle(
    "Impact of Equivariance on Robustness across Sample Sizes", fontsize=16, y=0.98
)
plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
plt.savefig("acc_vs_p_by_N.pdf", dpi=300)


# ==============================================================================
# --- 6. Grid 2: Test Accuracy vs N (Fixed Non-Equivariance) ---
# ==============================================================================

unique_ne = sorted(agg["non_equivariance"].unique())
unique_p = sorted(agg["p_err"].unique())

num_plots = len(unique_ne)
cols = 2
rows = math.ceil(num_plots / cols)

fig2, axs2 = plt.subplots(rows, cols, figsize=(14, 5 * rows), sharey=True)
axs2 = axs2.flatten()

colors = cm.viridis(np.linspace(0, 1, len(unique_p)))

for i, ne in enumerate(unique_ne):
    ax = axs2[i]
    subset_ne = agg[agg["non_equivariance"] == ne]

    for j, p in enumerate(unique_p):
        sub = subset_ne[subset_ne["p_err"] == p].sort_values("sample_size")
        if sub.empty:
            continue

        x = sub["sample_size"]
        y = sub["mean"]
        yerr = sub["std"].fillna(0)

        ax.plot(x, y, color=colors[j], marker="o", markersize=4)
        ax.fill_between(x, y - yerr, y + yerr, color=colors[j], alpha=0.1, linewidth=0)

    # Log2 Scale
    ax.set_xscale("log", base=2)
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(sample_sizes)
    ax.minorticks_off()

    title_label = styles.get(ne, {}).get("label", f"Non-Equivariance {ne}")
    ax.set_title(title_label)
    ax.set_ylim(0.4, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

for j in range(i + 1, len(axs2)):
    fig2.delaxes(axs2[j])

fig2.text(0.5, 0.02, r"Sample Size $N$ (Log Scale)", ha="center", fontsize=14)
fig2.text(0.02, 0.5, "Test Accuracy", va="center", rotation="vertical", fontsize=14)

legend_lines = [
    plt.Line2D([0], [0], color=colors[j], lw=2) for j in range(len(unique_p))
]
legend_labels = [f"$p_{{err}}={p}$" for p in unique_p]
fig2.legend(
    legend_lines,
    legend_labels,
    loc="center right",
    title="Noise ($p_{err}$)",
    bbox_to_anchor=(0.98, 0.5),
    fontsize=9,
)

plt.suptitle(
    "Impact of Sample Size on Noise Robustness (Fixed Non-Equivariance)",
    fontsize=16,
    y=0.98,
)
plt.tight_layout(rect=[0.03, 0.03, 0.88, 0.96])
plt.savefig("acc_vs_N_by_equiv.pdf", dpi=300)


# ==============================================================================
# --- 7. New Grid: Test Accuracy vs N (Fixed Noise, Varying Equivariance) ---
# ==============================================================================

# Calculate grid dimensions based on number of p_err values
num_plots_3 = len(unique_p)
cols_3 = 4  # Adjust columns as needed (4 looks good for ~11 plots)
rows_3 = math.ceil(num_plots_3 / cols_3)

fig3, axs3 = plt.subplots(rows_3, cols_3, figsize=(16, 4 * rows_3), sharey=True)
axs3 = axs3.flatten()

# Iterate over each Noise Probability (p_err)
for i, p in enumerate(unique_p):
    ax = axs3[i]
    subset_p = agg[agg["p_err"] == p]

    # Iterate over Equivariance levels (lines)
    # Using 'draw_order' (3,2,1,0) to keep layering consistent with Grid 1
    for ne in draw_order:
        sub = subset_p[subset_p["non_equivariance"] == ne].sort_values("sample_size")

        if sub.empty:
            continue

        x = sub["sample_size"]
        y = sub["mean"]
        yerr = sub["std"].fillna(0)
        s = styles.get(ne, styles[3])  # Get style (color/marker/dash)

        # Plot lines
        ax.plot(
            x,
            y,
            label=s["label"],
            color=s["color"],
            marker=s["marker"],
            linestyle=s["linestyle"],
        )
        ax.fill_between(x, y - yerr, y + yerr, color=s["color"], alpha=0.1, linewidth=0)

    # Styling
    ax.set_title(f"Noise $p_{{err}} = {p}$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(sample_sizes)
    ax.minorticks_off()
    ax.set_ylim(0.4, 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    # Add legend to the first plot only to save space
    if i == 0:
        ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.9,
            edgecolor="gray",
            fontsize=8,
        )

# Remove empty subplots
for j in range(i + 1, len(axs3)):
    fig3.delaxes(axs3[j])

# Global Labels for Grid 3
fig3.text(0.5, 0.02, r"Sample Size $N$ (Log Scale)", ha="center", fontsize=14)
fig3.text(0.02, 0.5, "Test Accuracy", va="center", rotation="vertical", fontsize=14)
plt.suptitle("Impact of Sample Size on Accuracy (Fixed Noise)", fontsize=16, y=0.98)
plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

plt.savefig("acc_vs_N_by_p.pdf", dpi=300)
plt.show()
