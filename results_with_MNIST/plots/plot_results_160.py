import re

import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Lettura e Parsing dei Dati ---
file_path = "test_accuracies_160.txt"
data = []
# Regex per estrarre i campi dal file di testo
pattern = re.compile(
    r"Seed: (\d+), Sample size: (\d+), Non equivariance: (\d+), Noise: ([\d\.]+), Test Accuracy: ([\d\.]+) \(at epoch (\d+)\)"
)

with open(file_path, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            data.append(
                {
                    "non_equivariance": int(match.group(3)),
                    "p_err": float(match.group(4)),
                    "test_accuracy": float(match.group(5)),
                }
            )

df = pd.DataFrame(data)

# --- 2. Aggregazione ---
# Calcoliamo media e deviazione standard per ogni gruppo
agg = (
    df.groupby(["non_equivariance", "p_err"])["test_accuracy"]
    .agg(["mean", "std"])
    .reset_index()
)

# --- 3. Impostazioni Grafico (Stile Pubblicazione) ---
plt.rcParams.update(
    {
        "font.size": 14,
        "font.family": "serif",  # Font serif (es. Times) per standard accademici
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (10, 6),
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

fig, ax = plt.subplots()

# --- 4. Definizione Stili (COLORI SCAMBIATI) ---
# 1 = Blu (prima Giallo), 3 = Giallo Oro (prima Blu)
styles = {
    0: {"color": "black", "marker": "o", "linestyle": "-", "label": r"Equiv"},
    1: {
        "color": "#1F77B4",
        "marker": "s",
        "linestyle": "--",
        "label": r"ApproxEquiv1",
    },  # Blu (Matplotlib default blue)
    2: {
        "color": "#D62728",
        "marker": "^",
        "linestyle": "-.",
        "label": r"ApproxEquiv2",
    },  # Rosso
    3: {
        "color": "#DAA520",
        "marker": "D",
        "linestyle": ":",
        "label": r"NonEquiv",
    },  # Giallo (GoldenRod)
}

# --- 5. Ordine di Disegno ---
# Disegniamo prima le curve con varianza maggiore (sul fondo) e poi quelle più sottili (sopra)
# Ordine: 3 (Larga) -> 2 -> 1 -> 0 (Sottile)
draw_order = [3, 2, 1, 0]

for ne in draw_order:
    sub = agg[agg["non_equivariance"] == ne].sort_values("p_err")
    x = sub["p_err"]
    y = sub["mean"]
    yerr = sub["std"].fillna(0)  # Se std è NaN (es. 1 solo seed), mettiamo 0

    s = styles[ne]

    # Banda di errore (semitrasparente)
    ax.fill_between(x, y - yerr, y + yerr, color=s["color"], alpha=0.2, linewidth=0)

    # Linea e Marcatori
    ax.plot(
        x,
        y,
        label=s["label"],
        color=s["color"],
        marker=s["marker"],
        linestyle=s["linestyle"],
    )

# --- 6. Etichette e Finiture ---
ax.set_xlabel(r"Noise Probability ($p_{err}$)")
ax.set_ylabel("Test Accuracy")
ax.set_title("Impact of Equivariance on Robustness to Noise")
ax.set_ylim(0.5, 1.0)

# Legenda
ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="gray")

# Rimuoviamo i bordi superiore e destro per pulizia
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# Salvataggio
plt.savefig("test_accuracies_vs_p_err_by_non_equivariance.png", dpi=300)
plt.savefig("test_accuracies_vs_p_err_by_non_equivariance.pdf", dpi=300)
plt.show()
