import re

import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Lettura e Parsing dei Dati ---
file_path = "summary_DONT_DELETE.txt"
data = []

# Regex aggiornata per chiarezza sui gruppi:
# Group 1: Seed
# Group 2: Sample size (N) -> DA ESTRARRE
# Group 3: Non equivariance
# Group 4: Noise
# Group 5: Test Accuracy
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
                    "sample_size": int(match.group(2)),  # Estraniamo N
                    "non_equivariance": int(match.group(3)),
                    "p_err": float(match.group(4)),
                    "test_accuracy": float(match.group(5)),
                }
            )

df = pd.DataFrame(data)

# --- 2. Aggregazione ---
# Raggruppiamo per Sample Size (N), Non Equivariance e Noise
agg = (
    df.groupby(["sample_size", "non_equivariance", "p_err"])["test_accuracy"]
    .agg(["mean", "std"])
    .reset_index()
)

# --- 3. Impostazioni Grafico (Stile Pubblicazione) ---
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# Definizione Stili (come da tua richiesta)
styles = {
    0: {"color": "black", "marker": "o", "linestyle": "-", "label": r"Equiv"},
    1: {
        "color": "#1F77B4",
        "marker": "s",
        "linestyle": "--",
        "label": r"ApproxEquiv1",
    },  # Blu
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
    },  # Giallo Oro
}

# Ordine di disegno: le curve più spesse/variabili sotto, quelle fini sopra
draw_order = [3, 2, 1, 0]
sample_sizes = [20, 40, 80, 160]

# Creiamo una griglia 2x2
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axs = axs.flatten()  # Appiattiamo l'array per iterarci facilmente

# --- 4. Loop sui Sample Size ---
for i, N in enumerate(sample_sizes):
    ax = axs[i]

    # Filtriamo i dati per il sample size corrente
    subset_N = agg[agg["sample_size"] == N]

    # Se non ci sono dati per questo N, saltiamo (o gestiamo l'errore)
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

        s = styles[ne]

        # Banda di errore
        ax.fill_between(x, y - yerr, y + yerr, color=s["color"], alpha=0.2, linewidth=0)

        # Linea principale
        ax.plot(
            x,
            y,
            label=s["label"],
            color=s["color"],
            marker=s["marker"],
            linestyle=s["linestyle"],
        )

    # --- 5. Etichette specifiche per Subplot ---
    ax.set_title(f"Sample Size $N = {N}$")
    ax.set_ylim(0.4, 1.02)  # Adatta questo range in base ai tuoi dati reali

    # Rimuoviamo spine superflue
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Griglia
    ax.grid(True, alpha=0.3)

    # La legenda la mettiamo solo nel primo grafico (o dove preferisci) per pulizia,
    # oppure in tutti. Qui la metto in tutti ma piccola, o solo nel primo.
    # Mettiamola in tutti per chiarezza immediata.
    ax.legend(
        loc="lower left", frameon=True, framealpha=0.9, edgecolor="gray", fontsize=9
    )

# Etichette globali per gli assi esterni
fig.text(0.5, 0.02, r"Noise Probability ($p_{err}$)", ha="center", fontsize=14)
fig.text(0.02, 0.5, "Test Accuracy", va="center", rotation="vertical", fontsize=14)

plt.suptitle(
    "Impact of Equivariance on Robustness across Sample Sizes", fontsize=16, y=0.98
)
plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])  # Lascia spazio per le label globali

# Salvataggio
plt.savefig("test_accuracies_by_N_grid.png", dpi=300)
plt.savefig("test_accuracies_by_N_grid.pdf", dpi=300)
plt.show()
