import os
import re
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- CONFIGURAZIONE PERCORSI ---
RESULTS_DIR = os.path.join("..", "results")
EXTRA_FILE_PATH = os.path.join("old", "test_accuracies_20.txt")
OUTPUT_CSV = "all_results_cleaned.csv"
MISSING_CSV = "missing_configs.csv"
LAUNCHER_SH = "run_missing_experiments.sh"

# --- PARAMETRI ATTESI ---
# Qui impostiamo 104 come il target ufficiale per il "quinto seed"
EXPECTED_SEEDS = [42, 1234, 5678, 999, 104] 
EXPECTED_NS = [20, 40, 80, 160, 320]
EXPECTED_PS = [round(x * 0.01, 2) for x in range(11)] 
EXPECTED_NES = [0, 1, 2, 3, 4]

# --- STILE PLOT ---
plt.rcParams.update({
    "font.size": 10, "font.family": "serif", "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    "xtick.direction": "in", "ytick.direction": "in"
})

def parse_folder_name_regex(folder_name):
    params = {}
    match_n = re.search(r"(?:DATA\.N|N\.DATA|N)\s*=\s*(\d+)", folder_name, re.IGNORECASE)
    if match_n: params["N"] = int(match_n.group(1))
    match_seed = re.search(r"(?:GENERAL\.)?seed\s*=\s*(\d+)", folder_name, re.IGNORECASE)
    if match_seed: params["seed"] = int(match_seed.group(1))
    match_ne = re.search(r"non[._]equivariance\s*=\s*(\d+)", folder_name, re.IGNORECASE)
    if match_ne: params["non_equivariance"] = int(match_ne.group(1))
    match_p = re.search(r"p[._]err\s*=\s*(\d+\.?\d*)", folder_name, re.IGNORECASE)
    if match_p: params["p_err"] = float(match_p.group(1))
    return params

def extract_metrics_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty or "val_acc" not in df.columns: return None
        max_idx = df["val_acc"].idxmax()
        return {"max_val_acc": df.loc[max_idx, "val_acc"], "max_epoch": int(df.loc[max_idx, "epoch"])}
    except: return None

def parse_txt_file(file_path):
    data = []
    if not os.path.exists(file_path): return data
    regex = r"Seed:\s*(\d+).*?Sample size:\s*(\d+).*?Non equivariance:\s*(\d+).*?Noise:\s*([\d\.]+).*?Test Accuracy:\s*([\d\.]+).*?epoch\s*(\d+)"
    with open(file_path, "r") as f:
        for line in f:
            match = re.search(regex, line)
            if match:
                data.append({"seed": int(match.group(1)), "N": int(match.group(2)), 
                             "non_equivariance": int(match.group(3)), "p_err": float(match.group(4)),
                             "max_val_acc": float(match.group(5)), "max_epoch": int(match.group(6))})
    return data

def generate_bash_launcher(df_missing):
    if df_missing.empty: return
    sh_content = [
        "#!/bin/bash",
        "GREEN='\\033[0;32m'", "BLUE='\\033[0;34m'", "NC='\\033[0m'",
        "echo -e \"${BLUE}=== Avvio Recupero Esperimenti Mancanti ===${NC}\\n\"",
        "source $HOME/miniconda3/etc/profile.d/conda.sh",
        "conda activate eqnn_env",
        "export OMP_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\nexport TORCH_NUM_THREADS=1\nexport OPENBLAS_NUM_THREADS=1",
        "echo -e \"${BLUE}Esecuzione Job Hydra...${NC}\"\n"
    ]
    # Ordiniamo per raggruppamento coerente
    df_missing = df_missing.sort_values(by=['N', 'non_equivariance'])
    grouped = df_missing.groupby(['N', 'non_equivariance'])
    
    for (n_val, ne_val), group in grouped:
        # Qui i seed saranno 104 se manca quel blocco, perché EXPECTED_SEEDS usa 104
        seeds = ",".join(map(str, sorted(group['seed'].unique())))
        ps = ",".join(map(str, sorted(group['p_err'].round(3).unique())))
        
        cmd = (f"poetry run python src/eqnn/main.py -m \\\n"
               f"    DATA.N={n_val} \\\n"
               f"    GENERAL.seed={seeds} \\\n"
               f"    QNN.non_equivariance={ne_val} \\\n"
               f"    QNN.p_err={ps} \\\n"
               f"    hydra/launcher=joblib \\\n"
               f"    hydra.launcher.n_jobs=10\n")
        sh_content.append(cmd)
        sh_content.append(f"echo -e \"${{GREEN}}✓ Completato blocco N={n_val} NE={ne_val}${{NC}}\"\n")
    
    sh_content.append("echo -e \"\\n${GREEN}=== TUTTI GLI ESPERIMENTI COMPLETATI ===${NC}\"")
    with open(LAUNCHER_SH, "w") as f: f.write("\n".join(sh_content))
    os.chmod(LAUNCHER_SH, 0o755)
    print(f"✅ Script di lancio creato: {LAUNCHER_SH}")

def analyze_coverage_and_launcher(df):
    # 1. Creiamo la griglia ideale usando esplicitamente 104
    all_combos = list(itertools.product(EXPECTED_SEEDS, EXPECTED_NS, EXPECTED_PS, EXPECTED_NES))
    df_expected = pd.DataFrame(all_combos, columns=["seed", "N", "p_err", "non_equivariance"])
    df_expected["p_err"] = df_expected["p_err"].round(3)
    
    # 2. Prepariamo i dati attuali per il confronto
    df_actual = df.copy()
    df_actual["p_err"] = df_actual["p_err"].round(3)
    
    # TRUCCO: Per il controllo di presenza, mappiamo temporaneamente 101 su 104.
    # Così se abbiamo 101, il merge lo vedrà come "104 presente".
    df_actual['seed'] = df_actual['seed'].replace(101, 104)

    # 3. Merge Left per trovare i buchi
    # Se df_expected ha 104 e df_actual ha 104 (che era 101), matchano.
    # Se df_actual non ha nulla, rimane il buco con seed=104.
    missing = pd.merge(df_expected, df_actual, on=["seed", "N", "p_err", "non_equivariance"], how="left")
    missing = missing[missing["max_val_acc"].isna()][["seed", "N", "p_err", "non_equivariance"]]
    
    if not missing.empty:
        print(f"❌ Mancano {len(missing)} configurazioni.")
        missing = missing.sort_values(by=["seed", "N", "non_equivariance", "p_err"])
        missing.to_csv(MISSING_CSV, index=False)
        generate_bash_launcher(missing)
    else: 
        print("✅ Copertura 100% (Considerando 101 equivalente a 104).")

def generate_plot(df_subset, output_filename, ylim=None):
    if df_subset.empty: return
    Ns = sorted(df_subset["N"].unique())
    rows = (len(Ns) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows), squeeze=False)
    axes = axes.flatten()
    labels_map = {0: "Equiv", 1: "Approx1", 2: "Approx2", 3: "NonEquiv", 4: "Equiv-ized"}
    for i, n_val in enumerate(Ns):
        ax = axes[i]
        sub = df_subset[df_subset["N"] == n_val]
        grouped = sub.groupby(["non_equivariance", "p_err"])["max_val_acc"].agg(["mean", "std"]).reset_index()
        for ne in sorted(grouped["non_equivariance"].unique()):
            data = grouped[grouped["non_equivariance"] == ne].sort_values("p_err")
            ax.plot(data["p_err"], data["mean"], marker='o', label=labels_map.get(ne, f"NE={ne}"))
            ax.fill_between(data["p_err"], data["mean"]-data["std"].fillna(0), data["mean"]+data["std"].fillna(0), alpha=0.2)
        ax.set_title(f"N = {n_val}")
        if ylim: ax.set_ylim(ylim)
    for k in range(i + 1, len(axes)): fig.delaxes(axes[k])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), frameon=False)
    plt.tight_layout(); plt.savefig(output_filename, bbox_inches="tight"); plt.close()

def main():
    data = []
    if os.path.exists(RESULTS_DIR):
        for folder in os.listdir(RESULTS_DIR):
            p = os.path.join(RESULTS_DIR, folder, "metrics.csv")
            if os.path.exists(p):
                params = parse_folder_name_regex(folder)
                m = extract_metrics_from_csv(p)
                if m and all(k in params for k in ["N", "p_err", "non_equivariance"]): data.append({**params, **m})

    data.extend(parse_txt_file(EXTRA_FILE_PATH))
    df = pd.DataFrame(data)
    if df.empty: return

    # --- PULIZIA DATI ---
    # Creiamo una colonna temporanea 'seed_equiv' dove 101 diventa 104 solo per il drop_duplicates
    df['seed_equiv'] = df['seed'].replace(101, 104)
    
    # Rimuoviamo duplicati basandoci su seed_equiv (quindi 101 e 104 collidono)
    df = df.drop_duplicates(subset=["seed_equiv", "N", "non_equivariance", "p_err"], keep='last')
    
    # Rimuoviamo la colonna temporanea. 
    # IMPORTANTE: Il 'seed' originale (che sia 101 o 104) rimane invariato nel CSV.
    df = df.drop(columns=['seed_equiv'])

    # Filtro p_err
    df = df[df["p_err"] <= 0.101]
    
    # ORDINAMENTO: seed, N, non_equivariance, p_err
    df = df.sort_values(by=["seed", "N", "non_equivariance", "p_err"])
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"📊 Dataset pulito salvato ({len(df)} record).")

    # Analisi Copertura
    analyze_coverage_and_launcher(df)

    # Plot
    y_min, y_max = df["max_val_acc"].min(), df["max_val_acc"].max()
    ylim = (max(0, y_min - 0.05), min(1, y_max + 0.05))
    generate_plot(df[df["non_equivariance"].isin([0,1,2])], "summary_equiv.pdf", ylim)
    generate_plot(df[df["non_equivariance"].isin([3,4])], "summary_nonequiv.pdf", ylim)

if __name__ == "__main__":
    main()
