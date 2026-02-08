import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# --- CONFIGURAZIONE PERCORSI ---
RESULTS_DIR = os.path.join('..', 'results')
EXTRA_FILE_PATH = os.path.join('old', 'test_accuracies_20.txt')

OUTPUT_CSV = 'all_results.csv'
OUTPUT_PLOT = 'summary_plot.pdf'

# --- STILE PLOT (PRL Style) ---
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':'
})

def parse_folder_name_regex(folder_name):
    """Estrae parametri dai nomi delle cartelle."""
    params = {}
    match_n = re.search(r'(?:DATA\.N|N\.DATA|N)\s*=\s*(\d+)', folder_name, re.IGNORECASE)
    if match_n: params['N'] = int(match_n.group(1))
    
    match_seed = re.search(r'(?:GENERAL\.)?seed\s*=\s*(\d+)', folder_name, re.IGNORECASE)
    if match_seed: params['seed'] = int(match_seed.group(1))
        
    match_ne = re.search(r'non[._]equivariance\s*=\s*(\d+)', folder_name, re.IGNORECASE)
    if match_ne: params['non_equivariance'] = int(match_ne.group(1))
        
    match_p = re.search(r'p[._]err\s*=\s*(\d+\.?\d*)', folder_name, re.IGNORECASE)
    if match_p: params['p_err'] = float(match_p.group(1))
    return params

def extract_metrics_from_csv(file_path):
    """Legge il csv metrics.csv."""
    try:
        df = pd.read_csv(file_path)
        if df.empty: return None
        max_idx = df['val_acc'].idxmax()
        row = df.loc[max_idx]
        return {'max_val_acc': row['val_acc'], 'max_epoch': int(row['epoch'])}
    except Exception:
        return None

def parse_txt_file(file_path):
    """Legge il file di testo manuale per i dati mancanti."""
    data = []
    if not os.path.exists(file_path):
        print(f"ATTENZIONE: File extra '{file_path}' non trovato.")
        return data

    print(f"Lettura file extra: {file_path}")
    regex_pattern = r"Seed:\s*(\d+).*?Sample size:\s*(\d+).*?Non equivariance:\s*(\d+).*?Noise:\s*([\d\.]+).*?Test Accuracy:\s*([\d\.]+).*?epoch\s*(\d+)"
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(regex_pattern, line)
            if match:
                data.append({
                    'seed': int(match.group(1)),
                    'N': int(match.group(2)),
                    'non_equivariance': int(match.group(3)),
                    'p_err': float(match.group(4)),
                    'max_val_acc': float(match.group(5)),
                    'max_epoch': int(match.group(6))
                })
    return data

def main():
    data = []
    
    # 1. SCANSIONE CARTELLE
    print(f"Scansione directory results: {os.path.abspath(RESULTS_DIR)}")
    if os.path.exists(RESULTS_DIR):
        subfolders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]
        for folder in subfolders:
            metrics_path = os.path.join(RESULTS_DIR, folder, 'metrics.csv')
            if not os.path.exists(metrics_path): continue
            
            params = parse_folder_name_regex(folder)
            if 'N' not in params or 'p_err' not in params or 'non_equivariance' not in params: continue

            metrics = extract_metrics_from_csv(metrics_path)
            if metrics:
                data.append({**params, **metrics})

    # 2. FILE EXTRA
    data.extend(parse_txt_file(EXTRA_FILE_PATH))

    # 3. DATAFRAME
    df = pd.DataFrame(data)
    if df.empty:
        print("Nessun dato valido trovato.")
        return

    desired_columns = ['seed', 'N', 'non_equivariance', 'p_err', 'max_val_acc', 'max_epoch']
    df = df[[c for c in desired_columns if c in df.columns]]
    df = df.sort_values(by=['seed', 'N', 'non_equivariance', 'p_err'])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"File '{OUTPUT_CSV}' creato.")
    
    unique_Ns = sorted(df['N'].unique())
    print(f"Valori di N totali: {unique_Ns}") 

    # 4. PLOTTING
    num_plots = len(unique_Ns)
    cols = 3 
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5 * rows), squeeze=False)
    axes = axes.flatten()

    unique_equiv = sorted(df['non_equivariance'].unique())
    cmap = plt.get_cmap('viridis', len(unique_equiv))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*'] 

    legend_labels = {
        0: "Equiv",
        1: "ApproxEquiv1",
        2: "ApproxEquiv2",
        3: "NonEquiv",
        4: "NonEquiv Equivariantized"
    }

    y_min = df['max_val_acc'].min()
    y_max = df['max_val_acc'].max()
    margin = (y_max - y_min) * 0.1

    for i, n_val in enumerate(unique_Ns):
        ax = axes[i]
        df_n = df[df['N'] == n_val]
        
        grouped = df_n.groupby(['non_equivariance', 'p_err'])['max_val_acc'].agg(['mean', 'std']).reset_index()
        
        for j, eq_val in enumerate(unique_equiv):
            subset = grouped[grouped['non_equivariance'] == eq_val].sort_values(by='p_err')
            if subset.empty: continue
            
            color = cmap(j)
            marker = markers[j % len(markers)]
            label = legend_labels.get(eq_val, f"NE={eq_val}")
            
            ax.plot(subset['p_err'], subset['mean'], 
                    marker=marker, markersize=5, 
                    label=label, color=color, alpha=0.9)
            
            subset['std'] = subset['std'].fillna(0)
            
            # --- MODIFICA QUI: alpha ridotto a 0.1 per shades meno visibili ---
            ax.fill_between(subset['p_err'], 
                            subset['mean'] - subset['std'], 
                            subset['mean'] + subset['std'], 
                            color=color, alpha=0.1, edgecolor='none')
            # ------------------------------------------------------------------

        ax.set_title(f"$N = {n_val}$", fontsize=12)
        ax.set_ylim(y_min - margin, y_max + margin)
        
        if i >= len(unique_Ns) - cols:
            ax.set_xlabel(r"$p_{\mathrm{err}}$")
        if i % cols == 0:
            ax.set_ylabel("Max Val. Accuracy")

    for k in range(i + 1, len(axes)):
        fig.delaxes(axes[k])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), 
               ncol=len(unique_equiv), frameon=False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, bbox_inches='tight')
    print(f"Grafico salvato: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
