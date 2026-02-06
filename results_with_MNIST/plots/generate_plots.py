import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# --- CONFIGURAZIONE PERCORSI ---
RESULTS_DIR = os.path.join('..', 'results')
OUTPUT_CSV = 'all_results.csv'
OUTPUT_PLOT = 'summary_plot.pdf'

# --- STILE PLOT (Physical Review Letters Standard) ---
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
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':'
})

def parse_folder_name_regex(folder_name):
    """
    Estrae i parametri usando Regex per gestire le inconsistenze nei nomi
    (es: gestisce sia 'DATA.N=160' che 'N.DATA=160').
    """
    params = {}
    
    # 1. Trova N: Cerca 'N=' preceduto opzionalmente da 'DATA.' o seguito da '.DATA'
    match_n = re.search(r'(?:DATA\.N|N\.DATA|N)\s*=\s*(\d+)', folder_name, re.IGNORECASE)
    if match_n:
        params['N'] = int(match_n.group(1))
    
    # 2. Trova Seed
    match_seed = re.search(r'(?:GENERAL\.)?seed\s*=\s*(\d+)', folder_name, re.IGNORECASE)
    if match_seed:
        params['seed'] = int(match_seed.group(1))
        
    # 3. Trova Non-Equivariance
    match_ne = re.search(r'non[._]equivariance\s*=\s*(\d+)', folder_name, re.IGNORECASE)
    if match_ne:
        params['non_equivariance'] = int(match_ne.group(1))
        
    # 4. Trova p_err
    match_p = re.search(r'p[._]err\s*=\s*(\d+\.?\d*)', folder_name, re.IGNORECASE)
    if match_p:
        params['p_err'] = float(match_p.group(1))

    return params

def extract_metrics(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty: return None
        max_idx = df['val_acc'].idxmax()
        row = df.loc[max_idx]
        return {'max_val_acc': row['val_acc'], 'max_epoch': int(row['epoch'])}
    except Exception:
        return None

def main():
    data = []
    print(f"Scansione directory: {os.path.abspath(RESULTS_DIR)}")
    
    if not os.path.exists(RESULTS_DIR):
        print(f"ERRORE: La cartella {RESULTS_DIR} non esiste.")
        return

    subfolders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]
    
    print(f"Analisi di {len(subfolders)} cartelle...")

    for folder in subfolders:
        folder_path = os.path.join(RESULTS_DIR, folder)
        metrics_path = os.path.join(folder_path, 'metrics.csv')
        
        if not os.path.exists(metrics_path):
            continue
            
        # Parsing con Regex
        params = parse_folder_name_regex(folder)
        
        # Verifica parametri
        if 'N' not in params or 'p_err' not in params or 'non_equivariance' not in params:
            continue

        metrics = extract_metrics(metrics_path)
        if metrics:
            data.append({**params, **metrics})

    # Creazione DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        print("Nessun dato valido trovato. Controlla i nomi delle cartelle.")
        return

    # --- MODIFICA: ORDINAMENTO COLONNE E RIGHE ---
    # 1. Definisci l'ordine esatto delle colonne
    desired_columns = ['seed', 'N', 'non_equivariance', 'p_err', 'max_val_acc', 'max_epoch']
    
    # Riordina le colonne del DataFrame (selezionando solo quelle desiderate)
    # (Gestiamo il caso in cui qualche colonna manchi per evitare crash, anche se non dovrebbe accadere)
    existing_cols = [c for c in desired_columns if c in df.columns]
    df = df[existing_cols]

    # 2. Ordina le righe secondo la gerarchia richiesta
    df = df.sort_values(by=['seed', 'N', 'non_equivariance', 'p_err'])

    # Salva CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Creato '{OUTPUT_CSV}' con le colonne ordinate: {existing_cols}")
    
    # Ordiniamo N per il plot (indipendente dal CSV)
    unique_Ns = sorted(df['N'].unique())
    print(f"Valori di N trovati: {unique_Ns}") 

    # --- PLOTTING ---
    num_plots = len(unique_Ns)
    cols = 3 
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5 * rows), squeeze=False)
    axes = axes.flatten()

    unique_equiv = sorted(df['non_equivariance'].unique())
    cmap = plt.get_cmap('viridis', len(unique_equiv))

    # Calcolo range Y uniforme
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
            label = f"$NE = {eq_val}$"
            
            ax.plot(subset['p_err'], subset['mean'], marker='o', markersize=3, 
                    label=label, color=color, alpha=0.9)
            
            ax.fill_between(subset['p_err'], 
                            subset['mean'] - subset['std'], 
                            subset['mean'] + subset['std'], 
                            color=color, alpha=0.2, edgecolor='none')

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
               ncol=len(unique_equiv), frameon=False, title="Non-Equivariance (NE)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, bbox_inches='tight')
    print(f"Grafico salvato: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
