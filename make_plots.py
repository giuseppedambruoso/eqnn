import pandas as pd
import matplotlib.pyplot as plt

def plot_qnn_results(csv_path):
    # Caricamento dati
    df = pd.read_csv(csv_path)
    
    # Ordiniamo per p_err per evitare linee che saltano avanti e indietro nel plot
    df = df.sort_values(by='p_err')

    # Identifichiamo le configurazioni uniche per le righe (seed, N)
    row_configs = df[['seed', 'N']].drop_duplicates().values
    # Identifichiamo i valori unici di non_equivariance per le colonne
    non_equiv_values = sorted(df['non_equivariance'].unique())

    num_rows = len(row_configs)
    num_cols = len(non_equiv_values)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), 
                             sharex=True, sharey=True, constrained_layout=True)

    # Se c'è solo una riga o una colonna, axes non è una matrice 2D, lo forziamo
    if num_rows == 1: axes = [axes]
    if num_cols == 1: axes = [[ax] for ax in axes]

    for r_idx, (seed, n_val) in enumerate(row_configs):
        for c_idx, neq in enumerate(non_equiv_values):
            ax = axes[r_idx][c_idx]
            
            # Filtriamo i dati per questa specifica sottotabella
            subset = df[(df['seed'] == seed) & (df['N'] == n_val) & (df['non_equivariance'] == neq)]
            
            if not subset.empty:
                ax.plot(subset['p_err'], subset['acc'], 'o-', label='Accuracy', markersize=4)
                ax.plot(subset['p_err'], subset['aug_acc'], 's--', label='Aug Accuracy', markersize=4)
                
                ax.set_title(f"Seed:{seed} | N:{n_val} | Non-Eq:{neq}")
                ax.grid(True, linestyle='--', alpha=0.7)
                
                if r_idx == num_rows - 1:
                    ax.set_xlabel('p_err')
                if c_idx == 0:
                    ax.set_ylabel('Accuracy Score')
                
                if r_idx == 0 and c_idx == 0:
                    ax.legend()

    plt.suptitle("Analisi Accuratezza QNN: Standard vs Augmented", fontsize=16)
    plt.savefig("qnn_analysis_grid.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Assicurati che il file si chiami esattamente così o cambia il percorso qui
    plot_qnn_results('merged_results.csv')
