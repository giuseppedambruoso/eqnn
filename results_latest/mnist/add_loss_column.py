import os
import pandas as pd

# Imposta il nome del file dei risultati
results_file = 'results.txt'

print(f"Lettura del file {results_file}...")
df_results = pd.read_csv(results_file)

# Lista per memorizzare i valori calcolati
last_losses = []

print("Estrazione della loss dalle cartelle in corso...")

# Itera su ogni riga di results.txt
for index, row in df_results.iterrows():
    # Controllo di sicurezza: se 'N' è NaN o la riga è vuota, saltiamo la lettura
    if pd.isna(row['N']):
        last_losses.append(None)
        continue
        
    try:
        # Estrai i parametri dalla riga per ricostruire il nome della cartella
        dataset = str(row['dataset']).strip()
        N = int(row['N'])
        seed = int(row['seed'])
        p_err = int(row['p_err'])
        non_eq = int(row['non_equivariance'])
        reps = int(row['reps'])
        
        # Dai tuoi nomi cartella vediamo che epochs=60 è fisso
        epochs = 60
        
        # Ricrea il nome esatto della cartella
        folder_name = f"DATA.N={N},DATA.dataset={dataset},GENERAL.seed={seed},QNN.non_equivariance={non_eq},QNN.p_err={p_err},QNN.reps={reps},TRAINING.epochs={epochs}"
        
        csv_path = os.path.join(folder_name, 'loss_history.csv')
        loss_val = None 
        
        if os.path.exists(csv_path):
            df_loss = pd.read_csv(csv_path)
            df_loss.columns = df_loss.columns.str.strip()
            
            if 'train_loss' in df_loss.columns:
                # Usa la media delle ultime 10 epoche
                loss_val = df_loss.tail(10)['train_loss'].mean()
                
        last_losses.append(loss_val)
        
    except Exception as e:
        print(f"Errore nell'elaborazione della riga {index}: {e}")
        last_losses.append(None)

# Aggiunge la nuova colonna al dataframe
df_results['last_loss'] = last_losses

# Salva il nuovo DataFrame
output_file = 'results_with_last_loss.txt'
df_results.to_csv(output_file, index=False)

print(f"\nFatto! Il file aggiornato è stato salvato come: '{output_file}'")
