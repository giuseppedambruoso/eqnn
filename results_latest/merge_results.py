import pandas as pd
import os

# Definisci i percorsi assoluti (o relativi) dei tuoi due file
file_nwpu = '/data/giuseppedambruoso/eqnn/results_latest/nwpu/results_with_last_loss.txt'
file_mnist = '/data/giuseppedambruoso/eqnn/results_latest/mnist/results_with_last_loss.txt'
file_augmnist = '/data/giuseppedambruoso/eqnn/results_latest/augmnist/results_with_last_loss.txt'

# Nome del file finale in cui salvare il risultato unito
output_file = '/data/giuseppedambruoso/eqnn/results_latest/merged_results_with_last_loss.txt'

print("Inizio la lettura dei file...")

try:
    # Leggi i due file CSV
    df_nwpu = pd.read_csv(file_nwpu)
    print(f"Letti {len(df_nwpu)} record da NWPU.")
    
    df_mnist = pd.read_csv(file_mnist)
    print(f"Letti {len(df_mnist)} record da MNIST.")

    df_augmnist = pd.read_csv(file_augmnist)
    print(f"Letti {len(df_augmnist)} record da AUGMNIST.")
    
    # Unisci i due dataframe (in automatico mantiene una sola intestazione)
    df_combined = pd.concat([df_nwpu, df_mnist, df_augmnist], ignore_index=True)
    
    # Pulizia: rimuovi le righe in cui il valore 'N' (o 'dataset') è nullo/NaN
    # Questo elimina le righe vuote o malformate
    df_combined = df_combined.dropna(subset=['N', 'dataset'])
    
    # Opzionale: converte 'N' di nuovo a int (a volte dropna lo converte in float per sicurezza)
    df_combined['N'] = df_combined['N'].astype(int)
    
    # Salva il risultato nel nuovo file (index=False evita che scriva i numeri di riga all'inizio)
    df_combined.to_csv(output_file, index=False)
    
    print(f"\nOperazione completata con successo!")
    print(f"Totale righe valide salvate: {len(df_combined)}")
    print(f"Il file unito si trova qui: {output_file}")

except FileNotFoundError as e:
    print(f"\nERRORE: Non riesco a trovare uno dei file. Controlla che i percorsi siano corretti.")
    print(e)
except Exception as e:
    print(f"\nSi è verificato un errore inaspettato: {e}")
