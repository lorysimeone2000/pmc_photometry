import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# --- 1. CONFIGURAZIONE ---
RUNS_TO_PROCESS = [1, 2, 3]
SIZES_TO_TEST = [3, 5]

# Colori per il grafico
colors_perse = {3: 'red', 5: 'blue'}
colors_fp = {3: 'darkred', 5: 'darkblue'}

print("--- Inizio Unificazione Dati e Plotting Medio ---")

# --- 2. CARICAMENTO E AGGREGAZIONE DATI ---
lista_dataframes = []

for run in RUNS_TO_PROCESS:
    # Costruisco il nome del file
    filename = f"risultati_scan_run_{run}_perse_mag_lt_10.csv"

    # Verifico se esiste
    if os.path.exists(filename):
        print(f"Caricamento dati da: {filename}")
        df_temp = pd.read_csv(filename)
        lista_dataframes.append(df_temp)
    else:
        print(f"ATTENZIONE: File {filename} non trovato. Lo salto.")

if not lista_dataframes:
    print("ERRORE: Nessun file dati trovato. Impossibile generare il grafico.")
    exit()

# Concateno tutti i dataframe in uno solo
df_totale = pd.concat(lista_dataframes)

# Calcolo la MEDIA raggruppando per FWHM
# Questo è il passaggio chiave: per ogni valore di FWHM, fa la media tra Run 1, 2 e 3
print("Calcolo delle medie su tutte le run...")
df_medio = df_totale.groupby('FWHM').mean().reset_index()

# --- 3. CONFIGURAZIONE GRAFICO ---
plt.figure(figsize=(12, 8))

# Variabile asse X
x_axis = df_medio['FWHM']

# --- 4. CICLO DI PLOTTING SUI DATI MEDI ---
for size_val in SIZES_TO_TEST:
    # Ricostruisco i nomi delle colonne
    col_perse_mean = f'Perse_MagLT10_Size{size_val}_Mean'
    col_fp_mean = f'FP_Size{size_val}_Mean'

    # Controllo se le colonne esistono (dovrebbero esserci se i CSV sono corretti)
    if col_perse_mean in df_medio.columns and col_fp_mean in df_medio.columns:

        # Plot Stelle Perse (Linea Continua) - MEDIA
        plt.plot(x_axis, df_medio[col_perse_mean],
                 color=colors_perse[size_val], linestyle='-', linewidth=2,
                 label=f'Media Stelle Perse (Mag < 10) - Size {size_val}')

        # Plot Falsi Positivi (Linea Tratteggiata) - MEDIA
        plt.plot(x_axis, df_medio[col_fp_mean],
                 color=colors_fp[size_val], linestyle='--', linewidth=2,
                 label=f'Media Falsi Positivi (FP) - Size {size_val}')
    else:
        print(f"Attenzione: Colonne per Size {size_val} non trovate nel dataset unito.")

# --- 5. FORMATTAZIONE ---
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.xlabel('FWHM')
plt.ylabel('Numero MEDIO per immagine (Log) - Media delle Run')
plt.yscale('log')

# Titolo che riflette l'unificazione
plt.title(f'Scan Parametri UNIFICATO (Media su Run {RUNS_TO_PROCESS})\nFalsi Positivi vs Stelle Perse < Mag 10')

# Formattazione assi logaritmici
ax = plt.gca()
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y')

plt.legend()
plt.tight_layout()

# --- 6. SALVATAGGIO ---
nome_file_output = 'scan_parametri_UNIFICATO_perse_replot.png'
plt.savefig(nome_file_output, dpi=300)
print(f"\nGrafico unificato salvato come: {nome_file_output}")

plt.show()
plt.close()