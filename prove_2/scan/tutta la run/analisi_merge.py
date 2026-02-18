import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# --- 1. CONFIGURAZIONE ---
RUNS_TO_PROCESS = [1, 2, 3]

# Imposto i colori per il grafico
colore_perse = 'red'
colore_fp = 'darkblue'

print("--- Inizio Unificazione Dati e Plotting Medio ---")

# --- 2. CARICAMENTO E AGGREGAZIONE DATI ---
lista_dataframes = []

for run in RUNS_TO_PROCESS:
    # Costruisco il nome del file corretto per i risultati del merge
    filename = f"risultati_scan_MERGE_run_{run}_perse_mag_lt_10.csv"

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

# Calcolo la MEDIA raggruppando per il nuovo asse x: MergeRadius_Arcsec
df_medio = df_totale.groupby('MergeRadius_Arcsec').mean().reset_index()

# --- 3. CREAZIONE GRAFICO ---
plt.figure(figsize=(12, 8))

# Definisco l'asse X
x_axis = df_medio['MergeRadius_Arcsec']

# --- 4. TRACCIAMENTO LINEE ---

# Plot Stelle Perse (Linea Continua) - MEDIA
plt.plot(x_axis, df_medio['Perse_MagLT10_Mean'],
         color=colore_perse, linestyle='-', linewidth=2,
         label='Media Stelle Perse (Mag < 10)')

# Plot Falsi Positivi (Linea Tratteggiata) - MEDIA
plt.plot(x_axis, df_medio['FP_Mean'],
         color=colore_fp, linestyle='--', linewidth=2,
         label='Media Falsi Positivi Reali (FP)')

# --- 5. FORMATTAZIONE ---
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.xlabel('Raggio di Merging Vizier/Hipparcos (Arcosecondi)')
plt.ylabel('Numero MEDIO per immagine (Log) - Media delle Run')
plt.yscale('log')

# Inserisco il titolo che riflette l'unificazione sul raggio di merging
plt.title(f'Scan Raggio Merging UNIFICATO (Media su Run {RUNS_TO_PROCESS})\nFalsi Positivi Reali vs Stelle Perse < Mag 10')

# Formatto gli assi logaritmici
ax = plt.gca()
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y')

plt.legend()
plt.tight_layout()

# Salvo il grafico generato
output_filename = 'scan_MERGE_UNIFICATO_perse_mag_lt_10.png'
plt.savefig(output_filename, dpi=300)
plt.show()
plt.close()

print(f"Grafico unificato salvato con successo in: {output_filename}")