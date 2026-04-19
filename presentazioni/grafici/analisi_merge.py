import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# --- 1. CONFIGURAZIONE ---
RUNS_TO_PROCESS = [1, 2, 3]

# imposto i colori per il grafico
colore_perse = 'red'
colore_fp = 'darkblue'

print("--- Inizio Unificazione Dati e Plotting Medio ---")

# --- 2. CARICAMENTO E AGGREGAZIONE DATI ---
lista_dataframes = []

for run in RUNS_TO_PROCESS:
    # costruisco il nome del file corretto per i risultati del merge
    filename = f"risultati_scan_MERGE_run_{run}_perse_mag_lt_10.csv"

    # verifico se esiste
    if os.path.exists(filename):
        print(f"Caricamento dati da: {filename}")
        df_temp = pd.read_csv(filename)
        lista_dataframes.append(df_temp)
    else:
        print(f"ATTENZIONE: File {filename} non trovato. Lo salto.")

if not lista_dataframes:
    print("ERRORE: Nessun file dati trovato. Impossibile generare il grafico.")
    exit()

# concateno tutti i dataframe in uno solo
df_totale = pd.concat(lista_dataframes)

# calcolo la media raggruppando per il nuovo asse x: MergeRadius_Arcsec
df_medio = df_totale.groupby('MergeRadius_Arcsec').mean().reset_index()

# --- 3. CREAZIONE GRAFICO ---
# creo la figura dimensionata in modo ottimale per 0.85\textwidth in un documento A4
plt.figure(figsize=(8.5, 5))

# definisco l'asse X
x_axis = df_medio['MergeRadius_Arcsec']

# --- 4. TRACCIAMENTO LINEE ---

# traccio le stelle perse (Linea Continua) - MEDIA, traducendo la label
plt.plot(x_axis, df_medio['Perse_MagLT10_Mean'],
         color=colore_perse, linestyle='-', linewidth=2,
         label='Mean lost stars (mag < 10)')

# traccio i falsi positivi (Linea Tratteggiata) - MEDIA, traducendo la label
plt.plot(x_axis, df_medio['FP_Mean'],
         color=colore_fp, linestyle='--', linewidth=2,
         label='Mean no matched clusters')

# --- 5. FORMATTAZIONE ---
plt.grid(True, which="both", linestyle='--', alpha=0.6)

# traduco e dimensiono le etichette degli assi
plt.xlabel('VizieR/Hipparcos merging radius (arcseconds)', fontsize=14)
plt.ylabel('Mean number per image', fontsize=14)

plt.yscale('log')

# formatto gli assi logaritmici
ax = plt.gca()
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y')

# dimensiono i tick degli assi
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=10)

# dimensiono la legenda
plt.legend(fontsize=12)

plt.tight_layout()

# salvo il grafico ad alta risoluzione evitando tagli sui bordi
output_filename = 'scan_MERGE.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# plt.show()
plt.close()

print(f"Grafico unificato salvato con successo in: {output_filename}")