import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# --- 1. CONFIGURAZIONE ---
RUNS_TO_PROCESS = [1, 2, 3]

# calcolo la soglia fissa originale in arcosecondi (0.003349 gradi * 3600)
SOGLIA_FISSA_DEG = 0.003349
SOGLIA_FISSA_ARCSEC = SOGLIA_FISSA_DEG * 3600

# imposto i colori per il grafico
colore_perse = 'red'
colore_fp = 'darkblue'

print("--- Inizio Unificazione Dati e Plotting Medio ---")

# --- 2. CARICAMENTO E AGGREGAZIONE DATI ---
lista_dataframes = []

for run in RUNS_TO_PROCESS:
    # costruisco il nome del file corretto per i risultati dello scan di correlazione
    filename = f"risultati_scan_correlazione_run_{run}.csv"

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

# calcolo la MEDIA raggruppando per l'asse x: Raggio_Corr_arcsec
df_medio = df_totale.groupby('Raggio_Corr_arcsec').mean().reset_index()

# --- 3. CREAZIONE GRAFICO ---
plt.figure(figsize=(12, 8))

# definisco l'asse X
# df_medio = df_medio[df_medio['Raggio_Corr_arcsec'] > 12]
x_axis = df_medio['Raggio_Corr_arcsec']

# --- 4. TRACCIAMENTO LINEE ---

# plot Stelle Perse (Linea Continua) - MEDIA
plt.plot(x_axis, df_medio['Perse_MagLT10_Mean'],
         color=colore_perse, linestyle='-', linewidth=2,
         label='Media Stelle Perse (Mag < 10)')

# plot Falsi Positivi (Linea Tratteggiata) - MEDIA
plt.plot(x_axis, df_medio['FP_Mean'],
         color=colore_fp, linestyle='--', linewidth=2,
         label='Media Falsi Positivi Reali (FP)')

# traccio la linea verticale verde tratteggiata per la soglia fissa originale
plt.axvline(x=SOGLIA_FISSA_ARCSEC, color='green', linestyle='--', linewidth=2,
            label=f'Soglia Fissa Originale ({SOGLIA_FISSA_ARCSEC:.2f}")')

# --- 5. FORMATTAZIONE ---
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.xlabel('Raggio di Correlazione Fotocamera/Catalogo (Arcosecondi)')
plt.ylabel('Numero MEDIO per immagine - Media delle Run')

ax = plt.gca()

# imposto l'asse Y in scala logaritmica
#ax.set_yscale('symlog', linthresh=1.0)

# forzo la visualizzazione di più numeri sull'asse Y abilitando le etichette anche per i minor ticks
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 10), numticks=100))
# forzo il limite inferiore esattamente a 0
ax.set_ylim(bottom=0)

# formatto le etichette dell'asse Y logaritmico per mostrarle come numeri normali
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y')

# inserisco il titolo coerente con l'analisi effettuata
plt.title(f'Scan Raggio Correlazione UNIFICATO (Media su Run {RUNS_TO_PROCESS})\nFalsi Positivi Reali vs Stelle Perse < Mag 10')

plt.legend()
plt.tight_layout()

# salvo l'immagine finale
nome_output = 'scan_correlazione_UNIFICATO.png'
plt.savefig(nome_output, dpi=300)
print(f"Grafico unificato salvato con successo in: {nome_output}")

plt.show()