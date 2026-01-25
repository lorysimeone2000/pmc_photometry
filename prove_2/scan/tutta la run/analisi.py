import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# --- 1. CONFIGURAZIONE ---
# Puoi inserire qui la lista delle run che vuoi graficare
RUNS_TO_PLOT = [1, 2, 3]

# Definiamo le Size usate nell'analisi
SIZES_TO_TEST = [3, 5]

# Colori per il grafico
colors_perse = {3: 'red', 5: 'blue'}
colors_fp = {3: 'darkred', 5: 'darkblue'}

for run in RUNS_TO_PLOT:
    print(f"\n--- Elaborazione Grafico Run {run} ---")

    # Nome del file generato dall'analisi precedente
    filename = f"risultati_scan_run_{run}_perse_mag_lt_10.csv"

    # Verifica esistenza file
    if not os.path.exists(filename):
        print(f"ERRORE: Il file {filename} non è stato trovato. Salto questa run.")
        continue

    # Lettura del CSV
    dataframe = pd.read_csv(filename)

    # --- 2. CONFIGURAZIONE GRAFICO ---
    plt.figure(figsize=(12, 8))

    # Variabile asse X (comune a tutti)
    x_axis = dataframe['FWHM']

    # --- 3. CICLO DI PLOTTING ---
    for size_val in SIZES_TO_TEST:
        # Costruiamo i nomi delle colonne basandoci sul NUOVO formato
        col_perse_mean = f'Perse_MagLT10_Size{size_val}_Mean'
        col_fp_mean = f'FP_Size{size_val}_Mean'

        # Verifica se le colonne esistono nel file
        if col_perse_mean not in dataframe.columns:
            print(f"Attenzione: Colonna {col_perse_mean} non trovata.")
            continue

        # Plot Falsi Positivi (Linea Tratteggiata)
        if col_fp_mean in dataframe.columns:

            # Plot Stelle Perse (Linea Continua)
            plt.plot(x_axis, dataframe[col_perse_mean],
                     color=colors_perse[size_val], linestyle='-', linewidth=2,
                     label=f'Media Stelle Perse (Mag < 10) - Size {size_val}')

            plt.plot(x_axis, dataframe[col_fp_mean],
                     color=colors_fp[size_val], linestyle='--', linewidth=2,
                     label=f'Media Falsi Positivi (FP) - Size {size_val}')

    # --- 4. FORMATTAZIONE ---
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.xlabel('FWHM')
    plt.ylabel('Numero MEDIO per immagine (Log)')
    plt.yscale('log')

    # Titolo dinamico
    plt.title(f'Scan Parametri (Falsi Positivi vs Stelle Perse < Mag 10): Run {run}')

    # Formattazione assi logaritmici
    ax = plt.gca()
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')

    plt.legend()
    plt.tight_layout()

    # --- 5. SALVATAGGIO ---
    nome_file_output = f'scan_parametri_run_{run}_replot_perse.png'
    plt.savefig(nome_file_output, dpi=300)
    print(f"Grafico salvato come: {nome_file_output}")

    plt.show() # Decommenta se vuoi vedere i grafici a schermo
    plt.close()  # Chiude la figura per liberare memoria