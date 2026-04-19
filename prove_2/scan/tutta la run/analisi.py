import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# --- 1. CONFIGURO I PARAMETRI ---
RUNS_TO_PROCESS = [1, 2, 3]
SIZES_TO_TEST = [3, 5]

# imposto i colori per il grafico
colors_perse = {3: 'red', 5: 'blue'}
colors_fp = {3: 'darkred', 5: 'darkblue'}

print("--- Starting Data Unification and Mean Plotting ---")

# --- 2. CARICO E AGGREGO I DATI ---
lista_dataframes = []

for run in RUNS_TO_PROCESS:
    # costruisco il nome del file
    filename = f"risultati_scan_run_{run}_perse_mag_lt_10.csv"

    # verifico se esiste
    if os.path.exists(filename):
        print(f"Loading data from: {filename}")
        df_temp = pd.read_csv(filename)
        lista_dataframes.append(df_temp)
    else:
        print(f"WARNING: File {filename} not found. Skipping it.")

if not lista_dataframes:
    print("ERROR: No data files found. Cannot generate the plot.")
    exit()

# concateno momentaneamente per trovare il minimo e massimo globale di FWHM
df_temp_totale = pd.concat(lista_dataframes)
fwhm_min = df_temp_totale['FWHM'].min()
fwhm_max = df_temp_totale['FWHM'].max()

# creo una griglia comune di 500 punti per l'interpolazione
fwhm_grid = np.linspace(fwhm_min, fwhm_max, 500)

lista_dataframes_interpolati = []

# interpolo ogni dataframe sulla nuova griglia comune per allinearli tutti
for df in lista_dataframes:
    # ordino il dataframe originale per sicurezza
    df = df.sort_values('FWHM')
    df_interp = pd.DataFrame({'FWHM': fwhm_grid})

    for col in df.columns:
        if col != 'FWHM':
            # uso np.interp per allineare i valori, metto NaN fuori dal range originale del mio singolo dataframe
            df_interp[col] = np.interp(fwhm_grid, df['FWHM'], df[col], left=np.nan, right=np.nan)

    lista_dataframes_interpolati.append(df_interp)

# concateno tutti i dataframe allineati in uno solo
df_totale = pd.concat(lista_dataframes_interpolati)

# calcolo la media raggruppando per la griglia FWHM (ignoro automaticamente i NaN)
df_medio = df_totale.groupby('FWHM').mean().reset_index()

# --- 4. TRACCIO IL GRAFICO ---
# imposto le dimensioni del testo per subfigure in LaTeX (senza titolo, adattate per A4)
plt.rc('font', size=11)          # dimensione base adatta a documenti LaTeX
plt.rc('axes', labelsize=11)     # etichette assi X e Y proporzionate al testo
plt.rc('xtick', labelsize=10)    # numeri asse X
plt.rc('ytick', labelsize=10)    # numeri asse Y
plt.rc('legend', fontsize=10)    # testo della legenda

# imposto dimensioni compatte equivalenti a circa 0.85\textwidth per un foglio A4 (circa 13.7 cm)
plt.figure(figsize=(5.4, 4))
x_axis = df_medio['FWHM']

for size_val in SIZES_TO_TEST:
    col_perse_mean = f'Perse_MagLT10_Size{size_val}_Mean'
    col_fp_mean = f'FP_Size{size_val}_Mean'

    if col_perse_mean in df_medio.columns and col_fp_mean in df_medio.columns:

        # traccio la linea continua per la media delle stelle perse
        plt.plot(x_axis, df_medio[col_perse_mean],
                 color=colors_perse[size_val], linestyle='-', linewidth=1.5,
                 label=f'Lost stars (mag < 10) (size {size_val})')

        # traccio la linea tratteggiata per la media dei falsi positivi
        plt.plot(x_axis, df_medio[col_fp_mean],
                 color=colors_fp[size_val], linestyle='--', linewidth=1.5,
                 label=f'Mean no matched clusters (size {size_val})')
    else:
        print(f"Warning: Columns for Size {size_val} not found in the merged dataset.")

# --- 5. FORMATTO IL GRAFICO ---
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.xlabel('FWHM')
plt.ylabel('Mean number per image')
plt.yscale('symlog', linthresh=5.0)

ax = plt.gca()

# definisco e applico le tacchette (ticks) personalizzate per l'asse y
y_ticks = list(range(0, 6)) + list(range(10, 25, 5)) + list(range(50, 85, 25))
ax.set_yticks(y_ticks)

# formatto gli assi mantenendo i numeri in formato normale senza notazione scientifica
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y')

# rimuovo i minor ticks per avere un asse pulito e ordinato con solo i miei valori
ax.yaxis.set_minor_locator(ticker.NullLocator())

plt.legend()
plt.tight_layout()

plt.savefig('scan_parametri_UNIFICATO.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'scan_parametri_UNIFICATO.png'")

#plt.show()