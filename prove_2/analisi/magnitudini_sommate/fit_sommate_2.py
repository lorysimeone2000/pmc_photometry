import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
import warnings

# Ignora warning numerici (es. log(0))
warnings.filterwarnings('ignore', category=RuntimeWarning)


def modello_lineare_generico(mag, m, q):
    """
    Modello: log10(Flux) = m * Mag + q
    """
    return m * mag + q


# --- CONFIGURAZIONE ---
percorso_file_csv = "/home/lorysimeone/tesi_magistrale/prove_2/analisi/magnitudini/dispersione_flussi/risultati_analisi_run_1.csv"

print(f"Elaborazione file: {os.path.basename(percorso_file_csv)}")

if not os.path.exists(percorso_file_csv):
    print("ERRORE: File non trovato. Verifica il percorso.")
    exit()

df = pd.read_csv(percorso_file_csv, comment='#')

# --- NOMI COLONNE ---
col_flux = 'media_flusso_fisso_max_run'
col_std = 'std_flusso_fisso_max_run'
col_mag = 'Mag_Integrata'

# --- 1. SEPARAZIONE DATI (LOGICA ROBUSTA) ---

# Identifichiamo i "Non Catalogati":
# Sono quelli che hanno la Magnitudine nulla (NaN) OPPURE l'ID che inizia con NOMATCH
mask_no_match = (df[col_mag].isna()) | (df['ID'].astype(str).str.startswith('NOMATCH'))

# DataFrame Oggetti Senza Corrispondenza
df_no_match = df[mask_no_match].copy()

# DataFrame Oggetti Con Corrispondenza (per il FIT)
df_match = df[~mask_no_match].copy()

# --- DEBUG: STAMPA COSA HAI TROVATO ---
print(f"\n--- STATISTICHE DATI ---")
print(f"Totale righe nel CSV: {len(df)}")
print(f"Oggetti CON corrispondenza (potenziali per fit): {len(df_match)}")
print(f"Oggetti SENZA corrispondenza (target visualizzazione): {len(df_no_match)}")

if len(df_no_match) > 0:
    print(f"   -> Esempio ID senza match: {df_no_match['ID'].iloc[0]}")
    print(f"   -> Esempio Flusso senza match: {df_no_match[col_flux].iloc[0]}")
else:
    print("   ATTENZIONE: Non sono stati trovati oggetti senza corrispondenza nel CSV!")
    print("   Controlla che nel file CSV le colonne 'Mag_Brightest' siano vuote per questi oggetti.")

# --- 2. PREPARAZIONE DATI PER IL FIT (SOLO MATCHATI) ---

# Filtro validità rigoroso SOLO per il fit
mask_valid_fit = (
        (df_match[col_flux] > 0) &
        (df_match[col_mag].notna()) &
        (df_match[col_std] > 0)  # Serve std > 0 per i pesi
)
data_fit = df_match[mask_valid_fit].copy()

# Dati per il fit
X = data_fit[col_mag].values
Y_linear = data_fit[col_flux].values
Y_log = np.log10(Y_linear)

# Pesi (Propagazione errore)
sigma_linear = data_fit[col_std].values
sigma_log = (1/np.log(10)) * (sigma_linear / Y_linear)

# --- 3. ESECUZIONE FIT ---

if len(X) > 2:
    # A. Fit Preliminare
    popt_prelim, _ = curve_fit(modello_lineare_generico, X, Y_log)
    m_prelim, q_prelim = popt_prelim

    # B. Sigma Clipping
    residui = Y_log - modello_lineare_generico(X, m_prelim, q_prelim)
    std_residui = np.std(residui)
    soglia_sigma = 2.5
    mask_inliers = np.abs(residui) < (soglia_sigma * std_residui)

    X_clean = X[mask_inliers]
    Y_clean_log = Y_log[mask_inliers]
    sigma_clean = sigma_log[mask_inliers]

    X_out = X[~mask_inliers]
    Y_out_log = Y_log[~mask_inliers]

    # C. Fit Finale Pesato
    popt, pcov = curve_fit(modello_lineare_generico, X_clean, Y_clean_log, sigma=sigma_clean, absolute_sigma=True)
    m_fit, q_fit = popt

    # Chi Quadro
    y_model_log = modello_lineare_generico(X_clean, m_fit, q_fit)
    residui_finali = Y_clean_log - y_model_log
    chi_squared = np.sum((residui_finali / sigma_clean) ** 2)
    chi_reduced = chi_squared / (len(X_clean) - 2)

    print(f"\n--- Risultati Fit ---")
    print(f"m: {m_fit:.4f}, q: {q_fit:.4f}, Chi2_red: {chi_reduced:.4f}")
else:
    print("Non abbastanza punti per il fit.")
    m_fit, q_fit, chi_reduced = 0, 0, 0
    X_clean, Y_clean_log = [], []
    X_out, Y_out_log = [], []

# --- 4. VISUALIZZAZIONE ---

plt.figure(figsize=(12, 9))

# A. Dati del Fit (Blu)
if len(X_clean) > 0:
    plt.errorbar(
        X_clean, 10 ** Y_clean_log, yerr=sigma_linear[mask_inliers],
        fmt='o', markersize=4, color='blue', ecolor='lightblue', alpha=0.7,
        label=f'Catalogati Validi ({len(X_clean)})'
    )

# B. Outliers (Rosso)
if len(X_out) > 0:
    plt.scatter(X_out, 10 ** Y_out_log, s=50, c='red', marker='x', label='Outliers', zorder=5)

# C. OGGETTI SENZA CORRISPONDENZA (Arancione)
# Assegniamo una magnitudine fittizia "fuori scala" per vederli
if len(df_no_match) > 0:
    # Troviamo un valore X a sinistra del grafico (es. min(X) - 1 o fisso a 4.0)
    mag_fittizia = 4.0
    if len(X) > 0:
        mag_fittizia = np.min(X) - 1.5  # Li mette un po' a sinistra dei dati reali

    # Prendiamo solo quelli con flusso valido (>0)
    # NON filtriamo per deviazione standard, perché potrebbero averne NaN se visti 1 sola volta
    df_no_match_plot = df_no_match[df_no_match[col_flux] > 0]

    if len(df_no_match_plot) > 0:
        plt.scatter(
            np.full(len(df_no_match_plot), mag_fittizia),
            df_no_match_plot[col_flux],
            s=40, c='orange', marker='D', edgecolors='black', alpha=0.8,
            label=f'NON Catalogati ({len(df_no_match_plot)})', zorder=10
        )

        # Aggiungiamo un testo per indicare che la posizione X è fittizia
        plt.annotate("Mag Fittizia", xy=(mag_fittizia, np.mean(df_no_match_plot[col_flux])),
                     xytext=(mag_fittizia, np.max(df_no_match_plot[col_flux]) * 1.5),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     ha='center')

# D. Retta di Fit
if len(X) > 2:
    # Estendiamo il range X per coprire anche la magnitudine fittizia
    x_min_plot = min(np.min(X), mag_fittizia - 0.5)
    x_max_plot = max(X)
    x_plot = np.linspace(x_min_plot, x_max_plot, 100)
    y_plot_linear = 10 ** modello_lineare_generico(x_plot, m_fit, q_fit)

    label_fit = f'Fit: log(F)={m_fit:.2f}M + {q_fit:.2f} ($\chi^2_R$={chi_reduced:.2f})'
    plt.plot(x_plot, y_plot_linear, 'k--', linewidth=2, label=label_fit)

# --- FORMATTAZIONE ---
plt.title(f'Calibrazione Fotometrica con Oggetti Non Catalogati', fontsize=14)
plt.xlabel('Magnitudine Catalogo (Brightest)', fontsize=12)
plt.ylabel('Media Flusso Fisso Max Run [ADU]', fontsize=12)
plt.yscale('log')
plt.gca().invert_xaxis()  # Importante per le magnitudini
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=11, loc='best')
plt.tight_layout()

# Salva
output_img = "fit_con_non_catalogati.png"
plt.savefig(output_img, dpi=300)
print(f"Grafico salvato in: {output_img}")
plt.show()