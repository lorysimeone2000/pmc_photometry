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
percorso_file_csv = "/home/lorysimeone/tesi_magistrale/prove_2/analisi/studio_flussi/dispersione_flussi/risultati_analisi_run_1.csv"

print(f"Elaborazione file: {os.path.basename(percorso_file_csv)}")

if not os.path.exists(percorso_file_csv):
    print("ERRORE: File non trovato. Verifica il percorso.")
    exit()

df = pd.read_csv(percorso_file_csv, comment='#')

# --- NOMI COLONNE ---
col_flux = 'media_flusso_fisso_max_run'
col_std = 'std_flusso_fisso_max_run'
col_mag = 'Mag_Brightest'
col_count = 'count_flusso_fisso_max_run'

# --- 1. SEPARAZIONE DATI (LOGICA ROBUSTA) ---

# A. Identifichiamo i "Non Catalogati" sul DF originale
mask_no_match = (df[col_mag].isna()) | (df['ID'].astype(str).str.startswith('NOMATCH'))
df_no_match = df[mask_no_match].copy()

# B. Identifichiamo i "Saturi" (per il plot delle X rosse) sul DF originale
# Nota: La maschera sature la calcoliamo sull'originale per estrarre il DF sature
mask_sature_original = df['saturazione'].astype(str).str.startswith('SI')
# Prendiamo le sature che HANNO corrispondenza (per avere la Mag sull'asse X)
df_sature = df[mask_sature_original & ~mask_no_match].copy()

# C. Creazione DF Match (Base per il fit)
# Partiamo da quelli che HANNO corrispondenza
df_match = df[~mask_no_match].copy()

# D. Filtri sequenziali su df_match (CORREZIONE USER WARNING)
# Filtriamo i saturi USANDO LA COLONNA DI df_match, non la maschera globale
mask_sature_match = df_match['saturazione'].astype(str).str.startswith('SI')
df_match = df_match[~mask_sature_match].copy()

# Filtriamo i deboli USANDO LA COLONNA DI df_match
mask_deboli_match = df_match[col_mag] >= 10
df_match = df_match[~mask_deboli_match].copy()

# --- DEBUG: STAMPA COSA HAI TROVATO ---
print(f"\n--- STATISTICHE DATI ---")
print(f"Totale righe nel CSV: {len(df)}")
print(f"Oggetti CON corrispondenza (potenziali per fit): {len(df_match)}")
print(f"Oggetti SENZA corrispondenza: {len(df_no_match)}")
print(f"Oggetti SATURI (esclusi dal fit): {len(df_sature)}")

# --- 2. PREPARAZIONE DATI PER IL FIT (SOLO MATCHATI E NON SATURI) ---

# Filtro validità rigoroso SOLO per il fit
mask_valid_fit = (
        (df_match[col_flux] > 0) &
        (df_match[col_mag].notna()) &
        (df_match[col_std] > 0) &
        (df_match[col_count] > 0)
)
data_fit = df_match[mask_valid_fit].copy()

# Dati per il fit
X = data_fit[col_mag].values
Y_linear = data_fit[col_flux].values
Y_log = np.log10(Y_linear)

# Sigma_mean = Sigma_std / sqrt(N)
sigma_std = data_fit[col_std].values
counts = data_fit[col_count].values
sigma_mean_linear = sigma_std / np.sqrt(counts)

# --- PROPAGAZIONE ERRORE SUL LOGARITMO ---
# Sigma_log = (1/ln(10)) * (Sigma_mean_linear / Flux)
sigma_log = (1 / np.log(10)) * (sigma_mean_linear / Y_linear)

# --- 3. ESECUZIONE FIT (SENZA SIGMA CLIPPING) ---

if len(X) > 2:
    # Eseguiamo direttamente il fit pesato su TUTTI i dati validi
    popt, pcov = curve_fit(modello_lineare_generico, X, Y_log, sigma=sigma_log, absolute_sigma=True)
    m_fit, q_fit = popt

    # Chi Quadro (Calcolato su tutti i punti X)
    y_model_log = modello_lineare_generico(X, m_fit, q_fit)
    residui_finali = Y_log - y_model_log
    chi_squared = np.sum((residui_finali / sigma_log) ** 2)
    chi_reduced = chi_squared / (len(X) - 2)

    print(f"\n--- Risultati Fit ---")
    print(f"m: {m_fit:.4f}, q: {q_fit:.4f}, Chi2_red: {chi_reduced:.4f}")
else:
    print("Non abbastanza punti per il fit.")
    m_fit, q_fit, chi_reduced = 0, 0, 0

# --- 4. VISUALIZZAZIONE ---

plt.figure(figsize=(12, 9))

# A. Dati del Fit (Blu)
if len(X) > 0:
    plt.errorbar(
        X, 10 ** Y_log, yerr=sigma_mean_linear,  # Plotto errore standard della media
        fmt='o', markersize=4, color='blue', ecolor='lightblue', alpha=0.7,
        label=f'Catalogati Validi ({len(X)})'
    )

# B. Stelle Sature (NUOVA AGGIUNTA)
# Plottiamo le stelle sature con una X rossa
if len(df_sature) > 0:
    # Filtro validità flussi
    mask_sat_valid = (df_sature[col_flux] > 0) & (df_sature[col_mag].notna())
    df_sature_plot = df_sature[mask_sat_valid]

    if len(df_sature_plot) > 0:
        plt.scatter(
            df_sature_plot[col_mag],
            df_sature_plot[col_flux],
            s=80, c='red', marker='x', linewidth=2,
            label=f'Sature (Escluse) ({len(df_sature_plot)})', zorder=20
        )

# C. OGGETTI SENZA CORRISPONDENZA (Arancione)
if len(df_no_match) > 0:
    mag_fittizia = 4.0
    if len(X) > 0:
        mag_fittizia = np.min(X) - 1.5

    df_no_match_plot = df_no_match[df_no_match[col_flux] > 0]

    if len(df_no_match_plot) > 0:
        plt.scatter(
            np.full(len(df_no_match_plot), mag_fittizia),
            df_no_match_plot[col_flux],
            s=40, c='orange', marker='D', edgecolors='black', alpha=0.8,
            label=f'NON Catalogati ({len(df_no_match_plot)})', zorder=10
        )

        plt.annotate("Mag Fittizia", xy=(mag_fittizia, np.mean(df_no_match_plot[col_flux])),
                     xytext=(mag_fittizia, np.max(df_no_match_plot[col_flux]) * 1.5),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     ha='center')

# D. Retta di Fit
if len(X) > 2:
    x_min_plot = min(np.min(X), mag_fittizia - 0.5)
    x_max_plot = max(X)
    x_plot = np.linspace(x_min_plot, x_max_plot, 100)
    y_plot_linear = 10 ** modello_lineare_generico(x_plot, m_fit, q_fit)

    # CORREZIONE SYNTAX WARNING: Usa rf"..." per raw f-string
    label_fit = rf'Fit: log(F)={m_fit:.2f}M + {q_fit:.2f} ($\chi^2_R$={chi_reduced:.2f})'
    plt.plot(x_plot, y_plot_linear, 'k--', linewidth=2, label=label_fit)

# --- FORMATTAZIONE ---
plt.title(f'Calibrazione Fotometrica (Fit Errore Std Media)', fontsize=14)
plt.xlabel('Magnitudine Catalogo (Brightest)', fontsize=12)
plt.ylabel('Media Flusso Fisso Max Run [ADU]', fontsize=12)
plt.yscale('log')
plt.gca().invert_xaxis()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=11, loc='best')
plt.tight_layout()

# Salva
output_img = "fit_con_sature_X.png"
plt.savefig(output_img, dpi=300)
print(f"Grafico salvato in: {output_img}")
plt.show()