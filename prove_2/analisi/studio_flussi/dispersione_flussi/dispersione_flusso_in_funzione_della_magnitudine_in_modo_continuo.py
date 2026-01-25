import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats
import warnings

# Ignora warning numerici
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- 1. FUNZIONI ---

def converti_valore(valore):
    """Converte una stringa nel tipo di dato appropriato."""
    valore = str(valore).strip()
    if not valore: return valore
    try:
        return int(valore)
    except ValueError:
        pass
    try:
        return float(valore)
    except ValueError:
        pass
    return valore


def leggi_header_da_csv(filename):
    """Legge l'header salvato nelle prime righe del file CSV."""
    header_dict = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#') and ':' in line:
                    clean_line = line.strip()[1:].strip()
                    if clean_line and ': ' in clean_line:
                        key, value = clean_line.split(': ', 1)
                        header_dict[key] = converti_valore(value)
                elif line.strip() == '#':
                    break
    except Exception as e:
        print(f"Warning: Impossibile leggere header da CSV: {e}")
    return header_dict


def modello_lineare_generico(mag, m, q):
    """ Modello: log10(Flux) = m * Mag + q """
    return m * mag + q


# --- 2. CONFIGURAZIONE ---

# Percorsi
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
percorso_file_csv_main = "/home/lorysimeone/tesi_magistrale/prove_2/analisi/studio_flussi/dispersione_flussi/risultati_analisi_run_1.csv"

# Parametri
run_list = [1, 2, 3]
INDICE_IMMAGINE_RIFERIMENTO = 35

# Dimensioni Sensore (per calcolo centro)
W, H = 3072, 2048
CENTER_X, CENTER_Y = W / 2, H / 2
R_MAX = np.hypot(CENTER_X, CENTER_Y)  # Diagonale per normalizzare i colori

# Nomi colonne nel file MAIN
col_flux = 'media_flusso_fisso_max_run'
col_std = 'std_flusso_fisso_max_run'
col_mag = 'Mag_Brightest'
col_count = 'count_flusso_fisso_max_run'

print(f"Elaborazione Main CSV: {os.path.basename(percorso_file_csv_main)}")

# --- 3. CARICAMENTO DATI ---

# A. Caricamento CSV Principale
if not os.path.exists(percorso_file_csv_main):
    print("ERRORE: File Main CSV non trovato.")
    exit()
df = pd.read_csv(percorso_file_csv_main, comment='#')

# B. Caricamento Coordinate da File Riferimento (Immagine 35)
cartella_ref = os.path.join(base_path, f"tabelle_unite_run_{run_list[0]}")
path_ref = ""

if os.path.exists(cartella_ref):
    files_ref = sorted([f for f in os.listdir(cartella_ref) if f.endswith('.csv')])

    if len(files_ref) > 0:
        if len(files_ref) <= INDICE_IMMAGINE_RIFERIMENTO:
            INDICE_IMMAGINE_RIFERIMENTO = 0

        path_ref = os.path.join(cartella_ref, files_ref[INDICE_IMMAGINE_RIFERIMENTO])
        print(f"File riferimento coordinate: {os.path.basename(path_ref)}")

        try:
            # Leggiamo solo le colonne necessarie
            df_coords = pd.read_csv(path_ref, comment='#', usecols=['ID', 'xcentroid', 'ycentroid'])

            # Merge con il dataframe principale
            rows_before = len(df)
            df = pd.merge(df, df_coords, on='ID', how='left')
            print(f"Coordinate unite. Righe totali: {len(df)}")

        except Exception as e:
            print(f"ERRORE lettura coordinate: {e}")
            df['xcentroid'] = np.nan
            df['ycentroid'] = np.nan
    else:
        print("ERRORE: Nessun file CSV trovato nella cartella riferimento.")
        df['xcentroid'] = np.nan
        df['ycentroid'] = np.nan
else:
    print(f"ERRORE: Cartella riferimento non trovata.")
    df['xcentroid'] = np.nan
    df['ycentroid'] = np.nan

# --- 4. PREPARAZIONE DATI E FIT ---

# Calcolo Distanza dal Centro
# Se xcentroid è NaN (stella non trovata nel ref), dist_center sarà NaN
df['dist_center'] = np.hypot(df['xcentroid'] - CENTER_X, df['ycentroid'] - CENTER_Y)

# Filtri Base (No Match e Sature)
mask_no_match = (df[col_mag].isna()) | (df['ID'].astype(str).str.startswith('NOMATCH'))
df_no_match = df[mask_no_match].copy()

mask_sature_original = df['saturazione'].astype(str).str.startswith('SI')
df_sature = df[mask_sature_original & ~mask_no_match].copy()

# Dataset per il Fit (Matchati e Validi)
df_match = df[~mask_no_match].copy()
mask_sature_match = df_match['saturazione'].astype(str).str.startswith('SI')
df_match = df_match[~mask_sature_match].copy()
mask_deboli = df_match[col_mag] >= 10
df_match = df_match[~mask_deboli].copy()

# Filtro validità valori numerici
mask_valid_fit = (
        (df_match[col_flux] > 0) &
        (df_match[col_mag].notna()) &
        (df_match[col_std] > 0) &
        (df_match[col_count] > 0)
)
data_fit = df_match[mask_valid_fit].copy()

# Esecuzione Fit
X = data_fit[col_mag].values
Y_linear = data_fit[col_flux].values
Y_log = np.log10(Y_linear)
sigma_std = data_fit[col_std].values
counts = data_fit[col_count].values
sigma_mean_linear = sigma_std / np.sqrt(counts)
sigma_log = (1 / np.log(10)) * (sigma_mean_linear / Y_linear)

if len(X) > 2:
    popt, pcov = curve_fit(modello_lineare_generico, X, Y_log, sigma=sigma_log, absolute_sigma=True)
    m_fit, q_fit = popt

    y_model_log = modello_lineare_generico(X, m_fit, q_fit)
    chi_squared = np.sum(((Y_log - y_model_log) / sigma_log) ** 2)
    chi_reduced = chi_squared / (len(X) - 2)
    print(f"\nRisultati Fit: m={m_fit:.4f}, q={q_fit:.4f}, Chi2_red={chi_reduced:.4f}")
else:
    m_fit, q_fit, chi_reduced = 0, 0, 0

# --- 5. VISUALIZZAZIONE CONTINUA ---

plt.figure(figsize=(12, 9))

# A. Retta di Fit
if len(X) > 2:
    x_plot = np.linspace(min(X), max(X), 100)
    y_plot_linear = 10 ** modello_lineare_generico(x_plot, m_fit, q_fit)
    label_fit = rf'Fit: $\log(F)={m_fit:.2f}M + {q_fit:.2f}$ ($\chi^2_R$={chi_reduced:.2f})'
    plt.plot(x_plot, y_plot_linear, 'k--', linewidth=2, label=label_fit, zorder=10)

# B. Stelle con Coordinate (Colormap Continua)
mask_coords = data_fit['dist_center'].notna()
subset_coords = data_fit[mask_coords]

if len(subset_coords) > 0:
    # 1. Disegna le Errorbars in grigio (senza marker)
    plt.errorbar(
        subset_coords[col_mag], subset_coords[col_flux],
        yerr=subset_coords[col_std] / np.sqrt(subset_coords[col_count]),
        fmt='none', ecolor='gray', alpha=0.5, zorder=4
    )

    # 2. Disegna i Markers colorati in base alla distanza
    sc = plt.scatter(
        subset_coords[col_mag], subset_coords[col_flux],
        c=subset_coords['dist_center'], cmap='jet', vmin=0, vmax=R_MAX,
        s=30, alpha=0.9, edgecolors='none', zorder=5, label='Catalogati (Run 1, Img 35)'
    )

    # Aggiungi Colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Distanza dal Centro (px)', rotation=270, labelpad=20)

# C. Stelle SENZA Coordinate (Grigio Trasparente)
subset_no_coords = data_fit[~mask_coords]
if len(subset_no_coords) > 0:
    plt.errorbar(
        subset_no_coords[col_mag], subset_no_coords[col_flux],
        yerr=subset_no_coords[col_std] / np.sqrt(subset_no_coords[col_count]),
        fmt='o', markersize=4, color='gray', alpha=0.3,
        label='Senza coord. in rif.', zorder=3
    )

# D. Stelle Sature (Rosse X)
if len(df_sature) > 0:
    mask_sat = (df_sature[col_flux] > 0) & (df_sature[col_mag].notna())
    df_sat_plot = df_sature[mask_sat]
    plt.scatter(df_sat_plot[col_mag], df_sat_plot[col_flux],
                s=70, c='red', marker='x', label='Sature (Escluse)', zorder=20)

# E. No Match (Arancioni)
if len(df_no_match) > 0:
    df_nm_plot = df_no_match[df_no_match[col_flux] > 0]
    mag_fittizia = np.min(X) - 1.5 if len(X) > 0 else 4.0
    plt.scatter(np.full(len(df_nm_plot), mag_fittizia), df_nm_plot[col_flux],
                s=30, c='orange', marker='D', edgecolors='black', label='No Match', zorder=10)

# F. Formattazione
plt.title(f'Calibrazione Fotometrica - Dipendenza Spaziale (Img {INDICE_IMMAGINE_RIFERIMENTO})', fontsize=14)
plt.xlabel('Magnitudine Catalogo', fontsize=12)
plt.ylabel('Media Flusso [ADU]', fontsize=12)
plt.yscale('log')
plt.gca().invert_xaxis()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(loc='best')
plt.tight_layout()

# Salva
output_img = "fit_calibrazione_continuous_distance.png"
plt.savefig(output_img, dpi=300)
print(f"Grafico salvato in: {output_img}")
plt.show()