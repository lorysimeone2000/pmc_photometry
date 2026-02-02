import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Wedge
from matplotlib.colors import LogNorm
import numpy as np
import os
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import warnings

# Ignora warning numerici e FITS
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- 1. FUNZIONI (Vanno sempre in alto) ---

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
    """Legge l'header FITS salvato nelle prime righe del file CSV."""
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


# --- 2. CONFIGURAZIONE (Deve stare PRIMA dell'uso delle variabili) ---

# Percorsi fondamentali
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
percorso_file_csv_main = "/home/lorysimeone/tesi_magistrale/prove_2/analisi/studio_flussi/dispersione_flussi/risultati_analisi_run_1.csv"

# Parametri Analisi
run_list = [1, 2, 3]
INDICE_IMMAGINE_RIFERIMENTO = 35  # Indice file da usare per le coordinate

# --- NUOVO PARAMETRO: NUMERO DI FASCE ---
NUM_FASCE = 3  # Impostato a 3 come richiesto (Aree Uguali)

# Nomi colonne
col_flux = 'media_flusso_fisso_max_run'
col_std = 'std_flusso_fisso_max_run'
col_mag = 'Mag_Brightest'
col_count = 'count_flusso_fisso_max_run'

print(f"Elaborazione Main CSV: {os.path.basename(percorso_file_csv_main)}")

# --- 3. CARICAMENTO DATI (Usa le variabili definite sopra) ---

# A. Caricamento CSV Principale
if not os.path.exists(percorso_file_csv_main):
    print("ERRORE: File Main CSV non trovato.")
    exit()
df = pd.read_csv(percorso_file_csv_main, comment='#')

# B. Caricamento Coordinate da File Riferimento
cartella_ref = os.path.join(base_path, f"tabelle_unite_run_{run_list[0]}")
path_fits_originale = ""
image_header_fits = {}

if os.path.exists(cartella_ref):
    files_ref = sorted([f for f in os.listdir(cartella_ref) if f.endswith('.csv')])

    if len(files_ref) > 0:
        if len(files_ref) <= INDICE_IMMAGINE_RIFERIMENTO:
            INDICE_IMMAGINE_RIFERIMENTO = 0
            print("Indice immagine fuori range, uso la 0.")

        path_ref = os.path.join(cartella_ref, files_ref[INDICE_IMMAGINE_RIFERIMENTO])
        print(f"File riferimento coordinate: {os.path.basename(path_ref)}")

        # 1. Leggi Header
        header_ref_csv = leggi_header_da_csv(path_ref)
        path_fits_originale = header_ref_csv.get('PERCORSO_FILE', '')

        # 2. Leggi le coordinate
        try:
            df_coords = pd.read_csv(path_ref, comment='#', usecols=['ID', 'xcentroid', 'ycentroid'])

            # Merge
            rows_before = len(df)
            df = pd.merge(df, df_coords, on='ID', how='left')
            print(f"Coordinate unite. Righe totali: {len(df)}")

        except Exception as e:
            print(f"ERRORE lettura coordinate da riferimento: {e}")
            df['xcentroid'] = np.nan
            df['ycentroid'] = np.nan
    else:
        print("ERRORE: Nessun file CSV trovato nella cartella riferimento.")
        df['xcentroid'] = np.nan
        df['ycentroid'] = np.nan
else:
    print(f"ERRORE: Cartella riferimento non trovata: {cartella_ref}")
    df['xcentroid'] = np.nan
    df['ycentroid'] = np.nan

# C. Caricamento Immagine FITS
# Inizializza con dimensioni default nel caso il FITS non esista, per far funzionare il calcolo aree
H_default, W_default = 2048, 3072
image_data_sub = np.zeros((H_default, W_default))

try:
    if path_fits_originale and os.path.exists(path_fits_originale):
        print(f"Caricamento FITS: {path_fits_originale}")
        hdu_list = fits.open(path_fits_originale)
        image_data = hdu_list[0].data
        image_header_fits = hdu_list[0].header

        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
        image_data_sub = image_data - median
        hdu_list.close()
    else:
        print(f"ATTENZIONE: File FITS non trovato al percorso: {path_fits_originale}")
        # Se non trova il file, usiamo la dimensione standard
        print(f"Uso dimensioni default per calcolo geometrico: {H_default}x{W_default}")
except Exception as e:
    print(f"Errore caricamento FITS: {e}")

# --- 4. ANALISI E FILTRI ---

mask_no_match = (df[col_mag].isna()) | (df['ID'].astype(str).str.startswith('NOMATCH'))
df_no_match = df[mask_no_match].copy()

mask_sature_original = df['saturazione'].astype(str).str.startswith('SI')
df_sature = df[mask_sature_original & ~mask_no_match].copy()

df_match = df[~mask_no_match].copy()
mask_sature_match = df_match['saturazione'].astype(str).str.startswith('SI')
df_match = df_match[~mask_sature_match].copy()
mask_deboli_match = df_match[col_mag] >= 10
df_match = df_match[~mask_deboli_match].copy()

mask_valid_fit = (
        (df_match[col_flux] > 0) &
        (df_match[col_mag].notna()) &
        (df_match[col_std] > 0) &
        (df_match[col_count] > 0)
)
data_fit = df_match[mask_valid_fit].copy()

# --- 5. CALCOLO ZONE (AREE UGUALI) ---

H, W = image_data_sub.shape
center_x, center_y = W / 2, H / 2

# Calcolo distanza delle stelle dal centro
data_fit['dist_center'] = np.hypot(data_fit['xcentroid'] - center_x, data_fit['ycentroid'] - center_y)

print(f"--- Calcolo Raggi per {NUM_FASCE} Aree Uguali ---")
# 1. Creiamo una griglia di coordinate per TUTTI i pixel dell'immagine
# Usa ogrid per risparmiare memoria (broadcasting)
Y_grid, X_grid = np.ogrid[:H, :W]

# 2. Calcoliamo la distanza dal centro per ogni pixel
dist_grid = np.hypot(X_grid - center_x, Y_grid - center_y)

# 3. Calcoliamo i percentili sulla distribuzione delle distanze dei pixel.
#    Poiché ogni pixel ha area 1, dividere i pixel in N gruppi uguali
#    equivale a dividere l'area geometrica valida (intersecata col rettangolo) in N parti uguali.
percentili = np.linspace(0, 100, NUM_FASCE + 1)
# Escludiamo lo 0 (raggio 0) e prendiamo i limiti superiori
valori_percentili = percentili[1:]

radii_limits = np.percentile(dist_grid, valori_percentili)

# Raggio massimo teorico per il grafico (angolo)
R_max_corner = np.sqrt((W / 2) ** 2 + (H / 2) ** 2)

# Colori
colors_zones = plt.cm.jet(np.linspace(0, 1, NUM_FASCE))

print(f"Limiti Raggi calcolati: {[f'{r:.1f} px' for r in radii_limits]}")
print(f"Nota: Il raggio più esterno tiene conto del taglio rettangolare dell'immagine.")

# --- 6. FIT ---

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

# --- 7. VISUALIZZAZIONE ---

fig = plt.figure(figsize=(18, 9))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

# --- A. Grafico Fit ---
ax1 = plt.subplot(gs[0])

if len(X) > 2:
    x_plot = np.linspace(min(X), max(X), 100)
    y_plot_linear = 10 ** modello_lineare_generico(x_plot, m_fit, q_fit)
    label_fit = rf'Fit: $\log(F)={m_fit:.2f}M + {q_fit:.2f}$ ($\chi^2_R$={chi_reduced:.2f})'
    ax1.plot(x_plot, y_plot_linear, 'k--', linewidth=2, label=label_fit, zorder=5)

# Fasce colorate (Loop dinamico sui raggi calcolati per area)
prev_r = 0
for i, (r_limit, color) in enumerate(zip(radii_limits, colors_zones)):
    mask_zone = (data_fit['dist_center'] >= prev_r) & (data_fit['dist_center'] < r_limit)
    subset = data_fit[mask_zone]

    # Calcolo quanti pixel (area) ci sono in questa fascia per verifica (opzionale)
    # area_pixels = np.sum((dist_grid >= prev_r) & (dist_grid < r_limit))
    # label_text = f'Fascia {i + 1} (Area~{area_pixels/1e6:.1f}M px)'
    label_text = f'Fascia {i + 1} (R<{r_limit:.0f})'

    if len(subset) > 0:
        ax1.errorbar(
            subset[col_mag], subset[col_flux],
            yerr=subset[col_std] / np.sqrt(subset[col_count]),
            fmt='o', markersize=5, color=color, markeredgecolor='none', alpha=0.9,
            label=label_text, zorder=4
        )
    prev_r = r_limit

# Fuori Fascia (Se ci sono stelle oltre l'ultimo raggio calcolato, teoricamente 0 se usiamo il 100 percentile)
# Ma per sicurezza controlliamo se qualche stella ha coordinate strane
mask_outside = (data_fit['dist_center'] >= radii_limits[-1]) | (data_fit['dist_center'].isna())
subset_out = data_fit[mask_outside]
if len(subset_out) > 0:
    ax1.errorbar(
        subset_out[col_mag], subset_out[col_flux],
        yerr=subset_out[col_std] / np.sqrt(subset_out[col_count]),
        fmt='o', markersize=4, color='gray', markeredgecolor='none', alpha=0.3,
        label='Fuori Range', zorder=3
    )

if len(df_sature) > 0:
    mask_sat = (df_sature[col_flux] > 0) & (df_sature[col_mag].notna())
    df_sat_plot = df_sature[mask_sat]
    ax1.scatter(df_sat_plot[col_mag], df_sat_plot[col_flux],
                s=70, c='red', marker='x', label='Sature', zorder=20)

if len(df_no_match) > 0:
    df_nm_plot = df_no_match[df_no_match[col_flux] > 0]
    mag_fittizia = np.min(X) - 1.5 if len(X) > 0 else 4.0
    ax1.scatter(np.full(len(df_nm_plot), mag_fittizia), df_nm_plot[col_flux],
                s=30, c='orange', marker='D', edgecolors='black', label='No Match', zorder=10)

ax1.set_title(f'Calibrazione Fotometrica ({NUM_FASCE} Aree Uguali)', fontsize=14)
ax1.set_xlabel('Magnitudine Catalogo', fontsize=12)
ax1.set_ylabel('Flusso [ADU]', fontsize=12)
ax1.set_yscale('log')
ax1.invert_xaxis()
ax1.grid(True, which="both", alpha=0.2)
ax1.legend(loc='best', fontsize=10)

# --- B. Immagine con Corone Piene (Loop dinamico) ---
ax2 = plt.subplot(gs[1])

# Plot immagine di sfondo
vmin = median
vmax = median + 10 * std
ax2.imshow(image_data_sub, cmap="gray_r", norm=LogNorm(vmin=max(1, vmin), vmax=vmax),
           interpolation='nearest', origin='lower')

prev_r = 0
for r, color in zip(radii_limits, colors_zones):
    width = r - prev_r
    # Disegna l'annulus
    # Wedge disegna un settore circolare, ma verrà visivamente "tagliato" dai limiti degli assi (xlim/ylim)
    # che corrispondono alla dimensione dell'immagine.
    wedge = Wedge((center_x, center_y), r, 0, 360, width=width,
                  facecolor=color, alpha=0.3, edgecolor=color, linewidth=1)
    ax2.add_patch(wedge)
    prev_r = r

ra_val = image_header_fits.get('RA', 'N/A')
dec_val = image_header_fits.get('DEC', 'N/A')
ax2.set_title(f"Immagine Rif: {INDICE_IMMAGINE_RIFERIMENTO} (RA: {ra_val}, DEC: {dec_val})")
ax2.set_xlim(0, W)
ax2.set_ylim(0, H)

plt.tight_layout()
output_img = f"fit_calibrazione_{NUM_FASCE}_aree_uguali.png"
plt.savefig(output_img, dpi=300)
print(f"Grafico salvato: {output_img}")

# --- VERIFICA AREE ---
# Ricalcolo le distanze (come fatto nello script)
Y_grid, X_grid = np.ogrid[:H, :W]
dist_grid = np.hypot(X_grid - center_x, Y_grid - center_y)

# Conto i pixel in ogni fascia usando i limiti trovati (radii_limits)
r1 = radii_limits[0] # Limite Fascia 1 (Blu)
r2 = radii_limits[1] # Limite Fascia 2 (Verde)

area_blu = np.sum(dist_grid < r1)
area_verde = np.sum((dist_grid >= r1) & (dist_grid < r2))
area_rossa = np.sum(dist_grid >= r2)

print(f"--- VERIFICA AREE (in pixel) ---")
print(f"Area Blu:   {area_blu} px")
print(f"Area Verde: {area_verde} px")
print(f"Area Rossa: {area_rossa} px")
print(f"Differenza Max: {max(area_blu, area_verde, area_rossa) - min(area_blu, area_verde, area_rossa)} px")

plt.show()