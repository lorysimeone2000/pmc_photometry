import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import numpy as np
import os
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import warnings
from pathlib import Path
from tqdm import tqdm

# Ignora warning numerici e FITS
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# 0. FUNZIONI DI GESTIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


def converti_valore(valore):
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


def modello_lineare(mag, m, q):
    """ Modello: log10(Flux) = m * Mag + q """
    return m * mag + q


# =============================================================================
# 1. CONFIGURAZIONE DINAMICA
# =============================================================================

BASE_DIR = trova_cartella_base("pmc_photometry")
print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")

# Parametri Analisi
RUN_DA_ANALIZZARE = 1
INDICE_IMMAGINE_RIFERIMENTO = 35

# Nomi colonne fondamentali
col_flux = 'media_flusso_fisso_max_run'
col_std = 'std_flusso_fisso_max_run'
col_mag = 'Mag'

# =============================================================================
# 2. CARICAMENTO DATI (SOLO RUN TARGET)
# =============================================================================

print(f"--- Caricamento dati per Run {RUN_DA_ANALIZZARE} ---")
lista_dfs = []

nome_cartella = f"tabelle_unite_run_{RUN_DA_ANALIZZARE}"
path_cartella = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella)

if path_cartella is None:
    print(f"Attenzione: Cartella {nome_cartella} non trovata.")
    exit()

files_csv = sorted(list(path_cartella.glob("*.csv")))
print(f"Trovati {len(files_csv)} file nella cartella {nome_cartella}. Caricamento in corso...")

for f in tqdm(files_csv, leave=False):
    try:
        # Carico il CSV completo
        df_temp = pd.read_csv(f, comment='#')

        if 'Mag_Brightest' in df_temp.columns and 'Mag' not in df_temp.columns:
            df_temp.rename(columns={'Mag_Brightest': 'Mag'}, inplace=True)

        cols_to_drop = [c for c in ['xcentroid', 'ycentroid'] if c in df_temp.columns]
        if cols_to_drop:
            df_temp.drop(columns=cols_to_drop, inplace=True)

        df_temp['run_origin'] = RUN_DA_ANALIZZARE
        lista_dfs.append(df_temp)
    except Exception as e:
        pass

if not lista_dfs:
    print("ERRORE: Nessun dato caricato.")
    exit()

df_total = pd.concat(lista_dfs, ignore_index=True)
print(f"Totale righe caricate: {len(df_total)}")

# =============================================================================
# 3. CARICAMENTO IMMAGINE DI RIFERIMENTO E COORDINATE SPECIFICHE
# =============================================================================

nome_cartella_ref = f"tabelle_unite_run_{RUN_DA_ANALIZZARE}"
path_cartella_ref = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_ref)

path_fits_originale = ""
image_header_fits = {}
H, W = 2048, 3072  # Default
image_data_sub = np.zeros((H, W))
median, std = 0, 1

df_coords_ref = pd.DataFrame()

if path_cartella_ref is not None:
    files_ref = sorted(list(path_cartella_ref.glob("*.csv")))
    if len(files_ref) > INDICE_IMMAGINE_RIFERIMENTO:
        path_ref = files_ref[INDICE_IMMAGINE_RIFERIMENTO]
        print(f"File riferimento coordinate: {path_ref.name}")

        try:
            df_coords_ref = pd.read_csv(path_ref, comment='#', usecols=['ID', 'xcentroid', 'ycentroid'])
            print(f"Coordinate caricate da riferimento: {len(df_coords_ref)} stelle.")
        except Exception as e:
            print(f"Errore caricamento coordinate riferimento: {e}")

        header_ref_csv = leggi_header_da_csv(path_ref)
        path_fits_str = header_ref_csv.get('PERCORSO_FILE', '')

        if path_fits_str:
            p_obj = Path(path_fits_str)
            if not os.path.exists(path_fits_str):
                try:
                    if "pmc_photometry" in p_obj.parts:
                        idx = p_obj.parts.index("pmc_photometry")
                        new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                        if new_path.exists(): path_fits_originale = str(new_path)
                except:
                    pass
            else:
                path_fits_originale = path_fits_str

        if path_fits_originale and os.path.exists(path_fits_originale):
            print(f"Caricamento FITS: {path_fits_originale}")
            try:
                hdu_list = fits.open(path_fits_originale)
                image_data = hdu_list[0].data
                image_header_fits = hdu_list[0].header
                mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
                image_data_sub = image_data - median
                hdu_list.close()
                H, W = image_data_sub.shape
            except Exception as e:
                print(f"Errore lettura FITS: {e}")

# =============================================================================
# 4. PREPARAZIONE DATI (MERGE E FILTRI)
# =============================================================================

df_total_sorted = df_total.sort_values(by=['label', 'Mag'], ascending=[True, True])
df_unique = df_total_sorted.drop_duplicates(subset=['ID'], keep='first').copy()
print(f"Oggetti UNICI nella Run {RUN_DA_ANALIZZARE}: {len(df_unique)}")

if not df_coords_ref.empty:
    df_unique = pd.merge(df_unique, df_coords_ref, on='ID', how='left')
else:
    print("ATTENZIONE: Nessuna coordinata di riferimento disponibile.")
    df_unique['xcentroid'] = np.nan
    df_unique['ycentroid'] = np.nan

mask_match = df_unique['Corrispondenza'].astype(str).str.startswith('SI')
df_no_match = df_unique[~mask_match].copy()

df_match = df_unique[mask_match].copy()

if 'saturazione' in df_match.columns:
    mask_sature = df_match['saturazione'].astype(str).str.startswith('SI')
    df_sature = df_match[mask_sature].copy()
    df_fit_potential = df_match[~mask_sature].copy()
else:
    df_sature = pd.DataFrame()
    df_fit_potential = df_match.copy()

col_count_final = 'count_flusso_fisso_max_run'
if col_count_final not in df_fit_potential.columns:
    cols = df_fit_potential.columns
    c_alt = [c for c in cols if 'ripetizioni' in c]
    if c_alt:
        col_count_final = c_alt[0]
    else:
        df_fit_potential['ones'] = 1
        col_count_final = 'ones'

mask_valid = (
        (df_fit_potential['Mag'].notna()) &
        (df_fit_potential[col_flux] > 0) &
        (df_fit_potential[col_std] > 0) &
        (df_fit_potential[col_count_final] > 0)
)
data_fit = df_fit_potential[mask_valid].copy()

SOGLIA_MAG_FIT = 10.0
data_fit = data_fit[data_fit['Mag'] <= SOGLIA_MAG_FIT].copy()
df_sature = df_sature[df_sature['Mag'] <= SOGLIA_MAG_FIT].copy()

mask_in_ref = (data_fit['xcentroid'].notna()) & (data_fit['ycentroid'].notna())
data_in_ref = data_fit[mask_in_ref].copy()
data_out_ref = data_fit[~mask_in_ref].copy()

# =============================================================================
# 5. CALCOLO ZONE (4 QUADRANTI) - Solo per chi è nel riferimento
# =============================================================================

center_x, center_y = W / 2, H / 2
print(f"--- Divisione in 4 Quadranti (Centro: {center_x:.1f}, {center_y:.1f}) ---")

# Definisco le proprietà dei 4 quadranti
# Struttura: (Nome, Maschera Logica, Colore, PuntoOrigineRettangolo)
quadrant_defs = []

if not data_in_ref.empty:
    x = data_in_ref['xcentroid']
    y = data_in_ref['ycentroid']

    quadrant_defs = [
        # Q1: Alto-Destra (X >= centro, Y >= centro)
        {
            'label': 'Q1 (Alto-Dx)',
            'mask': (x >= center_x) & (y >= center_y),
            'color': 'red',
            'rect_xy': (center_x, center_y)
        },
        # Q2: Alto-Sinistra (X < centro, Y >= centro)
        {
            'label': 'Q2 (Alto-Sx)',
            'mask': (x < center_x) & (y >= center_y),
            'color': 'green',
            'rect_xy': (0, center_y)
        },
        # Q3: Basso-Sinistra (X < centro, Y < centro)
        {
            'label': 'Q3 (Basso-Sx)',
            'mask': (x < center_x) & (y < center_y),
            'color': 'blue',
            'rect_xy': (0, 0)
        },
        # Q4: Basso-Destra (X >= centro, Y < centro)
        {
            'label': 'Q4 (Basso-Dx)',
            'mask': (x >= center_x) & (y < center_y),
            'color': 'orange',
            'rect_xy': (center_x, 0)
        }
    ]
else:
    print("ATTENZIONE: Nessuna stella nell'immagine di riferimento per definire i quadranti.")

# =============================================================================
# 6. VISUALIZZAZIONE
# =============================================================================

X_global_per_limits = data_fit['Mag'].values
if len(X_global_per_limits) == 0:
    X_global_per_limits = [0, 1]

fig = plt.figure(figsize=(18, 9))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

# --- A. Grafico Fit (Sinistra) ---
ax1 = plt.subplot(gs[0])

# 1. Loop sui Quadranti
if not data_in_ref.empty:
    for q_data in quadrant_defs:
        mask = q_data['mask']
        color = q_data['color']
        label_base = q_data['label']

        subset = data_in_ref[mask]

        if len(subset) > 0:
            # Plot Punti
            ax1.errorbar(
                subset['Mag'], subset[col_flux],
                yerr=subset[col_std],
                fmt='o', markersize=5, color=color, markeredgecolor='none', alpha=0.9,
                label=label_base, zorder=4
            )

            # FIT SPECIFICO PER QUESTO QUADRANTE
            if len(subset) > 2:
                X_sub = subset['Mag'].values
                Y_linear_sub = subset[col_flux].values
                Y_log_sub = np.log10(Y_linear_sub)

                sigma_flux_sub = subset[col_std].values
                sigma_log_sub = (1 / np.log(10)) * (sigma_flux_sub / Y_linear_sub)

                try:
                    popt_sub, _ = curve_fit(modello_lineare, X_sub, Y_log_sub, sigma=sigma_log_sub, absolute_sigma=True)
                    m_fit_sub, q_fit_sub = popt_sub

                    # Genero linea di fit
                    x_plot_sub = np.linspace(min(X_global_per_limits) - 0.5, max(X_global_per_limits) + 0.5, 100)
                    y_plot_linear_sub = 10 ** modello_lineare(x_plot_sub, m_fit_sub, q_fit_sub)

                    label_fit_sub = rf'Fit {label_base}: $m={m_fit_sub:.2f} , $q={q_fit_sub:.2f}$'
                    ax1.plot(x_plot_sub, y_plot_linear_sub, '--', color=color, linewidth=2, label=label_fit_sub,
                             zorder=5)
                except Exception as e:
                    print(f"Errore fit {label_base}: {e}")

# 3. PALLINI GRIGI (Chi NON è nell'immagine di riferimento)
if len(data_out_ref) > 0:
    ax1.errorbar(
        data_out_ref['Mag'], data_out_ref[col_flux],
        yerr=data_out_ref[col_std],
        fmt='o', markersize=4, color='gray', markeredgecolor='none', alpha=0.4,
        label='Non in Img Rif', zorder=3
    )

# 4. Sature
if not df_sature.empty:
    mask_sat_valid = (df_sature[col_flux] > 0) & (df_sature['Mag'].notna())
    df_sat_plot = df_sature[mask_sat_valid]
    if len(df_sat_plot) > 0:
        ax1.scatter(df_sat_plot['Mag'], df_sat_plot[col_flux],
                    s=70, c='red', marker='x', label='Sature', zorder=20)

# 5. Non Catalogati (SOLO quelli presenti nell'immagine di riferimento)
if not df_no_match.empty:
    mask_nm_in_ref = (
        (df_no_match[col_flux] > 0) &
        (df_no_match['xcentroid'].notna()) &
        (df_no_match['ycentroid'].notna())
    )
    df_nm_plot = df_no_match[mask_nm_in_ref]

    if len(df_nm_plot) > 0:
        mag_fittizia = (min(X_global_per_limits) - 1.5) if len(X_global_per_limits) > 0 else 4.0
        ax1.scatter(np.full(len(df_nm_plot), mag_fittizia), df_nm_plot[col_flux],
                    s=30, c='orange', marker='D', edgecolors='black', label='No Match (in Rif)', zorder=10)

ax1.set_title(f'Calibrazione Run {RUN_DA_ANALIZZARE} per Quadranti', fontsize=14)
ax1.set_xlabel('Magnitudine Catalogo', fontsize=12)
ax1.set_ylabel('Media Flusso Fisso [ADU]', fontsize=12)
ax1.set_yscale('log')
ax1.invert_xaxis()
ax1.grid(True, which="both", alpha=0.2)
ax1.legend(loc='best', fontsize=9)

# --- B. Immagine con Quadranti (Destra) ---
ax2 = plt.subplot(gs[1])

vmin = median
vmax = median + 10 * std
ax2.imshow(image_data_sub, cmap="gray_r", norm=LogNorm(vmin=max(1, vmin), vmax=vmax),
           interpolation='nearest', origin='lower')

# Disegno i rettangoli dei quadranti
for q_data in quadrant_defs:
    xy = q_data['rect_xy']
    # La larghezza e altezza sono sempre metà immagine per definizione
    w_rect = center_x
    h_rect = center_y
    color = q_data['color']

    rect = Rectangle(xy, w_rect, h_rect, facecolor=color, alpha=0.3, edgecolor=color, linewidth=1)
    ax2.add_patch(rect)

ra_val = image_header_fits.get('RA', 'N/A')
dec_val = image_header_fits.get('DEC', 'N/A')
ax2.set_title(f"Rif: Run {RUN_DA_ANALIZZARE} Img {INDICE_IMMAGINE_RIFERIMENTO}\n(RA: {ra_val}, DEC: {dec_val})")
ax2.set_xlim(0, W)
ax2.set_ylim(0, H)

plt.tight_layout()
output_img = f"fit_run_{RUN_DA_ANALIZZARE}_4_quadranti.png"
plt.savefig(output_img, dpi=300)
print(f"Grafico salvato: {output_img}")
plt.show()