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
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point, Polygon  # <--- Nuova importazione

# Ignora warning numerici e FITS
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# 0. FUNZIONI DI GESTIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
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
# 1. CONFIGURAZIONE DINAMICA E DEFINIZIONE POLIGONI (SHAPELY)
# =============================================================================

BASE_DIR = trova_cartella_base("Lorenzo")
print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")

# Parametri Analisi
RUN_DA_ANALIZZARE = 1
INDICE_IMMAGINE_RIFERIMENTO = 35

# Nomi colonne fondamentali
col_flux = 'media_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE'
col_std = 'std_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE'
col_mag = 'Mag'

# --- !!! DA COMPILARE !!! ---
# Sostituisci questi valori con le vere coordinate RA e DEC dei vertici
vertici_riferimento = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

dizionario_vertici_fasce = {
    'Fascia 1': [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
    'Fascia 2': [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
    'Fascia 3': [(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)]
}
# ----------------------------

# Inizializzo i poligoni Shapely
poligono_riferimento = Polygon(vertici_riferimento)
poligoni_fasce = {nome: Polygon(vertici) for nome, vertici in dizionario_vertici_fasce.items()}

# Colori per il plot (mappati sui nomi delle fasce)
colori_fasce_dict = {
    nome: plt.cm.jet(i / max(1, len(dizionario_vertici_fasce) - 1))
    for i, nome in enumerate(dizionario_vertici_fasce.keys())
}

# =============================================================================
# 2. CARICAMENTO DATI (SOLO RUN TARGET)
# =============================================================================

print(f"--- Caricamento dati per Run {RUN_DA_ANALIZZARE} ---")
lista_dfs = []

nome_cartella = f"tabelle_unite_run_{RUN_DA_ANALIZZARE}"
path_cartella = cerca_cartella_nel_progetto(BASE_DIR / "tabelle", nome_cartella)

if path_cartella is None:
    print(f"Attenzione: Cartella {nome_cartella} non trovata.")
    exit()

files_csv = sorted(list(path_cartella.glob("*.csv")))
print(f"Trovati {len(files_csv)} file nella cartella {nome_cartella}. Caricamento in corso...")

for f in tqdm(files_csv, leave=False):
    try:
        df_temp = pd.read_csv(f, comment='#')

        if 'Mag_Brightest' in df_temp.columns and 'Mag' not in df_temp.columns:
            df_temp.rename(columns={'Mag_Brightest': 'Mag'}, inplace=True)

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
# 3. CARICAMENTO IMMAGINE DI RIFERIMENTO (SOLO FITS, NO MERGE COORDINATE)
# =============================================================================

nome_cartella_ref = f"tabelle_unite_run_{RUN_DA_ANALIZZARE}"
path_cartella_ref = cerca_cartella_nel_progetto(BASE_DIR / "tabelle", nome_cartella_ref)

path_fits_originale = ""
image_header_fits = {}
H, W = 2048, 3072
image_data_sub = np.zeros((H, W))
median, std = 0, 1

if path_cartella_ref is not None:
    files_ref = sorted(list(path_cartella_ref.glob("*.csv")))
    if len(files_ref) > INDICE_IMMAGINE_RIFERIMENTO:
        path_ref = files_ref[INDICE_IMMAGINE_RIFERIMENTO]
        print(f"File riferimento per recupero FITS: {path_ref.name}")

        header_ref_csv = leggi_header_da_csv(path_ref)
        path_fits_str = header_ref_csv.get('PERCORSO_FILE', '')

        if path_fits_str:
            p_obj = Path(path_fits_str)
            if not os.path.exists(path_fits_str):
                try:
                    if "Lorenzo" in p_obj.parts:
                        idx = p_obj.parts.index("Lorenzo")
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
# 4. PREPARAZIONE DATI E CONTROLLO SPAZIALE SHAPELY
# =============================================================================

# A. DEDUPLICAZIONE PER ID UNIVOCO
df_total_sorted = df_total.sort_values(by=['label', 'Mag'], ascending=[True, True])
df_unique = df_total_sorted.drop_duplicates(subset=['ID'], keep='first').copy()
print(f"Oggetti UNICI nella Run {RUN_DA_ANALIZZARE}: {len(df_unique)}")

# B. APPLICAZIONE LOGICA SHAPELY AL POSTO DEL MERGE
print("Esecuzione controllo spaziale con Shapely (RA/DEC)...")


def verifica_appartenenza(ra, dec):
    if pd.isna(ra) or pd.isna(dec):
        return False, None

    punto_stella = Point(ra, dec)
    in_riferimento = poligono_riferimento.contains(punto_stella)

    fascia_appartenenza = None
    if in_riferimento:
        # Controllo a quale fascia appartiene
        for nome_fascia, poligono_fascia in poligoni_fasce.items():
            if poligono_fascia.contains(punto_stella):
                fascia_appartenenza = nome_fascia
                break  # Si ferma alla prima fascia che contiene il punto

    return in_riferimento, fascia_appartenenza


# Verifica che le colonne esistano
col_ra = 'RA_centroid'
col_dec = 'DEC_centroid'

if col_ra in df_unique.columns and col_dec in df_unique.columns:
    risultati = df_unique.apply(lambda row: verifica_appartenenza(row[col_ra], row[col_dec]), axis=1)
    df_unique['in_immagine_riferimento'] = [res[0] for res in risultati]
    df_unique['fascia_appartenenza'] = [res[1] for res in risultati]
else:
    print(f"ERRORE CRITICO: Colonne {col_ra} e/o {col_dec} non trovate nel DataFrame!")
    df_unique['in_immagine_riferimento'] = False
    df_unique['fascia_appartenenza'] = None

# C. SEPARAZIONE DATI
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

# Filtro Validità Numerica
col_count_final = 'count_flusso_fisso_max_run'
if col_count_final not in df_fit_potential.columns:
    cols = df_fit_potential.columns
    c_alt = [c for c in cols if 'ripetizioni' in c]
    if c_alt:
        col_count_final = c_alt[0]
    else:
        df_fit_potential['ones'] = 1; col_count_final = 'ones'

mask_valid = (
        (df_fit_potential['Mag'].notna()) &
        (df_fit_potential[col_flux] > 0) &
        (df_fit_potential[col_std] > 0) &
        (df_fit_potential[col_count_final] > 0)
)
data_fit = df_fit_potential[mask_valid].copy()

# Filtro Magnitudine
SOGLIA_MAG_FIT = 10.0
data_fit = data_fit[data_fit['Mag'] <= SOGLIA_MAG_FIT].copy()
df_sature = df_sature[df_sature['Mag'] <= SOGLIA_MAG_FIT].copy()

# SEPARAZIONE IN BASE AL NUOVO CONTROLLO SHAPELY
mask_in_ref = data_fit['in_immagine_riferimento'] == True
data_in_ref = data_fit[mask_in_ref].copy()
data_out_ref = data_fit[~mask_in_ref].copy()

# =============================================================================
# 5. VISUALIZZAZIONE
# =============================================================================

X_global_per_limits = data_fit['Mag'].values
if len(X_global_per_limits) == 0: X_global_per_limits = [0, 1]

fig = plt.figure(figsize=(18, 9))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

# --- A. Grafico Fit (Sinistra) ---
ax1 = plt.subplot(gs[0])

# 1. Plot per ogni Fascia (Shapely)
if not data_in_ref.empty:
    for nome_fascia, color in colori_fasce_dict.items():
        subset = data_in_ref[data_in_ref['fascia_appartenenza'] == nome_fascia]

        if len(subset) > 0:
            ax1.errorbar(
                subset['Mag'], subset[col_flux],
                yerr=subset[col_std],
                fmt='o', markersize=5, color=color, markeredgecolor='none', alpha=0.9,
                label=f'{nome_fascia} ({len(subset)} obj)', zorder=4
            )

            # FIT SPECIFICO PER QUESTA FASCIA
            if len(subset) > 2:
                X_sub = subset['Mag'].values
                Y_linear_sub = subset[col_flux].values
                Y_log_sub = np.log10(Y_linear_sub)

                sigma_flux_sub = subset[col_std].values
                sigma_log_sub = (1 / np.log(10)) * (sigma_flux_sub / Y_linear_sub)

                try:
                    popt_sub, _ = curve_fit(modello_lineare, X_sub, Y_log_sub, sigma=sigma_log_sub, absolute_sigma=True)
                    m_fit_sub, q_fit_sub = popt_sub

                    x_plot_sub = np.linspace(min(X_global_per_limits) - 0.5, max(X_global_per_limits) + 0.5, 100)
                    y_plot_linear_sub = 10 ** modello_lineare(x_plot_sub, m_fit_sub, q_fit_sub)

                    label_fit_sub = rf'Fit {nome_fascia}: $m={m_fit_sub:.2f}$ , $q={q_fit_sub:.2f}$'
                    ax1.plot(x_plot_sub, y_plot_linear_sub, '--', color=color, linewidth=2, label=label_fit_sub,
                             zorder=5)
                except Exception as e:
                    print(f"Errore fit {nome_fascia}: {e}")

# 2. PALLINI GRIGI (Fuori dal poligono di riferimento)
if len(data_out_ref) > 0:
    ax1.errorbar(
        data_out_ref['Mag'], data_out_ref[col_flux],
        yerr=data_out_ref[col_std],
        fmt='o', markersize=4, color='gray', markeredgecolor='none', alpha=0.4,
        label='Fuori Riquadro Rif.', zorder=3
    )

# 3. Sature
if not df_sature.empty:
    mask_sat_valid = (df_sature[col_flux] > 0) & (df_sature['Mag'].notna())
    df_sat_plot = df_sature[mask_sat_valid]
    if len(df_sat_plot) > 0:
        ax1.scatter(df_sat_plot['Mag'], df_sat_plot[col_flux], s=70, c='red', marker='x', label='Sature', zorder=20)

# 4. Non Catalogati (nel poligono di riferimento)
if not df_no_match.empty:
    df_no_match['in_immagine_riferimento'] = df_no_match.apply(
        lambda row: verifica_appartenenza(row.get(col_ra, np.nan), row.get(col_dec, np.nan))[0], axis=1
    )
    mask_nm_in_ref = (df_no_match[col_flux] > 0) & (df_no_match['in_immagine_riferimento'] == True)
    df_nm_plot = df_no_match[mask_nm_in_ref]

    if len(df_nm_plot) > 0:
        mag_fittizia = (min(X_global_per_limits) - 1.5) if len(X_global_per_limits) > 0 else 4.0
        ax1.scatter(np.full(len(df_nm_plot), mag_fittizia), df_nm_plot[col_flux],
                    s=30, c='orange', marker='D', edgecolors='black', label='No Match (in Riquadro)', zorder=10)

ax1.set_title(f'Calibrazione Run {RUN_DA_ANALIZZARE} (Metodo Poligoni)', fontsize=14)
ax1.set_xlabel('Magnitudine Catalogo', fontsize=12)
ax1.set_ylabel('Media Flusso Fisso [ADU]', fontsize=12)
ax1.set_yscale('log')
ax1.invert_xaxis()
ax1.grid(True, which="both", alpha=0.2)
ax1.legend(loc='best', fontsize=10)

# --- B. Immagine (Destra) ---
ax2 = plt.subplot(gs[1])

vmin = median
vmax = median + 10 * std
ax2.imshow(image_data_sub, cmap="gray_r", norm=LogNorm(vmin=max(1, vmin), vmax=vmax),
           interpolation='nearest', origin='lower')

# NOTA: La parte che disegnava i Wedge (cerchi colorati) è stata rimossa
# perché ora le fasce sono definite in RA/DEC e non più in pixel.
# Per disegnare i poligoni sull'immagine servirebbe convertire i vertici da RA/DEC
# a X/Y usando il WCS (World Coordinate System) dell'header FITS.

ra_val = image_header_fits.get('RA', 'N/A')
dec_val = image_header_fits.get('DEC', 'N/A')
ax2.set_title(f"Rif: Run {RUN_DA_ANALIZZARE} Img {INDICE_IMMAGINE_RIFERIMENTO}\n(RA: {ra_val}, DEC: {dec_val})")
ax2.set_xlim(0, W)
ax2.set_ylim(0, H)

plt.tight_layout()
output_img = f"fit_run_{RUN_DA_ANALIZZARE}_fasce_separate.png"
plt.savefig(output_img, dpi=300)
print(f"Grafico salvato: {output_img}")
plt.show()