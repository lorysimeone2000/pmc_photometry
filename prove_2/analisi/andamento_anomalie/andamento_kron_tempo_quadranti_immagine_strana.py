import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.table import Table
import warnings
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import FITSFixedWarning, WCS
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Patch
from matplotlib.gridspec import GridSpec
from pathlib import Path
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import simple_norm
import astropy.units as u
from matplotlib.lines import Line2D
from tqdm import tqdm

# importo i moduli per l'image segmentation
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel, SourceFinder

# sopprimo i warning
warnings.filterwarnings('ignore', category=FITSFixedWarning)


# =============================================================================
# --- FUNZIONI DI UTILITÀ PER LA RICERCA DINAMICA ---
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # cerco un file specifico in tutte le sottocartelle
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # cerco una cartella specifica in tutte le sottocartelle
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    if len(cartelle_trovate) > 1:
        print(
            f"INFO: Trovate {len(cartelle_trovate)} cartelle '{nome_cartella_esatto}'. Uso la prima: {cartelle_trovate[0].relative_to(base_dir)}")
    return cartelle_trovate[0]


def converti_valore(valore):
    # converto una stringa nel tipo di dato appropriato
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
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False
    return valore


def leggi_header_da_csv(filename):
    # leggo l'header FITS salvato nelle prime righe del file CSV
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':
                break
    return header_dict


def leggi_file_parametri(percorso):
    # leggo il file dei parametri per la segmentazione
    parametri = {}
    if not percorso or not os.path.exists(percorso): return {}
    with open(percorso, 'r') as file:
        next(file, None)  # salto l'intestazione
        for riga in file:
            riga = riga.split('#')[0].strip()
            if riga:
                parts = riga.split()
                if len(parts) >= 2:
                    try:
                        valore = float(parts[1]) if '.' in parts[1] else int(parts[1])
                        parametri[parts[0]] = valore
                    except ValueError:
                        pass
    return parametri


def salva_cutout(region, data_sub, r_corr_px, wcs_ref, img_idx, run_id, cartella_out, parametri_seg, base_dir_progetto):
    # estraggo i dati della stella
    xc, yc = region['coords']
    area = region['area']
    star_id = region['star_id']
    media_flusso = region['media_flusso_fisso_max_run']
    corr = region['corrispondenza']

    # calcolo il lato del riquadro basato sulla radice dell'area
    side = np.sqrt(area) * 1.5 * 2
    half_side = side / 2.0

    # imposto i limiti spaziali del cutout
    x_min = int(np.floor(xc - half_side))
    x_max = int(np.ceil(xc + half_side))
    y_min = int(np.floor(yc - half_side))
    y_max = int(np.ceil(yc + half_side))

    # prevengo sforamenti rispetto ai bordi dell'immagine reale
    ny, nx = data_sub.shape
    x_min = max(0, x_min)
    x_max = min(nx, x_max)
    y_min = max(0, y_min)
    y_max = min(ny, y_max)

    cutout = data_sub[y_min:y_max, x_min:x_max]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Inizializzo le liste per la legenda manuale
    legend_elements = []

    if cutout.size > 0:
        norm = simple_norm(cutout, 'log', percent=99.9)
        ax.imshow(cutout, cmap='gray_r', origin='lower', extent=[x_min, x_max, y_min, y_max], norm=norm)

        # applico l'image segmentation al cutout usando i parametri forniti
        fwhm = parametri_seg.get('fwhm', 3.0)
        size_kernel = int(parametri_seg.get('size', 5))
        threshold = parametri_seg.get('threshold_assoluta', 3.61)
        pixel_n = int(parametri_seg.get('pixel', 3))

        try:
            kernel = make_2dgaussian_kernel(fwhm, size=size_kernel)
            convolved_cutout = convolve(cutout, kernel)
            finder = SourceFinder(npixels=pixel_n, progress_bar=False)
            segment_map = finder(convolved_cutout, threshold)

            if segment_map is not None:
                # disegno i confini. Rimosso 'label' per evitare il warning
                ax.contour(segment_map.data > 0, levels=[0.5], colors='#00ff00', alpha=0.5, linewidths=1.5,
                           extent=[x_min, x_max, y_min, y_max], origin='lower', zorder=8)

                # Aggiungo l'elemento alla legenda (proxy artist)
                legend_elements.append(Patch(facecolor='none', edgecolor='#00ff00', alpha=0.5,
                                             label='Regione della segmentazione', linewidth=1.5))
        except Exception:
            pass

    # --- NUOVA LOGICA: RECUPERO IL CATALOGO ORIGINALE PER STAMPARE TUTTE LE STELLE ---
    # cerco la cartella delle sorgenti catalogate della run corrente
    nome_cartella_cat = f"sorgenti_catalogate_run_{run_id}"
    cartella_cat = cerca_cartella_nel_progetto(base_dir_progetto, nome_cartella_cat)

    tbl_cat_box = []  # inizializzo la lista vuota in caso di problemi

    if cartella_cat is not None:
        # costruisco il nome esatto del file csv in base all'indice
        nome_file_cat = f"run_{run_id}_stelle_catalogate_immagine_{img_idx:03d}.csv"
        path_file_cat = cerca_file_nel_progetto(cartella_cat, nome_file_cat)

        if path_file_cat is not None:
            try:
                # leggo TUTTO il catalogo per quell'immagine
                df_cat_full = pd.read_csv(path_file_cat, comment='#')
                tbl_cat_full = Table.from_pandas(df_cat_full)

                # se non ho xcentroid e ycentroid (poiché il catalogo puro potrebbe avere solo RA e DEC),
                # li genero al volo usando il WCS che ho passato alla funzione
                if 'xcentroid' not in tbl_cat_full.colnames or 'ycentroid' not in tbl_cat_full.colnames:
                    coords_cat_sky = u.Quantity([tbl_cat_full['RAJ2000'], tbl_cat_full['DEJ2000']], unit=u.deg)
                    x_pix, y_pix = wcs_ref.world_to_pixel_values(coords_cat_sky[0], coords_cat_sky[1])
                    tbl_cat_full['xcentroid'] = x_pix
                    tbl_cat_full['ycentroid'] = y_pix

                # filtro tenendo solo quelle che cadono visivamente dentro il mio riquadro specifico
                mask_in_box = (tbl_cat_full['xcentroid'] >= x_min) & (tbl_cat_full['xcentroid'] <= x_max) & \
                              (tbl_cat_full['ycentroid'] >= y_min) & (tbl_cat_full['ycentroid'] <= y_max)

                tbl_cat_box = tbl_cat_full[mask_in_box]

            except Exception as e:
                pass

    if len(tbl_cat_box) > 0:
        # fisso il limite inferiore della colorbar a 5, il superiore a 15 (o lo adatto ai dati)
        min_mag = min(np.nanmin(tbl_cat_box['Mag']), 5)
        max_mag = 15

        ax.scatter(tbl_cat_box['xcentroid'], tbl_cat_box['ycentroid'], c=tbl_cat_box['Mag'],
                   cmap='viridis_r', vmin=min_mag, vmax=max_mag, s=4, zorder=5)

        # Aggiungo alla legenda
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Stelle catalogate',
                                      markerfacecolor='gray', markersize=5))

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(vmin=min_mag, vmax=max_mag))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mag')

    # Cerchio di correlazione
    circle = Circle((xc, yc), r_corr_px, edgecolor='#ffff00', facecolor='none', linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    legend_elements.append(Patch(facecolor='none', edgecolor='#ffff00', label='Regione di correlazione', linewidth=1.5))

    # Croce per NO
    if str(corr) == 'NO':
        ax.plot(xc, yc, marker='+', color='red', markersize=15, markeredgewidth=2, zorder=15)
        legend_elements.append(Line2D([0], [0], marker='+', color='red', label='Centroide (No match)',
                                      markersize=10, linestyle='None', markeredgewidth=2))

    ax.set_title(f"Oggetto {star_id}\nkron medio {media_flusso}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Stampo la legenda corretta
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.7)

    nome_figura = f"{star_id}_immagine{img_idx:03d}_run_{run_id}.png"
    percorso_completo = cartella_out / nome_figura
    plt.savefig(percorso_completo, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# --- 1. IMPOSTAZIONE DINAMICA DEI PERCORSI E PARAMETRI ---
# =============================================================================

BASE_DIR = trova_cartella_base("pmc_photometry")
print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")

cartella_unite = cerca_cartella_nel_progetto(BASE_DIR, "tabelle_unite")
if cartella_unite is None:
    print("ERRORE: Cartella 'tabelle_unite' non trovata.")
    exit()

base_path = str(cartella_unite)
print(f"Uso la cartella dati: {base_path}")

file_parametri = cerca_file_nel_progetto(BASE_DIR, 'parametri_image_segmentation.txt')
parametri_seg = leggi_file_parametri(str(file_parametri)) if file_parametri else {}

print(f"------------------------------")

run_list = [1, 2, 3]
KRON_TARGET = 115
RUN_REF = 1  # Imposto l'indice 2 per lavorare sulla Run 3
MIN_COUNT_RUN1 = 75
MIN_COUNT_RUN1_NO = 25

# --- RICERCA DINAMICA IMMAGINE DI RIFERIMENTO ---
print(
    f"\nCerco l'immagine di riferimento nella Run {run_list[RUN_REF]} con > 20 oggetti non catalogati e singola ripetizione...")
cartella_ref_tmp = os.path.join(base_path, f"tabelle_unite_run_{run_list[RUN_REF]}")
files_ref_tmp = sorted([f for f in os.listdir(cartella_ref_tmp) if f.endswith('.csv')])

INDICE_IMMAGINE_RIFERIMENTO = 0  # imposto un default

for idx, f_csv in enumerate(files_ref_tmp):
    # carico solo le due colonne necessarie per velocizzare drasticamente la lettura
    df_tmp = pd.read_csv(os.path.join(cartella_ref_tmp, f_csv), comment='#', usecols=['Corrispondenza', 'ripetizioni'])

    # filtro per oggetti non catalogati apparsi 1 sola volta (i transienti del picco)
    mask_anomalie = (df_tmp['Corrispondenza'].astype(str) == 'NO') & (df_tmp['ripetizioni'] == 1)
    conteggio = mask_anomalie.sum()

    if conteggio > 20:
        INDICE_IMMAGINE_RIFERIMENTO = idx
        print(f"  > Trovata! Uso l'immagine all'indice {idx} ({f_csv}) che contiene {conteggio} anomalie isolate.")
        break
else:
    print("  > Nessuna immagine soddisfa il requisito dei transienti. Mantengo l'indice di default 0.")

# creo la cartella di output specifica per i salvataggi
nome_cartella_output = f"Kron_ref_{KRON_TARGET}_run_{run_list[RUN_REF]}_immagine_{INDICE_IMMAGINE_RIFERIMENTO:03d}"
cartella_output = Path(nome_cartella_output)
cartella_output.mkdir(exist_ok=True, parents=True)
print(f"Cartella di salvataggio creata/selezionata: {cartella_output.resolve()}")

H, W = 2048, 3072
CENTER_X, CENTER_Y = W / 2, H / 2

quadrants = [
    {'name': 'Q1 (Alto-Dx)', 'mask_func': lambda x, y: (x >= CENTER_X) & (y >= CENTER_Y), 'color': 'red'},
    {'name': 'Q2 (Alto-Sx)', 'mask_func': lambda x, y: (x < CENTER_X) & (y >= CENTER_Y), 'color': 'green'},
    {'name': 'Q3 (Basso-Sx)', 'mask_func': lambda x, y: (x < CENTER_X) & (y < CENTER_Y), 'color': 'blue'},
    {'name': 'Q4 (Basso-Dx)', 'mask_func': lambda x, y: (x >= CENTER_X) & (y < CENTER_Y), 'color': 'orange'}
]

# =============================================================================
# --- FASE 1: SELEZIONE STELLE NELL'IMMAGINE DI RIFERIMENTO ---
# =============================================================================

cartella_ref = os.path.join(base_path, f"tabelle_unite_run_{run_list[RUN_REF]}")
files_ref = sorted([f for f in os.listdir(cartella_ref) if f.endswith('.csv')])
if len(files_ref) <= INDICE_IMMAGINE_RIFERIMENTO:
    INDICE_IMMAGINE_RIFERIMENTO = 0
path_ref = os.path.join(cartella_ref, files_ref[INDICE_IMMAGINE_RIFERIMENTO])

df_ref = pd.read_csv(path_ref, comment='#')
tbl_ref = Table.from_pandas(df_ref)

header_ref_csv = leggi_header_da_csv(path_ref)
nome_fits = header_ref_csv.get('NOME_FILE_FITS', '')
path_fits_originale = cerca_file_nel_progetto(BASE_DIR, nome_fits)

mask_si = np.char.startswith(tbl_ref['Corrispondenza'].astype(str), 'SI')
mask_no = tbl_ref['Corrispondenza'].astype(str) == 'NO'

tbl_si = tbl_ref[mask_si]
tbl_no = tbl_ref[mask_no]

found_stars_si = []
found_stars_no = []

for quad in quadrants:
    mask_region = quad['mask_func'](tbl_si['xcentroid'], tbl_si['ycentroid'])
    candidates = tbl_si[mask_region]
    if len(candidates) > 0:
        candidates = candidates[candidates['ripetizioni'] >= MIN_COUNT_RUN1]
    if len(candidates) > 0:
        idx_best = np.argmin(np.abs(candidates['flusso_fisso_max_run'] - KRON_TARGET))
        best_star = candidates[idx_best]
        found_stars_si.append({
            'name': quad['name'], 'color': quad['color'], 'star_id': best_star['ID'],
            'coords': (best_star['xcentroid'], best_star['ycentroid']), 'area': best_star['area'],
            'media_flusso_fisso_max_run': best_star['media_flusso_fisso_max_run'],
            'corrispondenza': best_star['Corrispondenza']
        })

for quad in quadrants:
    mask_region = quad['mask_func'](tbl_no['xcentroid'], tbl_no['ycentroid'])
    candidates = tbl_no[mask_region]
    if len(candidates) > 0:
        candidates = candidates[candidates['ripetizioni'] >= MIN_COUNT_RUN1_NO]
    if len(candidates) > 0:
        idx_best = np.argmin(np.abs(candidates['flusso_fisso_max_run'] - KRON_TARGET))
        best_star = candidates[idx_best]
        found_stars_no.append({
            'name': quad['name'], 'color': quad['color'], 'star_id': best_star['ID'],
            'coords': (best_star['xcentroid'], best_star['ycentroid']), 'area': best_star['area'],
            'media_flusso_fisso_max_run': best_star['media_flusso_fisso_max_run'],
            'corrispondenza': best_star['Corrispondenza']
        })

# =============================================================================
# --- FASE 1.5: SALVATAGGIO CUTOUT ---
# =============================================================================
if path_fits_originale and os.path.exists(path_fits_originale):
    print("\nGenerazione immagini cutout dal file FITS di riferimento...")
    with fits.open(str(path_fits_originale), memmap=False) as hdu_list:
        image_data = hdu_list[0].data
        _, median, _ = sigma_clipped_stats(image_data, sigma=3.0)
        data_sub = image_data - median
        wcs_ref = WCS(hdu_list[0].header)
        r_corr_px = 0.003349 / np.mean(proj_plane_pixel_scales(wcs_ref))
        for reg in found_stars_si + found_stars_no:
            salva_cutout(reg, data_sub, r_corr_px, wcs_ref, INDICE_IMMAGINE_RIFERIMENTO, run_list[RUN_REF],
                         cartella_output, parametri_seg, BASE_DIR)

# =============================================================================
# --- FASE 2: CURVE DI LUCE ---
# =============================================================================
tutti_gli_id = [reg['star_id'] for reg in found_stars_si] + [reg['star_id'] for reg in found_stars_no]
stars_data = {sid: {'times': [], 'flux': []} for sid in tutti_gli_id}
run_boundaries = []
t0_global = None

for run in run_list:
    c_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")
    if not os.path.exists(c_csv): continue
    f_csv = sorted([os.path.join(c_csv, f) for f in os.listdir(c_csv) if f.endswith('.csv')])

    for p_csv in f_csv:
        try:
            df = pd.read_csv(p_csv, comment='#')
            header = leggi_header_da_csv(p_csv)
            t_curr = header.get('TSTART', 0)
            if t0_global is None: t0_global = t_curr
            t_rel = (t_curr - t0_global) / 1000.0
            for sid in stars_data.keys():
                row = df[df['ID'] == sid]
                val = row.iloc[0]['flusso_fisso_max_run'] if not row.empty else 0.0
                stars_data[sid]['flux'].append(val)
                stars_data[sid]['times'].append(t_rel)
        except Exception:
            pass
    if stars_data[tutti_gli_id[0]]['times']:
        run_boundaries.append((run, stars_data[tutti_gli_id[0]]['times'][-1]))

# =============================================================================
# --- FASE 3: PLOTTING GLOBALE ---
# =============================================================================
n_plots = max(len(found_stars_si), len(found_stars_no), 1)
fig = plt.figure(figsize=(18, n_plots * 2.5))
gs = GridSpec(n_plots, 2, figure=fig)

for i, list_stars in enumerate([(found_stars_si, 0, "SI"), (found_stars_no, 1, "NO")]):
    for j, region in enumerate(list_stars[0]):
        ax = fig.add_subplot(gs[j, list_stars[1]])
        data = stars_data[region['star_id']]
        t, f = np.array(data['times']), np.array(data['flux'])
        mask = (f > 0)
        mean_val = np.mean(f[mask]) if np.any(mask) else 0
        std_perc = (np.std(f[mask]) / mean_val * 100) if mean_val > 0 else 0
        ax.plot(t[mask], f[mask], 'o-', markersize=3, color=region['color'], alpha=0.8,
                label=rf"Avg: {mean_val:.0f}, $\sigma$: {std_perc:.2f}%")
        ax.set_title(f"{list_stars[2]} - {region['name']} - ID: {region['star_id']}", fontsize=10, loc='left',
                     fontweight='bold', color=region['color'])
        for r_num, t_end in run_boundaries:
            ax.axvline(x=t_end, color='gray', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8);
        ax.grid(True, linestyle=':', alpha=0.6)
        if j == len(list_stars[0]) - 1: ax.set_xlabel("Tempo (s)")

plt.tight_layout()
nome_figura_finale = f"andamento_kron_tempo_quadranti_{KRON_TARGET}_run_{run_list[RUN_REF]}.png"
plt.savefig(cartella_output / nome_figura_finale, dpi=300, bbox_inches='tight')
print(f"\nSalvataggio grafico globale completato: {nome_figura_finale}")

# =============================================================================
# --- FASE 4: SALVATAGGIO CUTOUT PER TUTTI GLI OGGETTI NON CATALOGATI ---
# =============================================================================
print(
    f"\n--- FASE 4: Generazione riquadri per TUTTI gli oggetti non catalogati nell'immagine {INDICE_IMMAGINE_RIFERIMENTO} (Corrispondenza = NO) ---")
cartella_tutti_no = Path(
    f"tutti_gli_oggetti_non_catalogati_immagine_{INDICE_IMMAGINE_RIFERIMENTO:03d}_run_{run_list[RUN_REF]}")
cartella_tutti_no.mkdir(exist_ok=True, parents=True)

if path_fits_originale and os.path.exists(path_fits_originale):
    print(f"Inizio salvataggio di {len(tbl_no)} oggetti nella cartella '{cartella_tutti_no.name}'...")

    with fits.open(str(path_fits_originale), memmap=False) as hdu_list:
        image_data_all = hdu_list[0].data
        _, median_all, _ = sigma_clipped_stats(image_data_all, sigma=3.0)
        data_sub_all = image_data_all - median_all
        wcs_ref_all = WCS(hdu_list[0].header)
        r_corr_px_all = 0.003349 / np.mean(proj_plane_pixel_scales(wcs_ref_all))

        # ciclo su tutte le stelle presenti in tbl_no (quelle senza corrispondenza)
        for riga in tqdm(tbl_no, desc="Salvataggio Cutout NO Match"):
            reg_no = {
                'star_id': riga['ID'],
                'coords': (riga['xcentroid'], riga['ycentroid']),
                'area': riga['area'],
                'media_flusso_fisso_max_run': riga['media_flusso_fisso_max_run'],
                'corrispondenza': riga['Corrispondenza']
            }
            # passo la cartella di output specifica e il WCS per la ricostruzione
            salva_cutout(reg_no, data_sub_all, r_corr_px_all, wcs_ref_all, INDICE_IMMAGINE_RIFERIMENTO,
                         run_list[RUN_REF],
                         cartella_tutti_no, parametri_seg, BASE_DIR)