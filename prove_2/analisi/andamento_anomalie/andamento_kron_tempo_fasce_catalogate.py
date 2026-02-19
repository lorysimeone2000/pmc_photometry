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
from matplotlib.patches import Circle, Wedge
from matplotlib.gridspec import GridSpec
from pathlib import Path
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import simple_norm
import astropy.units as u

# sopprimo il warning FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)


# --- funzioni di utilità per la ricerca dinamica ---

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


def salva_cutout(region, data_sub, r_corr_px, tbl_ref, img_idx, run_id):
    # estraggo i dati della stella
    xc, yc = region['coords']
    area = region['area']
    star_id = region['star_id']
    media_flusso = region['media_flusso_fisso_max_run']
    corr = region['corrispondenza']

    # calcolo il lato del riquadro basato sulla radice dell'area
    side = np.sqrt(area) * 1.1
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

    fig, ax = plt.subplots(figsize=(5, 5))
    if cutout.size > 0:
        norm = simple_norm(cutout, 'log', percent=99.9)
        ax.imshow(cutout, cmap='gray_r', origin='lower', extent=[x_min, x_max, y_min, y_max], norm=norm)

    # identifico le stelle catalogate presenti in tutto il file CSV
    mask_si = np.char.startswith(tbl_ref['Corrispondenza'].astype(str), 'SI')
    tbl_cat = tbl_ref[mask_si]

    # le filtro tenendo solo quelle che cadono visivamente dentro il mio riquadro
    mask_in_box = (tbl_cat['xcentroid'] >= x_min) & (tbl_cat['xcentroid'] <= x_max) & \
                  (tbl_cat['ycentroid'] >= y_min) & (tbl_cat['ycentroid'] <= y_max)
    tbl_cat_box = tbl_cat[mask_in_box]

    # estraggo la magnitudine minima (valore più brillante) per fissare il limite inferiore della colorbar
    min_mag = np.nanmin(tbl_ref[mask_si]['Mag']) if len(tbl_ref[mask_si]) > 0 else 0

    if len(tbl_cat_box) > 0:
        # disegno i pallini con dimensione 4 e associo la colorbar invertita
        sc = ax.scatter(tbl_cat_box['xcentroid'], tbl_cat_box['ycentroid'], c=tbl_cat_box['Mag'],
                        cmap='viridis_r', vmin=min_mag, vmax=15, s=4, zorder=5)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mag')

    # traccio il cerchio di correlazione in giallo acceso
    circle = Circle((xc, yc), r_corr_px, edgecolor='#ffff00', facecolor='none', linewidth=1.5, zorder=10)
    ax.add_patch(circle)

    # se è un falso positivo, aggiungo una vistosa croce rossa sul centroide
    if str(corr) == 'NO':
        ax.plot(xc, yc, marker='+', color='red', markersize=15, markeredgewidth=2, zorder=15)

    ax.set_title(f"Oggetto {star_id}\nkron medio {media_flusso}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # strutturo il nome del file esattamente come richiesto
    nome_figura = f"{star_id}_immagine{img_idx}_run_{run_id}.png"
    plt.savefig(nome_figura, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  > Salvata immagine cutout: {nome_figura}")


# --- impostazione dinamica dei percorsi ---

# imposto la cartella base dinamicamente
BASE_DIR = trova_cartella_base("pmc_photometry")
print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")

# 1. cerco prima la cartella 'tabelle'
cartella_tabelle = cerca_cartella_nel_progetto(BASE_DIR, "tabelle")
if cartella_tabelle is None:
    print("ERRORE: Cartella 'tabelle' non trovata.")
    exit()

# 2. scendo nelle sottocartelle per trovare 'tabelle_unite'
base_path = str(cartella_tabelle / "tabelle_unite")
if not os.path.exists(base_path):
    print(f"ERRORE: La cartella {base_path} non esiste.")
    exit()

# --- parametri configurazione ---

# imposto la lista delle run da analizzare
run_list = [1, 2, 3]

KRON_TARGET = 115  # imposto il flusso target per la selezione
RUN_REF = 0
INDICE_IMMAGINE_RIFERIMENTO = 35  # imposto l'indice del file da usare per la selezione
MIN_COUNT_RUN1 = 75  # imposto il minimo numero di ripetizioni per essere selezionata

# definisco le dimensioni del sensore
H, W = 2048, 3072
CENTER_X, CENTER_Y = W / 2, H / 2
# uso il raggio circoscritto (ipotenusa) per coprire anche gli angoli
MAX_RADIUS = W / 2

# --- configurazione automatica fasce ---
NUM_FASCE = 5  # imposto il numero di suddivisioni del raggio totale
PERC_RIDUZIONE = 1 / NUM_FASCE  # imposto il 5% del RAGGIO TOTALE come spessore costante

# genero i colori dinamici (uno per fascia)
colors_map = plt.cm.jet(np.linspace(0, 1, NUM_FASCE))

# creo due liste separate per le regioni di sinistra (SI) e destra (NO)
regions_info_si = []
regions_info_no = []

print(f"--- ANALISI MULTI-ZONA (Target: {KRON_TARGET} ADU) ---")
print(f"Configurazione: {NUM_FASCE} zone (Anelli spessi {int(PERC_RIDUZIONE * 100)}% del raggio max).")
print(f"Filtro: Stelle presenti almeno {MIN_COUNT_RUN1} volte (in base alla colonna 'ripetizioni').")

for i in range(NUM_FASCE):
    # 1. calcolo il limite esterno della fascia i-esima
    r_outer = MAX_RADIUS * ((i + 1) / NUM_FASCE)

    # 2. calcolo il limite interno sottraendo una quantità FISSA
    r_inner = r_outer - (MAX_RADIUS * PERC_RIDUZIONE)

    # applico un controllo di sicurezza
    if r_inner < 0: r_inner = 0

    name = f"Anello {i + 1} ({int(r_inner)}-{int(r_outer)} px)"

    # aggiungo la regione per la colonna di sinistra (SI)
    regions_info_si.append({
        'name': name,
        'r_min': r_inner,
        'r_max': r_outer,
        'color': colors_map[i],
        'star_id': None,
        'star_flux': 0
    })

    # aggiungo la regione per la colonna di destra (NO)
    regions_info_no.append({
        'name': name,
        'r_min': r_inner,
        'r_max': r_outer,
        'color': colors_map[i],
        'star_id': None,
        'star_flux': 0
    })
    print(f"  - {name}")

# --- FASE 1: SELEZIONE STELLE NELL'IMMAGINE DI RIFERIMENTO ---

cartella_ref = os.path.join(base_path, f"tabelle_unite_run_{run_list[RUN_REF]}")
# cerco i file CSV
files_ref = sorted([f for f in os.listdir(cartella_ref) if f.endswith('.csv')])
if len(files_ref) <= INDICE_IMMAGINE_RIFERIMENTO:
    INDICE_IMMAGINE_RIFERIMENTO = 0
path_ref = os.path.join(cartella_ref, files_ref[INDICE_IMMAGINE_RIFERIMENTO])

print(f"File riferimento per coordinate: {os.path.basename(path_ref)}")
df_ref = pd.read_csv(path_ref, comment='#')
tbl_ref = Table.from_pandas(df_ref)

# recupero il path FITS originale dall'header del CSV per utilizzarlo successivamente nei cutout
header_ref_csv = leggi_header_da_csv(path_ref)
nome_fits = header_ref_csv.get('NOME_FILE_FITS', '')
path_fits_originale = cerca_file_nel_progetto(BASE_DIR, nome_fits)

# divido subito la tabella in base alla Corrispondenza
mask_si = np.char.startswith(tbl_ref['Corrispondenza'].astype(str), 'SI')
mask_no = tbl_ref['Corrispondenza'].astype(str) == 'NO'

tbl_si = tbl_ref[mask_si]
tbl_no = tbl_ref[mask_no]

# calcolo le distanze dal centro per entrambi i gruppi
dists_from_center_si = np.hypot(tbl_si['xcentroid'] - CENTER_X, tbl_si['ycentroid'] - CENTER_Y)
dists_from_center_no = np.hypot(tbl_no['xcentroid'] - CENTER_X, tbl_no['ycentroid'] - CENTER_Y)

found_stars_si = []
found_stars_no = []

print("\nCerco le stelle catalogate (Corrispondenza = SI) per la colonna di sinistra:")
for region in regions_info_si:
    mask_region = (dists_from_center_si >= region['r_min']) & (dists_from_center_si < region['r_max'])
    candidates = tbl_si[mask_region]

    if len(candidates) > 0:
        mask_count = candidates['ripetizioni'] >= MIN_COUNT_RUN1
        candidates = candidates[mask_count]

    if len(candidates) > 0:
        diffs = np.abs(candidates['flusso_fisso_max_run'] - KRON_TARGET)
        idx_best = np.argmin(diffs)
        best_star = candidates[idx_best]
        region['star_id'] = best_star['ID']
        region['star_flux'] = best_star['flusso_fisso_max_run']

        # aggiungo i dati geometrici per i cutout
        region['coords'] = (best_star['xcentroid'], best_star['ycentroid'])
        region['area'] = best_star['area']
        region['media_flusso_fisso_max_run'] = best_star['media_flusso_fisso_max_run']
        region['corrispondenza'] = best_star['Corrispondenza']

        found_stars_si.append(region)
        print(f"  > {region['name']}: Trovata ID {best_star['ID']} (Flux: {best_star['flusso_fisso_max_run']:.1f})")
    else:
        print(f"  > {region['name']}: NESSUNA stella SI trovata con count >= {MIN_COUNT_RUN1}.")

MIN_COUNT_RUN1_NO = 25

print("\nCerco i falsi positivi stabili (Corrispondenza = NO) per la colonna di destra:")
for region in regions_info_no:
    mask_region = (dists_from_center_no >= region['r_min']) & (dists_from_center_no < region['r_max'])
    candidates = tbl_no[mask_region]

    if len(candidates) > 0:
        mask_count_no = candidates['ripetizioni'] >= MIN_COUNT_RUN1_NO
        candidates = candidates[mask_count_no]

    if len(candidates) > 0:
        diffs = np.abs(candidates['flusso_fisso_max_run'] - KRON_TARGET)
        idx_best = np.argmin(diffs)
        best_star = candidates[idx_best]
        region['star_id'] = best_star['ID']
        region['star_flux'] = best_star['flusso_fisso_max_run']

        # aggiungo i dati geometrici per i cutout
        region['coords'] = (best_star['xcentroid'], best_star['ycentroid'])
        region['area'] = best_star['area']
        region['media_flusso_fisso_max_run'] = best_star['media_flusso_fisso_max_run']
        region['corrispondenza'] = best_star['Corrispondenza']

        found_stars_no.append(region)
        print(f"  > {region['name']}: Trovata ID {best_star['ID']} (Flux: {best_star['flusso_fisso_max_run']:.1f})")
    else:
        print(f"  > {region['name']}: NESSUNA stella NO trovata con count >= {MIN_COUNT_RUN1_NO}.")

# --- FASE 1.5: SALVATAGGIO CUTOUT DELLE STELLE SELEZIONATE ---
if path_fits_originale and os.path.exists(path_fits_originale):
    print("\nGenerazione immagini cutout dal file FITS di riferimento...")
    with fits.open(str(path_fits_originale), memmap=False) as hdu_list:
        image_data = hdu_list[0].data
        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
        data_sub = image_data - median
        wcs_ref = WCS(hdu_list[0].header)

        # converto la scala pixel in gradi per ottenere la dimensione in pixel dell'area di correlazione
        pixel_scales = proj_plane_pixel_scales(wcs_ref)
        pixel_scale = np.mean(pixel_scales) * u.deg
        r_corr_px = 0.003349 / pixel_scale.value

        # ciclo su tutte le stelle trovate (sia SI che NO)
        for reg in found_stars_si + found_stars_no:
            salva_cutout(reg, data_sub, r_corr_px, tbl_ref, INDICE_IMMAGINE_RIFERIMENTO, run_list[RUN_REF])

# --- FASE 2: ESTRAZIONE CURVE DI LUCE ---
# unisco tutti gli ID trovati per estrarre i dati in un colpo solo
tutti_gli_id = [reg['star_id'] for reg in found_stars_si] + [reg['star_id'] for reg in found_stars_no]
stars_data = {sid: {'times': [], 'flux': []} for sid in tutti_gli_id if sid is not None}

run_boundaries = []
t0_global = None
total_times = []

run_files_map = {}
for run in run_list:
    c_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")
    if os.path.exists(c_csv):
        f_csv = sorted([f for f in os.listdir(c_csv) if f.endswith('.csv')])
        run_files_map[run] = [os.path.join(c_csv, f) for f in f_csv]

print("\n>>> Estrazione curve di luce...")

for run in run_list:
    file_paths = run_files_map.get(run, [])
    if not file_paths: continue

    for p_csv in file_paths:
        try:
            df = pd.read_csv(p_csv, comment='#')
            header = leggi_header_da_csv(p_csv)
            t_curr = header.get('TSTART', 0)
            if t0_global is None: t0_global = t_curr
            t_rel = (t_curr - t0_global) / 1000.0

            # aggiorno il tempo globale solo la prima volta
            if len(stars_data[list(stars_data.keys())[0]]['times']) == len(total_times):
                total_times.append(t_rel)

            for star_id in stars_data.keys():
                row = df[df['ID'] == star_id]
                val = 0.0
                if not row.empty:
                    val = row.iloc[0]['flusso_fisso_max_run']
                stars_data[star_id]['flux'].append(val)
                stars_data[star_id]['times'].append(t_rel)
        except Exception:
            pass

    if total_times:
        run_boundaries.append((run, total_times[-1]))

# --- FASE 3: PLOTTING COMPLESSO (Curve SI + Curve NO) ---

# configuro il layout dinamico con due colonne della stessa dimensione
n_plots = max(len(found_stars_si), len(found_stars_no), 1)
fig_height = max(8, n_plots * 2.5)
fig = plt.figure(figsize=(18, fig_height))
gs = GridSpec(n_plots, 2, figure=fig)

# --- COLONNA SINISTRA: Curve di Luce Catalogate (SI) ---
for i, region in enumerate(found_stars_si):
    ax_left = fig.add_subplot(gs[i, 0])

    sid = region['star_id']
    data = stars_data[sid]
    t_arr = np.array(data['times'])
    f_arr = np.array(data['flux'])
    mask = (f_arr > 0) & (~np.isnan(f_arr))

    mean_val, perc_err = 0, 0
    if np.sum(mask) > 0:
        mean_val = np.mean(f_arr[mask])
        std_val = np.std(f_arr[mask])
        perc_err = (std_val / mean_val) * 100

    col = region['color']
    ax_left.plot(t_arr[mask], f_arr[mask], marker='o', linestyle='-', linewidth=0.8, markersize=3,
                 color=col, alpha=0.8, label=rf"Avg: {mean_val:.0f}, $\sigma$: {perc_err:.2f}%")

    ax_left.set_title(f"SI - {region['name']} - ID: {sid}", fontsize=10, loc='left', fontweight='bold', color=col)

    # aggiungo le linee per delimitare le run
    for r_idx, (r_num, t_end) in enumerate(run_boundaries):
        if r_idx < len(run_boundaries):
            ax_left.axvline(x=t_end, color='gray', linestyle='--', alpha=0.5)
            y_min, y_max = ax_left.get_ylim()
            ax_left.text(t_end, y_max, f"Fine Run {r_num}", rotation=90, ha='right', va='top', color='#333333',
                         fontsize=7)

    ax_left.set_ylabel("Flusso Fisso")
    ax_left.grid(True, linestyle=':', alpha=0.6)
    ax_left.legend(loc='upper right', fontsize=8)

    if i < len(found_stars_si) - 1:
        ax_left.set_xticklabels([])
    else:
        ax_left.set_xlabel("Tempo dall'inizio Run 1 (s)")

# --- COLONNA DESTRA: Curve di Luce Falsi Positivi (NO) ---
for i, region in enumerate(found_stars_no):
    ax_right = fig.add_subplot(gs[i, 1])

    sid = region['star_id']
    data = stars_data[sid]
    t_arr = np.array(data['times'])
    f_arr = np.array(data['flux'])
    mask = (f_arr > 0) & (~np.isnan(f_arr))

    mean_val, perc_err = 0, 0
    if np.sum(mask) > 0:
        mean_val = np.mean(f_arr[mask])
        std_val = np.std(f_arr[mask])
        perc_err = (std_val / mean_val) * 100

    col = region['color']
    ax_right.plot(t_arr[mask], f_arr[mask], marker='o', linestyle='-', linewidth=0.8, markersize=3,
                  color=col, alpha=0.8, label=rf"Avg: {mean_val:.0f}, $\sigma$: {perc_err:.2f}%")

    ax_right.set_title(f"NO - {region['name']} - ID: {sid}", fontsize=10, loc='left', fontweight='bold', color=col)

    # aggiungo le linee per delimitare le run
    for r_idx, (r_num, t_end) in enumerate(run_boundaries):
        if r_idx < len(run_boundaries):
            ax_right.axvline(x=t_end, color='gray', linestyle='--', alpha=0.5)
            y_min, y_max = ax_right.get_ylim()
            ax_right.text(t_end, y_max, f"Fine Run {r_num}", rotation=90, ha='right', va='top', color='#333333',
                          fontsize=7)

    ax_right.set_ylabel("Flusso Fisso")
    ax_right.grid(True, linestyle=':', alpha=0.6)
    ax_right.legend(loc='upper right', fontsize=8)

    if i < len(found_stars_no) - 1:
        ax_right.set_xticklabels([])
    else:
        ax_right.set_xlabel("Tempo dall'inizio Run 1 (s)")

fig.suptitle(f'Analisi Stabilità Flusso per {NUM_FASCE} Anelli (kron di riferimento: {KRON_TARGET})',
             fontsize=16, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(wspace=0.15, hspace=0.15)

# salvo la figura impostando il nome dinamicamente in base al kron di riferimento
nome_figura = f"andamento_kron_tempo_fasce_{KRON_TARGET}_run_{RUN_REF + 1}.png"
plt.savefig(nome_figura, dpi=300, bbox_inches='tight')
print(f"\nSalvataggio completato: {nome_figura}")

plt.show()