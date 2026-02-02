import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.table import Table
import warnings
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import FITSFixedWarning
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Wedge
from matplotlib.gridspec import GridSpec

# Sopprime il warning FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)


# --- FUNZIONI DI UTILITÀ ---

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
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False
    return valore


def leggi_header_da_csv(filename):
    """Legge l'header FITS salvato nelle prime righe del file CSV."""
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


# --- PARAMETRI CONFIGURAZIONE ---

run_list = [1, 2, 3]  # Lista delle run da analizzare
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"

# Percorso del file che contiene i CONTEGGI TOTALI della Run 1
path_stats_run1 = "/home/lorysimeone/tesi_magistrale/prove_2/analisi/studio_flussi/dispersione_flussi/risultati_analisi_run_1.csv"

KRON_TARGET = 700  # Flusso target per la selezione
INDICE_IMMAGINE_RIFERIMENTO = 35  # Indice file da usare per la selezione
MIN_COUNT_RUN1 = 75  # Minimo numero di occorrenze nella Run 1 per essere selezionata

# Dimensioni sensore
H, W = 2048, 3072
CENTER_X, CENTER_Y = W / 2, H / 2
# Usa il raggio circoscritto (ipotenusa) per coprire anche gli angoli
MAX_RADIUS = W/2

# --- CONFIGURAZIONE AUTOMATICA FASCE ---
NUM_FASCE = 5  # Numero di suddivisioni del raggio totale
PERC_RIDUZIONE = 0.05  # 5% del RAGGIO TOTALE (Spessore costante)

# Generazione colori dinamici (uno per fascia)
colors_map = plt.cm.jet(np.linspace(0, 1, NUM_FASCE))

regions_info = []

print(f"--- ANALISI MULTI-ZONA (Target: {KRON_TARGET} ADU) ---")
print(f"Configurazione: {NUM_FASCE} zone (Anelli spessi {int(PERC_RIDUZIONE * 100)}% del raggio max).")
print(f"Filtro: Stelle presenti almeno {MIN_COUNT_RUN1} volte nella Run 1.")

for i in range(NUM_FASCE):
    # 1. Calcola il limite esterno della fascia i-esima
    r_outer = MAX_RADIUS * ((i + 1) / NUM_FASCE)

    # 2. Calcola il limite interno sottraendo una quantità FISSA
    r_inner = r_outer - (MAX_RADIUS * PERC_RIDUZIONE)

    # Safety check
    if r_inner < 0: r_inner = 0

    name = f"Anello {i + 1} ({int(r_inner)}-{int(r_outer)} px)"

    regions_info.append({
        'name': name,
        'r_min': r_inner,
        'r_max': r_outer,
        'color': colors_map[i],
        'star_id': None,
        'star_flux': 0
    })
    print(f"  - {name}")

# --- FASE 1: CARICAMENTO STATISTICHE E FILTRO ID ---

if os.path.exists(path_stats_run1):
    print(f"Caricamento statistiche Run 1: {os.path.basename(path_stats_run1)}")
    df_stats = pd.read_csv(path_stats_run1, comment='#')

    # Filtro gli ID che hanno count >= MIN_COUNT_RUN1
    # Assumiamo che la colonna sia 'count_flusso_fisso_max_run' come nei file precedenti
    if 'count_flusso_fisso_max_run' in df_stats.columns:
        valid_ids = df_stats[df_stats['count_flusso_fisso_max_run'] >= MIN_COUNT_RUN1]['ID'].values
        print(f"  > Trovate {len(valid_ids)} stelle con count >= {MIN_COUNT_RUN1}")
    else:
        print("ERRORE: Colonna 'count_flusso_fisso_max_run' non trovata nel file statistiche.")
        # Se non trovi la colonna, puoi decidere di uscire o continuare senza filtro (qui esco per sicurezza)
        exit()
else:
    print(f"ERRORE: File statistiche non trovato: {path_stats_run1}")
    exit()

# --- FASE 2: SELEZIONE STELLE NELL'IMMAGINE DI RIFERIMENTO ---

cartella_ref = os.path.join(base_path, f"tabelle_unite_run_{run_list[0]}")
files_ref = sorted([f for f in os.listdir(cartella_ref) if f.endswith('.csv')])
if len(files_ref) <= INDICE_IMMAGINE_RIFERIMENTO:
    INDICE_IMMAGINE_RIFERIMENTO = 0
path_ref = os.path.join(cartella_ref, files_ref[INDICE_IMMAGINE_RIFERIMENTO])

print(f"File riferimento per coordinate: {os.path.basename(path_ref)}")
df_ref = pd.read_csv(path_ref, comment='#')
tbl_ref = Table.from_pandas(df_ref)

# Recupero path FITS originale dall'header del CSV
header_ref_csv = leggi_header_da_csv(path_ref)
path_fits_originale = header_ref_csv.get('PERCORSO_FILE', '')

# Filtro solo catalogate
mask_cat = np.char.startswith(tbl_ref['Corrispondenza'].astype(str), 'SI')
tbl_valid = tbl_ref[mask_cat]

# Calcolo distanze dal centro
dists_from_center = np.hypot(tbl_valid['xcentroid'] - CENTER_X, tbl_valid['ycentroid'] - CENTER_Y)

found_stars = []

for region in regions_info:
    # 1. Filtro Spaziale (Anello)
    mask_region = (dists_from_center >= region['r_min']) & (dists_from_center < region['r_max'])
    candidates = tbl_valid[mask_region]

    # 2. Filtro Count (Intersezione con valid_ids)
    if len(candidates) > 0:
        mask_count = np.isin(candidates['ID'], valid_ids)
        candidates = candidates[mask_count]

    if len(candidates) > 0:
        # Trova la stella più vicina al target di flusso
        diffs = np.abs(candidates['flusso_fisso_max_run'] - KRON_TARGET)
        idx_best = np.argmin(diffs)
        best_star = candidates[idx_best]

        region['star_id'] = best_star['ID']
        region['star_flux'] = best_star['flusso_fisso_max_run']
        # Salviamo le coordinate per plottarle sull'immagine
        region['coords'] = (best_star['xcentroid'], best_star['ycentroid'])
        found_stars.append(region)
        print(f"  > {region['name']}: Trovata ID {best_star['ID']} (Flux: {best_star['flusso_fisso_max_run']:.1f})")
    else:
        print(f"  > {region['name']}: NESSUNA stella trovata con count >= {MIN_COUNT_RUN1}.")

if not found_stars:
    print("Nessuna stella trovata in nessuna fascia. Esco.")
    exit()

# --- FASE 3: ESTRAZIONE CURVE DI LUCE ---
stars_data = {reg['star_id']: {'times': [], 'flux': []} for reg in found_stars}
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

    start_idx = len(total_times)

    for n, p_csv in enumerate(file_paths):
        try:
            df = pd.read_csv(p_csv, comment='#')
            header = leggi_header_da_csv(p_csv)
            t_curr = header.get('TSTART', 0)
            if t0_global is None: t0_global = t_curr
            t_rel = (t_curr - t0_global) / 1000.0

            if len(stars_data[found_stars[0]['star_id']]['times']) == len(total_times):
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

# --- FASE 4: PLOTTING COMPLESSO (Curve + Immagine) ---

# Configurazione Layout Dinamico
n_plots = len(found_stars)
fig_height = max(8, n_plots * 2.5)
fig = plt.figure(figsize=(18, fig_height))
gs = GridSpec(n_plots, 2, width_ratios=[2, 1.2], figure=fig)

# --- COLONNA SINISTRA: Curve di Luce ---
axs_curves = []

for i, region in enumerate(found_stars):
    ax = fig.add_subplot(gs[i, 0])
    axs_curves.append(ax)

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
    ax.plot(t_arr[mask], f_arr[mask], marker='o', linestyle='-', linewidth=0.8, markersize=3,
            color=col, alpha=0.8, label=rf"Avg: {mean_val:.0f}, $\sigma$: {perc_err:.2f}%")

    ax.set_title(f"{region['name']} - ID: {sid}", fontsize=10, loc='left', fontweight='bold', color=col)

    # Linee Run
    for r_idx, (r_num, t_end) in enumerate(run_boundaries):
        if r_idx < len(run_boundaries):
            ax.axvline(x=t_end, color='gray', linestyle='--', alpha=0.5)
            y_min, y_max = ax.get_ylim()
            ax.text(t_end, y_max, f"Fine Run {r_num}", rotation=90, ha='right', va='top', color='#333333', fontsize=7)

    ax.set_ylabel("Flusso Fisso")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)

    if i < n_plots - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Tempo dall'inizio Run 1 (s)")

# --- COLONNA DESTRA: Immagine FITS con Regioni ---
ax_img = fig.add_subplot(gs[:, 1])

try:
    if os.path.exists(path_fits_originale):
        with fits.open(path_fits_originale) as hdu_list:
            image_data = hdu_list[0].data
            mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
            data_sub = image_data - median

            im = ax_img.imshow(data_sub, cmap="gray_r", norm=LogNorm(), interpolation='nearest', origin='lower')

            for i, region in enumerate(found_stars):
                col = region['color']
                r_in = region['r_min']
                r_out = region['r_max']
                width = r_out - r_in

                annulus = Wedge((CENTER_X, CENTER_Y), r_out, 0, 360, width=width,
                                facecolor=col, alpha=0.3, edgecolor=col, linewidth=1)
                ax_img.add_patch(annulus)

                sx, sy = region['coords']
                circle_star = Circle((sx, sy), radius=40, edgecolor=col, facecolor='none', linewidth=2, linestyle='-')
                ax_img.add_patch(circle_star)
                ax_img.text(sx + 50, sy + 50, f"Star {i + 1}", color=col, fontsize=9, fontweight='bold')

            ax_img.plot(CENTER_X, CENTER_Y, 'rx', markersize=10)

            ax_img.set_title(
                f"Zone Monitoraggio ({NUM_FASCE} anelli costanti)\nsull'immagine {INDICE_IMMAGINE_RIFERIMENTO}, run {run_list[0]}",
                fontsize=12)
            ax_img.set_xlim(0, W)
            ax_img.set_ylim(0, H)

    else:
        ax_img.text(0.5, 0.5, "File FITS non trovato", ha='center', va='center')

except Exception as e:
    ax_img.text(0.5, 0.5, f"Errore FITS: {e}", ha='center', va='center')

fig.suptitle(f'Analisi Stabilità Flusso per {NUM_FASCE} Anelli (kron di riferimento: {KRON_TARGET})',
             fontsize=16, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(wspace=0.15, hspace=0.15)
plt.show()