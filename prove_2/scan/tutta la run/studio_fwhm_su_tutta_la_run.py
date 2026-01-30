import pandas as pd
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel, SourceCatalog, SourceFinder
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import sys
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
from astropy.wcs import FITSFixedWarning
import matplotlib.ticker as ticker
from tqdm import tqdm

# --- IMPORT FONDAMENTALE PER LA PORTABILITÀ ---
from pathlib import Path

# --- GESTIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='Units from inserted quantities will be ignored.')

# Registra il tempo di inizio globale
start_time_global = time.time()


# =============================================================================
# 0. GESTIONE PERCORSI DINAMICA (PORTABILITÀ TOTALE)
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    """
    Risale la directory partendo dalla posizione dello script fino a trovare
    la cartella target (es. 'pmc_photometry').
    """
    path_corrente = Path(__file__).resolve()

    # Risaliamo fino a trovare la cartella target
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent

    # Fallback: se non la trova, usa la cartella dello script
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    """
    Cerca una CARTELLA ricorsivamente in tutte le sottocartelle di base_dir.
    """
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]

    if not cartelle_trovate:
        return None

    cartelle_trovate.sort(key=lambda p: len(str(p)))

    if len(cartelle_trovate) > 1:
        # Preferisci quella più vicina alla root se ce ne sono multiple
        pass

    return cartelle_trovate[0]


# Definisco la BASE_DIR dinamicamente
BASE_DIR = trova_cartella_base("pmc_photometry")

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")


# --- FUNZIONI DI UTILITÀ ---

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
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False
    return valore


def leggi_header_da_csv(filename):
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


def elabora_file_fits(percorso_file):
    # memmap=False per sicurezza su file network/condivisi
    with fits.open(percorso_file, memmap=False) as hdu:
        header = hdu[0].header
        data = hdu[0].data
        wcs = WCS(header)
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        data_sub = data - median
        return data_sub, wcs, median


def esegui_segmentazione_dinamica(data, fwhm, size, params):
    try:
        kernel = make_2dgaussian_kernel(fwhm, size=size)
        convolved_data = convolve(data, kernel)
    except Exception:
        return None

    finder = SourceFinder(npixels=params['pixel'], progress_bar=False)
    segment_map = finder(convolved_data, params['threshold_assoluta'])

    if segment_map is None: return None

    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    # Filtro rapido vettorializzato
    soglia_ass = params['soglia_filtro_ass']
    soglia_rel = params['soglia_filtro_rel']
    bordo = 7
    ny, nx = data.shape

    indici_validi = []

    for i, sorgente in enumerate(tbl):
        label = sorgente['label']
        mask_sorgente = (segment_map.data == label)
        valori_originali = data[mask_sorgente]
        xcentroid = sorgente['xcentroid']
        ycentroid = sorgente['ycentroid']

        dentro_riquadro = (xcentroid >= bordo) and (xcentroid < nx - bordo) and \
                          (ycentroid >= bordo) and (ycentroid < ny - bordo)

        if not dentro_riquadro:
            continue

        if np.sum(valori_originali > soglia_ass) >= 3:
            if np.sum(valori_originali > soglia_rel * sorgente['max_value']) >= 3:
                indici_validi.append(i)

    return tbl[indici_validi]


def filtra_vicini_saturi(tbl, median_bg, dist_limit=30, val_sat=254):
    if tbl is None or len(tbl) == 0: return tbl
    mask_sature = tbl['max_value'] >= val_sat
    idx_sature = np.where(mask_sature)[0]
    if len(idx_sature) == 0: return tbl

    x = tbl['xcentroid']
    y = tbl['ycentroid']
    to_remove = set()
    for i_sat in idx_sature:
        dists = np.hypot(x - x[i_sat], y - y[i_sat])
        vicini = np.where(dists <= dist_limit)[0]
        for v in vicini:
            if v != i_sat and not mask_sature[v]:
                to_remove.add(v)

    if len(to_remove) > 0:
        mask_keep = np.ones(len(tbl), dtype=bool)
        mask_keep[list(to_remove)] = False
        return tbl[mask_keep]
    return tbl


def unione_tabelle_ottimizzata(tbl_seg, tbl_cat, wcs, coords_catalogate, soglia_correlazione, mag_limit=10.0):
    """
    Esegue il matching e restituisce la tabella unita e il numero di stelle UNICHE trovate
    che soddisfano il criterio di magnitudine < mag_limit.
    """
    coords_trovate = wcs.pixel_to_world(tbl_seg['xcentroid'], tbl_seg['ycentroid'])

    # --- MATCHING CON SEARCH_AROUND_SKY ---
    idx_trovate, idx_catalogate, d2d, _ = coords_catalogate.search_around_sky(coords_trovate, soglia_correlazione)

    df_catalogate = tbl_cat.to_pandas()
    df_trovate = tbl_seg.to_pandas()

    matches = pd.DataFrame({
        'idx_t': idx_trovate,
        'idx_c': idx_catalogate,
        'dist': d2d.deg,
        'mag': df_catalogate.iloc[idx_catalogate]['Mag'].values
    })

    # --- LOGICA RANK ---
    matches.sort_values(by=['idx_t', 'mag'], inplace=True)
    matches['rank'] = matches.groupby('idx_t').cumcount() + 1
    matches['Corrispondenza'] = 'SI (Rank ' + matches['rank'].astype(str) + ')'

    # --- COSTRUZIONE TABELLA FINALE ---

    # A. Match SI
    part_trovate = df_trovate.iloc[matches['idx_t']].reset_index(drop=True)
    part_catalogate = df_catalogate.iloc[matches['idx_c']].reset_index(drop=True)
    part_rank = matches[['Corrispondenza']].reset_index(drop=True)

    df_si = pd.concat([part_trovate, part_rank, part_catalogate], axis=1)

    # B. Match NO
    all_indices = set(range(len(df_trovate)))
    matched_indices = set(matches['idx_t'].unique())
    unmatched_indices = list(all_indices - matched_indices)

    if unmatched_indices:
        df_no = df_trovate.iloc[unmatched_indices].copy()
        df_no['Corrispondenza'] = 'NO'

        for col in df_catalogate.columns:
            if col == 'Catalogo':
                df_no[col] = 'N/A'
            elif pd.api.types.is_integer_dtype(df_catalogate[col]):
                df_no[col] = -999
            elif pd.api.types.is_float_dtype(df_catalogate[col]):
                df_no[col] = np.nan
            else:
                df_no[col] = 'N/A'

        # Allinea colonne
        df_no = df_no.reindex(columns=df_si.columns, fill_value=np.nan)

    else:
        df_no = pd.DataFrame(columns=df_si.columns)

    # C. Unione
    df_finale = pd.concat([df_si, df_no], ignore_index=True)

    if 'label' in df_finale.columns:
        df_finale.sort_values('label', inplace=True)

    # --- RIORDINAMENTO FINALE (Catalogo prima di ID) ---
    colonne = df_finale.columns.tolist()
    if 'ID' in colonne and 'Catalogo' in colonne:
        colonne.remove('Catalogo')
        pos_id = colonne.index('ID')
        colonne.insert(pos_id, 'Catalogo')
        df_finale = df_finale[colonne]

    tbl_unite = Table.from_pandas(df_finale)

    # --- CONTEGGIO STELLE BRILLANTI TROVATE ---

    # 1. Identifichiamo gli ID trovati
    mask_si = np.char.startswith(tbl_unite['Corrispondenza'].astype(str), 'SI')
    ids_trovati_e_correlati = set(tbl_unite[mask_si]['ID'])

    # 2. Filtriamo il catalogo per le brillanti
    mask_bright_cat = tbl_cat['Mag'] < mag_limit
    stelle_catalogo_brillanti = tbl_cat[mask_bright_cat]

    # 3. Contiamo quante brillanti sono state trovate (intersezione)
    num_brillanti = 0
    for star_id in stelle_catalogo_brillanti['ID']:
        if star_id in ids_trovati_e_correlati:
            num_brillanti += 1

    # --- RITORNO COORDINATE E TABELLA PER LOGICA FP ---
    # Restituisco anche coords_trovate per facilitare il recupero delle RA/DEC dei NO
    return tbl_unite, num_brillanti, coords_trovate


# --- CONFIGURAZIONE GLOBALE ---
PARAMETRI_FISSI = {
    'threshold_assoluta': 3.61,
    'pixel': 3,
    'soglia_filtro_ass': 2.5,
    'soglia_filtro_rel': 0.05
}

# Parametri Scan
FWHM_RANGE = np.linspace(2., 3.5, 50)
SIZES_TO_TEST = [3, 5]
soglia_correlazione = 0.003349 * u.deg
MAG_LIMIT_ANALYSIS = 10.0  # Soglia per definire le stelle "importanti" (Stelle Perse < 10)


def analizza_singola_run(run_id):
    """
    Funzione principale che esegue l'intera analisi per una singola run.
    """
    print(f"\n{'=' * 60}")
    print(f"AVVIO ANALISI PER RUN {run_id}")
    print(f"{'=' * 60}")

    # --- RICERCA DINAMICA CARTELLA CSV ---
    nome_cartella_csv = f"sorgenti_catalogate_run_{run_id}"
    cartella_csv_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_csv)

    if cartella_csv_path is None:
        print(f"ERRORE: Cartella '{nome_cartella_csv}' non trovata in {BASE_DIR}")
        return

    print(f"Cartella sorgenti rilevata: {cartella_csv_path.relative_to(BASE_DIR)}")
    file_csv_cat = sorted([f for f in cartella_csv_path.glob('*.csv')])

    if not file_csv_cat:
        print(f"ATTENZIONE: Nessun file CSV trovato in {cartella_csv_path}")
        return

    # --- RESET STRUTTURE DATI PER QUESTA RUN ---
    raw_corr = {size: [[] for _ in range(len(FWHM_RANGE))] for size in SIZES_TO_TEST}
    raw_fp = {size: [[] for _ in range(len(FWHM_RANGE))] for size in SIZES_TO_TEST}

    # --- NUOVA STRUTTURA: Storage per le coordinate dei Falsi Positivi ---
    # fp_coords_storage[size][fwhm_idx] = lista di tuple (ra_arr, dec_arr) una per ogni immagine
    fp_coords_storage = {size: [[] for _ in range(len(FWHM_RANGE))] for size in SIZES_TO_TEST}

    # Contatore per le immagini effettivamente processate
    immagini_processate_correttamente = 0
    totale_immagini = len(file_csv_cat)

    # --- CALCOLO TOTALE PASSAGGI ---
    total_steps = totale_immagini * len(SIZES_TO_TEST) * len(FWHM_RANGE)

    print(f"Analisi focalizzata su stelle perse con Mag < {MAG_LIMIT_ANALYSIS}")
    print(f"Totale immagini: {totale_immagini}")
    print(f"Totale iterazioni previste: {total_steps}")
    print("-" * 60)

    # --- SETUP BARRA DI AVANZAMENTO TQDM ---
    pbar = tqdm(total=total_steps, desc=f"Run {run_id} Progress", unit="step")

    # --- CICLO SULLE IMMAGINI (ESTERNO) ---
    for idx_img, percorso_csv in enumerate(file_csv_cat):

        success = False

        try:
            header_info = leggi_header_da_csv(percorso_csv)
            percorso_fits_str = header_info.get('PERCORSO_FILE', '')

            # --- LOGICA RISOLUZIONE PATH FITS (PORTABILITÀ) ---
            if not os.path.exists(percorso_fits_str):
                p_obj = Path(percorso_fits_str)
                try:
                    if "pmc_photometry" in p_obj.parts:
                        idx = p_obj.parts.index("pmc_photometry")
                        new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                        if new_path.exists(): percorso_fits_str = str(new_path)
                    elif "prove_2" in p_obj.parts:
                        idx = p_obj.parts.index("prove_2")
                        new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                        if new_path.exists(): percorso_fits_str = str(new_path)
                except:
                    pass

            if os.path.exists(percorso_fits_str):
                dataframe = pd.read_csv(percorso_csv, comment='#')
                tbl_catalogate = Table.from_pandas(dataframe)

                if hasattr(tbl_catalogate['RAJ2000'], 'value'):
                    ra_vals, dec_vals = tbl_catalogate['RAJ2000'].value, tbl_catalogate['DEJ2000'].value
                else:
                    ra_vals, dec_vals = np.array(tbl_catalogate['RAJ2000']), np.array(tbl_catalogate['DEJ2000'])
                coords_catalogate = SkyCoord(ra=ra_vals * u.deg, dec=dec_vals * u.deg)

                data_sub, wcs, median_val = elabora_file_fits(percorso_fits_str)
                success = True
            else:
                pass

        except Exception:
            pass

        if success:
            immagini_processate_correttamente += 1

            # --- CALCOLO TARGET: Quante stelle < 10 ci sono nel catalogo di questa immagine? ---
            num_target_stars = np.sum(tbl_catalogate['Mag'] < MAG_LIMIT_ANALYSIS)

            # --- CICLO SUI PARAMETRI (INTERNO ALL'IMMAGINE) ---
            for size_val in SIZES_TO_TEST:
                for i_fwhm, fwhm_val in enumerate(FWHM_RANGE):

                    tbl_trovate = esegui_segmentazione_dinamica(data_sub, fwhm=fwhm_val, size=size_val,
                                                                params=PARAMETRI_FISSI)

                    val_perse = num_target_stars
                    val_fp_raw = 0

                    # Coordinate FP correnti (default vuote)
                    current_fp_ra = np.array([])
                    current_fp_dec = np.array([])

                    if tbl_trovate is not None and len(tbl_trovate) > 0:
                        tbl_trovate = filtra_vicini_saturi(tbl_trovate, median_bg=median_val)

                        # Match con il catalogo
                        tbl_matched, num_found_bright, coords_trovate_obj = unione_tabelle_ottimizzata(
                            tbl_trovate,
                            tbl_catalogate,
                            wcs,
                            coords_catalogate,
                            soglia_correlazione,
                            mag_limit=MAG_LIMIT_ANALYSIS
                        )

                        # --- CALCOLO STELLE PERSE (Solo quelle < 10) ---
                        val_perse = max(0, num_target_stars - num_found_bright)

                        # --- IDENTIFICAZIONE FP (Provvisoria) ---
                        mask_no = tbl_matched['Corrispondenza'] == 'NO'
                        val_fp_raw = np.sum(mask_no)

                        # --- RECUPERO COORDINATE FP PER FILTRAGGIO SUCCESSIVO ---
                        if val_fp_raw > 0:
                            # Nota: tbl_matched è ordinata come df_finale.
                            # Dobbiamo assicurarci di estrarre le coordinate corrette.
                            # La funzione unione_tabelle_ottimizzata ritorna coords_trovate (tutte le trovate)
                            # Ma tbl_matched potrebbe aver cambiato ordine.
                            # Tuttavia, se guardiamo unione_tabelle_ottimizzata, i 'NO' sono appesi in fondo
                            # o riordinati per label.
                            # Più robusto: ricalcoliamo RA/DEC dai centroidi della tabella matched filtrata NO.
                            tbl_fp = tbl_matched[mask_no]
                            coords_fp = wcs.pixel_to_world(tbl_fp['xcentroid'], tbl_fp['ycentroid'])
                            current_fp_ra = coords_fp.ra.deg
                            current_fp_dec = coords_fp.dec.deg

                    # Appendiamo i risultati
                    raw_corr[size_val][i_fwhm].append(val_perse)

                    # Salviamo il conto RAW (sarà sovrascritto dopo)
                    raw_fp[size_val][i_fwhm].append(val_fp_raw)

                    # Salviamo le coordinate per il post-processing
                    fp_coords_storage[size_val][i_fwhm].append((current_fp_ra, current_fp_dec))

                    # Aggiorna la barra di avanzamento di 1 step
                    pbar.update(1)

            # Aggiorna la descrizione della barra per mostrare a che immagine siamo
            pbar.set_postfix_str(f"Img {idx_img + 1}/{totale_immagini}")

        else:
            # Se l'immagine viene saltata, aggiorniamo comunque la barra per non falsare l'ETA
            step_saltati = len(SIZES_TO_TEST) * len(FWHM_RANGE)
            pbar.update(step_saltati)
            pbar.set_postfix_str(f"Img {idx_img + 1} SKIPPED (File not found)")

    pbar.close()  # Chiude la barra alla fine della run

    print(f"\nRun {run_id} terminata (Scan Immagini). Avvio Filtraggio Transienti...")

    # =========================================================================
    # --- FASE DI POST-PROCESSING: FILTRAGGIO Falsi Positivi (Logica Ripetizione) ---
    # =========================================================================

    if immagini_processate_correttamente > 0:
        for size_val in SIZES_TO_TEST:
            for i_fwhm in range(len(FWHM_RANGE)):

                # Lista di tuple (ra, dec) per ogni immagine processata per questo set di parametri
                list_of_img_coords = fp_coords_storage[size_val][i_fwhm]

                # Appiattiamo tutto per il cross-match globale
                all_ra = []
                all_dec = []
                map_idx_to_img = {}  # Mappa indice globale -> indice immagine originale

                global_counter = 0
                for img_idx, (ra_arr, dec_arr) in enumerate(list_of_img_coords):
                    if len(ra_arr) > 0:
                        all_ra.extend(ra_arr)
                        all_dec.extend(dec_arr)
                        for _ in range(len(ra_arr)):
                            map_idx_to_img[global_counter] = img_idx
                            global_counter += 1

                # Lista dei nuovi conteggi FP filtrati (inizializzata a 0 per ogni immagine)
                filtered_fp_counts = [0] * len(list_of_img_coords)

                if len(all_ra) > 0:
                    c_tot = SkyCoord(ra=all_ra * u.deg, dec=all_dec * u.deg)
                    dist_limit = 0.0011 * u.deg  # Soglia spaziale transienti

                    # Search around sky su se stesso
                    idx1, idx2, _, _ = c_tot.search_around_sky(c_tot, dist_limit)

                    indices_with_valid_neighbor = set()

                    for k in range(len(idx1)):
                        i1 = idx1[k]
                        i2 = idx2[k]
                        if i1 == i2: continue  # Se stesso

                        # Verifica se il vicino è in un'immagine diversa
                        img1 = map_idx_to_img[i1]
                        img2 = map_idx_to_img[i2]

                        if img1 != img2:
                            indices_with_valid_neighbor.add(i1)

                    # Conteggio finale: solo chi ha un vicino valido è un "FP Reale" (non transiente)
                    # Quelli isolati (transienti) vengono scartati
                    for idx_globale in indices_with_valid_neighbor:
                        img_origin = map_idx_to_img[idx_globale]
                        filtered_fp_counts[img_origin] += 1

                # --- SOVRASCRITTURA DI raw_fp ---
                # Sostituiamo i conti raw con quelli filtrati
                raw_fp[size_val][i_fwhm] = filtered_fp_counts

    print(f"Filtraggio Transienti completato.")
    print(f"Immagini valide: {immagini_processate_correttamente}/{totale_immagini}")

    # --- CALCOLO MEDIA E DEVIAZIONE STANDARD ---
    results_mean_corr = {}
    results_std_corr = {}
    results_mean_fp = {}
    results_std_fp = {}

    if immagini_processate_correttamente > 0:
        for size_val in SIZES_TO_TEST:
            mean_c = []
            std_c = []
            mean_f = []
            std_f = []

            for i_fwhm in range(len(FWHM_RANGE)):
                vals_c = raw_corr[size_val][i_fwhm]
                vals_f = raw_fp[size_val][i_fwhm]  # Ora contiene i valori filtrati

                mean_c.append(np.mean(vals_c))
                std_c.append(np.std(vals_c))

                mean_f.append(np.mean(vals_f))
                std_f.append(np.std(vals_f))

            results_mean_corr[size_val] = np.array(mean_c)
            results_std_corr[size_val] = np.array(std_c)
            results_mean_fp[size_val] = np.array(mean_f)
            results_std_fp[size_val] = np.array(std_f)
    else:
        print(f"ERRORE: Nessuna immagine elaborata correttamente per la Run {run_id}.")
        return

    # --- CREAZIONE DATAFRAME COMPLETO ---
    data_dict = {'FWHM': FWHM_RANGE}

    for size_val in SIZES_TO_TEST:
        data_dict[f'Perse_MagLT10_Size{size_val}_Mean'] = results_mean_corr[size_val]
        data_dict[f'Perse_MagLT10_Size{size_val}_Std'] = results_std_corr[size_val]
        data_dict[f'FP_Size{size_val}_Mean'] = results_mean_fp[size_val]
        data_dict[f'FP_Size{size_val}_Std'] = results_std_fp[size_val]

    df_risultati = pd.DataFrame(data_dict)

    csv_filename = f'risultati_scan_run_{run_id}_perse_mag_lt_10.csv'
    df_risultati.to_csv(csv_filename, index=False)
    print(f"Risultati statistici salvati in: {csv_filename}")

    # --- PLOTTING FINALE ---
    plt.figure(figsize=(12, 8))
    colors_corr = {3: 'red', 5: 'blue'}
    colors_fp = {3: 'darkred', 5: 'darkblue'}

    for size_val in SIZES_TO_TEST:
        plt.plot(FWHM_RANGE, results_mean_corr[size_val],
                 color=colors_corr[size_val], linestyle='-', linewidth=2,
                 label=f'Media Stelle Perse (Mag < 10) - Size {size_val}')

        plt.plot(FWHM_RANGE, results_mean_fp[size_val],
                 color=colors_fp[size_val], linestyle='--', linewidth=2,
                 label=f'Media Falsi Positivi (FP) - Size {size_val}')

    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.xlabel('FWHM')
    plt.ylabel('Numero MEDIO per immagine (Log)')
    plt.yscale('log')
    plt.title(f'Scan Parametri (Falsi Positivi vs Stelle Perse < Mag 10): Run {run_id}')

    ax = plt.gca()
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')

    plt.legend()
    plt.tight_layout()

    plt.savefig(f'scan_parametri_run_{run_id}_perse_mag_lt_10.png', dpi=300)
    plt.close()


# --- BLOCCO DI ESECUZIONE GLOBALE ---
if __name__ == "__main__":
    runs_to_process = [1, 2, 3]

    for r in runs_to_process:
        analizza_singola_run(r)

    print(f"\n\n{'=' * 60}")
    print(f"TUTTE LE RUN COMPLETATE.")
    print(f"Tempo totale esecuzione: {(time.time() - start_time_global) / 3600:.1f} ore")
    print(f"{'=' * 60}")