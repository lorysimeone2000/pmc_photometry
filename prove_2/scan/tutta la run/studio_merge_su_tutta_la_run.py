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
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
import astropy.coordinates as coord
import astropy.units as u
import warnings
from astropy.wcs import FITSFixedWarning
import matplotlib.ticker as ticker
from tqdm import tqdm
from astroquery.vizier import Vizier

# --- IMPORT FONDAMENTALE PER LA PORTABILITÀ ---
from pathlib import Path
from skyfield.api import load, wgs84
from astropy.time import Time
import requests
from datetime import timedelta
from astropy.io.fits.verify import VerifyWarning
import traceback

# --- GESTIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='Units from inserted quantities will be ignored.')
warnings.filterwarnings('ignore', category=VerifyWarning)

# Salvo il tempo di inizio globale
start_time_global = time.time()


# =============================================================================
# DOWNLOAD TLE SATELLITI
# =============================================================================

def scarica_tle_storici(tempo_astropy, username, password, cartella_output):
    # converto il tempo astropy in un oggetto datetime standard di python
    data_osservazione = tempo_astropy.datetime

    # creo una finestra temporale di sicurezza: il giorno prima e il giorno dopo lo scatto
    data_inizio = (data_osservazione - timedelta(days=0.5)).strftime('%Y-%m-%d')
    data_fine = (data_osservazione + timedelta(days=0.5)).strftime('%Y-%m-%d')

    # aggiungo un tag per far capire che sono solo i Payload (satelliti veri, niente spazzatura spaziale)
    nome_file = f"tle_storico_payload_{data_inizio}_to_{data_fine}.txt"
    percorso_output = cartella_output / nome_file

    # controllo se l'ho già scaricato per questa run, per non intasare i server
    if percorso_output.exists():
        print(f"TLE storici già presenti: {nome_file}")
        return str(percorso_output)

    print(f"Scaricando i TLE storici da Space-Track per le date {data_inizio} -> {data_fine}...")

    login_url = "https://www.space-track.org/ajaxauth/login"
    # uso la classe 'gp_history' per supportare la colonna OBJECT_TYPE
    query_url = f"https://www.space-track.org/basicspacedata/query/class/gp_history/EPOCH/{data_inizio}--{data_fine}/OBJECT_TYPE/PAYLOAD/format/tle"

    # apro una sessione per mantenere i cookie del login
    with requests.Session() as session:
        # eseguo l'autenticazione
        risposta_login = session.post(login_url, data={'identity': username, 'password': password})

        if risposta_login.status_code != 200:
            print("ERRORE: Login su Space-Track fallito. Controlla le credenziali.")
            return None

        # chiedo i dati
        risposta_tle = session.get(query_url, stream=True)

        if risposta_tle.status_code == 200:
            testo_risposta = risposta_tle.text

            # controllo se Space-Track mi ha restituito una pagina web di blocco invece del catalogo testuale
            if "<html" in testo_risposta.lower()[:50]:
                print("ERRORE: Space-Track ha bloccato il download.")
                print(
                    "-> Probabile causa: Devi accedere una volta a www.space-track.org dal browser e accettare l'User Agreement.")
                return None

            # se tutto è ok, salvo il file
            with open(percorso_output, 'w') as f:
                f.write(testo_risposta)
            print("Download TLE storici completato con successo!")
            return str(percorso_output)
        else:
            print(f"ERRORE: Download TLE fallito con codice HTTP {risposta_tle.status_code}")
            return None


# 1. Inizializzo Skyfield
ts = load.timescale()

# 2. Imposto le coordinate del mio telescopio
osservatorio = wgs84.latlon(28.3000, -16.505830555555555, elevation_m=2370)


# =============================================================================
# 0. GESTIONE PERCORSI DINAMICA (PORTABILITÀ TOTALE)
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # Risalgo la directory partendo dalla posizione dello script fino a trovare la cartella target
    path_corrente = Path(__file__).resolve()

    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent

    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # Cerco una CARTELLA ricorsivamente in tutte le sottocartelle
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate:
        return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # Cerco un file esatto all'interno del progetto
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


# Definisco la BASE_DIR dinamicamente
BASE_DIR = trova_cartella_base("pmc_photometry")

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)


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


def leggi_file_parametri(percorso):
    parametri = {}
    if not os.path.exists(percorso): return {}
    with open(percorso, 'r') as file:
        next(file, None)
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


def elabora_file_fits(percorso_file):
    # memmap=False per mia sicurezza su file network/condivisi
    with fits.open(percorso_file, memmap=False) as hdu:
        header = hdu[0].header
        data = hdu[0].data
        wcs = WCS(header)
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        data_sub = data - median
        return data_sub, wcs, median, header


def esegui_segmentazione_dinamica(data, fwhm, size, params):
    try:
        kernel = make_2dgaussian_kernel(fwhm, size=size)
        convolved_data = convolve(data, kernel)
    except Exception:
        return None

    # assumo i valori di default per le soglie se non trovo il parametro custom
    pixel_val = params.get('pixel', 5)
    thresh_val = params.get('threshold_assoluta', 3.0)

    finder = SourceFinder(npixels=pixel_val, progress_bar=False)
    segment_map = finder(convolved_data, thresh_val)

    if segment_map is None: return None

    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    # applico un filtro rapido vettorializzato
    soglia_ass = params.get('soglia_filtro_ass', 2.5)
    soglia_rel = params.get('soglia_filtro_rel', 0.05)
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
            if np.sum(valori_originali > soglia_rel * sorgente['max_value']) >= 2:
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


# =============================================================================
# ANALISI DELLA SINGOLA RUN E CICLO DI SCAN MERGING
# =============================================================================

def analizza_singola_run(run_id, satelliti_attivi, parametri_caricati,
                         coords_hipparco_global, tbl_catalogo_hipparco, exclusion_radii_deg):
    print(f"\n{'=' * 60}")
    print(f"AVVIO ANALISI SCAN MERGING PER RUN {run_id}")
    print(f"{'=' * 60}")

    # estraggo i valori fissi dal file dei parametri
    fwhm_fisso = parametri_caricati.get('fwhm', 3.0)
    size_fisso = parametri_caricati.get('size', 5)
    soglia_correlazione = 0.003349 * u.deg
    magnitudine_massima = 15.0
    MAG_LIMIT_ANALYSIS = 10.0

    # definisco l'array di scan per il raggio di merging
    mean_exclusion_radius_deg = np.mean(exclusion_radii_deg)
    start_rad_deg = 0.1 * mean_exclusion_radius_deg
    end_rad_deg = 3.0 / 3600.0  # 3 arcosecondi
    NUM_STEPS = 100  # numero di prove per il raggio di merging
    MERGE_RADIUS_RANGE_DEG = np.linspace(start_rad_deg, end_rad_deg, NUM_STEPS)
    MERGE_RADIUS_RANGE_ARCSEC = MERGE_RADIUS_RANGE_DEG * 3600.0

    nome_cartella_run = f"20250120_run{run_id}"
    found_folders = list(BASE_DIR.rglob(nome_cartella_run))
    if not found_folders:
        print(f"Run {run_id} non trovata, salto.")
        return
    run_folder = found_folders[0]

    estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
    file_list = []
    for ext in estensioni_valide: file_list.extend(run_folder.glob(ext))
    file_list = sorted([str(f) for f in file_list])
    if not file_list:
        print(f"Nessun FITS in Run {run_id}, salto.")
        return

    totale_immagini = len(file_list)

    # predispongo le strutture dati per salvare i risultati di ogni iterazione di raggio
    raw_corr = [[] for _ in range(NUM_STEPS)]
    raw_fp = [[] for _ in range(NUM_STEPS)]
    fp_coords_storage = [[] for _ in range(NUM_STEPS)]

    print(f"FWHM Fisso: {fwhm_fisso}, Size Fisso: {size_fisso}")
    print(f"Analisi focalizzata su stelle perse con Mag < {MAG_LIMIT_ANALYSIS}")
    print(f"Scan del Raggio di Merging: da {start_rad_deg * 3600:.3f}\" a 3.0\" ({NUM_STEPS} step)")
    print(f"Totale immagini: {totale_immagini}")
    print("-" * 60)

    # =========================================================================
    # FASE 1: CREAZIONE CATALOGO MADRE (RITAGLIO LARGO)
    # =========================================================================
    print("Preparo il Catalogo Madre Globale per la Run...")

    with fits.open(file_list[0], memmap=False) as hdu_ref:
        header_ref = hdu_ref[0].header
        wcs_ref = WCS(header_ref)

    ra_c, dec_c = header_ref["RA"], header_ref["DEC"]
    centro_run = SkyCoord(ra_c, dec_c, unit=u.deg)
    alto_destra = wcs_ref.pixel_to_world(3071, 2047)

    # calcolo 1.5 volte la metà della diagonale per avere un margine enorme
    raggio_ricerca_madre = Angle(centro_run.separation(alto_destra) * 1.5, "deg")

    print(f"Scarico il FOV allargato da Vizier (Raggio: {raggio_ricerca_madre.deg:.2f} deg)...")
    riquadro_esterno_vizier = vizier.query_region(
        coord.SkyCoord(ra=ra_c, dec=dec_c, unit=(u.deg, u.deg), frame='icrs'),
        radius=raggio_ricerca_madre,
        column_filters={'gmag': f'<{magnitudine_massima}'}
    )
    tbl_vizier_master = riquadro_esterno_vizier[0]
    coords_vizier_master = SkyCoord(ra=tbl_vizier_master['RAJ2000'], dec=tbl_vizier_master['DEJ2000'], unit=u.deg)

    print("Filtro il FOV allargato per Hipparcos...")
    distanze_hip_master = centro_run.separation(coords_hipparco_global)
    mask_hip_master = distanze_hip_master < raggio_ricerca_madre

    tbl_hipparco_master = tbl_catalogo_hipparco[mask_hip_master]
    coords_hipparco_master = coords_hipparco_global[mask_hip_master]
    exclusion_radii_master = exclusion_radii_deg[mask_hip_master]

    # =========================================================================
    # FASE 2: PRE-PROCESSING IMMAGINI (Estrazione + Satelliti 1 sola volta)
    # =========================================================================
    dati_immagini = []

    for idx_img, percorso_file in enumerate(tqdm(file_list, desc="Estrazione Sorgenti Fisse")):
        try:
            data_sub, wcs, median_val, header = elabora_file_fits(percorso_file)

            # eseguo l'estrazione
            tbl_trovate = esegui_segmentazione_dinamica(data_sub, fwhm_fisso, size_fisso, parametri_caricati)
            tbl_trovate = filtra_vicini_saturi(tbl_trovate, median_bg=median_val)

            # converto in coordinate assicurandomi di avere array 1D sicuri
            if tbl_trovate is not None and len(tbl_trovate) > 0:
                xc = np.atleast_1d(tbl_trovate['xcentroid'])
                yc = np.atleast_1d(tbl_trovate['ycentroid'])
                coords_trovate = wcs.pixel_to_world(xc, yc)
            else:
                coords_trovate = SkyCoord(np.array([]) * u.deg, np.array([]) * u.deg)

            # forzo la dimensionalità per evitare eccezioni su oggetti scalari singoli
            if coords_trovate.isscalar:
                coords_trovate = SkyCoord([coords_trovate])

            # trovo i satelliti una sola volta
            mask_is_satellite_trovate = np.zeros(len(coords_trovate), dtype=bool)

            if len(satelliti_attivi) > 0 and len(coords_trovate) > 0:
                tempo_scatto_astropy = Time(header['DATE-OBS'], format='isot', scale='utc')
                tempo_skyfield = ts.from_astropy(tempo_scatto_astropy)
                ra_sat_list, dec_sat_list = [], []

                for sat in satelliti_attivi:
                    topocentrica = (sat - osservatorio).at(tempo_skyfield)
                    ra_sat, dec_sat, _ = topocentrica.radec()
                    if np.isnan(ra_sat.hours) or np.isnan(dec_sat.degrees):
                        continue
                    ra_sat_list.append(ra_sat.hours * 15)
                    dec_sat_list.append(dec_sat.degrees)

                if ra_sat_list:
                    # blindo la moltiplicazione passando i valori tramite np.array
                    catalogo_satelliti_img = SkyCoord(ra=np.array(ra_sat_list) * u.deg,
                                                      dec=np.array(dec_sat_list) * u.deg)
                    idx_sat, d2d_sat, _ = coords_trovate.match_to_catalog_sky(catalogo_satelliti_img)
                    mask_is_satellite_trovate = d2d_sat < (3 / 60 * u.deg)

            # mi salvo tutto quello che serve in ram per il match ultraveloce
            dati_immagini.append({
                'wcs': wcs,
                'shape': data_sub.shape,
                'coords_trovate': coords_trovate,
                'mask_is_satellite': mask_is_satellite_trovate
            })

        except Exception as e:
            # Stampo esplicitamente l'errore per aiutarti a individuare eventuali anomalie nel FITS
            print(f"\n[ERRORE DURANTE L'ESTRAZIONE] Immagine {percorso_file}: {e}")
            traceback.print_exc()
            pass

    immagini_processate_correttamente = len(dati_immagini)
    if immagini_processate_correttamente == 0:
        print(f"ERRORE CRITICO: Nessuna immagine valida processata per la Run {run_id}.")
        return

    # =========================================================================
    # FASE 3: SCAN DEL RAGGIO DI MERGING (LOOP ESTERNO)
    # =========================================================================
    for i_rad, merge_rad in enumerate(tqdm(MERGE_RADIUS_RANGE_DEG, desc=f"Scan Merging Radius Run {run_id}")):

        # 3.1: RISOLVO I CONFLITTI SUL MASTER CATALOG
        mask_keep_hipparco = np.ones(len(tbl_hipparco_master), dtype=bool)
        mask_keep_vizier = np.ones(len(tbl_vizier_master), dtype=bool)

        seplimit = merge_rad * u.deg
        idx_A, idx_B, d2d_1, _ = coords_hipparco_master.search_around_sky(coords_vizier_master, seplimit)

        # bypass inversione indici astropy
        if len(idx_A) > 0 and np.max(idx_A) >= len(coords_hipparco_master):
            idx_viz_1, idx_hip_1 = idx_A, idx_B
        else:
            idx_hip_1, idx_viz_1 = idx_A, idx_B

        mask_threshold = d2d_1.deg <= merge_rad
        idx_hip_valid = idx_hip_1[mask_threshold]
        idx_viz_valid = idx_viz_1[mask_threshold]

        unique_hip_idx = np.unique(idx_hip_valid)

        for i_hip in unique_hip_idx:
            viz_matches = idx_viz_valid[idx_hip_valid == i_hip]
            if len(viz_matches) > 0:
                mag_viz_matches = np.nan_to_num(tbl_vizier_master['gmag'][viz_matches], nan=99.0)
                idx_min_mag = np.argmin(mag_viz_matches)
                best_viz_idx = viz_matches[idx_min_mag]
                best_viz_mag = mag_viz_matches[idx_min_mag]
                hip_mag = np.nan_to_num(tbl_hipparco_master['Vmag'][i_hip], nan=99.0)

                if best_viz_mag <= hip_mag:
                    mask_keep_hipparco[i_hip] = False
                else:
                    mask_keep_vizier[best_viz_idx] = False

        mask_keep_hipparco[tbl_hipparco_master['Vmag'] >= magnitudine_massima] = False

        tbl_hipparco_clean = tbl_hipparco_master[mask_keep_hipparco]
        tbl_vizier_clean = tbl_vizier_master[mask_keep_vizier]
        tbl_vizier_cut = tbl_vizier_clean[tbl_vizier_clean['gmag'] < magnitudine_massima]

        # costruisco i vettori veloci del Master Fuso assicurandomi di estrarre array numpy puliti
        ra_viz = np.array(tbl_vizier_cut['RAJ2000'])
        dec_viz = np.array(tbl_vizier_cut['DEJ2000'])
        mag_viz = np.array(
            tbl_vizier_cut['gmag'].filled(np.nan) if hasattr(tbl_vizier_cut['gmag'], 'filled') else tbl_vizier_cut[
                'gmag'])

        ra_hip = np.array(tbl_hipparco_clean['_RAJ2000'])
        dec_hip = np.array(tbl_hipparco_clean['_DEJ2000'])
        mag_hip = np.array(
            tbl_hipparco_clean['Vmag'].filled(np.nan) if hasattr(tbl_hipparco_clean['Vmag'], 'filled') else
            tbl_hipparco_clean['Vmag'])

        ra_cat_master = np.concatenate([ra_viz, ra_hip])
        dec_cat_master = np.concatenate([dec_viz, dec_hip])
        mag_cat_master = np.concatenate([mag_viz, mag_hip])

        coords_unita_master = SkyCoord(ra=ra_cat_master * u.deg, dec=dec_cat_master * u.deg)

        # 3.2: LOOP INTERNO SULLE IMMAGINI
        # (Uso il master fuso appena creato, lo ritaglio pixel per pixel e valuto)
        for img_data in dati_immagini:
            wcs = img_data['wcs']
            h, w_img = img_data['shape']
            coords_trovate = img_data['coords_trovate']
            mask_is_sat = img_data['mask_is_satellite']

            # taglio i bordi del Master Fuso per allinearmi ai pixel di questa immagine
            x_pix, y_pix = wcs.world_to_pixel(coords_unita_master)
            bordo = 7
            mask_bordo = ((x_pix >= bordo) & (x_pix < w_img - bordo) & (y_pix >= bordo) & (y_pix < h - bordo))

            coords_catalogate_img = coords_unita_master[mask_bordo]
            mag_catalogate_img = mag_cat_master[mask_bordo]

            # conto il target assoluto delle stelle < 10 nel FOV esatto
            target_mask = mag_catalogate_img < MAG_LIMIT_ANALYSIS
            num_target_stars = np.sum(target_mask)

            val_perse = num_target_stars
            val_fp_raw = 0
            current_fp_ra = np.array([])
            current_fp_dec = np.array([])

            if len(coords_trovate) > 0 and len(coords_catalogate_img) > 0:
                idx_t, idx_c, _, _ = coords_catalogate_img.search_around_sky(coords_trovate, soglia_correlazione)

                matched_catalog_indices = set(idx_c)
                num_found_bright = sum(1 for c_idx in matched_catalog_indices if target_mask[c_idx])
                val_perse = max(0, num_target_stars - num_found_bright)

                # identifico chi non ha fatto match (NO) ed escludo chi era mascherato come satellite
                unmatched_trovate = set(range(len(coords_trovate))) - set(idx_t)
                fp_indices = [idx for idx in unmatched_trovate if not mask_is_sat[idx]]

                val_fp_raw = len(fp_indices)
                if val_fp_raw > 0:
                    current_fp_ra = coords_trovate.ra.deg[fp_indices]
                    current_fp_dec = coords_trovate.dec.deg[fp_indices]

            elif len(coords_trovate) > 0 and len(coords_catalogate_img) == 0:
                # non c'erano stelle di catalogo nel FOV, tutto quello che trovo è FP se non è satellite
                fp_indices = [idx for idx in range(len(coords_trovate)) if not mask_is_sat[idx]]
                val_fp_raw = len(fp_indices)
                if val_fp_raw > 0:
                    current_fp_ra = coords_trovate.ra.deg[fp_indices]
                    current_fp_dec = coords_trovate.dec.deg[fp_indices]

            # salvo i risultati di questa immagine per questo step di raggio
            raw_corr[i_rad].append(val_perse)
            raw_fp[i_rad].append(val_fp_raw)
            fp_coords_storage[i_rad].append((current_fp_ra, current_fp_dec))

    print(f"\nRun {run_id} terminata (Scan Immagini). Avvio Filtraggio Transienti...")

    # =========================================================================
    # --- FASE DI POST-PROCESSING: FILTRAGGIO Falsi Positivi ---
    # =========================================================================

    if immagini_processate_correttamente > 0:
        for i_rad in range(NUM_STEPS):

            list_of_img_coords = fp_coords_storage[i_rad]

            # appiattisco tutto per il cross-match globale tra foto diverse
            all_ra = []
            all_dec = []
            map_idx_to_img = {}

            global_counter = 0
            for img_idx, (ra_arr, dec_arr) in enumerate(list_of_img_coords):
                if len(ra_arr) > 0:
                    all_ra.extend(ra_arr)
                    all_dec.extend(dec_arr)
                    for _ in range(len(ra_arr)):
                        map_idx_to_img[global_counter] = img_idx
                        global_counter += 1

            filtered_fp_counts = [0] * len(list_of_img_coords)

            if len(all_ra) > 0:
                c_tot = SkyCoord(ra=all_ra * u.deg, dec=all_dec * u.deg)
                dist_limit = 0.0011 * u.deg

                idx1, idx2, _, _ = c_tot.search_around_sky(c_tot, dist_limit)

                indices_with_valid_neighbor = set()

                for k in range(len(idx1)):
                    i1 = idx1[k]
                    i2 = idx2[k]
                    if i1 == i2: continue

                    # verifico se il vicino è presente in una fotografia diversa (non è transiente)
                    img1 = map_idx_to_img[i1]
                    img2 = map_idx_to_img[i2]

                    if img1 != img2:
                        indices_with_valid_neighbor.add(i1)

                for idx_globale in indices_with_valid_neighbor:
                    img_origin = map_idx_to_img[idx_globale]
                    filtered_fp_counts[img_origin] += 1

            # sovrascrivo i conti originali grezzi con quelli reali persistenti
            raw_fp[i_rad] = filtered_fp_counts

    print(f"Filtraggio Transienti completato.")
    print(f"Immagini valide valutate: {immagini_processate_correttamente}")

    # --- CALCOLO MEDIA E DEVIAZIONE STANDARD ---
    mean_c = []
    std_c = []
    mean_f = []
    std_f = []

    for i_rad in range(NUM_STEPS):
        vals_c = raw_corr[i_rad]
        vals_f = raw_fp[i_rad]

        mean_c.append(np.mean(vals_c))
        std_c.append(np.std(vals_c))
        mean_f.append(np.mean(vals_f))
        std_f.append(np.std(vals_f))

    # --- CREAZIONE DATAFRAME COMPLETO ---
    data_dict = {
        'MergeRadius_Arcsec': MERGE_RADIUS_RANGE_ARCSEC,
        'Perse_MagLT10_Mean': mean_c,
        'Perse_MagLT10_Std': std_c,
        'FP_Mean': mean_f,
        'FP_Std': std_f
    }

    df_risultati = pd.DataFrame(data_dict)

    csv_filename = f'risultati_scan_MERGE_run_{run_id}_perse_mag_lt_10.csv'
    df_risultati.to_csv(csv_filename, index=False)
    print(f"Risultati statistici salvati in: {csv_filename}")

    # --- PLOTTING FINALE ---
    plt.figure(figsize=(12, 8))

    plt.plot(MERGE_RADIUS_RANGE_ARCSEC, mean_c, color='red', linestyle='-', linewidth=2,
             label=f'Media Stelle Perse (Mag < 10)')
    plt.plot(MERGE_RADIUS_RANGE_ARCSEC, mean_f, color='darkblue', linestyle='--', linewidth=2,
             label=f'Media Falsi Positivi Reali (FP)')

    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.xlabel('Raggio di Merging Vizier/Hipparcos (Arcosecondi)')
    plt.ylabel('Numero MEDIO per immagine (Log)')
    plt.yscale('log')
    plt.title(f'Scan Raggio Merging (FP Reali vs Perse < 10): Run {run_id}\nFWHM={fwhm_fisso}, Size={size_fisso}')

    ax = plt.gca()
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')

    plt.legend()
    plt.tight_layout()

    plt.savefig(f'scan_MERGE_run_{run_id}_perse_mag_lt_10.png', dpi=300)
    plt.close()


# --- BLOCCO DI ESECUZIONE GLOBALE ---
if __name__ == "__main__":
    runs_to_process = [1, 2, 3]

    # Pre-carico Hipparcos globale
    file_hipparco = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
    hdu_list_hipparco = fits.open(file_hipparco)
    tbl_catalogo_hipparco = Table(hdu_list_hipparco[1].data)
    hdu_list_hipparco.close()

    dt = 2000.0 - 1991.25
    sigma_ra_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco['e_RAICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco['e_pmRA'])) ** 2) / 3600000.0
    sigma_dec_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco['e_DEICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco['e_pmDE'])) ** 2) / 3600000.0

    sigma_hip_deg = np.sqrt(sigma_ra_deg ** 2 + sigma_dec_deg ** 2)
    sigma_vizier_deg = 0.1 / 3600.0
    sigma_totale_deg = np.sqrt(sigma_hip_deg ** 2 + sigma_vizier_deg ** 2)

    exclusion_radii_deg_ = 3.0 * sigma_totale_deg
    exclusion_radii_deg = np.full(len(exclusion_radii_deg_), 2.5 / 3600.0)

    coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco['_RAJ2000'], dec=tbl_catalogo_hipparco['_DEJ2000'],
                                      unit=u.deg)

    # Carico i parametri di base
    nome_params = 'parametri_image_segmentation.txt'
    file_parametri = cerca_file_nel_progetto(BASE_DIR, nome_params)
    if file_parametri is None:
        print("File dei parametri non trovato")
        exit()
    parametri_caricati = leggi_file_parametri(file_parametri)

    # =================================================================
    # --- PRE-CALCOLO SATELLITI GLOBALE (UNA SOLA VOLTA) ---
    # =================================================================
    print("\nPreparo il catalogo satelliti storici globale...")
    file_fits_riferimento = None

    # cerco un FITS qualsiasi tra tutte le run per estrarre la data per il download
    for r in runs_to_process:
        nome_cartella_run = f"20250120_run{r}"
        cartelle_trovate = list(BASE_DIR.rglob(nome_cartella_run))
        if cartelle_trovate:
            f_list = list(cartelle_trovate[0].glob('*.fit')) + list(cartelle_trovate[0].glob('*.fits')) + list(
                cartelle_trovate[0].glob('*.FIT')) + list(cartelle_trovate[0].glob('*.FITS'))
            if f_list:
                file_fits_riferimento = str(f_list[0])
                break

    satelliti_attivi_globali = []

    if file_fits_riferimento:
        with fits.open(file_fits_riferimento, memmap=False) as hdu_ref:
            tempo_ref_astropy = Time(hdu_ref[0].header['DATE-OBS'], format='isot', scale='utc')

        # inserisco le mie credenziali Space-Track
        tuo_user = "lorenzo.simeone@studenti.unipg.it"
        tua_password = "Cazzata_2002348"

        cartella_tabelle = BASE_DIR / "tabelle"
        cartella_tabelle.mkdir(exist_ok=True)

        percorso_tle = scarica_tle_storici(tempo_ref_astropy, tuo_user, tua_password, cartella_tabelle)

        if percorso_tle:
            satelliti_attivi_globali = load.tle_file(percorso_tle)
            print(f"Download satelliti avvenuto: {len(satelliti_attivi_globali)} satelliti trovati nel catalogo")
        else:
            print("ATTENZIONE: Download fallito. Disabilito il filtro satelliti.")
    else:
        print("ATTENZIONE: Nessun FITS di riferimento trovato per la data. Disabilito il filtro satelliti.")
    # =================================================================

    # Faccio partire l'elaborazione ciclica passando il catalogo scaricato
    for r in runs_to_process:
        analizza_singola_run(r, satelliti_attivi_globali, parametri_caricati,
                             coords_hipparco_global, tbl_catalogo_hipparco, exclusion_radii_deg)

    print(f"\n\n{'=' * 60}")
    print(f"TUTTE LE RUN COMPLETATE.")
    print(f"Tempo totale esecuzione: {(time.time() - start_time_global) / 3600:.1f} ore")
    print(f"{'=' * 60}")