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
from astroquery.vizier import Vizier
import matplotlib.ticker as ticker
from tqdm import tqdm

# --- IMPORT FONDAMENTALE PER LA PORTABILITÀ ---
from pathlib import Path
from skyfield.api import load, wgs84
from astropy.time import Time
import requests
from datetime import timedelta

# --- GESTIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='Units from inserted quantities will be ignored.')

# salvo il tempo di inizio globale
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
# 0. GESTIONE PERCORSI DINAMICA E CATALOGHI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # risalgo la directory partendo dalla posizione dello script fino a trovare la cartella target
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
    # cerco una CARTELLA ricorsivamente in tutte le sottocartelle
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]

    if not cartelle_trovate:
        return None

    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


# definisco la BASE_DIR dinamicamente
BASE_DIR = trova_cartella_base("pmc_photometry")

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")

# Inizializzo Vizier per l'estrazione dinamica (senza limiti di righe)
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
    # memmap=False per mia sicurezza su file network/condivisi
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

    # filtro rapido vettorializzato che applico
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


def genera_tabella_catalogo_immagine(wcs, shape, bordo, tbl_vizier, tbl_hipparco):
    # genero dinamicamente la tabella unita di catalogo proiettata sul piano di questa specifica immagine
    h, w = shape

    nome_catalogo_vizier = np.array(["II/389/ps1_dr2"] * len(tbl_vizier), dtype=object)
    colonne_vizier = {
        'Catalogo': nome_catalogo_vizier,
        'ID': tbl_vizier['objID'],
        'RAJ2000': tbl_vizier['RAJ2000'],
        'DEJ2000': tbl_vizier['DEJ2000'],
        'Mag': tbl_vizier['gmag'],
    }

    nome_catalogo_hipparco = np.array(["I/239/hip_main"] * len(tbl_hipparco), dtype=object)
    colonne_hipparco = {
        'Catalogo': nome_catalogo_hipparco,
        'ID': tbl_hipparco['HIP'],
        'RAJ2000': tbl_hipparco['_RAJ2000'],
        'DEJ2000': tbl_hipparco['_DEJ2000'],
        'Mag': tbl_hipparco['Vmag'],
    }

    t1 = Table(colonne_vizier)
    t2 = Table(colonne_hipparco)

    tbl_unita = vstack([t1, t2])

    coords = SkyCoord(ra=tbl_unita['RAJ2000'], dec=tbl_unita['DEJ2000'], unit=u.deg)
    x_pix, y_pix = wcs.world_to_pixel(coords)

    mask_bordo = ((x_pix >= bordo) & (x_pix < (w - bordo)) & (y_pix >= bordo) & (y_pix < (h - bordo)))

    return tbl_unita[mask_bordo], coords[mask_bordo]


def unione_tabelle_ottimizzata(tbl_seg, tbl_cat, wcs, coords_catalogate, soglia_correlazione, mag_limit=10.0):
    # eseguo il matching e restituisco la tabella unita e il numero di stelle UNICHE
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

        # allineo le colonne
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

    # 1. identifico gli ID trovati
    mask_si = np.char.startswith(tbl_unite['Corrispondenza'].astype(str), 'SI')
    ids_trovati_e_correlati = set(tbl_unite[mask_si]['ID'])

    # 2. filtro il catalogo per le brillanti
    mask_bright_cat = tbl_cat['Mag'] < mag_limit
    stelle_catalogo_brillanti = tbl_cat[mask_bright_cat]

    # 3. conto quante brillanti sono state trovate (intersezione)
    num_brillanti = 0
    for star_id in stelle_catalogo_brillanti['ID']:
        if star_id in ids_trovati_e_correlati:
            num_brillanti += 1

    # restituisco coordinate per facilitare il calcolo FP
    return tbl_unite, num_brillanti


# --- CONFIGURAZIONE GLOBALE SCAN ---
PARAMETRI_FISSI = {
    'threshold_assoluta': 3.61,
    'pixel': 3,
    'soglia_filtro_ass': 2.5,
    'soglia_filtro_rel': 0.05
}

FWHM_RANGE = np.linspace(2.2, 3.0, 150)
SIZES_TO_TEST = [3, 5]
soglia_correlazione = 0.003349 * u.deg
MAG_LIMIT_ANALYSIS = 10.0  # soglia per definire le stelle "importanti" (Stelle Perse < 10)


def analizza_singola_run(run_id, satelliti_attivi):
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

    # =========================================================================
    # --- FASE 0: MERGE E PREPARAZIONE CATALOGO GLOBALE PER LA RUN ---
    # =========================================================================
    print("--- FASE 0: PREPARAZIONE CATALOGO MERGIATO PER LA RUN ---")
    primo_fits_valido = None

    # estraggo il primo file FITS disponibile per definire il Field of View
    for p_csv in file_csv_cat:
        header_info = leggi_header_da_csv(p_csv)
        p_fits = header_info.get('PERCORSO_FILE', '')
        p_obj = Path(p_fits)
        try:
            if "pmc_photometry" in p_obj.parts:
                idx = p_obj.parts.index("pmc_photometry")
                new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                if new_path.exists(): p_fits = str(new_path)
            elif "prove_2" in p_obj.parts:
                idx = p_obj.parts.index("prove_2")
                new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                if new_path.exists(): p_fits = str(new_path)
        except:
            pass

        if os.path.exists(p_fits):
            primo_fits_valido = p_fits
            break

    if primo_fits_valido is None:
        print(f"ERRORE: Nessun FITS valido trovato per la Run {run_id} per calcolare il merge.")
        return

    # calcolo centro e raggio di ricerca
    with fits.open(primo_fits_valido, memmap=False) as hdu_list:
        w_ref = WCS(hdu_list[0].header)
        ra_c = hdu_list[0].header["RA"]
        dec_c = hdu_list[0].header["DEC"]
        alto_destra = w_ref.pixel_to_world(hdu_list[0].header['NAXIS1'] - 1, hdu_list[0].header['NAXIS2'] - 1)
        centro = SkyCoord(ra_c, dec_c, unit=u.deg)
        raggio_ricerca = Angle(centro.separation(alto_destra) * 1.5, "deg")

    print(f"Esecuzione query Vizier per la Run {run_id}...")
    riquadro_esterno_vizier = vizier.query_region(
        coord.SkyCoord(ra=ra_c, dec=dec_c, unit=(u.deg, u.deg), frame='icrs'),
        radius=raggio_ricerca,
        column_filters={'gmag': f'<{15}'}
    )
    tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]

    # filtro Hipparcos per il FoV
    distanze_hip = centro.separation(coords_hipparco_global)
    mask_hip_fov = distanze_hip < raggio_ricerca

    tbl_hipparco_run_subset = tbl_catalogo_hipparco_globale[mask_hip_fov]
    coords_hipparco_run_subset = coords_hipparco_global[mask_hip_fov]
    exclusion_radii_run_subset = exclusion_radii_deg[mask_hip_fov]

    # --- FILTRAGGIO COMPETITIVO A SINGOLA FASE ---
    print("Avvio filtraggio competitivo a singola fase Vizier vs Hipparcos...")
    coords_vizier = SkyCoord(ra=tbl_riquadro_esterno_vizier['RAJ2000'],
                             dec=tbl_riquadro_esterno_vizier['DEJ2000'],
                             unit=u.deg)

    max_threshold_deg = np.max(exclusion_radii_run_subset) if len(exclusion_radii_run_subset) > 0 else 0
    seplimit = max_threshold_deg * u.deg

    idx_A, idx_B, d2d_1, _ = coords_hipparco_run_subset.search_around_sky(coords_vizier, seplimit)

    # aggiramento indice inverso di astropy
    if len(idx_A) > 0 and np.max(idx_A) >= len(coords_hipparco_run_subset):
        idx_viz_1, idx_hip_1 = idx_A, idx_B
    else:
        idx_hip_1, idx_viz_1 = idx_A, idx_B

    mask_threshold = d2d_1.deg <= exclusion_radii_run_subset[idx_hip_1]
    idx_hip_valid = idx_hip_1[mask_threshold]
    idx_viz_valid = idx_viz_1[mask_threshold]

    mask_keep_hipparco = np.ones(len(tbl_hipparco_run_subset), dtype=bool)
    mask_keep_vizier = np.ones(len(tbl_riquadro_esterno_vizier), dtype=bool)

    unique_hip_idx = np.unique(idx_hip_valid)

    # risolvo i conflitti
    for i_hip in unique_hip_idx:
        viz_matches = idx_viz_valid[idx_hip_valid == i_hip]
        if len(viz_matches) > 0:
            mag_viz_matches = np.nan_to_num(tbl_riquadro_esterno_vizier['gmag'][viz_matches], nan=99.0)
            idx_min_mag = np.argmin(mag_viz_matches)
            best_viz_idx = viz_matches[idx_min_mag]
            best_viz_mag = mag_viz_matches[idx_min_mag]
            hip_mag = np.nan_to_num(tbl_hipparco_run_subset['Vmag'][i_hip], nan=99.0)

            if best_viz_mag <= hip_mag:
                mask_keep_hipparco[i_hip] = False
            else:
                mask_keep_vizier[best_viz_idx] = False

    hipparco_escluse = np.sum(~mask_keep_hipparco)
    vizier_escluse = np.sum(~mask_keep_vizier)
    print(f"Risolti {len(unique_hip_idx)} conflitti spaziali:")
    print(f" -> Escluse {hipparco_escluse} stelle Hipparco (tenute Vizier perché più brillanti)")
    print(f" -> Escluse {vizier_escluse} stelle Vizier (tenute Hipparco perché più brillanti)")

    # applico taglio mag limite
    mask_keep_hipparco[tbl_hipparco_run_subset['Vmag'] >= 15] = False
    tbl_hipparco_run_clean = tbl_hipparco_run_subset[mask_keep_hipparco]
    tbl_riquadro_esterno_vizier_CLEAN = tbl_riquadro_esterno_vizier[mask_keep_vizier]
    tbl_vizier_cut = tbl_riquadro_esterno_vizier_CLEAN[tbl_riquadro_esterno_vizier_CLEAN['gmag'] < 15]

    # --- RESET STRUTTURE DATI PER LO SCAN DEL PARAMETRO ---
    raw_corr = {size: [[] for _ in range(len(FWHM_RANGE))] for size in SIZES_TO_TEST}
    raw_fp = {size: [[] for _ in range(len(FWHM_RANGE))] for size in SIZES_TO_TEST}
    fp_coords_storage = {size: [[] for _ in range(len(FWHM_RANGE))] for size in SIZES_TO_TEST}

    immagini_processate_correttamente = 0
    totale_immagini = len(file_csv_cat)
    total_steps = totale_immagini * len(SIZES_TO_TEST) * len(FWHM_RANGE)

    print(f"\nAnalisi focalizzata su stelle perse con Mag < {MAG_LIMIT_ANALYSIS}")
    print(f"Totale immagini: {totale_immagini}")
    print(f"Totale iterazioni previste: {total_steps}")
    print("-" * 60)

    pbar = tqdm(total=total_steps, desc=f"Run {run_id} Progress", unit="step")

    # --- CICLO SULLE IMMAGINI (ESTERNO) ---
    for idx_img, percorso_csv in enumerate(file_csv_cat):

        success = False

        try:
            header_info = leggi_header_da_csv(percorso_csv)
            percorso_fits_str = header_info.get('PERCORSO_FILE', '')

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
                data_sub, wcs, median_val = elabora_file_fits(percorso_fits_str)

                # Sostituisco la lettura statica dal file con la generazione in tempo reale del mio catalogo pulito
                tbl_catalogate, coords_catalogate = genera_tabella_catalogo_immagine(
                    wcs, data_sub.shape, 7, tbl_vizier_cut, tbl_hipparco_run_clean
                )

                success = True

        except Exception:
            pass

        if success:
            immagini_processate_correttamente += 1
            num_target_stars = np.sum(tbl_catalogate['Mag'] < MAG_LIMIT_ANALYSIS)

            # =========================================================
            # PRE-CALCOLO POSIZIONE SATELLITI PER LA SINGOLA IMMAGINE
            # =========================================================
            catalogo_satelliti_img = None

            if len(satelliti_attivi) > 0:
                with fits.open(percorso_fits_str, memmap=False) as hdu_sat:
                    tempo_scatto_astropy = Time(hdu_sat[0].header['DATE-OBS'], format='isot', scale='utc')

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
                    catalogo_satelliti_img = SkyCoord(ra=ra_sat_list * u.deg, dec=dec_sat_list * u.deg)

            # --- CICLO SUI PARAMETRI (INTERNO ALL'IMMAGINE) ---
            for size_val in SIZES_TO_TEST:
                for i_fwhm, fwhm_val in enumerate(FWHM_RANGE):

                    tbl_trovate = esegui_segmentazione_dinamica(data_sub, fwhm=fwhm_val, size=size_val,
                                                                params=PARAMETRI_FISSI)

                    val_perse = num_target_stars
                    val_fp_raw = 0

                    current_fp_ra = np.array([])
                    current_fp_dec = np.array([])

                    if tbl_trovate is not None and len(tbl_trovate) > 0:
                        tbl_trovate = filtra_vicini_saturi(tbl_trovate, median_bg=median_val)

                        # eseguo il merge spaziale
                        tbl_matched, num_found_bright = unione_tabelle_ottimizzata(
                            tbl_trovate,
                            tbl_catalogate,
                            wcs,
                            coords_catalogate,
                            soglia_correlazione,
                            mag_limit=MAG_LIMIT_ANALYSIS
                        )

                        val_perse = max(0, num_target_stars - num_found_bright)

                        # identificazione e filtro FP
                        mask_no = tbl_matched['Corrispondenza'] == 'NO'
                        tbl_fp = tbl_matched[mask_no]

                        if len(tbl_fp) > 0:
                            # trasformo in coordinate solo le stelle non catalogate
                            coords_fp = wcs.pixel_to_world(tbl_fp['xcentroid'], tbl_fp['ycentroid'])

                            # applico il filtro satelliti
                            if catalogo_satelliti_img is not None:
                                idx_sat, d2d_sat, _ = coords_fp.match_to_catalog_sky(catalogo_satelliti_img)
                                tolleranza_satellite = 3 / 60 * u.deg
                                mask_is_satellite = d2d_sat < tolleranza_satellite

                                # escludo i falsi positivi causati dai satelliti
                                coords_fp = coords_fp[~mask_is_satellite]

                            val_fp_raw = len(coords_fp)
                            if val_fp_raw > 0:
                                current_fp_ra = coords_fp.ra.deg
                                current_fp_dec = coords_fp.dec.deg

                    # salvo i risultati
                    raw_corr[size_val][i_fwhm].append(val_perse)
                    raw_fp[size_val][i_fwhm].append(val_fp_raw)
                    fp_coords_storage[size_val][i_fwhm].append((current_fp_ra, current_fp_dec))

                    # aggiorno la barra
                    pbar.update(1)

            pbar.set_postfix_str(f"Img {idx_img + 1}/{totale_immagini}")

        else:
            step_saltati = len(SIZES_TO_TEST) * len(FWHM_RANGE)
            pbar.update(step_saltati)
            pbar.set_postfix_str(f"Img {idx_img + 1} SKIPPED (File not found)")

    pbar.close()

    print(f"\nRun {run_id} terminata (Scan Immagini). Avvio Filtraggio Transienti...")

    # =========================================================================
    # --- FASE DI POST-PROCESSING: FILTRAGGIO Falsi Positivi ---
    # =========================================================================

    if immagini_processate_correttamente > 0:
        for size_val in SIZES_TO_TEST:
            for i_fwhm in range(len(FWHM_RANGE)):

                list_of_img_coords = fp_coords_storage[size_val][i_fwhm]

                # appiattisco tutto per il cross-match globale
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

                        # verifico se il vicino è in un'immagine diversa
                        img1 = map_idx_to_img[i1]
                        img2 = map_idx_to_img[i2]

                        if img1 != img2:
                            indices_with_valid_neighbor.add(i1)

                    for idx_globale in indices_with_valid_neighbor:
                        img_origin = map_idx_to_img[idx_globale]
                        filtered_fp_counts[img_origin] += 1

                # sostituisco i conti raw con quelli filtrati
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
                vals_f = raw_fp[size_val][i_fwhm]

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

    # =================================================================
    # PRE-CALCOLO HIPPARCOS GLOBALE (UNA SOLA VOLTA)
    # =================================================================
    print("\nPreparo il catalogo globale di Hipparcos...")
    file_hipparco = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
    with fits.open(file_hipparco) as hdu_list_hipparco:
        tbl_catalogo_hipparco_globale = Table(hdu_list_hipparco[1].data)

    # calcolo errori propagati al J2000
    dt = 2000.0 - 1991.25
    sigma_ra_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco_globale['e_RAICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco_globale['e_pmRA'])) ** 2) / 3600000.0
    sigma_dec_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco_globale['e_DEICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco_globale['e_pmDE'])) ** 2) / 3600000.0

    sigma_hip_deg = np.sqrt(sigma_ra_deg ** 2 + sigma_dec_deg ** 2)
    sigma_vizier_deg = 0.1 / 3600.0
    sigma_totale_deg = np.sqrt(sigma_hip_deg ** 2 + sigma_vizier_deg ** 2)

    # 3-SIGMA
    exclusion_radii_deg_ = 3.0 * sigma_totale_deg
    exclusion_radii_deg = np.full(len(exclusion_radii_deg_), 2.5 / 3600.0)

    # pre-calcolo SkyCoord globale
    coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco_globale['_RAJ2000'],
                                      dec=tbl_catalogo_hipparco_globale['_DEJ2000'],
                                      unit=u.deg)

    # =================================================================
    # --- PRE-CALCOLO SATELLITI GLOBALE (UNA SOLA VOLTA) ---
    # =================================================================
    print("\nPreparo il catalogo satelliti storici globale...")
    file_fits_riferimento = None

    # cerco un FITS qualsiasi tra tutte le run per estrarre la data per il download
    for r in runs_to_process:
        nome_cartella_csv = f"sorgenti_catalogate_run_{r}"
        cartella_csv_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_csv)

        if cartella_csv_path:
            file_csv_cat = sorted(list(cartella_csv_path.glob('*.csv')))
            if file_csv_cat:
                header_info_ref = leggi_header_da_csv(file_csv_cat[0])
                percorso_fits_str_ref = header_info_ref.get('PERCORSO_FILE', '')

                # risolvo il path per la mia sicurezza di portabilità
                if not os.path.exists(percorso_fits_str_ref):
                    p_obj = Path(percorso_fits_str_ref)
                    try:
                        if "pmc_photometry" in p_obj.parts:
                            idx = p_obj.parts.index("pmc_photometry")
                            new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                            if new_path.exists(): percorso_fits_str_ref = str(new_path)
                        elif "prove_2" in p_obj.parts:
                            idx = p_obj.parts.index("prove_2")
                            new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                            if new_path.exists(): percorso_fits_str_ref = str(new_path)
                    except:
                        pass

                if os.path.exists(percorso_fits_str_ref):
                    file_fits_riferimento = percorso_fits_str_ref
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

    # faccio partire l'elaborazione ciclica
    for r in runs_to_process:
        analizza_singola_run(r, satelliti_attivi_globali)

    print(f"\n\n{'=' * 60}")
    print(f"TUTTE LE RUN COMPLETATE.")
    print(f"Tempo totale esecuzione: {(time.time() - start_time_global) / 3600:.1f} ore")
    print(f"{'=' * 60}")