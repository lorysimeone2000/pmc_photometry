import pandas as pd
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import os
import sys
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
import warnings
from astropy.wcs import FITSFixedWarning
from photutils.datasets import make_100gaussians_image
from scipy.optimize import curve_fit
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.utils.data import download_file
from astropy.table import Table, vstack
from photutils.detection import find_peaks
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from shapely.geometry import Point, Polygon
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from pathlib import Path

# importo i moduli per il catalogo satelliti
from skyfield.api import load, wgs84
from astropy.time import Time
import requests
from datetime import timedelta

# importo il modulo per la ricerca spaziale ultra veloce
from scipy.spatial import cKDTree

# gestisco i warning ignorandoli
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    # risalgo l'albero delle directory per trovare la radice del progetto
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")

# aggiungo il percorso specifico al sys.path se la cartella funzioni si trova dentro pmc_photometry
PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

from funzioni.utilita import *
from funzioni.astrometria import *

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"Moduli esterni caricati con successo.")
print(f"------------------------------")

# inizializzo Skyfield
ts = load.timescale()

# imposto le coordinate del mio telescopio usando la funzione importata
lat_oss, lon_oss, alt_oss = ottieni_coordinate_telescopio('ASTRI 1', BASE_DIR / "pmc_photometry")

# creo l'oggetto geografico wgs84
osservatorio = wgs84.latlon(lat_oss, lon_oss, elevation_m=alt_oss)

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag'],
    row_limit=-1
)

# =============================================================================
# BLOCCO DI ESECUZIONE (MAIN)
# =============================================================================

if __name__ == "__main__":

    soglia_correlazione = 35 / 3600 * u.deg
    dist_ripetizione = soglia_correlazione
    magnitudine_massima = 15

    nome_params = 'parametri_image_segmentation.txt'
    file_parametri = cerca_file_nel_progetto(BASE_DIR, nome_params)
    if file_parametri is None:
        print("File dei parametri non trovato")
        exit()
    parametri_caricati = leggi_file_parametri(file_parametri)

    # pre-calcolo globalmente Hipparcos
    file_hipparco = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
    hdu_list_hipparco = fits.open(file_hipparco)
    tbl_catalogo_hipparco = Table(hdu_list_hipparco[1].data)
    hdu_list_hipparco.close()

    # calcolo gli errori propagati al J2000
    dt = 2000.0 - 1991.25
    sigma_ra_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco['e_RAICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco['e_pmRA'])) ** 2) / 3600000.0
    sigma_dec_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco['e_DEICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco['e_pmDE'])) ** 2) / 3600000.0

    # calcolo l'errore radiale totale di Hipparcos
    sigma_hip_deg = np.sqrt(sigma_ra_deg ** 2 + sigma_dec_deg ** 2)

    # considero l'errore stimato per Vizier
    sigma_vizier_deg = 0.1 / 3600.0

    # sommo in quadratura i due cataloghi
    sigma_totale_deg = np.sqrt(sigma_hip_deg ** 2 + sigma_vizier_deg ** 2)

    # calcolo il 3-sigma
    exclusion_radii_deg_ = 3.0 * sigma_totale_deg
    exclusion_radii_deg = np.full(len(exclusion_radii_deg_), 2.5 / 3600.0)

    print(f"Raggio di merging tra i cataloghi: {np.mean(exclusion_radii_deg)}")

    # creo lo SkyCoord Hipparcos
    coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco['_RAJ2000'],
                                      dec=tbl_catalogo_hipparco['_DEJ2000'],
                                      unit=u.deg)

    tutti_i_file_csv_generati = []
    tuo_user = "lorenzo.simeone@studenti.unipg.it"
    tua_password = "Cazzata_2002348"

    global_tracker_coords = None
    global_tracker_labels = []
    global_max_label = 0
    global_catalog_label_map = {}
    contatore_satelliti = 0
    contatore_satelliti_presenti = 0

    # =================================================================
    # --- RICERCA FILE E SUDDIVISIONE RUN ---
    # =================================================================
    print(f"\n--- DEBUG ESTRAZIONE BLAZAR E CREAZIONE RUN ---")
    # definisco le coordinate di Mrk 421
    coords_mrk421 = SkyCoord(ra=166.1138 * u.deg, dec=38.2088 * u.deg, frame='icrs')
    raggio_fov_tolleranza = 6 * u.deg

    PMC_DATA = cerca_cartella_nel_progetto(BASE_DIR, "PMC_DATA")
    file_validi = []

    if PMC_DATA:
        # cerco le cartelle risolvendo i symlink
        sottocartelle = [d for d in PMC_DATA.iterdir() if d.is_dir() or d.is_symlink()]
        print(f"Trovate {len(sottocartelle)} cartelle/link in PMC_DATA.")

        for cartella_giorno in sottocartelle:
            # risolvo il symlink per accedere ai file reali
            percorso_reale = cartella_giorno.resolve()

            file_fits_list = []
            for ext in ['*.fit', '*.fits', '*.FIT', '*.FITS']:
                file_fits_list.extend(percorso_reale.glob(ext))

            if not file_fits_list:
                continue

            for percorso_file in tqdm(file_fits_list, desc=f"Scansione {cartella_giorno.name}"):
                try:
                    # uso memmap=True per velocizzare la sola lettura dell'header
                    with fits.open(percorso_file, memmap=True) as hdu:
                        header = hdu[0].header

                        # provo diverse combinazioni di chiavi comuni
                        ra_val = header.get('RA') or header.get('RAJ2000') or header.get('OBJ-RA')
                        dec_val = header.get('DEC') or header.get('DEJ2000') or header.get('OBJ-DEC')
                        tempo_obs_str = header.get('DATE-OBS')

                        if ra_val is not None and dec_val is not None:
                            try:
                                if isinstance(ra_val, (int, float)):
                                    coords_centro = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame='icrs')
                                else:
                                    coords_centro = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg),
                                                             frame='icrs')

                                separazione = coords_centro.separation(coords_mrk421)

                                if separazione <= raggio_fov_tolleranza:
                                    file_validi.append({
                                        'percorso_originale': percorso_file,
                                        'nome_file': percorso_file.name,
                                        'nome_giorno': cartella_giorno.name,
                                        'tempo': Time(tempo_obs_str) if tempo_obs_str else Time(
                                            os.path.getmtime(percorso_file), format='unix'),
                                        'dej2000': coords_centro.dec.deg
                                    })
                            except Exception:
                                continue
                except Exception:
                    continue

    if not file_validi:
        print("\nERRORE: Nessun file trovato.")
        sys.exit()

    # ordino e creo i miei run
    file_validi.sort(key=lambda x: x['tempo'])

    soglia_tempo = 600.0
    soglia_spazio = 0.2
    contatore_run = 1
    tempo_precedente = None
    dej2000_precedente = None

    runs_dict = {}

    for dato in file_validi:
        if tempo_precedente is not None:
            delta_tempo = (dato['tempo'] - tempo_precedente).sec
            delta_spazio = abs(dato['dej2000'] - dej2000_precedente)
            if delta_tempo > soglia_tempo or delta_spazio > soglia_spazio:
                contatore_run += 1

        nome_run = f"run_{contatore_run:08d}"
        giorno = dato['nome_giorno']

        if giorno not in runs_dict:
            runs_dict[giorno] = {}
        if nome_run not in runs_dict[giorno]:
            runs_dict[giorno][nome_run] = []

        runs_dict[giorno][nome_run].append(dato['percorso_originale'])

        tempo_precedente = dato['tempo']
        dej2000_precedente = dato['dej2000']

    print(f"Creati raggruppamenti per {contatore_run} run.")

    # =================================================================
    # --- PRE-CALCOLO SATELLITI GLOBALE ---
    # =================================================================
    print("\nPreparo il catalogo satelliti storici...")

    file_fits_riferimento = file_validi[0]['percorso_originale']

    hdu_ref = fits.open(file_fits_riferimento)
    tempo_ref_astropy = Time(hdu_ref[0].header['DATE-OBS'], format='isot', scale='utc')
    hdu_ref.close()

    cartella_tabelle = BASE_DIR / "tabelle_blazar_e_non_blazar"
    cartella_tabelle.mkdir(parents=True, exist_ok=True)

    percorso_tle = scarica_tle_storici(tempo_ref_astropy, tuo_user, tua_password, cartella_tabelle)

    if percorso_tle:
        satelliti_attivi = load.tle_file(percorso_tle)
        print(f"Download satelliti avvenuto: {len(satelliti_attivi)} satelliti trovati")
    else:
        print("ATTENZIONE: Download fallito. Disabilito il filtro satelliti.")
        satelliti_attivi = []
    # =================================================================

    # eseguo il ciclo per ogni run trovato
    for cartella_giorno, runs_nel_giorno in runs_dict.items():
        for run_name, file_list in runs_nel_giorno.items():

            print(f"\n==================== ELABORAZIONE {cartella_giorno} - {run_name} ====================")

            file_list = sorted([str(f) for f in file_list])
            if not file_list:
                print(f"Nessun FITS in {run_name}, salto.")
                continue

            output_dir = cartella_tabelle / cartella_giorno / run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Cartella di output: {output_dir}")

            print(f"--- FASE 1: Segmentazione & Unione ({len(file_list)} files) ---")

            for n, percorso_file in enumerate(tqdm(file_list, desc=f"Fase 1 {run_name}"), 1):
                if n == 1:
                    hdu_list = fits.open(percorso_file)
                    w = WCS(hdu_list[0].header)
                    ra_c, dec_c = hdu_list[0].header["RA"], hdu_list[0].header["DEC"]

                    alto_destra = w.pixel_to_world(3071, 2047)
                    centro = SkyCoord(ra_c, dec_c, unit=u.deg)

                    raggio_ricerca = Angle(centro.separation(alto_destra) * 1.5, "deg")

                    hdu_list.close()

                    riquadro_esterno_vizier = vizier.query_region(
                        coord.SkyCoord(ra=ra_c, dec=dec_c, unit=(u.deg, u.deg), frame='icrs'),
                        radius=raggio_ricerca,
                        column_filters={'rmag': f'<{15}'}
                    )
                    tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]

                    distanze_hip = centro.separation(coords_hipparco_global)

                    mask_hip_fov = distanze_hip < raggio_ricerca

                    tbl_hipparco_run_subset = tbl_catalogo_hipparco[mask_hip_fov]
                    coords_hipparco_run_subset = coords_hipparco_global[mask_hip_fov]
                    exclusion_radii_run_subset = exclusion_radii_deg[mask_hip_fov]

                    # =================================================================
                    # --- FILTRAGGIO COMPETITIVO A SINGOLA FASE ---
                    # =================================================================
                    print("Avvio filtraggio competitivo a singola fase Vizier vs Hipparcos...")

                    coords_vizier = SkyCoord(ra=tbl_riquadro_esterno_vizier['RAJ2000'],
                                             dec=tbl_riquadro_esterno_vizier['DEJ2000'],
                                             unit=u.deg)

                    max_threshold_deg = np.max(exclusion_radii_run_subset)
                    seplimit = max_threshold_deg * u.deg

                    idx_A, idx_B, d2d_1, _ = coords_hipparco_run_subset.search_around_sky(coords_vizier, seplimit)

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

                    for i_hip in unique_hip_idx:
                        viz_matches = idx_viz_valid[idx_hip_valid == i_hip]

                        if len(viz_matches) > 0:
                            mag_viz_matches = np.nan_to_num(tbl_riquadro_esterno_vizier['rmag'][viz_matches], nan=99.0)

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

                    mask_keep_hipparco[tbl_hipparco_run_subset['Vmag'] >= 15] = False

                    tbl_hipparco_run_clean = tbl_hipparco_run_subset[mask_keep_hipparco].copy()
                    tbl_riquadro_esterno_vizier_CLEAN = tbl_riquadro_esterno_vizier[mask_keep_vizier]

                    # calcolo la mia magnitudine sintetica combinata usando rmag come base
                    colore_g_r_temp = tbl_riquadro_esterno_vizier_CLEAN['gmag'] - tbl_riquadro_esterno_vizier_CLEAN['rmag']
                    mag_sintetica_temp = tbl_riquadro_esterno_vizier_CLEAN['rmag'] - 0.587 * colore_g_r_temp - 0.011
                    tbl_vizier_cut = tbl_riquadro_esterno_vizier_CLEAN[
                        mag_sintetica_temp < magnitudine_massima].copy()

                    # applico la mia nuova formula lineare: rmag come base della combinazione
                    colore_g_r = tbl_vizier_cut['gmag'] - tbl_vizier_cut['rmag']
                    tbl_vizier_cut['Mag'] = tbl_vizier_cut['rmag'] - 0.587 * colore_g_r - 0.011

                    tbl_hipparco_run_clean['Mag'] = tbl_hipparco_run_clean['Vmag']

                tbl_catalogate = tabella_catalogo(percorso_file, tbl_vizier_cut, tbl_hipparco_run_clean)
                tbl_trovate, _ = analisi_image_segmentation(percorso_file, parametri_caricati)

                df_trovate = tbl_trovate.to_pandas()
                df_catalogate = tbl_catalogate.to_pandas()

                all_cols = df_trovate.columns.tolist()
                cols_keep = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']
                for c in ['saturazione', 'kron_flux']:
                    if c in all_cols: cols_keep.append(c)
                extra_flux = ['kron_manuale_seg', 'kron_manuale_aper', 'somma_apertura_ultimo_pixel',
                              'raggio_kron_aper']
                for c in extra_flux:
                    if c in all_cols: cols_keep.append(c)

                df_trovate = df_trovate[[c for c in cols_keep if c in df_trovate.columns]].copy()

                with fits.open(percorso_file, memmap=False) as hdu:
                    w = WCS(hdu[0].header)
                    header_date_obs = hdu[0].header['DATE-OBS']
                coords = w.pixel_to_world(df_trovate['xcentroid'], df_trovate['ycentroid'])
                df_trovate['RA_centroid'] = coords.ra.deg
                df_trovate['DEC_centroid'] = coords.dec.deg

                cols_order = df_trovate.columns.tolist()
                if 'ycentroid' in cols_order:
                    for c in ['RA_centroid', 'DEC_centroid']:
                        if c in cols_order: cols_order.remove(c)
                    idx_y = cols_order.index('ycentroid')
                    cols_order.insert(idx_y + 1, 'RA_centroid')
                    cols_order.insert(idx_y + 2, 'DEC_centroid')
                    df_trovate = df_trovate[cols_order]

                    if 'RAJ2000' in df_catalogate.columns:
                        c_cat = SkyCoord(ra=df_catalogate['RAJ2000'].values * u.deg,
                                         dec=df_catalogate['DEJ2000'].values * u.deg)
                        idx_t, idx_c, d2d, _ = c_cat.search_around_sky(coords, soglia_correlazione)

                        matches = pd.DataFrame(
                            {'idx_t': idx_t, 'idx_c': idx_c, 'dist': d2d.deg,
                             'mag': df_catalogate.iloc[idx_c]['Mag'].values})
                        matches.sort_values(by=['idx_t', 'mag'], inplace=True)
                        matches['rank'] = matches.groupby('idx_t').cumcount() + 1
                        matches['Corrispondenza'] = 'SI (Rank ' + matches['rank'].astype(str) + ')'

                        df_si = pd.concat([
                            df_trovate.iloc[matches['idx_t']].reset_index(drop=True),
                            matches[['Corrispondenza']].reset_index(drop=True),
                            df_catalogate.iloc[matches['idx_c']].reset_index(drop=True)
                        ], axis=1)

                        unmatched = list(set(range(len(df_trovate))) - set(matches['idx_t']))
                        df_no = df_trovate.iloc[unmatched].copy()
                        df_no['Corrispondenza'] = 'NO'
                        for c in df_catalogate.columns: df_no[c] = np.nan

                        '''tempo_scatto_astropy = Time(header_date_obs, format='isot', scale='utc')
                        tempo_skyfield = ts.from_astropy(tempo_scatto_astropy)

                        ra_sat_list, dec_sat_list = [], []
                        for sat in satelliti_attivi:
                            topocentrica = (sat - osservatorio).at(tempo_skyfield)
                            ra_sat, dec_sat, _ = topocentrica.radec()

                            if np.isnan(ra_sat.hours) or np.isnan(dec_sat.degrees):
                                continue

                            ra_sat_list.append(ra_sat.hours * 15)
                            dec_sat_list.append(dec_sat.degrees)

                        if ra_sat_list and len(df_no) > 0:
                            catalogo_satelliti = SkyCoord(ra=ra_sat_list * u.deg, dec=dec_sat_list * u.deg)

                            coords_oggetti_no = SkyCoord(ra=df_no['RA_centroid'].values * u.deg,
                                                         dec=df_no['DEC_centroid'].values * u.deg)
                            idx_sat, d2d_sat, _ = coords_oggetti_no.match_to_catalog_sky(catalogo_satelliti)

                            tolleranza_satellite = 3 / 60 * u.deg
                            mask_is_satellite = d2d_sat < tolleranza_satellite

                            # elimino i falsi positivi causati dai satelliti
                            contatore_satelliti = contatore_satelliti + np.sum(mask_is_satellite)
                            contatore_satelliti_presenti = contatore_satelliti_presenti + len(catalogo_satelliti)'''

                        df_final = pd.concat([df_si, df_no], ignore_index=True)

                    else:
                        df_final = df_trovate.copy()
                        df_final['Corrispondenza'] = 'NO'

                # =================================================================
                # INIZIO BLOCCO: TRACKING GLOBALE OTTIMIZZATO (cKDTree - SCIPY)
                # =================================================================
                ra_rad = np.radians(df_final['RA_centroid'].values)
                dec_rad = np.radians(df_final['DEC_centroid'].values)
                x = np.cos(dec_rad) * np.cos(ra_rad)
                y = np.cos(dec_rad) * np.sin(ra_rad)
                z = np.sin(dec_rad)
                coords_cart = np.column_stack((x, y, z))

                soglia_rad = np.radians(dist_ripetizione.value)
                soglia_3d = 2.0 * np.sin(soglia_rad / 2.0)

                final_labels = np.empty(len(df_final), dtype=object)

                if global_tracker_coords is None:
                    # al primo giro, inizializzo la mia memoria storica
                    global_tracker_coords = coords_cart
                    global_tracker_labels = [f"RA_{ra:.3f}DEC{dec:.3f}" for ra, dec in
                                             zip(df_final['RA_centroid'].values, df_final['DEC_centroid'].values)]
                    final_labels[:] = global_tracker_labels
                else:
                    # creo un albero KD per una ricerca spaziale ultra veloce
                    albero = cKDTree(global_tracker_coords)
                    distanze, indici = albero.query(coords_cart, distance_upper_bound=soglia_3d)

                    # trovo quali oggetti hanno un match sotto la mia soglia
                    mask_match = distanze <= soglia_3d

                    # assegno le etichette già note
                    for i in np.where(mask_match)[0]:
                        final_labels[i] = global_tracker_labels[indici[i]]

                    # isolo i miei nuovi oggetti non trovati
                    nuovi_idx = np.where(~mask_match)[0]
                    if len(nuovi_idx) > 0:
                        nuove_coords_cart = coords_cart[nuovi_idx]
                        nuove_labels = [
                            f"RA_{df_final['RA_centroid'].values[i]:.3f}__DEC_{df_final['DEC_centroid'].values[i]:.3f}"
                            for i in nuovi_idx
                        ]

                        for i, l_idx in enumerate(nuovi_idx):
                            final_labels[l_idx] = nuove_labels[i]

                        # aggiorno la mia memoria accodando i nuovi array in puro numpy
                        global_tracker_coords = np.vstack([global_tracker_coords, nuove_coords_cart])
                        global_tracker_labels.extend(nuove_labels)

                df_final['label'] = final_labels

                # aggiungo le colonne identificative
                df_final['run_id'] = run_name
                df_final['img_index'] = n

                # =================================================================
                # FINE BLOCCO TRACKING
                # =================================================================

                if 'label' in df_final.columns: df_final.sort_values('label', inplace=True)

                cols = df_final.columns.tolist()
                if 'ID' in cols and 'Catalogo' in cols:
                    cols.remove('Catalogo')
                    cols.insert(cols.index('ID'), 'Catalogo')

                final_cols = df_final.columns.tolist()
                for c in ['run_id', 'img_index']:
                    if c in final_cols: final_cols.remove(c)

                if 'Corrispondenza' in final_cols:
                    idx_corr = final_cols.index('Corrispondenza')
                    final_cols.insert(idx_corr, 'img_index')
                    final_cols.insert(idx_corr, 'run_id')
                else:
                    final_cols.insert(0, 'run_id')
                    final_cols.insert(1, 'img_index')

                df_final = df_final[final_cols]

                file_out = output_dir / f'{run_name}_stelle_trovate_e_catalogate_immagine_{n:03d}.csv'
                salva_csv_con_header_fits(df_final, dict(fits.getheader(percorso_file)),
                                          file_out, str(percorso_file), parametri_caricati)

            # =============================================================================
            # FASE 2 & 3: RAGGI MAX E FLUSSO FISSO (PER RUN)
            # =============================================================================
            print(f"--- FASE 2 & 3: Analisi Fotometria Fissa per {run_name} ---")

            file_csv_list = sorted([f for f in output_dir.glob('*.csv')])

            for f in file_csv_list:
                tutti_i_file_csv_generati.append((f, run_name))

            all_ids = []
            all_radii = []

            for file_csv in tqdm(file_csv_list, desc="Scansione Raggi"):
                try:
                    df_temp = pd.read_csv(file_csv, comment='#', usecols=['ID', 'raggio_kron_aper'])
                    df_temp = df_temp.dropna(subset=['ID', 'raggio_kron_aper'])
                    all_ids.append(df_temp['ID'].values)
                    all_radii.append(df_temp['raggio_kron_aper'].values)
                except Exception:
                    pass

            if len(all_ids) > 0:
                big_ids = np.concatenate(all_ids)
                big_radii = np.concatenate(all_radii)
                df_global = pd.DataFrame({'ID': big_ids, 'R': big_radii})
                map_raggi_max = df_global.groupby('ID')['R'].quantile(0.95).to_dict()
            else:
                map_raggi_max = {}


            def salva_csv_con_header_aggiornato(df, header_dict, output_file):
                with open(output_file, 'w') as f:
                    f.write("# Header FITS:\n")
                    for k, v in header_dict.items(): f.write(f"# {k}: {v}\n")
                    f.write("#\n")
                    df.to_csv(f, index=False)


            for file_csv in tqdm(file_csv_list, desc="Ricalcolo Flussi"):
                df_frame = pd.read_csv(file_csv, comment='#')
                header_info = leggi_header_da_csv(file_csv)

                path_fits = header_info.get('PERCORSO_FILE', '')
                nome_fits = header_info.get('NOME_FILE_FITS', '')

                if not path_fits or not os.path.exists(path_fits):
                    if path_fits:
                        p_obj = Path(path_fits)
                        try:
                            if "pmc_photometry" in p_obj.parts:
                                idx_part = p_obj.parts.index("pmc_photometry")
                                new_path = BASE_DIR.joinpath(*p_obj.parts[idx_part + 1:])
                                if new_path.exists():
                                    path_fits = str(new_path)
                        except:
                            pass

                    if (not path_fits or not os.path.exists(path_fits)) and nome_fits:
                        found = cerca_file_nel_progetto(BASE_DIR, str(nome_fits).strip())
                        if found:
                            path_fits = str(found)

                if not path_fits or not os.path.exists(path_fits):
                    print(f"ATTENZIONE: File FITS {path_fits} originale non trovato per {nome_fits}, salto.")
                    continue

                with fits.open(path_fits, memmap=False) as hdu:
                    data_fits = hdu[0].data
                    _, median_bg, _ = sigma_clipped_stats(data_fits[::10, ::10], sigma=3.0)
                    data_sub = data_fits - median_bg

                raggi_fissi = []
                ids_presenti = df_frame['ID'].values
                flussi_calcolati = []

                for idx_star, star_id in enumerate(ids_presenti):
                    r_globale = map_raggi_max.get(star_id, np.nan)
                    if np.isnan(r_globale) or r_globale <= 0:
                        if 'raggio_kron_aper' in df_frame.columns:
                            r_globale = df_frame.at[idx_star, 'raggio_kron_aper']
                        else:
                            r_globale = np.nan
                    raggi_fissi.append(r_globale)

                    if r_globale > 0 and not np.isnan(r_globale):
                        pos = (df_frame.at[idx_star, 'xcentroid'], df_frame.at[idx_star, 'ycentroid'])
                        aper = CircularAperture(pos, r=r_globale)
                        phot = aperture_photometry(data_sub, aper)
                        flussi_calcolati.append(phot['aperture_sum'][0])
                    else:
                        flussi_calcolati.append(np.nan)

                df_frame['flusso_fisso_max_run'] = flussi_calcolati
                df_frame['raggio_fisso_max_run'] = raggi_fissi

                df_frame['flusso_fisso_max_run'] = df_frame['flusso_fisso_max_run'].map(
                    lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')
                df_frame['raggio_fisso_max_run'] = df_frame['raggio_fisso_max_run'].map(
                    lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')

                if 'label' in df_frame.columns: df_frame.sort_values(by=['label', 'Corrispondenza'], inplace=True)
                salva_csv_con_header_aggiornato(df_frame, header_info, file_csv)

    # =============================================================================
    # FASE FINALE GLOBALE: STATISTICHE E ID SU TUTTE LE RUN
    # =============================================================================
    print("\n==================== FASE FINALE GLOBALE (TUTTE LE RUN) ====================")
    print(f"Elaborazione di {len(tutti_i_file_csv_generati)} file totali...")

    if not tutti_i_file_csv_generati:
        print("Nessun file generato. Esco.")
        exit()

    lista_df = []
    tutti_i_file_csv_generati = sorted(tutti_i_file_csv_generati, key=lambda x: str(x[0]))

    for idx_file, (file_csv, run_number) in enumerate(tqdm(tutti_i_file_csv_generati, desc="Lettura Dati Globali")):
        df_temp = pd.read_csv(file_csv, comment='#')
        df_temp['file_index'] = idx_file
        df_temp['run_number'] = run_number
        df_temp['original_file_path'] = str(file_csv)
        df_temp['original_idx'] = df_temp.index
        lista_df.append(df_temp)

    big_df = pd.concat(lista_df, ignore_index=True)

    big_df['run_unique_id'] = np.nan
    big_df['run_unique_id'] = big_df['run_unique_id'].astype(object)

    mask_si = big_df['Corrispondenza'].str.startswith('SI', na=False)
    big_df.loc[mask_si, 'run_unique_id'] = "CAT_" + big_df.loc[mask_si, 'ID'].astype(str)

    mask_no = big_df['Corrispondenza'] == 'NO'
    df_no = big_df[mask_no].copy()

    df_no.sort_values('file_index', inplace=True)

    known_clusters_coords = []
    known_clusters_ids = []
    threshold_deg = 35 / 3600
    unique_files = df_no['file_index'].unique()
    next_internal_id = 1
    no_mapping = {}

    for f_idx in tqdm(unique_files, desc="Matching oggetti NO (Multi-Run)"):
        subset = df_no[df_no['file_index'] == f_idx]
        if subset.empty: continue
        coords_subset = SkyCoord(ra=subset['RA_centroid'].values * u.deg,
                                 dec=subset['DEC_centroid'].values * u.deg)
        indices_subset = subset.index.tolist()

        if not known_clusters_coords:
            for i, (ra, dec) in enumerate(zip(subset['RA_centroid'], subset['DEC_centroid'])):
                cid = f"INT_{next_internal_id}"
                known_clusters_coords.append((ra, dec))
                known_clusters_ids.append(cid)
                no_mapping[indices_subset[i]] = cid
                next_internal_id += 1
        else:
            cluster_sc = SkyCoord(known_clusters_coords, unit=u.deg)
            idx_cluster, d2d, _ = coords_subset.match_to_catalog_sky(cluster_sc)
            for i, (match_idx, dist, ra_curr, dec_curr) in enumerate(
                    zip(idx_cluster, d2d, subset['RA_centroid'], subset['DEC_centroid'])):
                global_idx = indices_subset[i]

                # applico la modifica per aggiornare la memoria e risolvere il doppio append
                if dist.deg <= threshold_deg:
                    no_mapping[global_idx] = known_clusters_ids[match_idx]
                    # aggiorno le coordinate in memoria per seguire lo spostamento
                    known_clusters_coords[match_idx] = (ra_curr, dec_curr)
                else:
                    cid = f"INT_{next_internal_id}"
                    known_clusters_ids.append(cid)
                    known_clusters_coords.append((ra_curr, dec_curr))
                    no_mapping[global_idx] = cid
                    next_internal_id += 1

    for idx, uid in no_mapping.items(): big_df.at[idx, 'run_unique_id'] = uid

    # =================================================================
    # FILTRO TEMPORALE E FILTRO RIPETIZIONI (< 2)
    # =================================================================
    print("Eseguo il filtraggio temporale e delle ripetizioni minime...")

    # applico il filtro di prossimità temporale
    big_df['da_eliminare_temporale'] = False
    mask_no_temp = big_df['Corrispondenza'] == 'NO'

    for uid, group in big_df[mask_no_temp].groupby('run_unique_id'):
        indici_file = group['file_index'].values
        for idx_row, f_idx in zip(group.index, indici_file):
            # cerco se esiste almeno una rilevazione dello stesso oggetto nel range di 2 immagini adiacenti
            vicini = [x for x in indici_file if x != f_idx and abs(x - f_idx) <= 2]
            if len(vicini) == 0:
                # marco la singola rilevazione isolata per l'eliminazione
                big_df.at[idx_row, 'da_eliminare_temporale'] = True

    # elimino materialmente le righe isolate temporalmente
    big_df = big_df[~big_df['da_eliminare_temporale']].drop(columns=['da_eliminare_temporale'])

    # applico il filtro per numero totale di ripetizioni
    conteggi_aggiornati = big_df['run_unique_id'].value_counts()
    id_da_scartare = conteggi_aggiornati[conteggi_aggiornati < 2].index

    # creo la maschera ed elimino i NO con meno di 2 ripetizioni totali
    mask_da_scartare_rip = (big_df['Corrispondenza'] == 'NO') & (big_df['run_unique_id'].isin(id_da_scartare))
    big_df = big_df[~mask_da_scartare_rip]
    # =================================================================

    print("Calcolo statistiche globali e riorganizzazione colonne...")
    cols_flux = ['somma_apertura_ultimo_pixel', 'kron_manuale_seg', 'kron_manuale_aper', 'flusso_fisso_max_run']
    cols_flux_presenti = [c for c in cols_flux if c in big_df.columns]
    for c in cols_flux_presenti: big_df[c] = pd.to_numeric(big_df[c], errors='coerce')

    grouped_per_run = big_df.groupby(['run_unique_id', 'run_number'])

    stat_columns = []
    for c in cols_flux_presenti:
        col_mean = f'media_{c}'
        col_std = f'std_{c}'

        big_df[col_mean] = grouped_per_run[c].transform('mean')

        stds_sample = grouped_per_run[c].transform('std')
        counts_grouped = grouped_per_run[c].transform('count')
        big_df[col_std] = stds_sample / np.sqrt(counts_grouped)

        stat_columns.extend([col_mean, col_std])

    for c in stat_columns: big_df[c] = big_df[c].map(lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')

    big_df['ID'] = big_df['ID'].astype(object)

    mask_no_match = big_df['Corrispondenza'] == 'NO'
    big_df.loc[mask_no_match, 'ID'] = big_df.loc[mask_no_match, 'run_unique_id']

    files_groups = big_df.groupby('original_file_path')


    def salva_finale_global(df, header_dict, output_file, fp_count):
        nome_solo = os.path.basename(str(output_file))
        with open(output_file, 'w') as f:
            f.write("# Header FITS:\n")
            f.write(f"# Numero di falsi positivi esclusi sicuramente: {fp_count}\n")
            for k, v in header_dict.items():
                if k != 'PERCORSO_FILE':
                    f.write(f"# {k}: {v}\n")
            f.write(f"# NOME_FILE: {nome_solo}\n")
            f.write("#\n")
            df.to_csv(f, index=False)


    global_repetition_counts = big_df['run_unique_id'].value_counts()

    for file_path, df_file in tqdm(files_groups, desc="Salvataggio file globali aggiornati"):

        current_run_num = df_file['run_number'].iloc[0]

        col_rip_name = 'ripetizioni'

        df_file[col_rip_name] = df_file['run_unique_id'].map(global_repetition_counts)

        df_final_save = df_file.copy()
        num_falsi_positivi = 0

        header_orig = leggi_header_da_csv(file_path)

        cols = df_final_save.columns.tolist()
        for temp_c in ['file_index', 'original_file_path', 'original_idx', 'run_unique_id', 'run_number']:
            if temp_c in cols: cols.remove(temp_c)

        if col_rip_name in cols: cols.remove(col_rip_name)
        if 'saturazione' in cols:
            cols.insert(cols.index('saturazione') + 1, col_rip_name)
        else:
            cols.append(col_rip_name)

        for c_flux in cols_flux_presenti:
            c_mean, c_std = f'media_{c_flux}', f'std_{c_flux}'
            if c_flux in cols and c_mean in cols:
                cols.remove(c_mean)
                cols.remove(c_std)
                idx_flux = cols.index(c_flux)
                cols.insert(idx_flux + 1, c_mean)
                cols.insert(idx_flux + 2, c_std)

        df_final_save = df_final_save[cols]
        salva_finale_global(df_final_save, header_orig, file_path, num_falsi_positivi)

    print("\n--- ELABORAZIONE GLOBALE MULTI-RUN COMPLETATA CON SUCCESSO ---")