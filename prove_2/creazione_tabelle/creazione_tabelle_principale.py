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

# Eseguo l'import fondamentale per la mia portabilità
from pathlib import Path

'''
# Importo il catalogo dei satelliti
from skyfield.api import load, wgs84
from astropy.time import Time
import requests
from datetime import timedelta
'''

# Gestisco i warning ignorandoli
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")

PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

from funzioni.utilita import *
from funzioni.astrometria import *

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"Moduli esterni caricati con successo.")
print(f"------------------------------")

'''
# 1. Inizializzo Skyfield
ts = load.timescale()
'''

# 2. Imposto le coordinate del mio telescopio usando la mia funzione importata
lat_oss, lon_oss, alt_oss = ottieni_coordinate_telescopio('ASTRI 1', BASE_DIR)

'''
# Creo il mio oggetto geografico wgs84
osservatorio = wgs84.latlon(lat_oss, lon_oss, elevation_m=alt_oss)
'''

# Definisco le mie run da analizzare
RUN = [1, 2, 3]

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)

# =============================================================================
# BLOCCO DI ESECUZIONE (MAIN)
# =============================================================================

if __name__ == "__main__":

    # Impongo le unità di misura per risolvere il mio errore di matching
    soglia_correlazione = 35 / 3600 * u.deg
    dist_ripetizione = soglia_correlazione
    magnitudine_massima = 15

    nome_params = 'parametri_image_segmentation.txt'
    file_parametri = cerca_file_nel_progetto(BASE_DIR, nome_params)
    if file_parametri is None:
        print("File dei parametri non trovato")
        exit()
    parametri_caricati = leggi_file_parametri(file_parametri)

    # Eseguo il pre-calcolo globale di Hipparcos
    file_hipparco = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
    hdu_list_hipparco = fits.open(file_hipparco)
    tbl_catalogo_hipparco = Table(hdu_list_hipparco[1].data)
    hdu_list_hipparco.close()

    # Calcolo i miei errori propagati al J2000
    dt = 2000.0 - 1991.25
    sigma_ra_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco['e_RAICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco['e_pmRA'])) ** 2) / 3600000.0
    sigma_dec_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco['e_DEICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco['e_pmDE'])) ** 2) / 3600000.0

    # Calcolo il mio errore radiale totale di Hipparcos
    sigma_hip_deg = np.sqrt(sigma_ra_deg ** 2 + sigma_dec_deg ** 2)

    # Imposto l'errore stimato per Vizier
    sigma_vizier_deg = 0.1 / 3600.0

    # Sommo in quadratura i due cataloghi
    sigma_totale_deg = np.sqrt(sigma_hip_deg ** 2 + sigma_vizier_deg ** 2)

    # Imposto la mia soglia a 3-SIGMA
    exclusion_radii_deg_ = 3.0 * sigma_totale_deg
    exclusion_radii_deg = np.full(len(exclusion_radii_deg_), 2.5 / 3600.0)

    print(f"Raggio di merging tra i cataloghi: {np.mean(exclusion_radii_deg)}")

    # Creo il mio SkyCoord Hipparcos
    coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco['_RAJ2000'],
                                      dec=tbl_catalogo_hipparco['_DEJ2000'],
                                      unit=u.deg)

    tuo_user = "lorenzo.simeone@studenti.unipg.it"
    tua_password = "Cazzata_2002348"

    '''
    # =================================================================
    # Eseguo il pre-calcolo globale dei miei satelliti
    # =================================================================
    print("\nPreparo il catalogo satelliti storici...")

    file_fits_riferimento = None

    # Cerco un file fits di riferimento dalla mia prima run disponibile
    for r in RUN:
        cartella_run_temp = list(BASE_DIR.rglob(f"20250120_run{r}"))
        if cartella_run_temp:
            f_list = list(cartella_run_temp[0].glob('*.fit')) + list(cartella_run_temp[0].glob('*.fits')) + list(
                cartella_run_temp[0].glob('*.FIT')) + list(cartella_run_temp[0].glob('*.FITS'))
            if f_list:
                file_fits_riferimento = f_list[0]
                break

    if file_fits_riferimento:
        hdu_ref = fits.open(file_fits_riferimento)
        tempo_ref_astropy = Time(hdu_ref[0].header['DATE-OBS'], format='isot', scale='utc')
        hdu_ref.close()

        cartella_tabelle = cerca_cartella_nel_progetto(BASE_DIR, 'tabelle')
        if cartella_tabelle is None:
            cartella_tabelle = BASE_DIR / "tabelle"
        cartella_tabelle.mkdir(exist_ok=True)

        percorso_tle = scarica_tle_storici(tempo_ref_astropy, tuo_user, tua_password, cartella_tabelle)

        if percorso_tle:
            satelliti_attivi = load.tle_file(percorso_tle)
            print(f"Download satelliti avvenuto: {len(satelliti_attivi)} satelliti trovati")
        else:
            print("ATTENZIONE: Download fallito. Disabilito il filtro satelliti.")
            satelliti_attivi = []
    else:
        print("ATTENZIONE: Nessun FITS trovato per determinare la data. Disabilito il filtro satelliti.")
        satelliti_attivi = []
    # =================================================================
    '''

    next_internal_id = 1

    # Inizio il ciclo per ogni mia run
    for run in RUN:
        print(f"\n==================== ELABORAZIONE RUN {run} ====================")
        nome_cartella_run = f"20250120_run{run}"
        found_folders = list(BASE_DIR.rglob(nome_cartella_run))
        if not found_folders:
            print(f"Run {run} non trovata, salto.")
            continue
        run_folder = found_folders[0]

        estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
        file_list = []
        for ext in estensioni_valide:
            file_list.extend(run_folder.glob(ext))

        file_list = sorted([str(f) for f in file_list])
        if not file_list:
            print(f"Nessun FITS in Run {run}, salto.")
            continue

        # =============================================================================
        # DEFINIZIONE E CREAZIONE AUTOMATICA PERCORSI DI OUTPUT (CORRETTO)
        # =============================================================================
        # Sposto questo blocco fuori da ogni condizione restrittiva
        cartella_prove = BASE_DIR
        cartella_tabelle = cartella_prove / "tabelle"

        # Creo il percorso finale (mkdir con parents=True gestisce tutta la catena)
        output_dir = cartella_tabelle / "tabelle_unite" / f"tabelle_unite_run_{run}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Cartella di output verificata/creata: {output_dir}")

        # Inizializzo le mie variabili di tracking a zero per ogni singola run
        global_tracker_coords = None
        global_tracker_labels = []
        '''
        contatore_satelliti = 0
        contatore_satelliti_presenti = 0
        '''
        file_csv_generati_nella_run = []

        # Inizio la fase 1 per creare le mie tabelle unite
        print(f"--- FASE 1: Segmentazione & Unione ({len(file_list)} files) ---")

        for n, percorso_file in enumerate(tqdm(file_list, desc=f"Fase 1 Run {run}"), 1):
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
                    column_filters={'gmag': f'<{15}'}
                )
                tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]

                distanze_hip = centro.separation(coords_hipparco_global)

                mask_hip_fov = distanze_hip < raggio_ricerca

                tbl_hipparco_run_subset = tbl_catalogo_hipparco[mask_hip_fov]
                coords_hipparco_run_subset = coords_hipparco_global[mask_hip_fov]
                exclusion_radii_run_subset = exclusion_radii_deg[mask_hip_fov]

                # =================================================================
                # Avvio il filtraggio competitivo a singola fase
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

                mask_keep_hipparco[tbl_hipparco_run_subset['Vmag'] >= 15] = False

                tbl_hipparco_run_clean = tbl_hipparco_run_subset[mask_keep_hipparco]
                tbl_riquadro_esterno_vizier_CLEAN = tbl_riquadro_esterno_vizier[mask_keep_vizier]

                tbl_vizier_cut = tbl_riquadro_esterno_vizier_CLEAN[
                    tbl_riquadro_esterno_vizier_CLEAN['gmag'] < magnitudine_massima]

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

                    '''
                    tempo_scatto_astropy = Time(header_date_obs, format='isot', scale='utc')
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

                        # Elimino i miei falsi positivi causati dai satelliti
                        contatore_satelliti = contatore_satelliti + np.sum(mask_is_satellite)
                        contatore_satelliti_presenti = contatore_satelliti_presenti + len(catalogo_satelliti)
                    '''

                    df_final = pd.concat([df_si, df_no], ignore_index=True)

                else:
                    df_final = df_trovate.copy()
                    df_final['Corrispondenza'] = 'NO'

            # =================================================================
            # Avvio il mio blocco di tracking globale basato sulle coordinate
            # =================================================================

            # Creo il mio array astropy garantendo che sia sempre 1D
            coords_obj_all = SkyCoord(ra=np.atleast_1d(df_final['RA_centroid'].values) * u.deg,
                                      dec=np.atleast_1d(df_final['DEC_centroid'].values) * u.deg)
            final_labels = np.empty(len(df_final), dtype=object)

            if global_tracker_coords is None:
                # Inizializzo direttamente il mio tracker con tutte le coordinate
                global_tracker_coords = coords_obj_all
                global_tracker_labels = [f"RA_{ra:.3f}DEC{dec:.3f}" for ra, dec in
                                         zip(coords_obj_all.ra.deg, coords_obj_all.dec.deg)]
                final_labels[:] = global_tracker_labels
            else:
                # Eseguo il mio match vettorializzato su tutte le coordinate simultaneamente
                idx_match, d2d, _ = coords_obj_all.match_to_catalog_sky(global_tracker_coords)
                mask_match = d2d < dist_ripetizione

                # Assegno le mie etichette agli oggetti già noti
                for i in np.where(mask_match)[0]:
                    final_labels[i] = global_tracker_labels[idx_match[i]]

                # Isolo i miei nuovi oggetti non trovati
                nuovi_idx = np.where(~mask_match)[0]
                if len(nuovi_idx) > 0:
                    nuove_coords = coords_obj_all[nuovi_idx]
                    nuove_labels = [f"RA_{ra:.3f}__DEC_{dec:.3f}" for ra, dec in
                                    zip(nuove_coords.ra.deg, nuove_coords.dec.deg)]

                    for i, l_idx in enumerate(nuovi_idx):
                        final_labels[l_idx] = nuove_labels[i]

                    # Aggiorno il mio catalogo globale estraendo i valori grezzi per evitare errori di rappresentazione
                    nuovi_ra = np.concatenate([global_tracker_coords.ra.deg, nuove_coords.ra.deg])
                    nuovi_dec = np.concatenate([global_tracker_coords.dec.deg, nuove_coords.dec.deg])
                    global_tracker_coords = SkyCoord(ra=nuovi_ra * u.deg, dec=nuovi_dec * u.deg)

                    global_tracker_labels.extend(nuove_labels)

            df_final['label'] = final_labels

            # Aggiungo le mie colonne identificative
            df_final['run_id'] = run
            df_final['img_index'] = n

            # =================================================================
            # Termino il mio blocco di tracking
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

            file_out = output_dir / f'run_{run}_stelle_trovate_e_catalogate_immagine_{n:03d}.csv'
            salva_csv_con_header_fits(df_final, dict(fits.getheader(percorso_file)),
                                      file_out, str(percorso_file), parametri_caricati)

        # =============================================================================
        # Avvio la fase 2 e 3 per estrarre i raggi massimi e il flusso fisso per la mia run
        # =============================================================================
        print(f"--- FASE 2 & 3: Analisi Fotometria Fissa per Run {run} ---")

        file_csv_list = sorted([f for f in output_dir.glob('*.csv')])

        for f in file_csv_list:
            file_csv_generati_nella_run.append((f, run))

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

            # ricavo il nome esatto del file FITS dall'header
            nome_fits = header_info.get('NOME_FILE_FITS', '')

            # nel caso in cui NOME_FILE_FITS fosse vuoto, lo estraggo dal PERCORSO_FILE
            if not nome_fits:
                percorso_raw = header_info.get('PERCORSO_FILE', '')
                nome_fits = os.path.basename(str(percorso_raw))

            nome_fits = str(nome_fits).strip()

            # cerco il file FITS ovunque all'interno del progetto
            file_trovato = cerca_file_nel_progetto(BASE_DIR, nome_fits)

            if file_trovato is None:
                print(f"ATTENZIONE: File FITS originale '{nome_fits}' non trovato all'interno del progetto, salto.")
                continue

            # converto l'oggetto Path in stringa per passarlo ad astropy
            path_fits = str(file_trovato)

            with fits.open(path_fits, memmap=False) as hdu:
                data_fits = hdu[0].data
                _, median_bg, _ = sigma_clipped_stats(data_fits[::10, ::10], sigma=3.0)
                data_sub = data_fits - median_bg

            raggi_fissi = []
            ids_presenti = df_frame['ID'].values
            flussi_calcolati = []

            for idx_star, star_id in enumerate(ids_presenti):
                r_globale = map_raggi_max.get(star_id, np.nan)

                # se il raggio globale non è valido, provo a prendere quello dell'apertura calcolata nell'immagine corrente
                if np.isnan(r_globale) or r_globale <= 0:
                    if 'raggio_kron_aper' in df_frame.columns:
                        r_globale = df_frame.at[idx_star, 'raggio_kron_aper']
                    else:
                        r_globale = np.nan

                raggi_fissi.append(r_globale)

                # calcolo il flusso usando l'apertura circolare fissa
                if r_globale > 0 and not np.isnan(r_globale):
                    pos = (df_frame.at[idx_star, 'xcentroid'], df_frame.at[idx_star, 'ycentroid'])
                    aper = CircularAperture(pos, r=r_globale)
                    phot = aperture_photometry(data_sub, aper)
                    flussi_calcolati.append(phot['aperture_sum'][0])
                else:
                    flussi_calcolati.append(np.nan)

            df_frame['flusso_fisso_max_run'] = flussi_calcolati
            df_frame['raggio_fisso_max_run'] = raggi_fissi

            # formatto le colonne a 2 cifre decimali
            df_frame['flusso_fisso_max_run'] = df_frame['flusso_fisso_max_run'].map(
                lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')
            df_frame['raggio_fisso_max_run'] = df_frame['raggio_fisso_max_run'].map(
                lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')

            if 'label' in df_frame.columns:
                df_frame.sort_values(by=['label', 'Corrispondenza'], inplace=True)

            salva_csv_con_header_aggiornato(df_frame, header_info, file_csv)

        # =============================================================================
        # Avvio la fase 4 per calcolare statistiche e ID per la mia singola run corrente
        # =============================================================================
        print(f"\n--- FASE 4: Statistiche Locali e Ripetizioni per Run {run} ---")

        if not file_csv_generati_nella_run:
            print(f"Nessun file generato nella run {run}. Salto Fase 4.")
            continue

        lista_df_run = []
        file_csv_generati_nella_run = sorted(file_csv_generati_nella_run, key=lambda x: str(x[0]))

        for idx_file, (file_csv, run_number) in enumerate(
                tqdm(file_csv_generati_nella_run, desc="Lettura Dati Run")):
            df_temp = pd.read_csv(file_csv, comment='#')
            df_temp['file_index'] = idx_file
            df_temp['run_number'] = run_number
            df_temp['original_file_path'] = str(file_csv)
            df_temp['original_idx'] = df_temp.index
            lista_df_run.append(df_temp)

        run_df = pd.concat(lista_df_run, ignore_index=True)

        run_df['run_unique_id'] = np.nan
        run_df['run_unique_id'] = run_df['run_unique_id'].astype(object)

        mask_si = run_df['Corrispondenza'].str.startswith('SI', na=False)
        run_df.loc[mask_si, 'run_unique_id'] = "CAT_" + run_df.loc[mask_si, 'ID'].astype(str)

        mask_no = run_df['Corrispondenza'] == 'NO'
        df_no_run = run_df[mask_no].copy()

        df_no_run.sort_values('file_index', inplace=True)

        known_clusters_coords_run = []
        known_clusters_ids_run = []
        threshold_deg = 35/3600
        unique_files_run = df_no_run['file_index'].unique()

        # Rimuovo l'inizializzazione del contatore da qui, avendola spostata all'esterno del ciclo run
        no_mapping = {}

        for f_idx in tqdm(unique_files_run, desc="Matching oggetti NO (Intra-Run)"):
            subset = df_no_run[df_no_run['file_index'] == f_idx]
            if subset.empty: continue
            coords_subset = SkyCoord(ra=subset['RA_centroid'].values * u.deg,
                                     dec=subset['DEC_centroid'].values * u.deg)
            indices_subset = subset.index.tolist()

            if not known_clusters_coords_run:
                for i, (ra, dec) in enumerate(zip(subset['RA_centroid'], subset['DEC_centroid'])):
                    cid = f"INT_{next_internal_id}"
                    known_clusters_coords_run.append((ra, dec))
                    known_clusters_ids_run.append(cid)
                    no_mapping[indices_subset[i]] = cid

                    # Incremento il mio contatore
                    next_internal_id += 1
            else:
                cluster_sc = SkyCoord(known_clusters_coords_run, unit=u.deg)
                idx_cluster, d2d, _ = coords_subset.match_to_catalog_sky(cluster_sc)
                for i, (match_idx, dist, ra_curr, dec_curr) in enumerate(
                        zip(idx_cluster, d2d, subset['RA_centroid'], subset['DEC_centroid'])):
                    global_idx = indices_subset[i]
                    if dist.deg <= threshold_deg:
                        no_mapping[global_idx] = known_clusters_ids_run[match_idx]

                        # Aggiorno le coordinate in memoria con quelle del frame corrente per seguire lo spostamento graduale
                        known_clusters_coords_run[match_idx] = (ra_curr, dec_curr)

                    else:
                        cid = f"INT_{next_internal_id}"
                        known_clusters_ids_run.append(cid)
                        known_clusters_coords_run.append((ra_curr, dec_curr))

                        no_mapping[global_idx] = cid

                        # Incremento il mio contatore per il prossimo oggetto
                        next_internal_id += 1

        for idx, uid in no_mapping.items(): run_df.at[idx, 'run_unique_id'] = uid

        # =================================================================
        # FILTRO TEMPORALE E FILTRO RIPETIZIONI (< 2)
        # =================================================================
        print("Eseguo il filtraggio temporale e delle ripetizioni minime...")

        # 1. applico il filtro di prossimità temporale (+/- 2 immagini)
        run_df['da_eliminare_temporale'] = False
        mask_no_temp = run_df['Corrispondenza'] == 'NO'

        for uid, group in run_df[mask_no_temp].groupby('run_unique_id'):
            indici_file = group['file_index'].values
            for idx_row, f_idx in zip(group.index, indici_file):
                # cerco se esiste almeno una rilevazione dello stesso oggetto nel range di 2 immagini adiacenti
                vicini = [x for x in indici_file if x != f_idx and abs(x - f_idx) <= 2]
                if len(vicini) == 0:
                    # marco la singola rilevazione isolata per l'eliminazione
                    run_df.at[idx_row, 'da_eliminare_temporale'] = True

        # elimino materialmente le righe isolate temporalmente
        run_df = run_df[~run_df['da_eliminare_temporale']].drop(columns=['da_eliminare_temporale'])

        # 2. applico il filtro per numero totale di ripetizioni (< 2)
        # ricalcolo i conteggi effettivi dopo la pulizia temporale precedente
        conteggi_aggiornati = run_df['run_unique_id'].value_counts()
        id_da_scartare = conteggi_aggiornati[conteggi_aggiornati < 2].index

        # creo la maschera ed elimino i NO con meno di 2 ripetizioni totali
        mask_da_scartare_rip = (run_df['Corrispondenza'] == 'NO') & (run_df['run_unique_id'].isin(id_da_scartare))
        run_df = run_df[~mask_da_scartare_rip]
        # =================================================================

        print("Calcolo statistiche di run e riorganizzazione colonne...")
        cols_flux = ['somma_apertura_ultimo_pixel', 'kron_manuale_seg', 'kron_manuale_aper', 'flusso_fisso_max_run']
        cols_flux_presenti = [c for c in cols_flux if c in run_df.columns]
        for c in cols_flux_presenti: run_df[c] = pd.to_numeric(run_df[c], errors='coerce')

        grouped_per_run_id = run_df.groupby(['run_unique_id'])

        stat_columns = []
        for c in cols_flux_presenti:
            col_mean = f'media_{c}'
            col_std = f'std_{c}'

            run_df[col_mean] = grouped_per_run_id[c].transform('mean')

            stds_sample = grouped_per_run_id[c].transform('std')
            counts_grouped = grouped_per_run_id[c].transform('count')
            run_df[col_std] = stds_sample / np.sqrt(counts_grouped)

            stat_columns.extend([col_mean, col_std])

        for c in stat_columns: run_df[c] = run_df[c].map(
            lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')

        run_df['ID'] = run_df['ID'].astype(object)

        mask_no_match = run_df['Corrispondenza'] == 'NO'
        run_df.loc[mask_no_match, 'ID'] = run_df.loc[mask_no_match, 'run_unique_id']

        files_groups_run = run_df.groupby('original_file_path')


        def salva_finale_locale(df, header_dict, output_file, fp_count):
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


        run_repetition_counts = run_df['run_unique_id'].value_counts()

        for file_path, df_file in tqdm(files_groups_run, desc=f"Salvataggio file finali per Run {run}"):

            col_rip_name = 'ripetizioni'

            df_file[col_rip_name] = df_file['run_unique_id'].map(run_repetition_counts)

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
            # Salvo i miei file finali con conteggi locali
            salva_finale_locale(df_final_save, header_orig, file_path, num_falsi_positivi)

    print("\n--- ELABORAZIONE COMPLETATA CON SUCCESSO ---")