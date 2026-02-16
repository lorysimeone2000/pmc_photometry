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

# --- IMPORT FONDAMENTALE PER LA PORTABILITÀ ---
from pathlib import Path

# catalogo satelliti
from skyfield.api import load, wgs84
from astropy.time import Time


# --- GESTIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)

import requests
from datetime import timedelta


def scarica_tle_storici(tempo_astropy, username, password, cartella_output):
    # converto il tempo astropy in un oggetto datetime standard di python
    data_osservazione = tempo_astropy.datetime

    # creo una finestra temporale di sicurezza: il giorno prima e il giorno dopo lo scatto
    data_inizio = (data_osservazione - timedelta(days=1)).strftime('%Y-%m-%d')
    data_fine = (data_osservazione + timedelta(days=1)).strftime('%Y-%m-%d')

    nome_file = f"tle_storico_{data_inizio}_to_{data_fine}.txt"
    percorso_output = cartella_output / nome_file

    # controllo se l'ho già scaricato per questa run, per non intasare i server
    if percorso_output.exists():
        print(f"TLE storici già presenti: {nome_file}")
        return str(percorso_output)

    print(f"Scaricando i TLE storici da Space-Track per le date {data_inizio} -> {data_fine}...")

    # URL per le API di Space-Track
    login_url = "https://www.space-track.org/ajaxauth/login"
    query_url = f"https://www.space-track.org/basicspacedata/query/class/tle/EPOCH/{data_inizio}--{data_fine}/orderby/EPOCH desc/format/tle"

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
            with open(percorso_output, 'w') as f:
                f.write(risposta_tle.text)
            print("Download TLE storici completato con successo!")
            return str(percorso_output)
        else:
            print(f"ERRORE: Download TLE fallito con codice {risposta_tle.status_code}")
            return None

# 1. Inizializzazione Skyfield e download database satelliti attivi (Celestrak)
ts = load.timescale()
url_satelliti = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
satelliti_attivi = load.tle_file(url_satelliti)

# 2. Coordinate del tuo telescopio (sostituisci con Latitudine, Longitudine e Altitudine reali)
osservatorio = wgs84.latlon(28.3000, -16.505830555555555, elevation_m=2370)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI DINAMICA (PORTABILITÀ TOTALE)
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    if len(cartelle_trovate) > 1:
        print(
            f"INFO: Trovate {len(cartelle_trovate)} cartelle '{nome_cartella_esatto}'. Uso la prima: {cartelle_trovate[0].relative_to(base_dir)}")
    return cartelle_trovate[0]


BASE_DIR = trova_cartella_base("pmc_photometry")
print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")

RUN = [1, 2, 3]

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)


# =============================================================================
# 1. FUNZIONI HELPER
# =============================================================================

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


def elabora_file_fits(percorso_file_):
    with fits.open(percorso_file_, memmap=False) as hdu_list_:
        image_data_ = hdu_list_[0].data
        w_ = WCS(hdu_list_[0].header)
        mean_, median_, std_ = sigma_clipped_stats(image_data_, sigma=3.0)
        image_data_ = image_data_ - median_
        return image_data_, median_, w_


def calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pixel, k=2.5, r_min=3.5):
    somma_intensita = np.sum(valori_pixel)
    if somma_intensita <= 0: return np.nan, np.nan
    somma_momenti = np.sum(valori_pixel * distanze_pixel)
    r_1 = somma_momenti / somma_intensita
    r_kron_finale = max(k * r_1, r_min)
    aper = CircularAperture((xc, yc), r=r_kron_finale)
    phot = aperture_photometry(data, aper)
    return phot['aperture_sum'][0], r_kron_finale


def tabella_catalogo(image_file_):
    hdu_list_ = fits.open(image_file_)
    wcs = WCS(hdu_list_[0].header)
    data_ = hdu_list_[0].data
    h, w = data_.shape
    bordo = 7

    nome_catalogo_vizier = np.array(["II/389/ps1_dr2"] * len(tbl_vizier_cut), dtype=object)
    colonne_vizier = {
        'Catalogo': nome_catalogo_vizier,
        'ID': tbl_vizier_cut['objID'],
        'RAJ2000': tbl_vizier_cut['RAJ2000'],
        'DEJ2000': tbl_vizier_cut['DEJ2000'],
        'Mag': tbl_vizier_cut['gmag'],
    }

    nome_catalogo_hipparco = np.array(["I/239/hip_main"] * len(tbl_hipparco_run_clean), dtype=object)
    colonne_hipparco = {
        'Catalogo': nome_catalogo_hipparco,
        'ID': tbl_hipparco_run_clean['HIP'],
        'RAJ2000': tbl_hipparco_run_clean['_RAJ2000'],
        'DEJ2000': tbl_hipparco_run_clean['_DEJ2000'],
        'Mag': tbl_hipparco_run_clean['Vmag'],
    }

    t1 = Table(colonne_vizier)
    t2 = Table(colonne_hipparco)

    tbl_unita = vstack([t1, t2])

    coords = SkyCoord(ra=tbl_unita['RAJ2000'], dec=tbl_unita['DEJ2000'], unit=u.deg)
    x_pix, y_pix = wcs.world_to_pixel(coords)

    mask_bordo = ((x_pix >= bordo) & (x_pix < (w - bordo)) & (y_pix >= bordo) & (y_pix < (h - bordo)))

    hdu_list_.close()
    return tbl_unita[mask_bordo]


def esegui_fotometria_variabile(data, positions, raggi):
    flussi = []
    for (xc, yc), r in zip(positions, raggi):
        if r > 0 and not np.isnan(r):
            aper = CircularAperture((xc, yc), r=r)
            phot = aperture_photometry(data, aper)
            flussi.append(phot['aperture_sum'][0])
        else:
            flussi.append(np.nan)
    return flussi


def salva_csv_con_header_fits(dataframe, header_fits, filename, nome_file_fits, parametri_seg=None):
    nome_solo = os.path.basename(str(nome_file_fits))
    with open(filename, 'w') as f:
        f.write("# Header FITS:\n")
        for key, value in header_fits.items():
            clean_val = str(value).replace('\n', ' ')
            f.write(f"# {key}: {clean_val}\n")
        f.write(f"# NOME_FILE_FITS: {nome_solo}\n")
        f.write("#\n# PARAMETRI SEGMENTAZIONE:\n")
        if parametri_seg:
            for key, value in parametri_seg.items():
                f.write(f"# {key}: {value}\n")
        f.write("#\n")
        dataframe.to_csv(f, index=False)


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
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            else:
                break
    return header_dict


# =============================================================================
# 2. FUNZIONE DI ANALISI PRINCIPALE
# =============================================================================

def analisi_image_segmentation(percorso_file_, parametri_globali):
    data, fondo_iniziale, w = elabora_file_fits(percorso_file_)
    fwhm = parametri_globali.get('fwhm', 3.0)
    size = parametri_globali.get('size', 5)
    threshold = parametri_globali.get('threshold_assoluta', 3.0)
    pixel_n = parametri_globali.get('pixel', 5)

    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    finder = SourceFinder(npixels=pixel_n, progress_bar=False)
    segment_map = finder(convolved_data, threshold)
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    if len(tbl) == 0: return tbl, parametri_globali

    for col in ['xcentroid', 'ycentroid', 'kron_flux']:
        tbl[col].info.format = '.2f'

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    livello_saturazione = 255 - fondo_iniziale - median
    tbl['saturazione'] = np.where(tbl['max_value'] >= livello_saturazione, 'SI', 'NO')

    K_KRON, R_MIN_KRON = 2.5, 3.5
    soglia_assoluta, soglia_relativa = 2.5, 0.05
    bordo, ny, nx = 7, data.shape[0], data.shape[1]

    lista_raggi_max, kron_manuale_seg, kron_manuale_aper, raggi_kron_aper, mask_keep = [], [], [], [], []

    for prop in cat:
        xc, yc = prop.xcentroid, prop.ycentroid
        if not ((xc >= bordo) and (xc < nx - bordo) and (yc >= bordo) and (yc < ny - bordo)):
            lista_raggi_max.append(0.5);
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan);
            raggi_kron_aper.append(np.nan);
            mask_keep.append(False)
            continue

        slices = prop.slices
        valori_pixel = data[prop.slices][segment_map.data[prop.slices] == prop.label]

        if len(valori_pixel) == 0:
            lista_raggi_max.append(0.5);
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan);
            raggi_kron_aper.append(np.nan);
            mask_keep.append(False)
            continue

        y_idx, x_idx = np.indices(segment_map.data[prop.slices].shape)
        ypix = y_idx[segment_map.data[prop.slices] == prop.label] + slices[0].start
        xpix = x_idx[segment_map.data[prop.slices] == prop.label] + slices[1].start
        distanze_pix = np.hypot(xpix - xc, ypix - yc)

        r_max_pix = max(np.max(distanze_pix) if len(distanze_pix) > 0 else 0.5, 0.5)
        lista_raggi_max.append(r_max_pix)

        r_int = int(np.ceil(r_max_pix))
        y_min, y_max = max(0, int(yc - r_int)), min(data.shape[0], int(yc + r_int + 1))
        x_min, x_max = max(0, int(xc - r_int)), min(data.shape[1], int(xc + r_int + 1))
        cutout = data[y_min:y_max, x_min:x_max]
        y_g, x_g = np.ogrid[y_min:y_max, x_min:x_max]
        dist_box = np.hypot(x_g - xc, y_g - yc)
        mask_circle = dist_box <= r_max_pix

        fl_aper, r_used = calcola_flusso_kron_completo(data, xc, yc, cutout[mask_circle], dist_box[mask_circle], K_KRON,
                                                       R_MIN_KRON)
        kron_manuale_aper.append(fl_aper)
        raggi_kron_aper.append(r_used)

        fl_seg, _ = calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pix, K_KRON, R_MIN_KRON)
        kron_manuale_seg.append(fl_seg)

        is_good = (np.sum(valori_pixel > soglia_assoluta) >= 3) and (
                np.sum(valori_pixel > soglia_relativa * prop.max_value) >= 2)
        mask_keep.append(is_good)

    tbl['kron_manuale_seg'] = kron_manuale_seg
    tbl['kron_manuale_aper'] = kron_manuale_aper
    tbl['raggio_kron_aper'] = raggi_kron_aper
    tbl['somma_apertura_ultimo_pixel'] = esegui_fotometria_variabile(data,
                                                                     np.transpose((tbl['xcentroid'], tbl['ycentroid'])),
                                                                     lista_raggi_max)

    for col in ['somma_apertura_ultimo_pixel', 'kron_manuale_seg', 'kron_manuale_aper', 'raggio_kron_aper']:
        tbl[col].info.format = '%.2f'

    tbl_filtrato = tbl[mask_keep]
    if len(tbl_filtrato) > 0: tbl_filtrato['label'] = np.arange(1, len(tbl_filtrato) + 1)

    return tbl_filtrato, parametri_globali


# =============================================================================
# 3. BLOCCO DI ESECUZIONE (MAIN)
# =============================================================================

if __name__ == "__main__":

    soglia_correlazione = 0.003349 * u.deg
    dist_ripetizione = 0.0011 * u.deg
    magnitudine_massima = 15

    nome_params = 'parametri_image_segmentation.txt'
    file_parametri = cerca_file_nel_progetto(BASE_DIR, nome_params)
    if file_parametri is None:
        print("File dei parametri non trovato")
        exit()
    parametri_caricati = leggi_file_parametri(file_parametri)

    # --- PRE-CALCOLO GLOBALE HIPPARCOS ---
    file_hipparco = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
    hdu_list_hipparco = fits.open(file_hipparco)
    tbl_catalogo_hipparco = Table(hdu_list_hipparco[1].data)
    hdu_list_hipparco.close()

    # Calcolo errori propagati al J2000
    dt = 2000.0 - 1991.25
    sigma_ra_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco['e_RAICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco['e_pmRA'])) ** 2) / 3600000.0
    sigma_dec_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco['e_DEICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco['e_pmDE'])) ** 2) / 3600000.0

    # Errore radiale totale Hipparcos
    sigma_hip_deg = np.sqrt(sigma_ra_deg ** 2 + sigma_dec_deg ** 2)

    # Errore stimato Vizier (0.1 arcsec di sicurezza)
    sigma_vizier_deg = 0.1 / 3600.0

    # Somma in quadratura dei due cataloghi
    sigma_totale_deg = np.sqrt(sigma_hip_deg ** 2 + sigma_vizier_deg ** 2)

    # 3-SIGMA
    exclusion_radii_deg_ = 3.0 * sigma_totale_deg
    exclusion_radii_deg = np.full(len(exclusion_radii_deg_), 2.5 / 3600.0)


    print(f"Raggio di merging tra i cataloghi: {np.mean(exclusion_radii_deg)}")

    # Creazione SkyCoord Hipparcos (UNA VOLTA SOLA - Molto pesante)
    coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco['_RAJ2000'],
                                      dec=tbl_catalogo_hipparco['_DEJ2000'],
                                      unit=u.deg)

    tutti_i_file_csv_generati = []

    global_tracker_coords = None
    global_tracker_labels = []
    global_max_label = 0
    global_catalog_label_map = {}
    contatore_satelliti = 0
    contatore_satelliti_presenti = 0

    # --- CICLO PER OGNI RUN ---
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
        for ext in estensioni_valide: file_list.extend(run_folder.glob(ext))
        file_list = sorted([str(f) for f in file_list])
        if not file_list:
            print(f"Nessun FITS in Run {run}, salto.")
            continue

        cartella_tabelle = cerca_cartella_nel_progetto(BASE_DIR, "tabelle")
        if cartella_tabelle is None:
            cartella_tabelle = BASE_DIR / "tabelle"
            cartella_tabelle.mkdir(exist_ok=True)

        output_dir = cartella_tabelle / f"tabelle_unite/tabelle_unite_run_{run}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_str = str(output_dir)

        print(f"Cartella di output: {output_dir_str}")

        # --- FASE 1: CREAZIONE TABELLE UNITE ---
        print(f"--- FASE 1: Segmentazione & Unione ({len(file_list)} files) ---")

        for n, percorso_file in enumerate(tqdm(file_list, desc=f"Fase 1 Run {run}"), 1):
            if n == 1:
                hdu_list = fits.open(percorso_file)
                w = WCS(hdu_list[0].header)
                ra_c, dec_c = hdu_list[0].header["RA"], hdu_list[0].header["DEC"]

                alto_destra = w.pixel_to_world(3071, 2047)
                centro = SkyCoord(ra_c, dec_c, unit=u.deg)

                raggio_ricerca = Angle(centro.separation(alto_destra) * 1.5, "deg")

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
                # --- FILTRAGGIO COMPETITIVO A SINGOLA FASE ---
                # =================================================================
                print("Avvio filtraggio competitivo a singola fase Vizier vs Hipparcos...")

                # preparo le coordinate Vizier
                coords_vizier = SkyCoord(ra=tbl_riquadro_esterno_vizier['RAJ2000'],
                                         dec=tbl_riquadro_esterno_vizier['DEJ2000'],
                                         unit=u.deg)

                # ricavo il limite massimo di ricerca per coprire tutte le tolleranze
                max_threshold_deg = np.max(exclusion_radii_run_subset)
                seplimit = max_threshold_deg * u.deg

                # cerco tutte le stelle Vizier attorno a ogni stella Hipparcos
                idx_A, idx_B, d2d_1, _ = coords_hipparco_run_subset.search_around_sky(coords_vizier, seplimit)

                # implemento un controllo di sicurezza per aggirare l'inversione degli indici di astropy
                # se l'indice massimo trovato supera la grandezza del mio catalogo Hipparcos,
                # significa che astropy mi ha restituito il catalogo Vizier come primo elemento
                if len(idx_A) > 0 and np.max(idx_A) >= len(coords_hipparco_run_subset):
                    idx_viz_1, idx_hip_1 = idx_A, idx_B
                else:
                    idx_hip_1, idx_viz_1 = idx_A, idx_B

                # applico la mia tolleranza dinamica esatta
                mask_threshold = d2d_1.deg <= exclusion_radii_run_subset[idx_hip_1]

                # filtro gli indici per tenere solo quelli entro la tolleranza
                idx_hip_valid = idx_hip_1[mask_threshold]
                idx_viz_valid = idx_viz_1[mask_threshold]

                # genero le maschere di mantenimento inizializzate a True
                mask_keep_hipparco = np.ones(len(tbl_hipparco_run_subset), dtype=bool)
                mask_keep_vizier = np.ones(len(tbl_riquadro_esterno_vizier), dtype=bool)

                # estraggo gli indici univoci di Hipparcos che hanno almeno un match
                unique_hip_idx = np.unique(idx_hip_valid)

                # itero su ogni stella Hipparcos coinvolta
                for i_hip in unique_hip_idx:
                    # trovo gli indici delle stelle Vizier associate a questa specifica stella Hipparcos
                    viz_matches = idx_viz_valid[idx_hip_valid == i_hip]

                    if len(viz_matches) > 0:
                        # estraggo le magnitudini delle stelle Vizier associate
                        mag_viz_matches = np.nan_to_num(tbl_riquadro_esterno_vizier['gmag'][viz_matches], nan=99.0)

                        # individuo la stella Vizier più luminosa (valore di magnitudine minore)
                        idx_min_mag = np.argmin(mag_viz_matches)
                        best_viz_idx = viz_matches[idx_min_mag]
                        best_viz_mag = mag_viz_matches[idx_min_mag]

                        # estraggo la magnitudine della stella Hipparcos in esame
                        hip_mag = np.nan_to_num(tbl_hipparco_run_subset['Vmag'][i_hip], nan=99.0)

                        # confronto e scarto solo la seconda più luminosa tra le due
                        if best_viz_mag <= hip_mag:
                            # Vizier è più luminosa (o uguale), scarto la stella Hipparcos
                            mask_keep_hipparco[i_hip] = False
                        else:
                            # Hipparcos è più luminosa, scarto la Vizier più luminosa
                            mask_keep_vizier[best_viz_idx] = False

                hipparco_escluse = np.sum(~mask_keep_hipparco)
                vizier_escluse = np.sum(~mask_keep_vizier)
                print(f"Risolti {len(unique_hip_idx)} conflitti spaziali:")
                print(f" -> Escluse {hipparco_escluse} stelle Hipparco (tenute Vizier perché più brillanti)")
                print(f" -> Escluse {vizier_escluse} stelle Vizier (tenute Hipparco perché più brillanti)")

                mask_keep_hipparco[tbl_hipparco_run_subset['Vmag'] >= 15] = False

                tbl_hipparco_run_clean = tbl_hipparco_run_subset[mask_keep_hipparco]
                tbl_riquadro_esterno_vizier_CLEAN = tbl_riquadro_esterno_vizier[mask_keep_vizier]

                # applico il filtro magnitudine massima
                tbl_vizier_cut = tbl_riquadro_esterno_vizier_CLEAN[
                    tbl_riquadro_esterno_vizier_CLEAN['gmag'] < magnitudine_massima]
                # =================================================================

            tbl_catalogate = tabella_catalogo(percorso_file)
            tbl_trovate, _ = analisi_image_segmentation(percorso_file, parametri_caricati)

            df_trovate = tbl_trovate.to_pandas()
            df_catalogate = tbl_catalogate.to_pandas()

            all_cols = df_trovate.columns.tolist()
            cols_keep = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']
            for c in ['saturazione', 'kron_flux']:
                if c in all_cols: cols_keep.append(c)
            extra_flux = ['kron_manuale_seg', 'kron_manuale_aper', 'somma_apertura_ultimo_pixel', 'raggio_kron_aper']
            for c in extra_flux:
                if c in all_cols: cols_keep.append(c)

            df_trovate = df_trovate[[c for c in cols_keep if c in df_trovate.columns]].copy()

            with fits.open(percorso_file, memmap=False) as hdu:
                w = WCS(hdu[0].header)
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

                    '''# 3. Ottengo l'orario esatto dello scatto dall'header FITS (usando DATE-OBS in formato ISO UTC)
                    # NOTA: assicurati che 'DATE-OBS' sia la keyword giusta nel tuo FITS per la data/ora UTC
                    tempo_scatto_astropy = Time(hdu_list[0].header['DATE-OBS'], format='isot', scale='utc')
                    tempo_skyfield = ts.from_astropy(tempo_scatto_astropy)

                    # 4. Calcolo le coordinate RA/DEC di tutti i satelliti visti dal telescopio in quel millisecondo
                    ra_sat_list, dec_sat_list = [], []
                    for sat in satelliti_attivi:
                        topocentrica = (sat - osservatorio).at(tempo_skyfield)
                        ra_sat, dec_sat, _ = topocentrica.radec()

                        # controllo che il calcolo dell'orbita sia valido e non restituisca dei NaN
                        if np.isnan(ra_sat.hours) or np.isnan(dec_sat.degrees):
                            continue

                        # Skyfield restituisce RA in ore, lo moltiplico per 15 per averlo in gradi
                        ra_sat_list.append(ra_sat.hours * 15)
                        dec_sat_list.append(dec_sat.degrees)

                    catalogo_satelliti = SkyCoord(ra=ra_sat_list * u.deg, dec=dec_sat_list * u.deg)

                    # 5. Eseguo il match tra gli oggetti 'NO' (non a catalogo) e i satelliti
                    coords_oggetti_no = SkyCoord(ra=df_no['RA_centroid'].values * u.deg,
                                                 dec=df_no['DEC_centroid'].values * u.deg)
                    idx_sat, d2d_sat, _ = coords_oggetti_no.match_to_catalog_sky(catalogo_satelliti)

                    # 6. Escludo gli oggetti vicini alla traiettoria di un satellite (tolleranza larga, es. 2-3 arcminuti per via della scia)
                    tolleranza_satellite = 1/60 * u.deg
                    mask_is_satellite = d2d_sat < tolleranza_satellite

                    # Elimino i falsi positivi causati dai satelliti
                    # df_no.loc[mask_is_satellite, 'Corrispondenza'] = 'SCARTO_SATELLITE'
                    df_no = df_no[~mask_is_satellite]
                    contatore_satelliti = contatore_satelliti + np.sum(mask_is_satellite)'''

                    df_final = pd.concat([df_si, df_no], ignore_index=True)

                else:
                    df_final = df_trovate.copy()
                    df_final['Corrispondenza'] = 'NO'

            final_labels = np.zeros(len(df_final), dtype=int)

            df_final['temp_group_id'] = list(zip(df_final['xcentroid'], df_final['ycentroid']))

            grouped = df_final.groupby('temp_group_id')

            for _, group in grouped:
                indices = group.index.values

                mask_rank1 = group['Corrispondenza'] == 'SI (Rank 1)'
                mask_any_cat = group['Corrispondenza'] != 'NO'

            # =================================================================
            # INIZIO BLOCCO: TRACKING GLOBALE OTTIMIZZATO (BASATO SU COORDINATE)
            # =================================================================

            # preparo la colonna label finale come array di stringhe
            final_labels = np.empty(len(df_final), dtype=object)

            # analizzo riga per riga per garantire l'assoluta indipendenza di ogni stella
            for i in range(len(df_final)):
                row = df_final.iloc[i]

                # Caso 1: Oggetto catalogato
                if row['Corrispondenza'] != 'NO':
                    cat_id = row['ID']
                    catalogo_nome = str(row['Catalogo']).lower()
                    ra_cat = row['RAJ2000']
                    dec_cat = row['DEJ2000']

                    # verifico se ho già generato l'ID testuale per questa specifica stella
                    if cat_id in global_catalog_label_map:
                        assigned_label = global_catalog_label_map[cat_id]
                    else:
                        # distinguo la precisione di arrotondamento in base all'errore del catalogo
                        if "hip" in catalogo_nome:
                            # Hipparcos: errore ~0.0002 deg -> 4 cifre decimali
                            assigned_label = f"RA_{ra_cat:.4f}DEC{dec_cat:.4f}"
                        else:
                            # Vizier: errore ~0.00003 deg -> 5 cifre decimali
                            assigned_label = f"RA_{ra_cat:.5f}DEC{dec_cat:.5f}"

                        global_catalog_label_map[cat_id] = assigned_label

                # Caso 2: Oggetto senza corrispondenza (NO)
                else:
                    ra_obj = row['RA_centroid']
                    dec_obj = row['DEC_centroid']
                    coord_obj = SkyCoord(ra=ra_obj * u.deg, dec=dec_obj * u.deg)

                    if global_tracker_coords is None:
                        # Tracking soglia = dist_ripetizione (0.0011 deg) -> 3 cifre decimali
                        assigned_label = f"RA_{ra_obj:.3f}DEC{dec_obj:.3f}"

                        global_tracker_coords = SkyCoord([coord_obj])
                        global_tracker_labels = [assigned_label]
                    else:
                        idx, d2d, _ = coord_obj.match_to_catalog_sky(global_tracker_coords)
                        if d2d < dist_ripetizione:
                            # uso il label testuale precedentemente memorizzato per questo falso positivo
                            assigned_label = global_tracker_labels[idx]
                        else:
                            assigned_label = f"RA_{ra_obj:.3f}DEC{dec_obj:.3f}"

                            temp_coords = SkyCoord([global_tracker_coords, SkyCoord([coord_obj])])
                            global_tracker_coords = temp_coords
                            global_tracker_labels.append(assigned_label)

                final_labels[i] = assigned_label

            df_final['label'] = final_labels

            # aggiungo colonne identificative Run e Immagine
            df_final['run_id'] = run
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
            # rimuovo temporaneamente
            for c in ['run_id', 'img_index']:
                if c in final_cols: final_cols.remove(c)

            # cerco l'indice
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

        print(f"Contati {contatore_satelliti} oggetti senza corrispondenza matchati con i satelliti nella run {run}")

        # =============================================================================
        # FASE 2 & 3: RAGGI MAX E FLUSSO FISSO (PER RUN)
        # =============================================================================
        print(f"--- FASE 2 & 3: Analisi Fotometria Fissa per Run {run} ---")

        file_csv_list = sorted([f for f in output_dir.glob('*.csv')])

        # salvo la tupla
        for f in file_csv_list:
            tutti_i_file_csv_generati.append((f, run))

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

            # recupero percorso
            path_fits = header_info.get('PERCORSO_FILE', '')
            nome_fits = header_info.get('NOME_FILE_FITS', '')

            if not path_fits or not os.path.exists(path_fits):

                if path_fits:
                    p_obj = Path(path_fits)
                    try:
                        if "pmc_photometry" in p_obj.parts:
                            idx = p_obj.parts.index("pmc_photometry")
                            new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                            if new_path.exists():
                                path_fits = str(new_path)
                    except:
                        pass

                # uso il nome esatto
                if (not path_fits or not os.path.exists(path_fits)) and nome_fits:
                    found = cerca_file_nel_progetto(BASE_DIR, str(nome_fits).strip())
                    if found:
                        path_fits = str(found)

            if not path_fits or not os.path.exists(path_fits):
                print(f"ATTENZIONE: File FITS {path_fits} originale non trovato per {nome_fits}, salto.")
                continue

            with fits.open(path_fits, memmap=False) as hdu:
                data = hdu[0].data
                _, median_bg, _ = sigma_clipped_stats(data[::10, ::10], sigma=3.0)
                data_sub = data - median_bg

            raggi_fissi = []
            ids_presenti = df_frame['ID'].values
            flussi_calcolati = []

            for i, star_id in enumerate(ids_presenti):
                r_globale = map_raggi_max.get(star_id, np.nan)
                if np.isnan(r_globale) or r_globale <= 0:
                    if 'raggio_kron_aper' in df_frame.columns:
                        r_globale = df_frame.at[i, 'raggio_kron_aper']
                    else:
                        r_globale = np.nan
                raggi_fissi.append(r_globale)

                if r_globale > 0 and not np.isnan(r_globale):
                    pos = (df_frame.at[i, 'xcentroid'], df_frame.at[i, 'ycentroid'])
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
    # ordino per percorso
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
    threshold_deg = 0.0011
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
                if dist.deg <= threshold_deg:
                    no_mapping[global_idx] = known_clusters_ids[match_idx]
                else:
                    cid = f"INT_{next_internal_id}"
                    known_clusters_ids.append(cid)
                    known_clusters_coords.append((ra_curr, dec_curr))
                    known_clusters_ids.append(cid)
                    no_mapping[global_idx] = cid
                    next_internal_id += 1

    for idx, uid in no_mapping.items(): big_df.at[idx, 'run_unique_id'] = uid

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

    # converto ID in object
    big_df['ID'] = big_df['ID'].astype(object)

    mask_no_match = big_df['Corrispondenza'] == 'NO'
    big_df.loc[mask_no_match, 'ID'] = big_df.loc[mask_no_match, 'run_unique_id']

    files_groups = big_df.groupby('original_file_path')


    def salva_finale_global(df, header_dict, output_file, fp_count):
        # estraggo solo il nome
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
                cols.remove(c_mean);
                cols.remove(c_std)
                idx_flux = cols.index(c_flux)
                cols.insert(idx_flux + 1, c_mean);
                cols.insert(idx_flux + 2, c_std)

        df_final_save = df_final_save[cols]
        salva_finale_global(df_final_save, header_orig, file_path, num_falsi_positivi)

    print("\n--- ELABORAZIONE GLOBALE MULTI-RUN COMPLETATA CON SUCCESSO ---")