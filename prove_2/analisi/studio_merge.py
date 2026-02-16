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

# --- GESTIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


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


# --- PRE-CALCOLO GLOBALE HIPPARCOS ---
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
exclusion_radii_deg = 3.0 * sigma_totale_deg

print(f"Errore calcolato per 3 sigma: {np.mean(exclusion_radii_deg)*3600} arcosecondi")

# =============================================================================
# RICERCA PRIMA IMMAGINE FITS DELLA RUN 1
# =============================================================================

run_target = 1
nome_cartella_run = f"20250120_run{run_target}"

found_folders = list(BASE_DIR.rglob(nome_cartella_run))

if found_folders:
    run_folder = found_folders[0]

    estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
    file_list = []
    for ext in estensioni_valide:
        file_list.extend(run_folder.glob(ext))

    file_list = sorted([str(f) for f in file_list])

    if file_list:
        prima_immagine_fits = file_list[0]
        print(f"\nPercorso della prima immagine FITS della Run {run_target}:")
        print(prima_immagine_fits)
    else:
        print(f"\nNessun file FITS trovato nella cartella '{nome_cartella_run}'.")
else:
    print(f"\nCartella '{nome_cartella_run}' non trovata all'interno del progetto.")

percorso_fits = prima_immagine_fits

hdu_list = fits.open(percorso_fits)
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

coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco['_RAJ2000'],
                                      dec=tbl_catalogo_hipparco['_DEJ2000'],
                                      unit=u.deg)

distanze_hip = centro.separation(coords_hipparco_global)

mask_hip_fov = distanze_hip < raggio_ricerca

tbl_hipparco_run_subset = tbl_catalogo_hipparco[mask_hip_fov]
coords_hipparco_run_subset = coords_hipparco_global[mask_hip_fov]
exclusion_radii_run_subset = exclusion_radii_deg[mask_hip_fov]

mag_lim = 10.0
mask_mag_hip = tbl_hipparco_run_subset['Vmag'] <= mag_lim
mask_mag_viz = tbl_riquadro_esterno_vizier['gmag'] <= mag_lim

tbl_hipparco_run_subset = tbl_hipparco_run_subset[mask_mag_hip]
coords_hipparco_run_subset = coords_hipparco_run_subset[mask_mag_hip]
exclusion_radii_run_subset = exclusion_radii_run_subset[mask_mag_hip]
tbl_riquadro_esterno_vizier = tbl_riquadro_esterno_vizier[mask_mag_viz]

# --- FILTRAGGIO VIZIER COMPETITIVO (ANALISI DINAMICA CON LOGICA 1-A-1) ---
print("Preparazione per l'analisi dinamica Vizier vs Hipparcos...")

# preparo le coordinate Vizier
coords_vizier = SkyCoord(ra=tbl_riquadro_esterno_vizier['RAJ2000'],
                         dec=tbl_riquadro_esterno_vizier['DEJ2000'],
                         unit=u.deg)

# calcolo i limiti per l'array di 1.000.000 di valori
limite_inferiore = np.mean(exclusion_radii_run_subset) * 0.01
limite_superiore = 10.0 / 3600.0  # 10 secondi d'arco in gradi

# eseguo la ricerca generale al limite massimo per non sovraccaricare la CPU nel ciclo
seplimit_max = limite_superiore * u.deg
idx_hip_all, idx_viz_all, d2d_all, _ = coords_hipparco_run_subset.search_around_sky(coords_vizier, seplimit_max)

# definisco la funzione dinamica
def calcola_corrispondenze_dinamiche(soglia_deg):
    # filtro le distanze per la soglia attuale
    mask_threshold = d2d_all.deg <= soglia_deg
    idx_hip_valid = idx_hip_all[mask_threshold]

    # applicando la logica competitiva 1-a-1, ogni singola stella Hipparcos
    # che trova almeno un match (unique) innescherà esattamente 1 scontro.
    # di conseguenza, ci sarà esattamente 1 stella eliminata (o Hipparcos o Vizier).
    # quindi il numero di scontri/eliminazioni corrisponde ai valori unici di Hipparcos.
    return len(np.unique(idx_hip_valid))

# creo l'array di soglie da testare
soglie_test = np.linspace(limite_inferiore, limite_superiore, 10000000)
risultati_corrispondenze = []

print(f"Avvio il ciclo per le {len(soglie_test)} iterazioni...")

# ciclo la funzione sui valori creati
for soglia in tqdm(soglie_test, desc="Analisi soglie"):
    risultati_corrispondenze.append(calcola_corrispondenze_dinamiche(soglia))

# grafico il tutto
plt.figure(figsize=(12, 7))

# metto le varie soglie_test come asse x e i risultati come asse y
# moltiplico per 3600 l'asse X in modo da leggere l'estensione in secondi d'arco
plt.plot(soglie_test * 3600.0, risultati_corrispondenze, color='purple')

plt.xlabel('Soglia di ricerca competitiva (arcosecondi)')
plt.ylabel('Numero di scontri inter-catalogo (Stelle eliminate)')
plt.title('Scontri competitivi Vizier-Hipparcos al variare della soglia')
plt.grid(True, linestyle='--', alpha=0.7)
output_img = f"studio_merge_hipparco_vs_vizier_da_{mag_lim}.png"
plt.savefig(output_img, dpi=300)

plt.show()