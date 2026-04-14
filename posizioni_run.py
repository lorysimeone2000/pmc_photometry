import pandas as pd
import matplotlib
import argparse
import json
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import concurrent.futures
from astropy.config import paths

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import time
import os
import sys
import gc
from scipy.optimize import curve_fit
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
import warnings
from astropy.wcs import FITSFixedWarning
from photutils.datasets import make_100gaussians_image
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
from astropy.coordinates import search_around_sky
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from shapely.geometry import Point, Polygon
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import label
import re
from pathlib import Path
from astropy.time import Time

# gestisco i warning ignorandoli per mantenere pulito il mio output
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='.*deblending mode.*')


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory del mio script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")
PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

# importo i moduli per il salvataggio in parquet e la relativa utilità
from funzioni.utilita_parquet import *
from funzioni.astrometria_parquet import *

# definisco le mie liste per ospitare le coordinate estratte
lista_ra = []
lista_dec = []

# cerco tutti i miei file parquet relativi agli oggetti non catalogati
file_parquet_non_cat = list(BASE_DIR.rglob("*oggetti_non_catalogati.parquet"))

print(f"Trovati {len(file_parquet_non_cat)} file parquet di oggetti non catalogati. Estraggo gli header...")

# analizzo ogni file trovato
for file_pq in tqdm(file_parquet_non_cat, desc="Estrazione coordinate dagli header"):
    # estraggo il mio dizionario dell'header
    header = leggi_header_da_parquet(file_pq)

    # recupero le mie coordinate considerando le possibili chiavi del FITS
    ra = header.get('RA') or header.get('RAJ2000') or header.get('OBJ-RA')
    dec = header.get('DEC') or header.get('DEJ2000') or header.get('OBJ-DEC')

    # aggiungo i dati alle mie liste se sono validi
    if ra is not None and dec is not None:
        try:
            lista_ra.append(float(ra))
            lista_dec.append(float(dec))
        except ValueError:
            pass

# creo i miei grafici se ho raccolto dati sufficienti
if len(lista_ra) > 0 and len(lista_dec) > 0:
    print(f"Generazione dei plot in corso con {len(lista_ra)} punti...")

    # ---------------------------------------------------------
    # 1. SCATTER PLOT RA/DEC
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    plt.scatter(lista_ra, lista_dec, color='blue', alpha=0.6, edgecolors='black', s=20)

    # inverto il mio asse X come da convenzione astronomica per l'Ascensione Retta
    plt.gca().invert_xaxis()

    plt.xlabel('RA (gradi)')
    plt.ylabel('DEC (gradi)')
    plt.title('Scatter Plot dei file Parquet (Oggetti Non Catalogati)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # salvo il mio plot nella cartella base
    percorso_salvataggio = BASE_DIR / 'scatter_plot_RA_DEC_non_catalogati.png'
    plt.savefig(percorso_salvataggio, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot scatter salvato con successo in: {percorso_salvataggio}")

    # ---------------------------------------------------------
    # 2. SCANSIONE DISTANZA DALLA NEBULOSA DEL GRANCHIO
    # ---------------------------------------------------------
    # definisco le coordinate esatte della Nebulosa del Granchio
    crab_coord = SkyCoord(ra=83.633083, dec=22.0145, unit='deg', frame='icrs')

    # strutturo le mie coordinate estratte in un oggetto SkyCoord
    oggetti_coords = SkyCoord(ra=lista_ra, dec=lista_dec, unit='deg', frame='icrs')

    # calcolo la mia separazione angolare in gradi per tutti i punti
    distanze_crab = oggetti_coords.separation(crab_coord).deg

    # genero il mio array di raggi limite entro cui cercare, impostando esplicitamente il range da 0 a 50 gradi
    raggi_scansione = np.linspace(0, 100, 100)
    conteggi_cumulativi = []

    # calcolo quanti file/oggetti ricadono entro il mio raggio limite corrente
    for r in raggi_scansione:
        conteggio = np.sum(distanze_crab <= r)
        conteggi_cumulativi.append(conteggio)

    plt.figure(figsize=(10, 8))
    plt.plot(raggi_scansione, conteggi_cumulativi, color='red', linewidth=2)
    plt.fill_between(raggi_scansione, conteggi_cumulativi, color='red', alpha=0.2)

    plt.xlabel('Distanza dalla Nebulosa del Granchio (gradi)')
    plt.ylabel('Numero cumulativo di file/oggetti')
    plt.title('Scansione Cumulativa: Distanza dalla Nebulosa del Granchio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')

    # salvo il mio secondo grafico
    percorso_salvataggio_distanze = BASE_DIR / 'scansione_distanze_crab.png'
    plt.savefig(percorso_salvataggio_distanze, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot distanze salvato con successo in: {percorso_salvataggio_distanze}")

else:
    print("Nessuna coordinata valida trovata negli header dei file.")