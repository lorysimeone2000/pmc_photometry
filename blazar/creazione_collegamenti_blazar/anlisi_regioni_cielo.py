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

# --- IMPORT FONDAMENTALE PER LA PORTABILITÀ ---
from pathlib import Path

# catalogo satelliti
from skyfield.api import load, wgs84
from astropy.time import Time
import requests
from datetime import timedelta

# --- GESTIONE WARNING ---
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


BASE_DIR = trova_cartella_base("pmc_photometry")

# Aggiungo la BASE_DIR al sys.path per permettere l'importazione della cartella 'funzioni'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Importo le mie funzioni di utilità e astrometria
from funzioni.utilita import *
from funzioni.astrometria import *

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"Moduli esterni caricati con successo.")
print(f"------------------------------")

PMC_DATA_BLAZAR = cerca_cartella_nel_progetto(BASE_DIR, "PMC_DATA")
print(f"Cartella PMC_DATA_BLAZAR trovata in {PMC_DATA_BLAZAR}")

if PMC_DATA_BLAZAR:
    # estraggo tutte le sottocartelle presenti dentro PMC_DATA_BLAZAR
    sottocartelle = [d for d in PMC_DATA_BLAZAR.iterdir() if d.is_dir()]

    for cartella_run in sottocartelle:
        # cerco tutti i file FITS nella sottocartella corrente
        estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
        file_fits_list = []
        for ext in estensioni_valide:
            file_fits_list.extend(cartella_run.glob(ext))

        if not file_fits_list:
            print(f"Nessun file FITS trovato nella sottocartella {cartella_run.name}, la salto.")
            continue

        print(f"\nElaborazione della sottocartella: {cartella_run.name} ({len(file_fits_list)} file trovati)")

        tempi = []
        ralinazioni = []

        # apro ogni FITS e leggo l'header
        for percorso_file in tqdm(file_fits_list, desc=f"Lettura header {cartella_run.name}"):
            try:
                with fits.open(percorso_file, memmap=False) as hdu:
                    header = hdu[0].header
                    date_obs = header.get('DATE-OBS')

                    # cerco la chiave ra (solitamente usata negli header), oppure RAJ2000 come fallback
                    ra = header.get('ra')
                    if ra is None:
                        ra = header.get('RAJ2000')

                    if date_obs is not None and ra is not None:
                        # converto la stringa di tempo nel formato nativo di Astropy
                        tempi.append(Time(date_obs, format='isot', scale='utc'))
                        ralinazioni.append(float(ra))
            except Exception:
                # ignoro silenziosamente i file corrotti o non leggibili
                pass

        if not tempi:
            print(f"Nessun dato temporale o di coordinata valido estratto per {cartella_run.name}.")
            continue

        # ordino cronologicamente i dati per sicurezza
        indici_ordinati = np.argsort(tempi)
        tempi_ordinati = np.array(tempi)[indici_ordinati]
        ralinazioni_ordinate = np.array(ralinazioni)[indici_ordinati]

        # calcolo il tempo trascorso in secondi, impostando il primo scatto come istante zero (0)
        t0 = tempi_ordinati[0]
        tempi_relativi_sec = [(t - t0).sec for t in tempi_ordinati]

        # genero il grafico per la sottocartella in esame
        plt.figure(figsize=(12, 6))
        plt.plot(tempi_relativi_sec, ralinazioni_ordinate, marker='o', linestyle='-', color='teal', markersize=4)

        plt.title(f"Andamento della coordinata RAJ2000 nel tempo\nSottocartella: {cartella_run.name}", fontsize=14)
        plt.xlabel("Tempo trascorso dalla prima immagine (secondi)", fontsize=12)
        plt.ylabel("Coordinata RAJ2000 (Gradi)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # salvo il grafico e lo mostro
        nome_grafico = f"andamento_RAJ2000_{cartella_run.name}.png"
        plt.savefig(nome_grafico, dpi=300)
        print(f"Grafico salvato: {nome_grafico}")

        plt.show()
        plt.close()
else:
    print("Elaborazione interrotta: cartella PMC_DATA_BLAZAR non trovata.")