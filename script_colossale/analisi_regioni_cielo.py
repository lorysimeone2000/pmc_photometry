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

def trova_cartella_base(nome_target="Lorenzo"):
    # risalgo l'albero delle directory per trovare la radice del progetto
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

cartella_ASTRI1 = BASE_DIR.parent / "PMC_DATA" / "ASTRI1"

file_fits_riferimento = None
PMC_DATA = cartella_ASTRI1
print(f"Cartella PMC_DATA trovata in {PMC_DATA}")

if PMC_DATA:
    # estraggo tutte le sottocartelle presenti dentro PMC_DATA
    sottocartelle = [d for d in PMC_DATA.iterdir() if d.is_dir()]

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
        declinazioni = []

        # apro ogni FITS e leggo l'header
        for percorso_file in tqdm(file_fits_list, desc=f"Lettura header {cartella_run.name}"):
            try:
                with fits.open(percorso_file, memmap=False) as hdu:
                    header = hdu[0].header
                    date_obs = header.get('DATE-OBS')

                    # cerco le chiavi RA e DEC
                    ra = header.get('ra')
                    if ra is None:
                        ra = header.get('RAJ2000') or header.get('OBJ-RA')

                    dec = header.get('dec')
                    if dec is None:
                        dec = header.get('DEJ2000') or header.get('OBJ-DEC')

                    if date_obs is not None and ra is not None and dec is not None:
                        # converto la stringa di tempo nel formato nativo di Astropy
                        tempi.append(Time(date_obs, format='isot', scale='utc'))
                        ralinazioni.append(float(ra))
                        declinazioni.append(float(dec))
            except Exception:
                # ignoro silenziosamente i file corrotti o non leggibili
                pass

        if not tempi:
            print(f"Nessun dato temporale o di coordinata valido estratto per {cartella_run.name}.")
            continue

        # ordino cronologicamente i dati per sicurezza
        indici_ordinati = np.argsort(tempi)
        tempi_ordinati = np.array(tempi)[indici_ordinati]
        ra_ordinate = np.array(ralinazioni)[indici_ordinati]
        dec_ordinate = np.array(declinazioni)[indici_ordinati]

        # calcolo il tempo trascorso in secondi, impostando il primo scatto come istante zero (0)
        t0 = tempi_ordinati[0]
        tempi_relativi_sec = np.array([(t - t0).sec for t in tempi_ordinati])

        # creo le coordinate per valutare le separazioni
        coordinate = SkyCoord(ra=ra_ordinate * u.deg, dec=dec_ordinate * u.deg, frame='icrs')

        # calcolo le distanze dal punto precedente e dal successivo
        distanze_prec = np.zeros(len(coordinate))
        distanze_succ = np.zeros(len(coordinate))

        if len(coordinate) > 2:
            distanze_prec[2:] = coordinate[2:].separation(coordinate[:-2]).deg
            distanze_succ[:-2] = coordinate[:-2].separation(coordinate[2:]).deg

        # individuo i punti isolati distanti più di 0.5 gradi da entrambi
        da_escludere = (distanze_prec > 0.5) & (distanze_succ > 0.5)

        # applico la maschera per escluderli dalle rappresentazioni
        coordinate_filtrate = coordinate[~da_escludere]
        tempi_filtrati = tempi_relativi_sec[~da_escludere]

        # ricalcolo le distanze angolari definitive sui punti validi
        distanze_angolari_filtrate = np.zeros(len(coordinate_filtrate))
        if len(coordinate_filtrate) > 2:
            distanze_angolari_filtrate[2:] = coordinate_filtrate[2:].separation(coordinate_filtrate[:-2]).deg

        # genero il grafico per la sottocartella in esame
        plt.figure(figsize=(12, 6))
        plt.plot(tempi_filtrati, distanze_angolari_filtrate, marker='o', linestyle='-', color='teal', markersize=4)

        plt.title(f"Distanza angolare tra immagini consecutive\nSottocartella: {cartella_run.name}", fontsize=14)
        plt.xlabel("Tempo trascorso dalla prima immagine (secondi)", fontsize=12)
        plt.ylabel("Distanza angolare (Gradi)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # salvo il grafico e lo chiudo
        nome_grafico = f"distanza_angolare_{cartella_run.name}.png"
        plt.savefig(nome_grafico, dpi=300)
        print(f"Grafico salvato: {nome_grafico}")

        plt.close()
else:
    print("Elaborazione interrotta: cartella PMC_DATA non trovata.")