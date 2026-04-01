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

# importo il modulo fondamentale per la portabilità
from pathlib import Path

# importo il catalogo satelliti
from skyfield.api import load, wgs84
from astropy.time import Time
import requests
from datetime import timedelta

# gestisco i warning
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# configuro i percorsi e importo i moduli esterni

def trova_cartella_base(nome_target="Lorenzo"):
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

PMC_DATA_BLAZAR = cerca_cartella_nel_progetto(BASE_DIR, "pmc_photometry/run_vecchie")
print(f"Cartella PMC_DATA_BLAZAR trovata in {PMC_DATA_BLAZAR}")

bool = False
asse_y = []
tempo = []

if PMC_DATA_BLAZAR:
    # estraggo tutte le sottocartelle presenti dentro PMC_DATA_BLAZAR
    sottocartelle = [d for d in PMC_DATA_BLAZAR.iterdir() if d.is_dir()]

    # inizializzo l'unica figura prima di far partire il ciclo
    plt.figure(figsize=(12, 6))
    plt.title("Trend of DEJ2000 coordinate over time (3 runs)", fontsize=14)
    plt.xlabel("Time elapsed since the first image (minutes)", fontsize=12)
    plt.ylabel("DEJ2000 coordinate (Degrees)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

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

                    # cerco la chiave ra (solitamente usata negli header), oppure DEJ2000 come fallback
                    ra = header.get('DEC')
                    if ra is None:
                        ra = header.get('DEJ2000')

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
        asse_y.append(ralinazioni_ordinate)

        # calcolo il tempo trascorso in secondi, impostando il primo scatto come istante zero (0)
        if not bool:
            t0 = tempi_ordinati[0]
            bool = True
        tempi_relativi_sec = [(t - t0).sec for t in tempi_ordinati]
        tempo.append(tempi_relativi_sec)

    # unisco tutti i sotto-array in un unico array monodimensionale
    tempi_uniti = np.concatenate(tempo)
    asse_y_uniti = np.concatenate(asse_y)

    # ottengo gli indici per ordinare i dati in base al tempo crescente
    indici_ordinamento = np.argsort(tempi_uniti)

    # eseguo uno shift per far sì che il valore minore assuma il valore 0
    tempi_uniti = tempi_uniti - np.min(tempi_uniti)

    # applico l'ordinamento per tracciare una linea unica e continua e divido l'asse x per 60
    tempi = tempi_uniti[indici_ordinamento] / 60.0
    asse_y = asse_y_uniti[indici_ordinamento]

    # calcolo le differenze di tempo e di posizione in y tra un frame e il successivo
    diff_tempi = np.diff(tempi)
    diff_y = np.abs(np.diff(asse_y))

    # trovo gli indici dove la differenza di tempo è superiore a 5 minuti o la differenza in y è di almeno 0.4 gradi
    indici_gap = np.where((diff_tempi > 5.0) | (diff_y >= 0.4))[0]

    inizio_segmento = 0

    # traccio i segmenti di linea continua e quelli tratteggiati per i gap
    for gap_idx in indici_gap:
        # disegno la linea continua fino al gap
        plt.plot(tempi[inizio_segmento:gap_idx + 1], asse_y[inizio_segmento:gap_idx + 1], color='tab:blue',
                 linestyle='-', markersize=1.5)
        # disegno la linea tratteggiata durante il gap
        plt.plot(tempi[gap_idx:gap_idx + 2], asse_y[gap_idx:gap_idx + 2], color='tab:blue', linestyle='--',
                 markersize=1.5)
        inizio_segmento = gap_idx + 1

    # disegno l'ultimo segmento continuo dopo l'ultimo gap
    plt.plot(tempi[inizio_segmento:], asse_y[inizio_segmento:], color='tab:blue', linestyle='-', markersize=1.5)

    # concludo le operazioni sul grafico dopo che il ciclo ha processato tutte le cartelle
    plt.legend()
    plt.tight_layout()

    # salvo l'unico grafico e lo chiudo
    nome_grafico = "andamento_DEJ2000_globale.png"
    plt.savefig(nome_grafico, dpi=300)
    print(f"\nGrafico finale salvato: {nome_grafico}")
    plt.close()

else:
    print("Elaborazione interrotta: cartella PMC_DATA_BLAZAR non trovata.")