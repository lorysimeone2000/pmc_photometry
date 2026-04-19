import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
import numpy as np
import os
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture

# Set up wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
import astropy.wcs.utils as wcs_utils  # importo le utils per calcolare la scala dei pixel

from astroquery.vizier import Vizier
from astropy.coordinates import Angle

from shapely.geometry import Point, Polygon
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.optimize import curve_fit
import warnings
from pathlib import Path
from tqdm import tqdm
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


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


# =============================================================================
# FUNZIONI DI GESTIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    # cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # cerco un file ricorsivamente assicurandomi che il pattern sia una stringa
    files_trovati = list(base_dir.rglob(str(nome_file_esatto)))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


# trovo la cartella base del mio progetto
BASE_DIR = trova_cartella_base("Lorenzo")

# --- CARICAMENTO E SOMMA DI TUTTE LE RUN ---

# inizializzo le matrici vuote che conterranno la somma totale e la proiezione
total_data = None
total_coverage = None
wcs_totale = None

# eseguo il ciclo su tutte le 3 run
runs = [1, 2, 3]

for run in runs:
    # definisco i nomi dei file salvati nello step precedente per la run corrente
    nome_file_c = f"coverage_map_mrk421_{2025}.fits"
    nome_file_sum = f"stacked_sum_mrk421_{2025}.fits"

    # cerco i file dinamicamente
    percorso_c = cerca_file_nel_progetto(BASE_DIR, nome_file_c)
    percorso_sum = cerca_file_nel_progetto(BASE_DIR, nome_file_sum)

    if not percorso_c or not percorso_sum:
        print(f"Attenzione: file mancanti per la run {run}. Salto questa run.")
        continue

    # 1. carico e sommo la Coverage Map
    with fits.open(str(percorso_c)) as hdu_c:
        image_data_c = hdu_c[0].data
        if total_coverage is None:
            # creo la matrice di zeri basandomi sulla forma della prima immagine trovata
            total_coverage = np.zeros_like(image_data_c, dtype=float)
        # aggiungo i dati della run corrente al totale
        total_coverage += image_data_c

    # 2. carico e sommo l'Immagine Stacked
    with fits.open(str(percorso_sum)) as hdu_sum:
        image_data_sum = hdu_sum[0].data

        # estraggo il WCS dalla prima immagine utile per usarlo nel grafico
        if wcs_totale is None:
            wcs_totale = WCS(hdu_sum[0].header)

        if total_data is None:
            # creo la matrice di zeri per l'immagine
            total_data = np.zeros_like(image_data_sum, dtype=float)
        # aggiungo i conteggi della run corrente al totale
        total_data += image_data_sum

if total_data is None or total_coverage is None:
    raise ValueError("Errore: non è stato possibile caricare i dati di nessuna run.")

# estraggo il valore massimo di copertura globale raggiunto unendo le 3 run
full_coverage_value = np.max(total_coverage)
print(f"Copertura massima totale raggiunta: {full_coverage_value} immagini")

# --- ESTRAZIONE E VISUALIZZAZIONE ---

# calcolo le statistiche sull'immagine totale
mean, median, std = sigma_clipped_stats(total_data, sigma=3.0)
print("Mediana totale: ", median)

# sottraggo il fondo mediano
data_finale = total_data - median

# preparo la visualizzazione
norm = simple_norm(data_finale, 'sqrt')

# creo la figura con le dimensioni ottimizzate per LaTeX
fig = plt.figure(figsize=(8.5, 5))

# imposto il sistema di riferimento celeste tramite WCS
ax = fig.add_subplot(111, projection=wcs_totale)

# genero l'immagine assicurando le stesse proporzioni per RA e DEC
im = ax.imshow(data_finale, cmap="viridis", norm=norm, interpolation='nearest', origin='lower', aspect='equal')

# forzo l'ascensione retta e la declinazione in gradi per la coerenza della scala
ax.coords[0].set_format_unit(u.deg)
ax.coords[1].set_format_unit(u.deg)

# applico la scala decimale esplicita per entrambi gli assi celesti
ax.coords[0].set_major_formatter('d.ddd')
ax.coords[1].set_major_formatter('d.ddd')

# mi assicuro in modo esplicito che le proporzioni dell'asse siano mantenute uguali
ax.set_aspect('equal')

# dimensiono i tick degli assi
ax.tick_params(axis='both', which='major', labelsize=12)

# configuro la colorbar aggiornando l'etichetta in inglese britannico
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Total counts (ADU)', fontsize=14)

# aggiorno le etichette con le coordinate celesti in inglese britannico
ax.set_xlabel('RA (deg)', fontsize=14)
ax.set_ylabel('DEC (deg)', fontsize=14)

plt.savefig('stacking_crab_totale.png', dpi=300, bbox_inches='tight')
# plt.show()


# --- CALCOLO FOV SPECIFICO PER run_1_stacked_sum_crab.fits ---

# cerco specificamente il file della run 1
percorso_run_1 = cerca_file_nel_progetto(BASE_DIR, f"stacked_sum_mrk421_{2025}.fits")

if percorso_run_1:
    with fits.open(str(percorso_run_1)) as hdu_1:
        dati_run_1 = hdu_1[0].data
        wcs_run_1 = WCS(hdu_1[0].header)

        # ottengo le dimensioni dell'immagine in pixel
        dim_y, dim_x = dati_run_1.shape

        # estraggo la scala dei pixel in gradi per ciascun asse dal WCS
        scala_pixel_x, scala_pixel_y = wcs_utils.proj_plane_pixel_scales(wcs_run_1)

        # calcolo il campo di vista in gradi e lo converto in arcominuti
        fov_x_arcmin_1 = (dim_x * scala_pixel_x * u.deg).to(u.arcmin)
        fov_y_arcmin_1 = (dim_y * scala_pixel_y * u.deg).to(u.arcmin)

        print("\n--- Dati FOV per run_1_stacked_sum_crab.fits ---")
        print(f"Campo di vista RA (X): {fov_x_arcmin_1:.2f}")
        print(f"Campo di vista DEC (Y): {fov_y_arcmin_1:.2f}")
else:
    print("\nNon ho trovato il file run_1_stacked_sum_crab.fits per calcolare il FOV.")