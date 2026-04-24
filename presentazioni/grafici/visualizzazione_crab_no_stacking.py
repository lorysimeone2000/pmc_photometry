import pandas as pd
import matplotlib
import argparse
import json
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import concurrent.futures
from astropy.config import paths
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
from astropy.nddata import Cutout2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.nddata.utils import NoOverlapError
from matplotlib.axes import Axes

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

image_file = cerca_file_nel_progetto(BASE_DIR, '20250120_213400.fits')

hdu_list = fits.open(image_file)
# stampo le informazioni del file
hdu_list.info()

# creo la matrice dei valori dei pixel
image_data = hdu_list[0].data
# ritaglio un'area tot x tot pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438]
# stampo tutti i dati dell'header
#print(hdu_list[0].header)
image_header = hdu_list[0].header
# stampo la RA
print("RA",image_header["RA"])
# stampo il DEC
print("DEC",image_header["DEC"])

# stampo la matrice dei valori dei pixel
print(image_data)
# stampo il valore del pixel di coordinate [0,0]
#print(image_data[0,0])
# stampo le dimensioni della matrice
print(image_data.shape)
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data

# estraggo il sistema di coordinate (WCS) dall'header
wcs = WCS(image_header)

# ottengo le coordinate della Nebulosa del Granchio dal suo nome
crab_coord = SkyCoord.from_name("Crab Nebula")

# imposto la dimensione del riquadro a 7x7 arcmin
dimensione_riquadro = u.Quantity([7, 7], u.arcmin)

try:
    # tento di creare il ritaglio dell'immagine centrato sulle coordinate richieste
    cutout = Cutout2D(data, crab_coord, dimensione_riquadro, wcs=wcs)
except NoOverlapError:
    # stampo un messaggio di errore ed esco se l'immagine non contiene il target
    print("Errore: Le coordinate della Nebulosa del Granchio non si sovrappongono all'immagine FITS fornita.")
    sys.exit()

# creo la figura
fig = plt.figure(figsize=(8, 6))

# aggiungo gli assi utilizzando la proiezione WCS del ritaglio per mostrare le coordinate reali
ax = fig.add_subplot(1, 1, 1, projection=cutout.wcs)

# genero l'immagine ritagliata con scala di colori bianco e nero
im = ax.imshow(cutout.data, cmap="grey_r", norm=LogNorm(), interpolation='nearest')

# inverto l'asse y
ax.invert_yaxis()

# imposto i formati degli assi per mostrare i gradi decimali
ax.coords[0].set_major_formatter('d.ddd')
ax.coords[1].set_major_formatter('d.ddd')

# imposto le etichette degli assi specificando che sono in gradi
ax.coords[0].set_axislabel('DEC (deg)', fontsize=14)
ax.coords[1].set_axislabel('RA (deg)', fontsize=14)

# posiziono un puntino rosso esattamente al centro del ritaglio
centro_x = cutout.data.shape[1] / 2
centro_y = cutout.data.shape[0] / 2
ax.plot(centro_x, centro_y, marker='o', color='red')

# configuro la colorbar affinché abbia la stessa altezza della figura passando la classe standard per gli assi
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=Axes)
fig.colorbar(im, cax=cax)

# salvo la figura
plt.savefig("crab_senza_stacking")