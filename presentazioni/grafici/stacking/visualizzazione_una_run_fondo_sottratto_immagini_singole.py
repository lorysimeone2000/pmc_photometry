import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
#pd.set_option('display.show_dimensions', False)
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
# Importo i moduli necessari per disegnare e calcolare il raggio del cerchio
from matplotlib.patches import Circle
from astropy.wcs.utils import proj_plane_pixel_scales

# Set up wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from astroquery.vizier import Vizier
from astropy.coordinates import Angle

from shapely.geometry import Point, Polygon
# warning
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning) # sopprimo il warning FITSFixedWarning

from pathlib import Path

# --- DEFINIZIONE FILE ---
image_file_c = "run_1_coverage_map_fondo_sottratto_nelle_singole_immagini.fits"
image_file   = "run_1_stacked_sum_fondo_sottratto_nelle_singole_immagini.fits"


# --- CARICAMENTO E CREAZIONE MASCHERA ---

# 1. Caricamento Coverage Map
hdu_list_c = fits.open(image_file_c)
print("Informazioni Coverage Map:")
hdu_list_c.info()
image_data_c = hdu_list_c[0].data
# imposto la maschera a True dove la copertura è massima
full_coverage_value = np.max(image_data_c) # il valore che voglio mascherare
mask_max_coverage = image_data_c == full_coverage_value
hdu_list_c.close()

# 2. Caricamento Immagine Sommata e WCS
hdu_list = fits.open(image_file)
print("\nInformazioni Immagine Sommata:")
hdu_list.info()
image_data = hdu_list[0].data
# estraggo l'header per definire il sistema di coordinate celesti
header = hdu_list[0].header
# inizializzo il mio oggetto WCS
wcs = WCS(header)
data = image_data
hdu_list.close()

# --- ESTRAZIONE E VISUALIZZAZIONE ---

# applico la maschera (commentata come nell'originale)
# data = ma.masked_array(image_data, mask=~mask_max_coverage)
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
print("Mediana: " , median)
data = data - median

# Visualizzazione
# creo la figura con le dimensioni ottimizzate per LaTeX
fig = plt.figure(figsize=(8.5, 5))

# aggiungo il subplot passando il mio oggetto WCS come proiezione
ax = fig.add_subplot(111, projection=wcs)

# genero l'immagine con scala logaritmica, mantenendo il rapporto 1:1 dei pixel
im = ax.imshow(data, cmap="grey_r", norm=LogNorm(), interpolation='nearest', origin='lower', aspect='equal')

# forzo l'ascensione retta in gradi per avere la stessa unità di misura e scala della declinazione
ax.coords[0].set_format_unit(u.deg)
ax.coords[1].set_format_unit(u.deg)

# mi assicuro in modo esplicito che le proporzioni dell'asse siano mantenute uguali
ax.set_aspect('equal')

# dimensiono i tick degli assi (RA e DEC)
ax.tick_params(axis='both', which='major', labelsize=12)

# Ottengo le coordinate della Nebulosa del Granchio
crab_coord = SkyCoord.from_name("Crab Nebula")

# Converto le coordinate celesti in coordinate pixel
x_crab, y_crab = wcs.world_to_pixel(crab_coord)

# Calcolo la dimensione del raggio in pixel impostandolo a 2.5 arcmin per un'ampiezza totale di 5
pixel_scales = proj_plane_pixel_scales(wcs) * u.deg
pixel_scale_arcmin = pixel_scales[0].to(u.arcmin)
raggio_arcmin = 7.5 * u.arcmin
raggio_pixel = (raggio_arcmin / pixel_scale_arcmin).value

# imposto le nuove etichette in inglese britannico
ax.set_xlabel('RA (deg)', fontsize=14)
ax.set_ylabel('DEC (deg)', fontsize=14)

# Richiamo la figura e gli assi correnti
fig = plt.gcf()
ax = plt.gca()

# Aggiungo il cerchio rosso vuoto al centro della nebulosa
cerchio = Circle((x_crab, y_crab), raggio_pixel, edgecolor='red', facecolor='none')
ax.add_patch(cerchio)

# aggiungo la colorbar utilizzando fraction e pad per mantenerla in proporzione con l'asse
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# imposto l'etichetta direttamente sulla colorbar
cbar.set_label('ADU sum', fontsize=14)

plt.savefig("somma_una_run_RA_DEC.png", dpi=300, bbox_inches='tight')
# plt.show()