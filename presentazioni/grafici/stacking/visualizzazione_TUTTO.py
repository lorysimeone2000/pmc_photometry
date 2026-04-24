import numpy as np
import numpy.ma as ma # Aggiunto per i Masked Arrays
import matplotlib.pyplot as plt
import pandas as pd
#pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
from scipy.optimize import curve_fit
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
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

from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from astropy.nddata import Cutout2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from shapely.geometry import Point, Polygon
# warning
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning) # Sopprime il warning FITSFixedWarning

from pathlib import Path

# Importo i moduli necessari per disegnare e calcolare il raggio del cerchio
from matplotlib.patches import Circle
from astropy.wcs.utils import proj_plane_pixel_scales

# --- DEFINIZIONE FILE ---
image_file_c = "master_coverage_map.fits"
image_file   = "master_stacked_sum.fits"


# --- CARICAMENTO E CREAZIONE MASCHERA ---

# 1. Caricamento Coverage Map
hdu_list_c = fits.open(image_file_c)
print("Informazioni Coverage Map:")
hdu_list_c.info()
image_data_c = hdu_list_c[0].data
# La maschera è True dove la copertura è massima (112)
full_coverage_value = np.max(image_data_c) # Il valore che voglio mascherare
mask_max_coverage = image_data_c == full_coverage_value
hdu_list_c.close()

# 2. Caricamento Immagine Sommata
hdu_list = fits.open(image_file)
print("\nInformazioni Immagine Sommata:")
hdu_list.info()
image_data = hdu_list[0].data

# Estraggo l'header e il sistema di coordinate (WCS)
image_header = hdu_list[0].header
wcs = WCS(image_header)

data = image_data
hdu_list.close()

# --- ESTRAZIONE E VISUALIZZAZIONE ---

# Applichiamo la maschera:
# Usiamo ~mask_max_coverage per nascondere tutti i pixel NON UGUALI a 115.
# data = ma.masked_array(image_data, mask=~mask_max_coverage)
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
print("Mediana: " , median)
data = data - median

# Ottengo le coordinate della Nebulosa del Granchio
crab_coord = SkyCoord.from_name("Crab Nebula")

# Converto le coordinate celesti in coordinate pixel
x_crab, y_crab = wcs.world_to_pixel(crab_coord)

# Calcolo la dimensione del raggio in pixel impostandolo a 2.5 arcmin per un'ampiezza totale di 5
pixel_scales = proj_plane_pixel_scales(wcs) * u.deg
pixel_scale_arcmin = pixel_scales[0].to(u.arcmin)
raggio_arcmin = 5 * u.arcmin
raggio_pixel = (raggio_arcmin / pixel_scale_arcmin).value

# Visualizzazione
plt.figure(figsize=(10, 8))
im = plt.imshow(data, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y

# Rimuovo plt.colorbar() isolato per evitare conflitti con la barra successiva
plt.title(f'Image Sum (coverage={int(full_coverage_value)}) runs')
plt.xlabel('X (pixel)')
plt.ylabel('Y (pixel)')

# Richiamo la figura e gli assi correnti
fig = plt.gcf()
ax = plt.gca()

# Aggiungo il cerchio rosso vuoto al centro della nebulosa
cerchio = Circle((x_crab, y_crab), raggio_pixel, edgecolor='red', facecolor='none')
ax.add_patch(cerchio)

# configuro la colorbar affinché abbia la stessa altezza della figura
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)

plt.savefig("SOMMA_TUTTO.png")
#plt.show()