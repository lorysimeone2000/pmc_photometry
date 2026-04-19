import numpy.ma as ma # Aggiunto per i Masked Arrays
import matplotlib.pyplot as plt
import pandas as pd
#pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
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

from astroquery.vizier import Vizier
from astropy.coordinates import Angle

from shapely.geometry import Point, Polygon
# warning
import warnings
from astropy.io.fits.verify import VerifyWarning
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning) # Sopprime il warning FITSFixedWarning

from pathlib import Path

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
data = image_data
hdu_list.close()

# --- ESTRAZIONE E VISUALIZZAZIONE ---

# Applichiamo la maschera:
# Usiamo ~mask_max_coverage per nascondere tutti i pixel NON UGUALI a 115.
# data = ma.masked_array(image_data, mask=~mask_max_coverage)
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
print("Mediana: " , median)
data = data - median

# Visualizzazione
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.colorbar()
plt.title(f'Immagine Sommata (Copertura={int(full_coverage_value)}) run')
plt.xlabel('X (pixel)')
plt.ylabel('Y (pixel)')

plt.savefig("SOMMA_TUTTO.png")
#plt.show()