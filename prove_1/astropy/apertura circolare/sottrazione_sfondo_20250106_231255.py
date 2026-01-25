import numpy as np


# Set up astropy
from astropy.stats import sigma_clipped_stats
import astropy.visualization
from astropy.visualization import simple_norm

# Set up photutils
from photutils.aperture import ApertureStats, CircularAperture
from photutils.datasets import make_4gaussians_image
from photutils.datasets import make_100gaussians_image
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.aperture import aperture_photometry

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

# Sottrazione sfondo locale

image_file = "20250106_231255.fits"

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel

# definisco l'apertura e l'anello