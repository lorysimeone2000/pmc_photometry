import numpy as np
import os

#%matplotlib inline

from astropy.io import fits
from astropy.utils.data import download_file

from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clipped_stats

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.data import download_file

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

# Set up wcs
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"  # prima immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine

hdu_list = fits.open(image_file)

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
w = WCS(hdu_list[0].header) # creo un oggetto WCS usando l'header del file FITS,
# che contiene le informazioni per le trasformazioni di coordinate
#print(hdu_list[0].header) #mette tutti i dati dell'header
print(hdu_list[0])

coord_cel = SkyCoord(ra=89, dec=-2, unit='deg')
coordinata = w.world_to_pixel(coord_cel)
print(coordinata)
