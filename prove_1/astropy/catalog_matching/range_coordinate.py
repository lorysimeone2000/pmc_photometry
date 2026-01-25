import numpy as np
import os

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

#%matplotlib inline

from astropy.io import fits
from astropy.utils.data import download_file

from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"  # prima immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'headerimport numpy as np
import os

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

#%matplotlib inline

from astropy.io import fits
from astropy.utils.data import download_file
from astropy.table import Table

from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture

# Set up wcs
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
import warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning) # cancello gli avvertimenti non rilevanti dovuti alle modifiche di astropy

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"  # prima immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'header

print(image_data.shape)

w = WCS(hdu_list[0].header) # creo un oggetto WCS usando l'header del file FITS,
# che contiene le informazioni per le trasformazioni di coordinate

# trovo gli estremi

alto_destra = w.pixel_to_world(3072, 2048)
print(f"Coordinate in alto a destra: {alto_destra}")
aperture1 = CircularAperture((3072,2048), r=300)
basso_sinistra = w.pixel_to_world(0,0)
print(f"Coordinate in basso a sinistra: {basso_sinistra}")
aperture2 = CircularAperture((0,0), r=300)

# rappresento gli estremi
RA_min = alto_destra.ra.deg  # oppure .hour per avere in ore
print(f"RA_min: {RA_min}°")
RA_max = basso_sinistra.ra.deg
print(f"RA_max: {RA_max}°")
DEC_min = alto_destra.dec.deg
print(f"DEC_min: {DEC_min}°")
DEC_max = basso_sinistra.dec.deg
print(f"DEC_max: {DEC_max}°")

'''plt.imshow(image_data, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')
aperture1.plot(color='blue', lw=0.8, alpha=0.5)
aperture2.plot(color='blue', lw=0.8, alpha=0.5)

plt.show()'''

print(w.pixel_to_world(2353, 1026))