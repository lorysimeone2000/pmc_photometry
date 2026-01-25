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

# supponiamo ad esempio di aver precedentemente calcolato l'errore su ciascuno valore pixel e salvato nell'array

positions = [(30.0, 30.0), (40.0, 40.0)]
aperture = CircularAperture(positions, r=3.0)
data = np.ones((100, 100)) # matrice di uni
error = 0.1 * data # matrice

phot_table = aperture_photometry(data, aperture, error=error)
for col in phot_table.colnames:
    phot_table[col].info.format = '%.8g'  # for consistent table output

print('Non mettendoci il guadagno: ')
print(phot_table)

# ora suppongo di avere un'immagine solo di sfondo chiamata bkg_error.
# se i dati sono in unità di elettroni/s, si utilizzerà il tempo d'esposizione come guadagno

from photutils.utils import calc_total_error

bkg_error = 0.1 * data # matrice
effective_gain = 500  # guadagno del rivelatore espresso in tempo d'esposizione in secondi
error = calc_total_error(data, bkg_error, effective_gain) # la funzione considera sia il rumore di Poisson
# dei fotoni che l'errore del fondo cielo
# phot_table = aperture_photometry(data - bkg, aperture, error=error)
phot_table = aperture_photometry(data, aperture, error=error)

print('Mettendoci il guadagno: ')
print(phot_table)

