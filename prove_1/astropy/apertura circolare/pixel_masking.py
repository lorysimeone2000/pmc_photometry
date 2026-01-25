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


data = np.ones((5, 5)) # matrice di uni
aperture = CircularAperture((2, 2), 2.0) # apertura circolare
mask = np.zeros(data.shape, dtype=bool) # creo una matrice booleana con valori inizialmente tutti falsi
data[2, 2] = 100.0  # simulo un pixel anomalo
mask[2, 2] = True # imposto True il pixel anomalo

t1 = aperture_photometry(data, aperture) # mask esclude i pixel anomali dal calcolo della fotometria
t1['aperture_sum'].info.format = '%.8g'

print('aperture_sum senza mask: ')
print(t1['aperture_sum'])

t1 = aperture_photometry(data, aperture, mask=mask) # mask esclude i pixel anomali dal calcolo della fotometria
t1['aperture_sum'].info.format = '%.8g'

print('aperture_sum con mask: ')
print(t1['aperture_sum'])
print('Funziona')