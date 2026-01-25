from photutils.aperture import CircularAperture
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture
from photutils.datasets import make_wcs
import numpy as np
from photutils.aperture import aperture_photometry

# Creare due aperture circolari di raggio r fissato

positions = [(30.0, 30.0), (40.0, 40.0)]
aperture = CircularAperture(positions, r=3.0)

data = np.ones((100, 100)) # creo una matrice di uni

phot_table = aperture_photometry(data, aperture) # creo la tabella da stampare su terminale
phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output
print(phot_table) # ottengo una tabella di quattro colonne, dove ''aperture_sum'' è la somma dei valori dell'apertura
# In questo caso i valori sono tutti uni, quindi la somma è l'area del cerchio di raggio r.

# Sovrapposizione pixel: ogni pixel viene diviso in 5

print('Ora ho diviso ogni pixel in 5')
phot_table = aperture_photometry(data, aperture, method='subpixel', subpixels=5)
print(phot_table)

# Creo tre aperture circolari su ciascuna posizione

print('Ora creo tre aperture di raggi diversi per ognuna delle posizioni')
radii = [3.0, 4.0, 5.0] # definisco i raggi
positions = [(30.0, 30.0), (40.0, 40.0)] # definisco le posizioni
apertures = [CircularAperture(positions, r=r) for r in radii] # creo le aperture con le posizioni di prima
phot_table = aperture_photometry(data, apertures) # creo la tabella da stampare su terminale
for col in phot_table.colnames:
    phot_table[col].info.format = '%.8g'  # for consistent table output


print(phot_table)