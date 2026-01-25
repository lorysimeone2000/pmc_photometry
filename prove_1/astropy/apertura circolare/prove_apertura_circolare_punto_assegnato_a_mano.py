from matplotlib.colors import LogNorm
from photutils.aperture import ApertureStats
from photutils.utils import calc_total_error
import numpy as np

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.datasets import make_100gaussians_image
import astropy.visualization
import astropy.units as u
from astropy.wcs import WCS
from astropy.table import Table


# Set up matplotlib
import matplotlib.pyplot as plt

#%matplotlib inline

from astropy.io import fits

from astropy.utils.data import download_file

image_file = "20250106_231255.fits"

hdu_list = fits.open(image_file)
hdu_list.info()



image_data = hdu_list[0].data
#print(hdu_list[0].header)
'''image_header = hdu_list[0].header
print("RA",image_header["RA"])'''

print(image_data)
print(image_data[0,0])
print(image_data.shape)



plt.imshow(image_data, cmap="viridis", norm = LogNorm())
cbar = plt.colorbar(ticks=[5.0e3, 1.0e4, 2.0e4])
cbar.ax.set_yticklabels(["5,000", "10,000", "20,000"])



plt.colorbar()

plt.show()

# creo apertura e anello su una stella specifica

positions = [(2352.0, 1024.9)] # centro apertura
aperture = CircularAperture(positions, r=15.0) # apertura
annulus_aperture = CircularAnnulus(positions, r_in=30.0, r_out=40.0) # anello

norm = LogNorm()
plt.imshow(image_data, norm=norm, cmap="gray", interpolation='nearest')
#plt.xlim(0, 170)
#plt.ylim(130, 250)

ap_patches = aperture.plot(color='white', lw=2, label='Photometry aperture')
ann_patches = annulus_aperture.plot(color='red', lw=2, label='Background annulus')
handles = (ap_patches[0], ann_patches[0])
plt.legend(loc=(0.17, 0.05), facecolor='#458989', labelcolor='white',
           handles=handles, prop={'weight': 'bold', 'size': 11})
plt.show()

# Mediana sigma-clipped dentro un anello circolare

from astropy.stats import SigmaClip

sigclip = SigmaClip(sigma=3.0, maxiters=10) # scarto i valori che deviano più di 3σ dalla media e lo itera al massimo maxiters volte

# utilizzo il metodo ApertureStats

aper_stats = ApertureStats(image_data, aperture, sigma_clip=None) # non applica la pulizia dentro l'apertura circolare
bkg_stats = ApertureStats(image_data, annulus_aperture, sigma_clip=sigclip) # applica la pulizia nell'anello

print('Sfondo sigma-clippato per pixel: ')
print(bkg_stats.median)

# ora calcolo lo sfondo totale all'interno dell'apertura

total_bkg_sigma = bkg_stats.median * aper_stats.sum_aper_area.value # sfondo calcolato dall'anello per l'area dell'apertura
print('Sfondo sigma-clippato totale: ')
print(total_bkg_sigma)

# calcolo la fotometria con lo sfondo sottratto

apersum_bkgsub_sigma = aper_stats.sum - total_bkg_sigma
print('Fotometria con sfondo sigma-clippato sottratto: ')
print(apersum_bkgsub_sigma)