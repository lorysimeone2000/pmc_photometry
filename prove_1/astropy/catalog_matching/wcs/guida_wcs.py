from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.data import download_file
# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

fn = get_pkg_data_filename('data/j94f05bgq_flt.fits', package='astropy.wcs.tests') # crea il percorso a un file FITS di esempio incluso nell'installazione di Astropy per scopi di test
f = fits.open(fn) # come sempre
w = WCS(f[1].header) # creo un oggetto WCS usando l'header della seconda estensione (f[1]) del file FITS,
# che contiene le informazioni per le trasformazioni di coordinate
sky = w.pixel_to_world(30, 40)
print(sky)
f.close()

print('------------------------------------')

from astropy.utils.data import get_pkg_data_filename
fn = get_pkg_data_filename('data/j94f05bgq_flt.fits', package='astropy.wcs.tests')
f = fits.open(fn)
w = WCS(f[1].header)
x, y = w.world_to_pixel(sky)
print(x, y)
f.close()

# suddivisione oggetti wcs

filename = get_pkg_data_filename('l1448/l1448_13co.fits')
wcs = WCS(fits.getheader(filename, ext=0))

from astropy.wcs.wcsapi import SlicedLowLevelWCS

slices = [10, slice(30, 100), slice(30, 100)]
subwcs = SlicedLowLevelWCS(wcs, slices=slices)
# in alternativa
print(wcs[10, 30:100, 30:100])

