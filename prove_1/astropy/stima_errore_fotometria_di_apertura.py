from photutils.aperture import ApertureStats
from photutils.aperture import aperture_photometry
from matplotlib.colors import LogNorm
from photutils.aperture import ApertureStats
import numpy as np

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.datasets import make_100gaussians_image
from photutils.aperture import ApertureStats

# Set up matplotlib
import matplotlib.pyplot as plt

#%matplotlib inline

from astropy.io import fits

from astropy.utils.data import download_file

image_file = "20250106_231255.fits"

hdu_list = fits.open(image_file)
hdu_list.info()

image_data = hdu_list[0].data

positions = [(2352.0, 1024.9)]
aperture = CircularAperture(positions, r=15.0)
annulus_aperture = CircularAnnulus(positions, r_in=10, r_out=15)

data = np.ones((100, 100))
error = 50 * data

phot_table = aperture_photometry(data, aperture, error=error)
for col in phot_table.colnames:
    phot_table[col].info.format = '%.8g'  # for consistent table output
print(phot_table)