from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
import numpy as np
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel


data = make_100gaussians_image()
bkg_estimator = MedianBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)

data -= bkg.background  # subtract the background

# Convoluzione

kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
convolved_data = convolve(data, kernel)

plt.imshow(convolved_data, cmap='grey', origin='lower', norm=LogNorm(), interpolation='nearest')
plt.title('Dati Convoluti con Kernel Gaussiano\n(FWHM = 3.0 pixels)', fontsize=14, fontweight='bold')
plt.colorbar()

plt.show()

# Sourcefinder

norm = ImageNormalize(stretch=SqrtStretch())

threshold = 1.5 * bkg.background_rms # definisco la threshold

from photutils.segmentation import SourceFinder
finder = SourceFinder(npixels=10, progress_bar=True) # inserisco n. minimo di pixel
segment_map = finder(convolved_data, threshold)
print(segment_map)

plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
plt.title('Sorgenti Rilevate\n(Threshold = 1.5σ, npixels = 10)', fontsize=14, fontweight='bold')

plt.show()

tbl = cat.to_table()
tbl['xcentroid'].info.format = '.2f'  # optional format
tbl['ycentroid'].info.format = '.2f'
tbl['kron_flux'].info.format = '.2f'
print(tbl)
