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


data = make_100gaussians_image()
bkg_estimator = MedianBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)

data -= bkg.background  # subtract the background
threshold = 1.5 * bkg.background_rms # definisco la threshold

kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
convolved_data = convolve(data, kernel)

plt.imshow(convolved_data, cmap='grey', origin='lower', norm=LogNorm(), interpolation='nearest')
plt.colorbar()

plt.show()

'''segment_map = detect_sources(convolved_data, threshold, npixels=10)
print(segment_map)

# sfusione sorgenti

segm_deblend = deblend_sources(convolved_data, segment_map,
                               npixels=10, nlevels=32, contrast=0.001,
                               progress_bar=False)
segment_map = segm_deblend

norm = ImageNormalize(stretch=SqrtStretch())
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
ax1.set_title('Background-subtracted Data')
ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
ax2.set_title('Segmentation Image')

plt.show()'''
norm = ImageNormalize(stretch=SqrtStretch())
# Sourcefinder

from photutils.segmentation import SourceFinder
finder = SourceFinder(npixels=10, progress_bar=False)
segment_map = finder(convolved_data, threshold)
print(segment_map)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
ax1.set_title('Background-subtracted Data')
ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
ax2.set_title('Segmentation Image con Sourcefinder')

plt.show()

# fotometria

segm_deblend = deblend_sources(convolved_data, segment_map, npixels=10, nlevels=32, contrast=0.001, progress_bar=False)

cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
print(cat)

# tabella sorgenti
tbl = cat.to_table()
tbl['xcentroid'].info.format = '.2f'  # optional format
tbl['ycentroid'].info.format = '.2f'
tbl['kron_flux'].info.format = '.2f'
print(tbl)


norm = simple_norm(data, 'sqrt')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
ax1.set_title('Data')
ax2.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap, interpolation='nearest')
ax2.set_title('Segmentation Image')
# creo le aperture ellittiche
cat.plot_kron_apertures(ax=ax1, color='white', lw=1.5)
cat.plot_kron_apertures(ax=ax2, color='white', lw=1.5)

plt.show()

from photutils.utils import calc_total_error
effective_gain = 500.0
error = calc_total_error(data, bkg.background_rms, effective_gain)
cat = SourceCatalog(data, segm_deblend, error=error)
labels = [1, 5, 20, 50, 75, 80]
cat_subset = cat.get_labels(labels)  # select a subset of objects
columns = ['label', 'xcentroid', 'ycentroid', 'segment_flux',
           'segment_fluxerr']
tbl5 = cat_subset.to_table(columns=columns)
tbl5['xcentroid'].info.format = '{:.4f}'  # optional format
tbl5['ycentroid'].info.format = '{:.4f}'
tbl5['segment_flux'].info.format = '{:.4f}'
tbl5['segment_fluxerr'].info.format = '{:.4f}'
for col in tbl5.colnames:
    tbl5[col].info.format = '%.8g'  # for consistent table output
print(tbl5)