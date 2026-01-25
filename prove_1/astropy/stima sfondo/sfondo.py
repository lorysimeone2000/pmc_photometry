import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.datasets import make_100gaussians_image
import numpy as np
from matplotlib.colors import LogNorm #permette di avere la scala logaritmica
from astropy.stats import sigma_clipped_stats
from photutils.datasets import load_star_image #carica immagine di esempio
from astropy.stats import sigma_clipped_stats

from photutils.detection import DAOStarFinder # trovare stelle

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
import numpy as np
from astropy.stats import biweight_location


data = make_100gaussians_image() # simulazione con media 5 e sigma 2
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data, norm=norm, origin='lower', cmap='Greys_r', interpolation='nearest')
plt.title('Data')

# stimma della mediana e della MAD da tutti i dati

print(np.median(data)) # mediana
print(biweight_location(data)) # biweight location
from astropy.stats import mad_std
print(mad_std(data))

# metodo sigma clipping
from astropy.stats import sigma_clipped_stats
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
print(np.array((mean, median, std)))

# mascherare le sorgenti

from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint

sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
threshold = detect_threshold(data, nsigma=2.0, sigma_clip=sigma_clip)
segment_img = detect_sources(data, threshold, npixels=10)
footprint = circular_footprint(radius=10)
mask = segment_img.make_source_mask(footprint=footprint)
mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
print(np.array((mean, median, std)))

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.background import Background2D
from photutils.datasets import make_100gaussians_image
from scipy.ndimage import rotate

data = make_100gaussians_image()
ny, nx = data.shape
y, x = np.mgrid[:ny, :nx]
gradient = x * y / 5000.0
data2 = data + gradient
data3 = rotate(data2, -45.0)
coverage_mask = (data3 == 0)
bkg3 = Background2D(data3, (15, 15), filter_size=(3, 3), coverage_mask=coverage_mask, fill_value=0.0, exclude_percentile=50.0)
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data3, origin='lower', cmap='Greys_r', norm=norm, interpolation='nearest')
bkg3.plot_meshes(outlines=True, marker='.', color='cyan', alpha=0.3)
plt.xlim(0, 250)
plt.ylim(0, 250)


plt.show()