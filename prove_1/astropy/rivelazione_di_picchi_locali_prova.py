import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture
from photutils.datasets import make_100gaussians_image
from photutils.detection import find_peaks

data = make_100gaussians_image()
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
threshold = median + (5.0 * std)
tbl = find_peaks(data, threshold, box_size=11)

positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
apertures = CircularAperture(positions, r=5.0)
norm = simple_norm(data, 'sqrt', percent=99.9)
plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm,
           interpolation='nearest')
apertures.plot(color='#0547f9', lw=1.5)
plt.xlim(0, data.shape[1] - 1)
plt.ylim(0, data.shape[0] - 1)

print(tbl[:10])  # print only the first 10 peaks

plt.show()