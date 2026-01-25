import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture, RectangularAperture
from photutils.datasets import load_star_image
from photutils.detection import DAOStarFinder

hdu = load_star_image()
data = hdu.data[0:401, 0:401]
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
mask = np.zeros(data.shape, dtype=bool)
mask[50:151, 50:351] = True
mask[250:351, 150:351] = True
sources = daofind(data - median, mask=mask)
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.0)
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
           interpolation='nearest')
plt.title('Star finder with a mask to exclude regions')
apertures.plot(color='blue', lw=1.5, alpha=0.5)
rect1 = RectangularAperture((200, 100), 300, 100, theta=0)
rect2 = RectangularAperture((250, 300), 200, 100, theta=0)
rect1.plot(color='salmon', ls='dashed')
rect2.plot(color='salmon', ls='dashed')

plt.show()