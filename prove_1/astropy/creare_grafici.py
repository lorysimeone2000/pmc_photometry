from matplotlib.colors import LogNorm

import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt

#%matplotlib inline

from astropy.io import fits

from astropy.utils.data import download_file

image_file = "20250106_231255.fits"

hdu_list = fits.open(image_file)
hdu_list.info()



image_data = hdu_list[0].data

print(image_data)
print(image_data[0,0])
print(image_data.shape)

plt.imshow(image_data, cmap="grey", norm = LogNorm())
#cbar = plt.colorbar(ticks=[5.0e3, 1.0e4, 2.0e4])
#cbar.ax.set_yticklabels(["5,000", "10,000", "20,000"])

#plt.colorbar()

plt.show()


print(type(image_data.flatten()))
print(image_data.flatten().shape)

for i in 3072:
    for j in 3072:
        

histogram = plt.hist(image_data.flatten(), bins=256,range=(-0.5,255.5))
plt.yscale("log")


plt.show()