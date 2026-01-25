import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import (mark_inset, zoomed_inset_axes)
from photutils.centroids import (centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic)
from photutils.datasets import make_4gaussians_image

data = make_4gaussians_image()
data -= np.median(data[0:30, 0:125]) # tolgo sfondo
data = data[40:80, 70:110]
xycen1 = centroid_com(data)
xycen2 = centroid_quadratic(data)
xycen3 = centroid_1dg(data)
xycen4 = centroid_2dg(data)
xycens = [xycen1, xycen2, xycen3, xycen4]
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(data, origin='lower', interpolation='nearest')
marker = '+'
ms = 60
colors = ('white', 'cyan', 'red', 'blue')
labels = ('Center of Mass', 'Quadratic', '1D Gaussian', '2D Gaussian')
for xycen, color, label in zip(xycens, colors, labels):
    ax.scatter(*xycen, color=color, marker=marker, s=ms, label=label)

ax.legend(loc='lower right', fontsize=12)

ax2 = zoomed_inset_axes(ax, zoom=6, loc=9)
ax2.imshow(data, vmin=190, vmax=220, origin='lower', interpolation='nearest')
ms = 1000
for xycen, color in zip(xycens, colors):
    ax2.scatter(*xycen, color=color, marker=marker, s=ms)
ax2.set_xlim(19, 21)
ax2.set_ylim(19, 21)
mark_inset(ax, ax2, loc1=3, loc2=4, fc='none', ec='black')
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
ax.set_xlim(0, data.shape[1] - 1)
ax.set_ylim(0, data.shape[0] - 1)

plt.show()