import matplotlib.pyplot as plt
import numpy as np
from photutils.centroids import (centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic)
from photutils.datasets import make_4gaussians_image

data = make_4gaussians_image()
data -= np.median(data[0:30, 0:125]) # tolgo sfondo
data = data[40:80, 70:110] # prendo un singolo oggetto
plt.figure(figsize=(8, 4))
plt.imshow(data, origin='lower', interpolation='nearest')
plt.tight_layout()

print(f"Centroide vero: [20.0000000 20.00000000]")
x1, y1 = centroid_com(data)
print(f"Centroide com: {np.array((x1, y1))}")
x2, y2 = centroid_quadratic(data)
print(f"Centroide quadratico: {np.array((x2, y2))}")
x3, y3 = centroid_1dg(data)
print(f"Centroide 1dg: {np.array((x3, y3))}")
x4, y4 = centroid_2dg(data)
print(f"Centroide 2dg: {np.array((x4, y4))}")

plt.show()