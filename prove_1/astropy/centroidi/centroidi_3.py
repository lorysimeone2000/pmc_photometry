import matplotlib.pyplot as plt
import numpy as np
from photutils.centroids import centroid_2dg, centroid_sources
from photutils.datasets import make_4gaussians_image

data = make_4gaussians_image()
data -= np.median(data[0:30, 0:125]) # tolgo sfondo

# queste sono stime iniziali che verranno affinate dal algoritmo di centroiding
x_init = (25, 91, 151, 160)
y_init = (40, 61, 24, 71)
x, y = centroid_sources(data, x_init, y_init, box_size=25, centroid_func=centroid_2dg) # restituisce le coordinate x e y precise dei centroidi
# box_size=25 usa una finestra di 25×25 pixel attorno a ogni posizione iniziale per il calcolo
print(x,y)

plt.figure(figsize=(8, 4))
plt.imshow(data, origin='lower', interpolation='nearest')
plt.scatter(x, y, marker='+', s=80, color='red', label='Centroids')
plt.legend()
plt.tight_layout()

plt.show()