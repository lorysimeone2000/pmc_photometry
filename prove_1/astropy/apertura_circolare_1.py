from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from photutils.aperture import ApertureStats, CircularAperture
from photutils.datasets import make_4gaussians_image
from photutils.aperture import ApertureStats

data = make_4gaussians_image()
_, median, _ = sigma_clipped_stats(data, sigma=3.0)
data -= median  # subtract background from the data
aper = CircularAperture((150, 25), 8)
aperstats = ApertureStats(data, aper)  # doctest: +FLOAT_CMP
print(aperstats.xcentroid)  # doctest: +FLOAT_CMP
print(aperstats.mean, aperstats.median, aperstats.std)  # doctest: +FLOAT_CMP
print(aperstats.sum)  # doctest: +FLOAT_CMP

plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='viridis', origin='lower')  # 'viridis' è una colormap adatta
plt.colorbar(label='Intensità')
plt.title('4 Gaussiane sintetiche (Photutils)')
plt.xlabel('X')
plt.ylabel('Y')

# Aggiungi cerchi per evidenziare le aperture (opzionale)
from photutils.aperture import CircularAperture
positions = [(150, 25), (50, 50), (100, 100), (25, 150)]  # Posizioni approssimative
apertures = CircularAperture(positions, r=8)
apertures.plot(color='red', lw=2, alpha=0.7)

plt.show()