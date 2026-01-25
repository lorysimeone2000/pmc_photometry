from astropy.stats import sigma_clipped_stats
from photutils.aperture import ApertureStats, CircularAperture
from photutils.datasets import make_4gaussians_image

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

data = make_4gaussians_image() # creo un'immagine sintetica di test contenente 4 sorgenti gaussiane
_, median, _ = sigma_clipped_stats(data, sigma=3.0) # rimuovo iterativamente i pixel che deviano più di 3σ dalla media
data = data - median  # sottraggo lo sfondo dai dati. L'immagine ora ha fondo circa zero
aper = CircularAperture((150, 25), 8) # creo un'apertura circolare
aperstats = ApertureStats(data, aper)
print('Centroide x: ' , aperstats.xcentroid)
print('Centroide y: ' , aperstats.ycentroid)
print('Centroide: ' , aperstats.centroid)
print('Media :' , aperstats.mean , ', Mediana: ' , aperstats.median, ', σ: ' , aperstats.std)

plt.imshow(data, cmap="coolwarm", norm=LogNorm()) # genero l'immagine con scala di colori bianco e nero
plt.colorbar()
aper.plot(color='blue', lw=1.5, alpha=0.5) # aggiungo l'immagine dell'apertura

# l'apertura di input può avere più posizioni

print('Ora faccio due aperture')
aper2 = CircularAperture(((150, 25), (90, 60)), 10)
aperstats2 = ApertureStats(data, aper2)
print('centroidi x =' , aperstats2.xcentroid) # ogni colonna è un'apertura
print(aperstats2.sum)
columns = ('id', 'mean', 'median', 'std', 'var', 'sum')
stats_table = aperstats2.to_table(columns)
for col in stats_table.colnames: stats_table[col].info.format = '%.8g'  # for consistent table output

print(stats_table)

aper2.plot(color='blue', lw=1.5, alpha=0.5) # aggiungo le immagini delle aperture

plt.show()