import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm #permette di avere la scala logaritmica
from astropy.stats import sigma_clipped_stats
from photutils.datasets import load_star_image #carica immagine di esempio

from photutils.detection import DAOStarFinder # trovare stelle

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture


hdu = load_star_image()  # Carica un'immagine FITS di esempio con stelle
data = hdu.data[0:401, 0:401]  # Ritaglia un'area tot x tot pixel
mean, median, std = sigma_clipped_stats(data, sigma=3.0)  # doctest: +REMOTE_DATA
print(np.array((mean, median, std)))

#trovare stelle
daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
# la fwhm dà Larghezza a Metà Altezza Massima
# la threshold dà il limite di deviazioni standard sopra lo sfondo

sources = daofind(data - median) #creo una tabella con le informazioni per ogni stella
for col in sources.colnames:
    if col not in ('id', 'npix'):
        sources[col].info.format = '%.2f'  # questo ciclo for rende più leggibili i numeri nella tabella

sources.pprint(max_width=76) # stampa la tabella delle sorgenti trovate con una larghezza massima di 76 caratteri per riga



plt.imshow(data, cmap="gray", norm=LogNorm()) #genero l'immagine con scala di colori bianco e nero
plt.colorbar()

positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.0)
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
apertures.plot(color='blue', lw=1.5, alpha=0.5)

plt.show()

