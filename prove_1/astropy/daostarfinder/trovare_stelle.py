#Guida: https://learn.astropy.org/tutorials/FITS-images.html#

import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm #permette di avere la scala logaritmica

#%matplotlib inline

from astropy.io import fits
from astropy.utils.data import download_file

from astropy.stats import sigma_clipped_stats

# Set up photutils
from photutils.datasets import load_star_image #carica immagine di esempio
from photutils.detection import DAOStarFinder # trovare stelle

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture

image_file = "20250106_231255.fits"

hdu_list = fits.open(image_file)
hdu_list.info() #dà le informazioni del file

image_data = hdu_list[0].data #creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'header
image_header = hdu_list[0].header
print("RA",image_header["RA"]) #dà la RA
print("DEC",image_header["DEC"]) #dà il DEC

print(image_data) #dà la matrice dei valori dei pixel
#print(image_data[0,0]) #dà il valore dell'pixel di coordinate [0,0]
print(image_data.shape) #dà le dimensioni della matrice

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
print(np.array((mean, median, std)))

# cavo lo sfondo

# metodo sigma clipping
from astropy.stats import sigma_clipped_stats
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
print("Media Mediana Deviazione standard")
print(np.array((mean, median, std)))

image_data = image_data - median

# Trovare stelle

t = float(input('Threshold (n. sigma) = ')) # threshold

threshold = t*std
print(f"Threshold: {threshold}")
fwhm = float(input("FWHM: "))
daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold)
# la fwhm dà Larghezza a Metà Altezza Massima
# la threshold dà il limite di deviazioni standard sopra lo sfondo

sources = daofind(image_data - median) # creo una tabella con le informazioni per ogni stella
for col in sources.colnames:
    if col not in ('id', 'npix'):
        sources[col].info.format = '%.2f'  # questo ciclo for rende più leggibili i numeri nella tabella

sources.pprint(max_width=76) # stampa la tabella delle sorgenti trovate con una larghezza massima di 76 caratteri per riga


plt.imshow(image_data, cmap="gray", norm=LogNorm(), interpolation='nearest') # genero l'immagine con scala di colori bianco e nero
plt.colorbar()

#contrassegno la posizione delle stelle trovate

positions = np.transpose((sources['xcentroid'], sources['ycentroid'])) # estrae le coordinate delle stelle dalla tabella sources
apertures = CircularAperture(positions, r=4.0) # crea cerchi di raggio r pixel
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(image_data, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')
apertures.plot(color='blue', lw=0.8, alpha=0.5)

plt.show()

'''print(type(image_data.flatten())) #verifico di aver creato un array 1D
print(image_data.flatten().shape) #dà le dimensioni dell'array

histogram = plt.hist(image_data.flatten(), bins=256,range=(-0.5,255.5)) #genero l'istogramma dell'array con i valori
plt.yscale("log")

plt.show()'''