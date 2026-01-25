import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

#%matplotlib inline

from astropy.io import fits
from astropy.utils.data import download_file

from astropy.stats import sigma_clipped_stats

#-------------

from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture
from photutils.detection import find_peaks # permette di trovare i picchi locali

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



plt.imshow(image_data, cmap="gray", norm=LogNorm()) #genero l'immagine con scala di colori bianco e nero
plt.colorbar()

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
threshold = median + (5.0 * std)
''' Definisce una soglia di rilevazione: qualsiasi picco con valore maggiore di median + 5·std
verrà considerato rilevante (cioè 5-sigma sopra il fondo). Questo aiuta a non rilevare rumore casuale come stelle'''

tbl = find_peaks(image_data, threshold, box_size=11) # individua picchi locali che superano la threshold, il risultato è una tabella
# box_size=11 indica la dimensione della finestra (11×11 pixel) usata per cercare massimi locali:
# ogni picco è il massimo all’interno della relativa finestra.

positions = np.transpose((tbl['x_peak'], tbl['y_peak'])) # costruisco un array delle posizioni
apertures = CircularAperture(positions, r=5.0) # crea delle aperture circolari centrate sulle posizioni trovate
norm = simple_norm(image_data, 'sqrt', percent=99.9) # definisce una normalizzazione per la visualizzazione
plt.imshow(image_data, cmap='Greys_r', origin='lower', norm=norm,
           interpolation='nearest') # mostra l'immagine
apertures.plot(color='#0547f9', lw=1.5) # mostra le aperture circolari

# fisso i limiti degli assi
plt.xlim(0, image_data.shape[1] - 1)
plt.ylim(0, image_data.shape[0] - 1)

print(tbl[:10])  # stampa sul terminale i primi 10 picchi

plt.show()