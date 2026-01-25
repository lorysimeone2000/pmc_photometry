import numpy as np
import os

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

#%matplotlib inline

from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.table import Table

from photutils.detection import find_peaks
from photutils.aperture import CircularAperture


image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'header

# cavo lo sfondo

# metodo sigma clipping
from astropy.stats import sigma_clipped_stats
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
print("Media Mediana Deviazione standard")
print(np.array((mean, median, std)))
data = image_data - median
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
print(f"Deviazione standard fondo tolto: {std}")

# Local Peak Detection

# singola prova

threshold = 9 * std # threshold
print(f"Threshold= {threshold}")
box_size =int(input("Inserisci il numero di pixel del lato della regione minima: ")) # lato regione minima

tbl = find_peaks(data, threshold, box_size=box_size) # trovo i picchi
tbl['peak_value'].info.format = '%.8g'  # per rappresentare meglio la tabella
print(tbl[:10])  # rappresento i primi 10 picchi
print(len(tbl))

positions = np.transpose((tbl['x_peak'], tbl['y_peak'])) # creo un array di posizioni
apertures = CircularAperture(positions, r=5.0) # creo le aperture per ogni posizione
norm = simple_norm(data, 'sqrt', percent=99.9)
plt.imshow(data, cmap='grey_r', origin='lower', norm=LogNorm(), interpolation='nearest')
apertures.plot(color='#0547f9', lw=1.5)
plt.title(f'Versione originale per box di {box_size} pixel', fontsize=14)

plt.show()


# VERSIONE OTTIMIZZATA CON ARRAY OPERATIONS
def azzera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3):
    """
    Versione ottimizzata per azzerare pixel isolati
    """
    from scipy import ndimage

    data_pulita = data.copy()

    # Soglia per considerare un pixel significativo
    soglia_minima = 5 * std
    mask_significativi = data > soglia_minima

    # Crea una mappa dei vicini forti
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Esclude il pixel centrale

    # Per ogni pixel, conta quanti vicini sono sopra threshold_ratio del pixel centrale
    mask_vicini_forti = np.zeros_like(data, dtype=bool)

    y_coords, x_coords = np.where(mask_significativi)
    for y, x in zip(y_coords, x_coords):
        pixel_val = data[y, x]
        threshold_locale = threshold_ratio * pixel_val

        # Controlla regione 3x3
        y_min, y_max = max(0, y - 1), min(data.shape[0], y + 2)
        x_min, x_max = max(0, x - 1), min(data.shape[1], x + 2)

        region = data[y_min:y_max, x_min:x_max]
        mask_forti = region > threshold_locale

        # Conta vicini forti (escludendo il centro)
        center_y, center_x = 1 if y_min < y else 0, 1 if x_min < x else 0
        mask_forti[center_y, center_x] = False  # Esclude il centro

        vicini_forti = np.sum(mask_forti)

        if vicini_forti < min_neighbors:
            data_pulita[y, x] = 0  # Azzera il pixel isolato

    return data_pulita


# Uso la versione pulita

data_pulita = azzera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)

# cerco i picchi nella versione pulita

tbl = find_peaks(data_pulita, threshold, box_size=box_size) # trovo i picchi
tbl['peak_value'].info.format = '%.8g'  # per rappresentare meglio la tabella

positions = np.transpose((tbl['x_peak'], tbl['y_peak'])) # creo un array di posizioni
apertures = CircularAperture(positions, r=5.0) # creo le aperture per ogni posizione
plt.imshow(data_pulita, cmap='grey_r', origin='lower', norm=LogNorm(), interpolation='nearest')
plt.gca().invert_yaxis() # inverto asse y
apertures.plot(color='#0547f9', lw=1.5)
plt.title(f'Versione pulita per box di {box_size} pixel', fontsize=14)

plt.show()

