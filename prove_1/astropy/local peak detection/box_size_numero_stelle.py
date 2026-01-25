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

# Local peak detection

# box_size vs numero stelle

threshold = 10 * std # threshold

def numero_stelle(dati , bs):
    tbl = find_peaks(dati, threshold, box_size=bs)  # trovo i picchi
    tbl['peak_value'].info.format = '%.8g'  # per rappresentare meglio la tabella

    return len(tbl)

'''box_size = int(input(f"Box size: "))
print(f"Numero di stelle trovate: {numero_stelle(data , box_size)}")'''

numero_di_pixel = 250
x = np.arange(3, numero_di_pixel) # asse x fatta dal numero di pixel che scorre
y = []
for box in x:
    num_stars = numero_stelle(data,box)  # chiama la funzione per ogni singolo valore
    y.append(num_stars)
    #print(f"FWHM={fwhm_value:.2f}, Stelle trovate: {num_stars}")

y = np.array(y)

# Crea il grafico
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'r-', linewidth=2, markersize=4, label='Stelle vs dimensioni box (versione originale)')
plt.xlabel('Dimensione box', fontsize=12)
plt.ylabel('Numero di stelle trovate', fontsize=12)
plt.title('Stelle vs dimensioni box (versione originale)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()

import time

# Test con un singolo valore per stimare il tempo
start_time = time.time()
num_stars = numero_stelle(data,3.0)
end_time = time.time()

tempo_per_chiamata = end_time - start_time
tempo_totale_stimato = tempo_per_chiamata * numero_di_pixel

print(f"Tempo stimato per {numero_di_pixel} chiamate: {tempo_totale_stimato:.1f} secondi")

# Faccio la stessa cosa con i dati puliti

# pulizia

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

data_pulita = azzera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)
data = data_pulita

# creazione grafico

x = np.arange(3, numero_di_pixel) # asse x fatta dal numero di pixel che scorre
y = []
for box in x:
    num_stars = numero_stelle(data,box)  # chiama la funzione per ogni singolo valore
    y.append(num_stars)
    #print(f"FWHM={fwhm_value:.2f}, Stelle trovate: {num_stars}")

y = np.array(y)

# Crea il grafico
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'r-', linewidth=2, markersize=4, label='Stelle vs dimensioni box (versione pulita)')
plt.xlabel('Dimensione box', fontsize=12)
plt.ylabel('Numero di stelle trovate', fontsize=12)
plt.title('Stelle vs dimensioni box (versione pulita)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()