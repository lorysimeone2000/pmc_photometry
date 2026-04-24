#Guida: https://learn.astropy.org/tutorials/FITS-images.html#

import numpy as np
import os

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

#%matplotlib inline

from astropy.io import fits
from astropy.utils.data import download_file

from astropy.stats import sigma_clipped_stats

image_file = "/home/lorysimeone/tesi_magistrale/Lorenzo/pmc_photometry/run_vecchie/20250120_run1/20250120_212815.fits"

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'header
image_header = hdu_list[0].header
print("RA",image_header["RA"]) #dà la RA
print("DEC",image_header["DEC"]) #dà il DEC

# cavo lo sfondo

# metodo sigma clipping
from astropy.stats import sigma_clipped_stats
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
print("Media Mediana Deviazione standard")
print(np.array((mean, median, std)))

#data_senza_fondo = image_data - median
data_senza_fondo = image_data


print(data_senza_fondo) #dà la matrice dei valori dei pixel
#print(data_senza_fondo[0,0]) #dà il valore dell'pixel di coordinate [0,0]
print(data_senza_fondo.shape) #dà le dimensioni della matrice

# Verifica i risultati
mean_new, median_new, std_new = sigma_clipped_stats(data_senza_fondo, sigma=3.0)
print("\nDOPO la sottrazione del fondo:")
print(f"Media: {mean_new:.2f}")
print(f"Mediana: {median_new:.2f}")
print(f"Deviazione standard: {std_new:.2f}")

# La mediana dovrebbe essere molto vicina a zero
print(f"\nLa mediana dopo la sottrazione è: {median_new:.2f}")
print("(Dovrebbe essere molto vicina a 0)")

plt.imshow(data_senza_fondo, cmap="grey_r", norm=LogNorm()) #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.colorbar()

# plt.show()
plt.close()

print(type(data_senza_fondo.flatten())) #verifico di aver creato un array 1D
print(data_senza_fondo.flatten().shape) #dà le dimensioni dell'array

histogram = plt.hist(data_senza_fondo.flatten(), bins=256,range=(-0.5,255.5)) #genero l'istogramma dell'array con i valori
plt.yscale("log")
plt.xlabel("Valore pixel")
plt.ylabel("Frequenza valore pixel")

plt.savefig('histogram.png')
#plt.show()