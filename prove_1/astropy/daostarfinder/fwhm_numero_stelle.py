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

# cavo lo sfondo

# metodo sigma clipping
from astropy.stats import sigma_clipped_stats
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
print("Media Mediana Deviazione standard")
print(np.array((mean, median, std)))

image_data = image_data - median

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

'''daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
# la fwhm dà Larghezza a Metà Altezza Massima
# la threshold dà il limite di deviazioni standard sopra lo sfondo

sources = daofind(image_data - median) # creo una tabella con le informazioni per ogni stella
for col in sources.colnames:
    if col not in ('id', 'npix'):
        sources[col].info.format = '%.2f'  # questo ciclo for rende più leggibili i numeri nella tabella

sources.pprint(max_width=76) # stampa la tabella delle sorgenti trovate con una larghezza massima di 76 caratteri per riga'''

#numero_di_stelle = len(sources)
#print(f"Numero di stelle trovate: {numero_di_stelle}")

def quadrato(a):
    b = np.pow(a,2)
    return b

c = quadrato(7)
print(c)

'''l = np.linspace(0.5, 15, 10000) # asse x fatta dalle FWHM che scorrono
m = quadrato(l)
plt.plot(l, m, 'b-', linewidth=2, label='y = x²')
plt.show()'''

# definisco la funzione che restituisce il numero di stelle, data una FWHM
n_sigma = 10

def numero_stelle(fw):
    daofind = DAOStarFinder(fwhm=fw, threshold=n_sigma * std)
    # la fwhm dà Larghezza a Metà Altezza Massima
    # la threshold dà il limite di deviazioni standard sopra lo sfondo

    sources = daofind(image_data - median)  # creo una tabella con le informazioni per ogni stella
    for col in sources.colnames:
        if col not in ('id', 'npix'):
            sources[col].info.format = '%.2f'  # questo ciclo for rende più leggibili i numeri nella tabella
    return len(sources)

print(f"Numero di stelle trovate: {numero_stelle(3.0)}")

numero_di_prove = 50
x = np.logspace(np.log10(0.5), np.log10(15), 200)
#x = np.linspace(0.5, 15, numero_di_prove) # asse x fatta dalle FWHM che scorrono
y = []
for fwhm_value in x:
    num_stars = numero_stelle(fwhm_value)  # Chiama la funzione per ogni singolo valore
    y.append(num_stars)
    #print(f"FWHM={fwhm_value:.2f}, Stelle trovate: {num_stars}")

y = np.array(y)

# Crea il grafico
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'r-', linewidth=2, markersize=4, label='Stelle vs FWHM')
plt.xlabel('FWHM', fontsize=12)
plt.ylabel('Numero di stelle trovate', fontsize=12)
plt.title('Impatto del parametro FWHM sul numero di stelle rilevate', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
# Scale più graduate
plt.xticks(np.arange(0.5, 15, 0.1))  # Tick ogni tot sull'asse x
plt.yticks(np.arange(0, 4000, 100))  # Tick ogni tot sull'asse y
plt.show()

import time

# Test con un singolo valore per stimare il tempo
start_time = time.time()
num_stars = numero_stelle(3.0)
end_time = time.time()

tempo_per_chiamata = end_time - start_time
tempo_totale_stimato = tempo_per_chiamata * numero_di_prove

print(f"Tempo stimato per {numero_di_prove} chiamate: {tempo_totale_stimato:.1f} secondi")