#In questo codice costruisco la matrice dei valori di un file FITS, ne ricavo l'immagine in scala logaritmica e un istogramma dei valori dei pixel
import pandas as pd
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
import numpy as np
import os
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture

# Set up wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from astropy.visualization import (ZScaleInterval, AsinhStretch,
                                   ImageNormalize, LinearStretch)

from shapely.geometry import Point, Polygon
# warning
import warnings
from astropy.io.fits.verify import VerifyWarning
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

from pathlib import Path

def converti_valore(valore):
    valore = str(valore).strip()  # Assicura che sia stringa prima dello strip
    if not valore: return valore
    try:
        return int(valore)
    except ValueError:
        pass
    try:
        return float(valore)
    except ValueError:
        pass
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False
    return valore


def leggi_header_da_csv(filename):
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':
                break
    return header_dict


image_file = "/home/lorysimeone/tesi_magistrale/prove_1/20250106_231255.fits"  # prima immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine

run = 1

cartella_csv_cat = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}"
file_csv_cat = sorted([f for f in os.listdir(cartella_csv_cat) if f.endswith('.csv')])
lista_percorsi_csv_cat = [os.path.join(cartella_csv_cat, file) for file in file_csv_cat]

n_immagine = 35

percorso_file_csv_cat = lista_percorsi_csv_cat[n_immagine]
dataframe = pd.read_csv(percorso_file_csv_cat, comment='#')
tbl_catalogate = Table.from_pandas(dataframe) # tabella di tutte le stelle catalogate

header_info = leggi_header_da_csv(percorso_file_csv_cat)
image_file = header_info.get('PERCORSO_FILE', '')

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'header
image_header = hdu_list[0].header
print("RA",image_header["RA"]) #dà la RA
print("DEC",image_header["DEC"]) #dà il DEC

print(image_data) #dà la matrice dei valori dei pixel
#print(image_data[0,0]) #dà il valore dell'pixel di coordinate [0,0]
print(image_data.shape) #dà le dimensioni della matrice
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data

np.savetxt("dati_output.csv", data, delimiter=",", fmt="%.2f")

plt.imshow(image_data, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.colorbar()

plt.show()

# Creazione istogramma

'''print(type(image_data.flatten())) #verifico di aver creato un array 1D
print(image_data.flatten().shape) #dà le dimensioni dell'array

histogram = plt.hist(image_data.flatten(), bins=256,range=(-0.5,255.5)) #genero l'istogramma dell'array con i valori
plt.yscale("log")

plt.show()'''

# Rimozione pixel isolati

def rimuovi_pixel_isolati_media(data, min_neighbors=2, threshold_ratio=0.3):
    """
    Rileva i pixel isolati (che non hanno abbastanza vicini forti)
    e li sostituisce con la media dei pixel adiacenti.
    """
    data_pulita = data.copy()

    # Calcolo soglia minima per considerare un pixel come "candidato" (hot pixel o sorgente)
    # Assumiamo che 'std' sia definita globalmente o la ricalcoliamo qui se necessario.
    # Se std non è passata, usiamo una stima locale o passala come argomento.
    # Qui uso sigma_clipped_stats per sicurezza se non è globale, altrimenti usa pure quella globale.
    from astropy.stats import sigma_clipped_stats
    _, _, std_dev = sigma_clipped_stats(data, sigma=3.0)

    soglia_minima = 5 * std_dev
    mask_significativi = data > soglia_minima

    y_coords, x_coords = np.where(mask_significativi)

    for y, x in zip(y_coords, x_coords):
        pixel_val = data[y, x]
        threshold_locale = threshold_ratio * pixel_val

        # Definisce i bordi della regione 3x3 (gestendo i bordi dell'immagine)
        y_min, y_max = max(0, y - 1), min(data.shape[0], y + 2)
        x_min, x_max = max(0, x - 1), min(data.shape[1], x + 2)

        region = data[y_min:y_max, x_min:x_max]

        # Crea maschera dei vicini forti
        mask_forti = region > threshold_locale

        # Trova la posizione relativa del pixel centrale nella regione ritagliata
        center_y = y - y_min
        center_x = x - x_min

        # Esclude il centro dal conteggio dei vicini forti
        mask_forti[center_y, center_x] = False

        vicini_forti = np.sum(mask_forti)

        # SE IL PIXEL È ISOLATO:
        if vicini_forti < min_neighbors:
            # Calcolo la media della regione escludendo il pixel centrale
            somma_totale = np.sum(region)
            numero_pixel = region.size

            # La media è (Somma regione - Valore pixel centrale) / (Numero pixel regione - 1)
            # (numero_pixel - 1) serve a non contare il pixel centrale nel denominatore
            valore_medio = (somma_totale - pixel_val) / (numero_pixel - 1)

            data_pulita[y, x] = valore_medio

    return data_pulita

data = image_data
data_pulita = rimuovi_pixel_isolati_media(data, min_neighbors=2, threshold_ratio=0.3)

# Modifica questa parte nel tuo blocco di visualizzazione

from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize

# 1. Calcoliamo i limiti ideali usando ZScale SOLO per trovare il massimo
interval = ZScaleInterval()
_, max_limit = interval.get_limits(data_pulita)

# 2. Creiamo la normalizzazione forzando vmin a 0
# Usiamo vmin=0 (bianco) e vmax=max_limit (nero)
norm = ImageNormalize(data_pulita,
                      vmin=0,              # <--- IL FONDO DEVE ESSERE 0
                      vmax=max_limit,      # <--- IL MASSIMO LO CALCOLA ZSCALE
                      stretch=AsinhStretch(a=0.1))

# 3. Visualizzazione
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(data_pulita, cmap='gray_r', origin='lower', norm=norm, interpolation='nearest')

plt.colorbar(im, label='Flux')
plt.title("Fondo Bianco (vmin=0 forzato)")
plt.show()