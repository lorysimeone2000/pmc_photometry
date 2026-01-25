import pandas as pd
#pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
from astropy.coordinates import match_coordinates_sky
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

from shapely.geometry import Point, Polygon
# warning
import warnings
from astropy.io.fits.verify import VerifyWarning
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning) # Sopprime il warning FITSFixedWarning

from pathlib import Path

def converti_valore(valore):
    """
    Converte una stringa nel tipo di dato appropriato.
    Prova in ordine: int, float, mantiene stringa se non è convertibile.
    """
    valore = valore.strip()

    # Se è vuoto, restituisci stringa vuota
    if not valore:
        return valore

    # Prova a convertire in int
    try:
        return int(valore)
    except ValueError:
        pass

    # Prova a convertire in float
    try:
        return float(valore)
    except ValueError:
        pass

    # Prova a riconoscere booleani FITS
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False

    # Altrimenti restituisci la stringa originale
    return valore

def leggi_header_da_csv(filename):
    """Legge l'header FITS dal file CSV"""
    header_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                # Rimuovi il '#' e dividi chiave-valore
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':  # Fine dell'header
                break

    return header_dict

# Lettura parametri
parametri = {}
with open('/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt', 'r') as file:
    next(file)  # Salta intestazione
    for riga in file:
        riga = riga.strip()
        if riga and not riga.startswith('#'):
            parametro, valore = riga.split()
            parametri[parametro] = float(valore) if '.' in valore else int(valore)


fwhm = parametri['fwhm']
size = parametri['size']
t = parametri['threshold_sigma']
# threshold = t * std # per adesso lascio stare questo metodo
threshold = parametri['threshold_assoluta']
n = parametri['pixel']

#run = int(input("Quale run vuoi elaborare: ")) # numero run: 1, 2 o 3
run = 1

# cartella contenente i file CSV delle stelle catalogate
# cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')]) # creo lista nomi
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv] # creo lista di percorsi

# n_immagine = int(input(f"Qual immagine vuoi elaborare (da 1 a {len(lista_percorsi_csv)}): "))

n_immagine = 35

percorso_file_csv = lista_percorsi_csv[n_immagine]

dataframe = pd.read_csv(percorso_file_csv, skiprows=60)
tbl = Table.from_pandas(dataframe)

print("Tabella completa:\n", tbl.colnames)

# quit()

tbl_correlate = tbl[tbl['Corrispondenza'] == 'SI']
tbl_non_corr = tbl[tbl['Corrispondenza'] == 'NO']

colonne_da_selezionare = ['xcentroid', 'ycentroid', 'kron_flux', 'RA_centroid', 'DEC_centroid']
tbl_non_corr = tbl_non_corr[colonne_da_selezionare]
tbl_correlate = tbl_correlate[colonne_da_selezionare]

df_non_corr_ = tbl_non_corr.to_pandas()
print("Tabella stelle non correlate: ")
print(df_non_corr_)

header = leggi_header_da_csv(percorso_file_csv)
image_file = header['PERCORSO_FILE']

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data


'''ax1.imshow(data, cmap="grey_r", norm=LogNorm()) #genero l'immagine con scala di colori bianco e nero
ax1.colorbar()
ax1.gca().invert_yaxis()

positions = np.transpose((tbl_non_corr['xcentroid'], tbl_non_corr['ycentroid']))  # creo un array di posizioni
apertures = CircularAperture(positions, r=5.0)  # creo le aperture per ogni posizione
apertures.plot(color='red', lw=1.)

ax1.show()'''

# quit()

# inserisco le coordinate invertite rispetto a quelle che vedo nell'immagine
#y_centro_non_corr = int(input("x centro: "))
#x_centro_non_corr = int(input("y centro: "))

n = 3

x_centro_non_corr = round(tbl_non_corr[n]['ycentroid'])
y_centro_non_corr = round(tbl_non_corr[n]['xcentroid'])
kron = tbl_non_corr[n]['kron_flux']


raggio = int(input("raggio: "))

print(data[x_centro_non_corr,y_centro_non_corr]) # valore centrale

#ax1.imshow(data, cmap="grey_r", norm=LogNorm()) #genero l'immagine con scala di colori bianco e nero
#ax1.gca().invert_yaxis()
#ax1.colorbar()
#ax1.show()



profilo_non_corr = hdu_list[0].data[x_centro_non_corr, y_centro_non_corr-raggio:y_centro_non_corr+raggio]

print(profilo_non_corr)
porzione_stella_non_corr = hdu_list[0].data[x_centro_non_corr-raggio:x_centro_non_corr+raggio , y_centro_non_corr-raggio:y_centro_non_corr+raggio]

print(len(profilo_non_corr))

'''# istogramma semplice

ax1.bar(range(len(profilo_non_corr)), profilo_non_corr, color='skyblue', edgecolor='navy', alpha=0.7, width=0.8) # istogramma stella

ax1.set_xlabel('profilo_non_corr')
ax1.set_ylabel('valori pixel')
ax1.grid(axis='y', alpha=0.3)

ax1.show()'''


# CALCOLO FWHM

def calculate_fwhm(profile):
    # Trova il valore massimo e minimo
    max_val = np.max(profile)
    min_val = np.min(profile)

    # Calcola metà altezza
    half_max = min_val + (max_val - min_val) / 2

    # Trovo dove il profilo_non_corr attraversa la metà altezza

    # creo un array booleano che indica per ogni posizione se il valore del profilo_non_corr è sopra (True) o sotto (False) la metà altezza
    above_half_max = profile > half_max

    # crovo gli indici dove il profilo_non_corr supera la metà altezza
    indices = np.where(above_half_max)[0] # prendo il primo indice True, ovvero sopra la metà altezza

    # Controllo se non ci sono punti sopra la metà altezza. Se vero, restituisce valori di default per evitare errori
    if len(indices) == 0:
        return 0, half_max, max_val, min_val

    # Primo e ultimo indice sopra la metà altezza
    left_index = indices[0] # primo pixel sopra la metà altezza
    right_index = indices[-1] # ultimo pixel sotto la metà altezza

    # Interpolazione per maggiore precisione
    x = np.arange(len(profile))

    # Interpolazione a sinistra
    if left_index > 0:
        left_x = [x[left_index - 1], x[left_index]]
        left_y = [profile[left_index - 1], profile[left_index]]
        f_left = interp1d(left_y, left_x, kind='linear')
        try:
            left_interp = float(f_left(half_max))
        except:
            left_interp = left_index
    else:
        left_interp = left_index

    # Interpolazione a destra
    if right_index < len(profile) - 1:
        right_x = [x[right_index], x[right_index + 1]]
        right_y = [profile[right_index], profile[right_index + 1]]
        f_right = interp1d(right_y, right_x, kind='linear')
        try:
            right_interp = float(f_right(half_max))
        except:
            right_interp = right_index
    else:
        right_interp = right_index

    fwhm = right_interp - left_interp

    return fwhm, half_max, max_val, min_val, left_interp, right_interp


# Calcola FWHM
# Chiama la funzione con il profilo_non_corr della stella e salva tutti i valori restituiti in variabili separate

fwhm, half_max, max_val, min_val, left_edge, right_edge = calculate_fwhm(profilo_non_corr)

# Plot del profilo_non_corr con FWHM
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# profilo_non_corr
ax1.bar(range(len(profilo_non_corr)), profilo_non_corr, color='skyblue', edgecolor='navy', alpha=0.7, width=0.8, label='profilo_non_corr stellare')

# Linee per FWHM
ax1.axhline(y=half_max, color='red', linestyle='--', linewidth=2, label=f'Metà altezza: {half_max:.2f}')
ax1.axhline(y=max_val, color='green', linestyle=':', linewidth=1, label=f'Massimo: {max_val:.2f}')
#ax1.axhline(y=min_val, color='orange', linestyle=':', linewidth=1, label=f'Minimo: {min_val:.2f}')

# Segna i bordi del FWHM
ax1.axvline(x=left_edge, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.axvline(x=right_edge, color='red', linestyle='--', linewidth=1, alpha=0.7)

ax1.set_xlabel('Posizione (pixel)')
ax1.set_ylabel('Intensità pixel')
ax1.set_title(f'profilo_non_corr stellare - FWHM = {fwhm:.2f} pixel')
ax1.grid(axis='y', alpha=0.3)
ax1.legend()

# Aggiungi testo con i valori
ax1.text(0.02, 0.98, f'FWHM: {fwhm:.2f} pixel\nMassimo: {max_val:.2f}\nMetà altezza: {half_max:.2f}',
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ax1.imshow()

print(f"\n=== RISULTATI FWHM ===")
print(f"Valore massimo: {max_val:.2f}")
print(f"Valore minimo: {min_val:.2f}")
print(f"Metà altezza: {half_max:.2f}")
print(f"Bordo sinistro: {left_edge:.2f} pixel")
print(f"Bordo destro: {right_edge:.2f} pixel")
print(f"FWHM: {fwhm:.2f} pixel")


target_flux = kron
tolleranza = 5  # Puoi modificare questo valore a seconda di quanto "vicino" vuoi

# Trova l'oggetto con kron_flux più vicino a 55
differenze = np.abs(tbl_correlate['kron_flux'] - target_flux)
indice_piu_vicino = np.argmin(differenze)
oggetto_correlato = tbl_correlate[indice_piu_vicino]

x_centro_corr = round(oggetto_correlato['ycentroid'])
y_centro_corr = round(oggetto_correlato['xcentroid'])

profilo_corr = hdu_list[0].data[x_centro_corr, y_centro_corr-raggio:y_centro_corr+raggio]

print(profilo_corr)

porzione_stella_corr = hdu_list[0].data[x_centro_corr-raggio:x_centro_corr+raggio , y_centro_corr-raggio:y_centro_corr+raggio]

fwhm, half_max, max_val, min_val, left_edge, right_edge = calculate_fwhm(profilo_corr)

# profilo_corr
ax2.bar(range(len(profilo_corr)), profilo_corr, color='skyblue', edgecolor='navy', alpha=0.7, width=0.8, label='profilo_corr stellare')

# Linee per FWHM
ax2.axhline(y=half_max, color='red', linestyle='--', linewidth=2, label=f'Metà altezza: {half_max:.2f}')
ax2.axhline(y=max_val, color='green', linestyle=':', linewidth=1, label=f'Massimo: {max_val:.2f}')
#ax2.axhline(y=min_val, color='orange', linestyle=':', linewidth=1, label=f'Minimo: {min_val:.2f}')

# Segna i bordi del FWHM
ax2.axvline(x=left_edge, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax2.axvline(x=right_edge, color='red', linestyle='--', linewidth=1, alpha=0.7)

ax2.set_xlabel('Posizione (pixel)')
ax2.set_ylabel('Intensità pixel')
ax2.set_title(f'profilo_corr stellare di kron simile - FWHM = {fwhm:.2f} pixel')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

# Aggiungi testo con i valori
ax2.text(0.02, 0.98, f'FWHM: {fwhm:.2f} pixel\nMassimo: {max_val:.2f}\nMetà altezza: {half_max:.2f}',
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

print(f"\n=== RISULTATI FWHM ===")
print(f"Valore massimo: {max_val:.2f}")
print(f"Valore minimo: {min_val:.2f}")
print(f"Metà altezza: {half_max:.2f}")
print(f"Bordo sinistro: {left_edge:.2f} pixel")
print(f"Bordo destro: {right_edge:.2f} pixel")
print(f"FWHM: {fwhm:.2f} pixel")

fig.suptitle(f'Kron {kron}')

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

im1 = ax1.imshow(porzione_stella_non_corr, cmap="grey_r", norm=LogNorm()) #genero porzione immagine con scala di colori bianco e nero
ax1.invert_yaxis() # inverto asse y
ax1.set_title(f'Stella non correlata')

im2 = ax2.imshow(porzione_stella_corr, cmap="grey_r", norm=LogNorm()) #genero porzione immagine con scala di colori bianco e nero
ax2.invert_yaxis() # inverto asse y
ax2.set_title(f'Stella correlata di kron simile')

fig.suptitle(f'Kron {kron}')

plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)

plt.show()