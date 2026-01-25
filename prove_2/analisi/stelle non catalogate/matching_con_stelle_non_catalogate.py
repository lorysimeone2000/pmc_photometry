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

n_immagine_1 = 35
n_immagine_2 = 95

percorso_file_csv_1 = lista_percorsi_csv[n_immagine_1]
percorso_file_csv_2 = lista_percorsi_csv[n_immagine_2]

dataframe = pd.read_csv(percorso_file_csv_1, skiprows=60)
tbl_1 = Table.from_pandas(dataframe)
dataframe = pd.read_csv(percorso_file_csv_2, skiprows=60)
tbl_2 = Table.from_pandas(dataframe)

print("Tabella completa:\n", tbl_1.colnames)
# quit()

tbl_correlate_1 = tbl_1[tbl_1['Corrispondenza'] == 'SI']
tbl_non_corr_1 = tbl_1[tbl_1['Corrispondenza'] == 'NO']
tbl_correlate = tbl_correlate_1
tbl_non_corr = tbl_non_corr_1

tbl_correlate_2 = tbl_2[tbl_2['Corrispondenza'] == 'SI']
tbl_non_corr_2 = tbl_2[tbl_2['Corrispondenza'] == 'NO']

'''print(f"Tabella stelle trovate:\n")
print(tbl)
print(f"Tabella stelle correlate:")
print(tbl_correlate)'''

# creo sottocataloghi giusto per analisi
tbl_vizier_correlate_1 = tbl_1[(tbl_1['Corrispondenza'] == 'SI') & (tbl_1['Catalogo'] == 'II/389/ps1_dr2')]
tbl_hipparco_correlate_1 = tbl_1[
    (tbl_1['Corrispondenza'] == 'SI') & (tbl_1['Catalogo'] == 'I/239/hip_main')]

num_correlate_1 = len(tbl_correlate_1)
num_non_correlate_1 = len(tbl_1) - len(tbl_correlate_1)
num_vizier_correlate_1 = len(tbl_vizier_correlate_1)
num_hipparco_correlate_1 = len(tbl_hipparco_correlate_1)

# creo sottocataloghi giusto per analisi
tbl_vizier_correlate_2 = tbl_2[(tbl_2['Corrispondenza'] == 'SI') & (tbl_2['Catalogo'] == 'II/389/ps1_dr2')]
tbl_hipparco_correlate_2 = tbl_2[
    (tbl_2['Corrispondenza'] == 'SI') & (tbl_2['Catalogo'] == 'I/239/hip_main')]

num_correlate_2 = len(tbl_correlate_2)
num_non_correlate_2 = len(tbl_1) - len(tbl_correlate_2)
num_vizier_correlate_2 = len(tbl_vizier_correlate_2)
num_hipparco_correlate_2 = len(tbl_hipparco_correlate_2)


colonne_da_selezionare = ['xcentroid', 'ycentroid', 'kron_flux', 'RA_centroid', 'DEC_centroid']
tbl_non_corr_pixel_1 = tbl_non_corr_1[colonne_da_selezionare]

df_non_corr_1 = tbl_non_corr_pixel_1.to_pandas()
print(f"Tabella stelle non correlate run {n_immagine_1}: \n {df_non_corr_1}")

tbl_non_corr_pixel_2 = tbl_non_corr_2[colonne_da_selezionare]

df_non_corr_2 = tbl_non_corr_pixel_2.to_pandas()
print(f"Tabella stelle non correlate run {n_immagine_2}: \n {df_non_corr_2}")

posizioni_1 = SkyCoord(tbl_non_corr_pixel_1['RA_centroid'], tbl_non_corr_pixel_1['DEC_centroid'], unit=(u.deg, u.deg))
posizioni_2 = SkyCoord(tbl_non_corr_pixel_2['RA_centroid'], tbl_non_corr_pixel_2['DEC_centroid'], unit=(u.deg, u.deg))

print(posizioni_1,"\n",posizioni_2)

idx, distanze, _ = match_coordinates_sky(posizioni_1, posizioni_2)

print(f"Distanze: \n{distanze}")

quit()


# Matching con l'immagine della PMC

header_dal_csv = leggi_header_da_csv(percorso_file_csv)
percorso_file_fits = header_dal_csv['PERCORSO_FILE']
hdu_list = fits.open(percorso_file_fits)
header = hdu_list[0].header
w = WCS(header)

posizioni_vere_celesti = SkyCoord(ra=tbl_correlate['RAJ2000'],
                                  dec=tbl_correlate['DEJ2000'],
                                  unit='deg',
                                  frame='icrs')

posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti)  # converto da celesti a pixel
posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

magnitudini = tbl_correlate['Mag']

# Parametri per i raggi
raggio_min = 4.0
raggio_max = 20.0
raggi = raggio_max - (magnitudini - magnitudini.min()) * (raggio_max - raggio_min) / (
        magnitudini.max() - magnitudini.min())

# Crea una scala di colori
cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())

fig = plt.figure(figsize=(12, 8))
ax = plt.subplot()

# Disegno cerchi colorati delle stelle catalogate
for i, (position, radius) in enumerate(zip(posizioni_vere_pixel, raggi)):
    color = cmap(norm(magnitudini[i]))
    aperture = CircularAperture(position, r=radius)
    aperture.plot(color=color, lw=1.0, alpha=0.6, fill=True)

# Aggiungo la legenda - ORA specificando l'asse
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label='Magnitudine V')

# Rappresento il matching aggiungendoci l'image segmentation

image_data = hdu_list[0].data
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
data_pmc = image_data - median  # tolgo il fondo

ax.imshow(data_pmc, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')

tbl['xcentroid'].info.format = '.2f'  # optional format
tbl['ycentroid'].info.format = '.2f'
tbl['kron_flux'].info.format = '.2f'
# print(tbl)

positions = np.transpose((tbl_non_corr['xcentroid'], tbl_non_corr['ycentroid']))  # creo un array di posizioni
# positions_sky = SkyCoord(positions, unit=u.deg, frame='icrs')
posizioni_celesti_segmentation = w.pixel_to_world(positions)
posizioni_celesti_segmentation_ra = np.array(posizioni_celesti_segmentation.ra)
posizioni_celesti_segmentation_dec = np.array(posizioni_celesti_segmentation.dec)
ra_segmentation_max = np.max(posizioni_celesti_segmentation_ra)
'''
print("RA max segmentazione: ", ra_segmentation_max)
print("RA max catalogo: ", ra_max)'''

apertures = CircularAperture(positions, r=5.0)  # creo le aperture per ogni posizione
apertures.plot(color='red', lw=1.)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(
f'Matching: {len(tbl)} stelle del catalogo II/389/ps1_dr2 + Hipparco (se magnitudine < 7)\n Cerchi dimensionati per magnitudine (<{15})\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')


'''plt.title(
    f'Matching: {len(tbl)} stelle del catalogo II/389/ps1_dr2 \n Cerchi dimensionati per magnitudine (<{mag_max})\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')'''


# legenda
legend_elements = [

    # Sorgenti rilevate (aperture rosse)
    Line2D([0], [0], marker='o', color='red', linestyle='None',
           markersize=8, markerfacecolor='none', markeredgewidth=1,
           label=f'Sorgenti rilevate senza corrispondenza: {len(tbl)} oggetti'),

    # Stelle catalogate (cerchi colorati)
    Circle((0.5, 0.5), 0.4, facecolor='blue', alpha=0.7, edgecolor='black', linewidth=1,
           label=f'Sorgenti rilevate con corrispondenza: {len(tbl)}'),

    Line2D([0], [0], marker='', color='green', linestyle='None',
           markersize=8, markerfacecolor='green', markeredgewidth=1,
           label=f'\n----------------------------------------------\nCorrispondenze trovate: {num_correlate}'
                 f' di cui\n- {num_vizier_correlate} di II/389/ps1_dr2\n- {num_hipparco_correlate} di Hipparco\n'),

    Line2D([0], [0], marker='', color='orange', linestyle='None',
           markersize=8, markerfacecolor='orange', markeredgewidth=1,
           label=f'Corrispondenze non trovate: {num_non_correlate}')
]

# Aggiungi la legenda
ax.legend(handles=legend_elements, loc='upper right',
          framealpha=0.85, fancybox=True, shadow=True)

hdu_list.close()

plt.show()

