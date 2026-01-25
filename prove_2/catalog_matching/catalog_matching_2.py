import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
import os

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

# Set up astropy
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from astropy.visualization import SqrtStretch
from astropy.table import Table
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.table import MaskedColumn, QTable
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve

# Set up wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table, vstack
import warnings
from astropy.wcs import FITSFixedWarning

# Soppressione warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)

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

# run = int(input("Quale run vuoi elaborare: ")) # numero run: 1, 2 o 3
run = 1
immagine = 35
max_sep = 0.003349 * u.deg

# cartella contenente i file CSV delle stelle catalogate
# cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
file_csv =  f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_1/run_1_stelle_trovate_e_catalogate_immagine_{immagine:03d}.csv"
dataframe = pd.read_csv(file_csv, comment='#')
header = leggi_header_da_csv(file_csv)
image_file = header['PERCORSO_FILE']

tbl_trovate = Table.from_pandas(dataframe)

cartella_csv_catalogate = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}"
file_csv_catalogate = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}/run_1_stelle_catalogate_immagine_{immagine:03d}.csv"

dataframe2 = pd.read_csv(file_csv_catalogate, comment='#')
tbl_catalogate = Table.from_pandas(dataframe2)

hdu_list = fits.open(image_file)
w = WCS(hdu_list[0].header) # creo un oggetto WCS usando l'header del file FITS,

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'headerimport numpy as np

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median # tolgo il fondo
data = image_data

magnitudini = tbl_catalogate['Mag']

# Colori basati sulla magnitudine
colors = magnitudini

posizioni_vere_celesti = SkyCoord(ra=tbl_catalogate['RAJ2000'],
                                 dec=tbl_catalogate['DEJ2000'],
                                 unit = 'deg',
                                 frame= 'icrs')

# Matching con l'immagine della PMC

posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti) # converto da celesti a pixel
posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

# Crea una scala di colori
cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())

fig = plt.figure(figsize=(12, 8))
ax = plt.subplot()

# Disegno cerchi colorati delle stelle catalogate
for i, position in enumerate(zip(posizioni_vere_pixel)):
    color = cmap(norm(magnitudini[i]))

# Aggiungo la legenda - ORA specificando l'asse
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label='Magnitudine V')

# Rappresento il matching aggiungendoci l'image segmentation

ax.imshow(data, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')

tbl_trovate['xcentroid'].info.format = '.2f'  # optional format
tbl_trovate['ycentroid'].info.format = '.2f'
tbl_trovate['kron_flux'].info.format = '.2f'

positions = np.transpose((tbl_trovate['xcentroid'], tbl_trovate['ycentroid'])) # creo un array di posizioni

posizioni_celesti_segmentation = w.pixel_to_world(positions)
posizioni_celesti_segmentation_ra = np.array(posizioni_celesti_segmentation.ra)
posizioni_celesti_segmentation_dec = np.array(posizioni_celesti_segmentation.dec)
ra_segmentation_max = np.max(posizioni_celesti_segmentation_ra)

apertures = CircularAperture(positions, r=5.0) # creo le aperture per ogni posizione
# apertures.plot(color='red', lw=1.)

# C. Aperture (Cerchi gialli)
scales = proj_plane_pixel_scales(w)
pixel_scale_deg = np.mean(scales)
r_in_pixels = max_sep.to(u.deg).value / pixel_scale_deg
aperture = CircularAperture(positions, r=r_in_pixels)
aperture.plot(color='yellow', lw=1.5, alpha=0.8, label='Regione di correlazione')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Matching: {len(tbl_catalogate)} stelle del catalogo II/389/ps1_dr2 + Hipparco\n Magnitudine <{15}\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')

# legenda
legend_elements = [
    # Stelle catalogate (cerchi colorati)
    Circle((0.5, 0.5), 0.4,facecolor='blue', alpha=0.7, edgecolor='black', linewidth=1,
          label=f'Stelle catalogo ({len(tbl_catalogate)} oggetti)'),

    # Sorgenti rilevate (aperture gialle)
    Line2D([0], [0], marker='o', color='yellow', linestyle='None',
           markersize=8, markerfacecolor='none', markeredgewidth=1,
           label=f'Sorgenti rilevate ({len(tbl_trovate)} oggetti)')
]

# Aggiungi la legenda
ax.legend(handles=legend_elements, loc='upper right',
           framealpha=0.85, fancybox=True, shadow=True)

ax.scatter(posizioni_vere_pixel[:, 0], posizioni_vere_pixel[:, 1], c=colors, s = 36, alpha=0.7, cmap='viridis_r')

plt.show()