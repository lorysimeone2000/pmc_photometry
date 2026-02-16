import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
import os

# Imposto matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # permetto di avere la scala logaritmica
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

# Imposto astropy
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

# Imposto wcs
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

# Importo Path per la gestione dinamica dei percorsi
from pathlib import Path

# Sopprimo i warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# FUNZIONI DI GESTIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # Cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # Cerco un file ricorsivamente
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # Cerco una cartella specifica ricorsivamente
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


# Trovo la cartella base del mio progetto
BASE_DIR = trova_cartella_base("pmc_photometry")

# Leggo i parametri
parametri = {}
file_parametri_path = cerca_file_nel_progetto(BASE_DIR, 'parametri_image_segmentation.txt')

if file_parametri_path is not None:
    with open(file_parametri_path, 'r') as file:
        next(file)  # Salto l'intestazione
        for riga in file:
            riga = riga.strip()
            if riga and not riga.startswith('#'):
                # Divido la riga usando solo il primo spazio come separatore
                parti = riga.split(maxsplit=1)
                if len(parti) == 2:
                    parametro = parti[0]
                    valore = parti[1]
                    try:
                        parametri[parametro] = float(valore) if '.' in valore else int(valore)
                    except ValueError:
                        # Se non riesco a convertire in numero puro, lo lascio come stringa
                        parametri[parametro] = valore
else:
    print("ERRORE: File dei parametri non trovato.")
    exit()

fwhm = parametri['fwhm']
size = parametri['size']
t = parametri['threshold_sigma']
# threshold = t * std # per adesso lascio stare questo metodo
threshold = parametri['threshold_assoluta']
n = parametri['pixel']


def converti_valore(valore):
    """
    Converto una stringa nel tipo di dato appropriato.
    Provo in ordine: int, float, mantengo stringa se non è convertibile.
    """
    valore = valore.strip()

    # Se è vuoto, restituisco stringa vuota
    if not valore:
        return valore

    # Provo a convertire in int
    try:
        return int(valore)
    except ValueError:
        pass

    # Provo a convertire in float
    try:
        return float(valore)
    except ValueError:
        pass

    # Provo a riconoscere booleani FITS
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False

    # Altrimenti restituisco la stringa originale
    return valore


def leggi_header_da_csv(filename):
    """Leggo l'header FITS dal file CSV"""
    header_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                # Rimuovo il '#' e divido chiave-valore
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

# Cerco la cartella in cui trovo i file CSV delle tabelle unite
nome_cartella_csv = f"tabelle_unite_run_{run}"
cartella_csv_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_csv)

if cartella_csv_path is None:
    print(f"ERRORE: Cartella '{nome_cartella_csv}' non trovata.")
    exit()

cartella_csv = str(cartella_csv_path)

# Cerco il file CSV specifico delle tabelle unite
nome_file_csv = f"run_{run}_stelle_trovate_e_catalogate_immagine_{immagine:03d}.csv"
file_csv_path = cerca_file_nel_progetto(BASE_DIR, nome_file_csv)

if file_csv_path is None:
    print(f"ERRORE: File '{nome_file_csv}' non trovato in pmc_photometry.")
    exit()

file_csv = str(file_csv_path)

dataframe = pd.read_csv(file_csv, comment='#')
header = leggi_header_da_csv(file_csv)

# Ricavo il nome del file FITS dall'header e cerco il suo percorso completo
nome_file_fits = header.get('PERCORSO_FILE_FITS', header.get('NOME_FILE_FITS', header.get('PERCORSO_FILE')))
nome_solo_fits = os.path.basename(str(nome_file_fits).strip())
file_trovato = cerca_file_nel_progetto(BASE_DIR, nome_solo_fits)

if file_trovato is None:
    print(f"ERRORE: File '{nome_solo_fits}' non trovato all'interno di {BASE_DIR}.")
    exit()

image_file = str(file_trovato)

tbl_trovate = Table.from_pandas(dataframe)

# Cerco la cartella in cui trovo i file CSV delle stelle catalogate
nome_cartella_catalogate = f"sorgenti_catalogate_run_{run}"
cartella_catalogate_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_catalogate)

if cartella_catalogate_path is None:
    print(f"ERRORE: Cartella '{nome_cartella_catalogate}' non trovata.")
    exit()

cartella_csv_catalogate = str(cartella_catalogate_path)

# Cerco il file CSV specifico delle stelle catalogate
nome_file_catalogate = f"run_{run}_stelle_catalogate_immagine_{immagine:03d}.csv"
file_catalogate_path = cerca_file_nel_progetto(BASE_DIR, nome_file_catalogate)

if file_catalogate_path is None:
    print(f"ERRORE: File '{nome_file_catalogate}' non trovato in pmc_photometry.")
    exit()

file_csv_catalogate = str(file_catalogate_path)

dataframe2 = pd.read_csv(file_csv_catalogate, comment='#')
tbl_catalogate = Table.from_pandas(dataframe2)

hdu_list = fits.open(image_file)
w = WCS(hdu_list[0].header)  # creo un oggetto WCS usando l'header del file FITS,

image_data = hdu_list[0].data  # creo la matrice dei valori dei pixel
# image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglio un'area tot x tot pixel
# print(hdu_list[0].header) # metto tutti i dati dell'header

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median  # tolgo il fondo
data = image_data

magnitudini = tbl_catalogate['Mag']

# Baso i colori sulla magnitudine
colors = magnitudini

posizioni_vere_celesti = SkyCoord(ra=tbl_catalogate['RAJ2000'],
                                  dec=tbl_catalogate['DEJ2000'],
                                  unit='deg',
                                  frame='icrs')

# Faccio il matching con l'immagine della PMC

posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti)  # converto da celesti a pixel
posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

# Creo una scala di colori
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

positions = np.transpose((tbl_trovate['xcentroid'], tbl_trovate['ycentroid']))  # creo un array di posizioni

posizioni_celesti_segmentation = w.pixel_to_world(positions)
posizioni_celesti_segmentation_ra = np.array(posizioni_celesti_segmentation.ra)
posizioni_celesti_segmentation_dec = np.array(posizioni_celesti_segmentation.dec)
ra_segmentation_max = np.max(posizioni_celesti_segmentation_ra)

apertures = CircularAperture(positions, r=5.0)  # creo le aperture per ogni posizione
# apertures.plot(color='red', lw=1.)

# C. Aperture (Cerchi gialli)
scales = proj_plane_pixel_scales(w)
pixel_scale_deg = np.mean(scales)
r_in_pixels = max_sep.to(u.deg).value / pixel_scale_deg
aperture = CircularAperture(positions, r=r_in_pixels)
aperture.plot(color='yellow', lw=1.5, alpha=0.8, label='Regione di correlazione')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(
    f'Matching: {len(tbl_catalogate)} stelle del catalogo II/389/ps1_dr2 + Hipparco\n Magnitudine <{15}\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')

# legenda
legend_elements = [
    # Stelle catalogate (cerchi colorati)
    Circle((0.5, 0.5), 0.4, facecolor='blue', alpha=0.7, edgecolor='black', linewidth=1,
           label=f'Stelle catalogo ({len(tbl_catalogate)} oggetti)'),

    # Sorgenti rilevate (aperture gialle)
    Line2D([0], [0], marker='o', color='yellow', linestyle='None',
           markersize=8, markerfacecolor='none', markeredgewidth=1,
           label=f'Sorgenti rilevate ({len(tbl_trovate)} oggetti)')
]

# Aggiungo la legenda
ax.legend(handles=legend_elements, loc='upper right',
          framealpha=0.85, fancybox=True, shadow=True)

ax.scatter(posizioni_vere_pixel[:, 0], posizioni_vere_pixel[:, 1], c=colors, s=36, alpha=0.7, cmap='viridis_r')

plt.show()