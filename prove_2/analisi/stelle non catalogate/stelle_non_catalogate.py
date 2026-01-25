import pandas as pd
#pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
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

tbl_corr = tbl[tbl['Corrispondenza'] == 'SI']
tbl_non_corr = tbl[tbl['Corrispondenza'] == 'NO']

posizioni = np.transpose((tbl_non_corr['xcentroid'], tbl_non_corr['ycentroid'], tbl_non_corr['kron_flux'])) # creo un array di posizioni
np.set_printoptions(suppress=True) # tolgo la dicitura e+01,02, ecc
print("Posizioni: \n" , posizioni)

i = 0
for indice in range(len(tbl_non_corr)):
    i = i + 1
    print(indice)