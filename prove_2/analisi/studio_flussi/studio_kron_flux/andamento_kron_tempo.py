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

# run = int(input("Quale run vuoi elaborare: ")) # numero run: 1, 2 o 3
run = 1

# cartella contenente i file CSV delle stelle catalogate
# cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')]) # creo lista nomi
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv] # creo lista di percorsi

id_stella_target = 133680839188884406
kron = 1000
kron_flux_array = []
file_trovati = []
t = []
coordinate2 = []

i=0
j=0
trovatoj4 = False
for n in range(len(lista_percorsi_csv)):

    percorso_csv_stelle_trovate = lista_percorsi_csv[n]
    dataframe = pd.read_csv(percorso_csv_stelle_trovate, skiprows=60)
    header_dal_csv = leggi_header_da_csv(percorso_csv_stelle_trovate)
    percorso_file_fits = header_dal_csv['PERCORSO_FILE']
    i = i + 1

    tbl_trovate = Table.from_pandas(dataframe)
    # if i==1: print(f"tbl_trovate: \n {tbl_trovate}")

    mask_si = np.char.startswith(tbl_trovate['Corrispondenza'], 'SI')
    tbl_catalogate = tbl_trovate[mask_si]

    '''if i==35:

        print("Percorso file: " , percorso_file_fits)
        # Calcola la differenza assoluta tra kron_flux e 80
        differenze = np.abs(tbl_catalogate['kron_flux'] - kron)

        # Trova l'indice della stella con la differenza minima
        idx_min_diff = np.argmin(differenze)

        # Seleziona la stella con kron_flux più vicino a 80
        stella = tbl_catalogate[idx_min_diff]
        id_stella = stella['ID']
        print(f"ID stella con kron intorno a {kron}: ",id_stella)
        print(f"Area: {stella['area']}")
        print(f"Kron: {stella['kron_flux']}")
        print("Coordinate: \n", stella['xcentroid'], stella['ycentroid'])
        quit()'''


    # Cerca la stella con l'ID specificato
    stella_target = tbl_trovate[tbl_trovate['ID'] == id_stella_target]
    id_stella = stella_target['ID']
    print(f"ID stella con kron intorno a {kron}: ", id_stella)
    print(f"Area: {stella_target['area']}")
    print(f"Kron: {stella_target['kron_flux']}")
    print("Coordinate: \n", stella_target['xcentroid'], stella_target['ycentroid'])

    # Se la stella è stata trovata e corrisponde al catalogo
    if len(stella_target) > 0 and stella_target['Corrispondenza'][0].startswith('SI'):
        kron_flux = stella_target['kron_flux'][0]  # Usa [0] invece di .iloc[0] per Astropy Table
        kron_flux_array.append(kron_flux)
        file_trovati.append(percorso_csv_stelle_trovate)
        coordinate = np.array([stella_target['xcentroid'], stella_target['ycentroid']])

        if i == 2:
            print(f"Stella trovata nel file {percorso_csv_stelle_trovate}:")
            print(f"kron_flux: {kron_flux}")
            print("Coordinate: ", coordinate)

    else:
        kron_flux_array.append(0)
        j = j + 1

    if j==2 and not trovatoj4:
        print(f"Percorso file senza stella: {percorso_csv_stelle_trovate}")
        #print(f"Coordinate: \n {coordinate}")
        trovatoj4 = True

    # Stampa di controllo ogni 10 file
    if i % 50 == 0:
        print(f"Elaborati {i} file, trovati {len(kron_flux_array)} valori kron_flux")

    if i == 1:
        t.append(0)
        t0 = header_dal_csv['TSTART']
        print("Tempo iniziale:",t0, " secondi")
    else: t.append((header_dal_csv['TSTART']-t0)/np.float64(1e3))

# Converti l'array in numpy array per facilità di utilizzo
t = np.array(t)
kron_flux_array = np.array(kron_flux_array)

# Stampa i risultati finali
print(f"\n=== RISULTATI FINALI ===")
print(f"ID stella cercata: {id_stella_target}")
media = np.mean(kron_flux_array[kron_flux_array != 0])
std = np.std(kron_flux_array[kron_flux_array != 0])
print(f"kron_flux medio: {media}")
print(f"Numero totale di file elaborati: {i}")
print(f"Numero di valori kron_flux trovati: {len(kron_flux_array[kron_flux_array != 0])}")
# print(f"Valori kron_flux \n: {kron_flux_array}")


plt.plot(t , kron_flux_array, marker='o', linestyle='-', linewidth=2, markersize=6)
plt.title(f'Andamento kron_flux di media {media} e std {std}')
plt.xlabel('Secondi')
plt.ylabel('Kron_flux')
plt.ylim(0, None)

plt.show()

