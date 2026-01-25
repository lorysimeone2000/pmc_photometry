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

from shapely.geometry import Point, Polygon
# warning
import warnings
from astropy.io.fits.verify import VerifyWarning
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

from pathlib import Path

def converti_valore(valore):
    valore = valore.strip()
    if not valore: return valore
    try: return int(valore)
    except ValueError: pass
    try: return float(valore)
    except ValueError: pass
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']: return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']: return False
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

def freedman_diaconis_bins(data):
    data_clean = data.compressed() if hasattr(data, 'mask') else data
    if len(data_clean) < 2: return 1
    iqr = np.percentile(data_clean, 75) - np.percentile(data_clean, 25)
    if iqr == 0: return 1
    bin_width = 2 * iqr / (len(data_clean) ** (1/3))
    data_range = np.max(data_clean) - np.min(data_clean)
    bins = int(np.ceil(data_range / bin_width))
    return max(bins, 1)

# --- INIZIO CODICE ---
run = 1
cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

cartella_csv_cat = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}"
file_csv_cat = sorted([f for f in os.listdir(cartella_csv_cat) if f.endswith('.csv')])
lista_percorsi_csv_cat = [os.path.join(cartella_csv_cat, file) for file in file_csv_cat]

n_immagine = 35

percorso_file_csv = lista_percorsi_csv[n_immagine]
dataframe = pd.read_csv(percorso_file_csv, skiprows=60)
tbl = Table.from_pandas(dataframe)

percorso_file_csv_cat = lista_percorsi_csv_cat[n_immagine]
dataframe_cat = pd.read_csv(percorso_file_csv_cat, skiprows=59)
tbl_cat = Table.from_pandas(dataframe_cat)

print("Tabella completa:\n", tbl)

mask_si = np.char.startswith(tbl['Corrispondenza'], 'SI')
tbl_catalogate_corr = tbl[mask_si]
magnitudini = tbl_catalogate_corr['Mag']
magnitudini_cat = tbl_cat['Mag']

# Gestione dei dati (conversione in array numpy e rimozione masked/NaN)
if hasattr(magnitudini, 'compressed'):
    mag_data = magnitudini.compressed()
else:
    mag_data = np.array(magnitudini)
    mag_data = mag_data[~np.isnan(mag_data)]

if hasattr(magnitudini_cat, 'compressed'):
    mag_cat_data = magnitudini_cat.compressed()
else:
    mag_cat_data = np.array(magnitudini_cat)
    mag_cat_data = mag_cat_data[~np.isnan(mag_cat_data)]


# Calcolo bin comuni
dati_totali = np.concatenate((mag_data, mag_cat_data))
n_bin = freedman_diaconis_bins(dati_totali)
hist_range = (np.min(dati_totali), np.max(dati_totali))
bins = np.histogram_bin_edges(dati_totali, bins=n_bin, range=hist_range)

# plt.figure(figsize=(20, 10))

# 1. Calcolo bin comuni (come nel tuo codice)
dati_totali = np.concatenate((mag_data, mag_cat_data))
n_bin = freedman_diaconis_bins(dati_totali)*2
hist_range = (np.min(dati_totali), np.max(dati_totali))
bins = np.histogram_bin_edges(dati_totali, bins=n_bin, range=hist_range)

# 2. Calcolo dei conteggi (altezze) e dei centri senza disegnare ancora nulla
# np.histogram restituisce (conteggi, bordi_bin)
counts_cat, bin_edges = np.histogram(mag_cat_data, bins=bins)
counts_corr, _ = np.histogram(mag_data, bins=bins)

# Calcolo i centri dei bin: (bordo_sinistro + bordo_destro) / 2
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 3. Creazione del grafico a linee
plt.figure(figsize=(14, 8)) # Dimensioni aumentate per leggibilità

# Plot delle Sorgenti Catalogate (Totali)
plt.plot(bin_centers, counts_cat,
         color='purple',
         linestyle='-',    # Linea continua
         linewidth=.5,
         label='Sorgenti Catalogate (Totali)')

# Plot delle Sorgenti Correlate (Filtrate)
plt.plot(bin_centers, counts_corr,
         color='red',
         linestyle='-',    # Linea continua
         linewidth=.5,
         label='Sorgenti Correlate (Filtrate)')

# Impostazioni Assi
plt.yscale('log')
plt.xlabel('Magnitudine (Centri dei Bin)')
plt.ylabel('Frequenza (Conteggi)')
plt.title(f'Distribuzione delle magnitudini: Confronto Catalogate vs Correlate\n({len(tbl_catalogate_corr)} oggetti correlati)')

# Invertiamo l'asse X (magnitudini astronomiche: valori più alti = oggetti più deboli)
plt.gca().invert_xaxis()

# Griglia per facilitare la lettura (utile con scala logaritmica)
plt.grid(True, which="both", linestyle='--', alpha=0.6)

# Legenda
plt.legend()

plt.tight_layout()
plt.show()