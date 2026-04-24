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

import pandas as pd
import matplotlib
import argparse
import json
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import concurrent.futures
from astropy.config import paths

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import time
import os
import sys
import gc
from scipy.optimize import curve_fit
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
import warnings
from astropy.wcs import FITSFixedWarning
from photutils.datasets import make_100gaussians_image
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.utils.data import download_file
from astropy.table import Table, vstack
from photutils.detection import find_peaks
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.coordinates import search_around_sky
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from shapely.geometry import Point, Polygon
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import label
import re
from pathlib import Path
from astropy.time import Time

# gestisco i warning ignorandoli per mantenere pulito il mio output
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='.*deblending mode.*')


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory del mio script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")
PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

# importo i moduli per il salvataggio in parquet e la relativa utilità
from funzioni.utilita_parquet import *
from funzioni.astrometria_parquet import *

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
n_immagine = 35
cartella_csv = cerca_cartella_nel_progetto(BASE_DIR,f"tabelle_alleggerite/tabelle_unite/tabelle_unite_run_{run}")
# filtro i miei file verificando che contengano la stringa fissa e che terminino con l'estensione corretta
file_csv = sorted([f for f in os.listdir(cartella_csv) if '_stelle_trovate_e_catalogate_immagine_' in f and f.endswith('.parquet')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

cartella_csv_cat = cerca_cartella_nel_progetto(BASE_DIR,f"tabelle_alleggerite/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}")
file_csv_cat = sorted([f for f in os.listdir(cartella_csv_cat) if f.endswith('.parquet')])
lista_percorsi_csv_cat = [os.path.join(cartella_csv_cat, file) for file in file_csv_cat]



percorso_file_csv = lista_percorsi_csv[n_immagine]
dataframe = pd.read_parquet(percorso_file_csv)
tbl = Table.from_pandas(dataframe)

percorso_file_csv_cat = lista_percorsi_csv_cat[n_immagine]
dataframe_cat = pd.read_parquet(percorso_file_csv_cat)
tbl_cat = Table.from_pandas(dataframe_cat)

print("Tabella completa:\n", tbl)

mask_si = tbl['Corrispondenza'] == True
tbl_catalogate_corr = tbl[mask_si]
magnitudini = tbl_catalogate_corr['Mag']
magnitudini_cat = tbl_cat['Mag']

# Gestisco i dati (converto in array numpy e rimuovo masked/NaN)
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


# Calcolo i bin comuni
dati_totali = np.concatenate((mag_data, mag_cat_data))
n_bin = freedman_diaconis_bins(dati_totali)
hist_range = (np.min(dati_totali), np.max(dati_totali))
bins = np.histogram_bin_edges(dati_totali, bins=n_bin, range=hist_range)

# plt.figure(figsize=(20, 10))

# 1. Calcolo i bin comuni
dati_totali = np.concatenate((mag_data, mag_cat_data))
n_bin = freedman_diaconis_bins(dati_totali)*2
hist_range = (np.min(dati_totali), np.max(dati_totali))
bins = np.histogram_bin_edges(dati_totali, bins=n_bin, range=hist_range)

# 2. Calcolo i conteggi (altezze) e i centri senza disegnare ancora nulla
# np.histogram mi restituisce (conteggi, bordi_bin)
counts_cat, bin_edges = np.histogram(mag_cat_data, bins=bins)
counts_corr, _ = np.histogram(mag_data, bins=bins)

# Calcolo i centri dei bin: (bordo_sinistro + bordo_destro) / 2
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 3. Creo il grafico a linee
# Riduco la dimensione della figura per far risaltare maggiormente i testi quando inserita in 0.65\textwidth
plt.figure(figsize=(7, 5))

# Traccio le Sorgenti Catalogate (Totali)
plt.plot(bin_centers, counts_cat,
         color='purple',
         linestyle='-',
         linewidth=1,
         label='Catalogued sources (Total)')

# Traccio le Sorgenti Correlate (Filtrate)
plt.plot(bin_centers, counts_corr,
         color='red',
         linestyle='-',
         linewidth=1,
         label='Correlated sources (Filtered)')

# Imposto gli assi
plt.yscale('log')
plt.xlabel('Magnitude (Bin centres)', fontsize=14)
plt.ylabel('Frequency (Counts)', fontsize=14)

# Inverto l'asse X (magnitudini astronomiche: valori più alti = oggetti più deboli)
plt.gca().invert_xaxis()

# Aumento la dimensione dei valori sugli assi
plt.tick_params(axis='both', which='major', labelsize=12)

# Inserisco la griglia per facilitare la lettura (utile con scala logaritmica)
plt.grid(True, which="both", linestyle='--', alpha=0.6)

# Aggiungo la legenda con dimensione adeguata al resto del testo
plt.legend(fontsize=12)

# plt.xlim(right=4)

# Ottimizzo i margini per evitare tagli delle etichette e salvo
plt.tight_layout()
plt.savefig('sensibilità_senza_rapporti_merge_senza_taglio.png', dpi=300, bbox_inches='tight')
# plt.show()