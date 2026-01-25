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

n_immagine = 36

percorso_file_csv = lista_percorsi_csv[n_immagine]
dataframe = pd.read_csv(percorso_file_csv, comment='#')
tbl = Table.from_pandas(dataframe)

percorso_file_csv_cat = lista_percorsi_csv_cat[n_immagine]
dataframe_cat = pd.read_csv(percorso_file_csv_cat, comment='#')
tbl_cat = Table.from_pandas(dataframe_cat)

print("Tabella completa:\n", tbl)

header_dal_csv = leggi_header_da_csv(percorso_file_csv)

parametri = {
    'fwhm': header_dal_csv.get('seg_fwhm', header_dal_csv.get('SEG_FWHM')),
    'size': header_dal_csv.get('seg_size', header_dal_csv.get('SEG_SIZE')),
    'threshold_sigma': header_dal_csv.get('seg_threshold_sigma', header_dal_csv.get('SEG_THRESHOLD_SIGMA')),
    'threshold_assoluta': header_dal_csv.get('seg_threshold_assoluta', header_dal_csv.get('SEG_THRESHOLD_ASSOLUTA')),
    'pixel': header_dal_csv.get('seg_pixel', header_dal_csv.get('SEG_PIXEL')),
    'soglia_filtro_ass': header_dal_csv.get('seg_soglia_filtro_ass', header_dal_csv.get('SEG_SOGLIA_FILTRO_ASS')),
    'soglia_filtro_rel': header_dal_csv.get('seg_soglia_filtro_rel', header_dal_csv.get('SEG_SOGLIA_FILTRO_REL')),
}

fwhm = parametri['fwhm']
size = parametri['size']

# Preparazione dati (come prima)
mask_si = np.char.startswith(tbl['Corrispondenza'].astype(str), 'SI')
ids_trovati_e_correlati = set(tbl[mask_si]['ID'])

# 2. Prendiamo dal catalogo completo tutte le stelle brillanti (Mag < 10)
mask_bright_cat = tbl_cat['Mag'] < 10
stelle_catalogo_brillanti = tbl_cat[mask_bright_cat]

# 3. Contiamo quante di queste NON sono nella lista delle trovate
num_perse_brillanti = 0
for star_id in stelle_catalogo_brillanti['ID']:
    if star_id not in ids_trovati_e_correlati:
        num_perse_brillanti += 1

print(f"Stelle di catalogo (Mag < 10) NON correlate/trovate: {num_perse_brillanti}")

tbl_catalogate_corr = tbl[mask_si]
magnitudini = tbl_catalogate_corr['Mag']
magnitudini_cat = tbl_cat['Mag']

# Conversione in array puliti
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

# 1. Calcolo bin comuni
dati_totali = np.concatenate((mag_data, mag_cat_data))
n_bin = freedman_diaconis_bins(dati_totali)
hist_range = (np.min(dati_totali), np.max(dati_totali))
bins = np.histogram_bin_edges(dati_totali, bins=n_bin, range=hist_range)

# 2. Calcolo dei conteggi (altezze)
counts_cat, bin_edges = np.histogram(mag_cat_data, bins=bins)
counts_corr, _ = np.histogram(mag_data, bins=bins)

# 3. CALCOLO DELLA PERCENTUALE (COMPLETEZZA)
# Gestiamo la divisione per zero (se un bin del catalogo è vuoto)
with np.errstate(divide='ignore', invalid='ignore'):
    percentuale_completezza = (counts_corr / counts_cat) * 100

# Sostituiamo i NaN (0/0) e gli Inf con 0
percentuale_completezza = np.nan_to_num(percentuale_completezza)

# Calcolo centri dei bin
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 4. Creazione del grafico
plt.figure(figsize=(14, 8))

plt.plot(bin_centers, percentuale_completezza,
         color='green',
         marker='o',
         markersize=4,
         linestyle='-',
         linewidth=2,
         label='Percentuale di rilevamento (Completezza)')

# Linea di riferimento al 100% e al 50%
plt.axhline(100, color='gray', linestyle='--', alpha=0.5, label='100% Completezza')
plt.axhline(50, color='red', linestyle=':', alpha=0.5, label='50% Completezza')

# Impostazioni Assi
plt.xlabel('Magnitudine (Centri dei Bin)')
plt.ylabel('Percentuale stelle correlate / totali (%)')
plt.title(f'Funzione di Completezza: Efficienza di rilevamento per magnitudine (FWHM = {fwhm}, size = {size})')

# Invertiamo l'asse X (magnitudini astronomiche)
plt.gca().invert_xaxis()

# Impostiamo il limite Y da 0 a poco più di 100 per chiarezza
plt.ylim(0, 110)

# Griglia
plt.grid(True, which="both", linestyle='--', alpha=0.6)

# Formattazione Tick Asse X (come richiesto prima)
tick_labels = [f'{c:.2f}' for c in bin_centers]
step = 4
subset_ticks = bin_centers[::step]
subset_labels = tick_labels[::step]
plt.gca().set_xticks(subset_ticks)
plt.gca().set_xticklabels(subset_labels, rotation=45, ha='right')

plt.legend()
plt.tight_layout()
plt.show()