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
dataframe = pd.read_csv(percorso_file_csv, comment="#")
tbl = Table.from_pandas(dataframe)

percorso_file_csv_cat = lista_percorsi_csv_cat[n_immagine]
dataframe_cat = pd.read_csv(percorso_file_csv_cat, comment="#")
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

# Istogrammi
counts_cat, bins_used_cat, patches_cat = plt.hist(
    mag_cat_data,
    bins=bins,
    alpha=1.0,
    color='purple',
    edgecolor='black',
    label='Sorgenti Catalogate (Totali)'
)

counts_corr, bins_used_corr, patches_corr = plt.hist(
    mag_data,
    bins=bins,
    alpha=0.4,
    color='red',
    edgecolor='black',
    label='Sorgenti Correlate (Filtrate)'
)

plt.yscale('log')
plt.xlabel('Magnitudine')
plt.ylabel('Frequenza')
plt.title(f'Distribuzione delle magnitudini delle stelle catalogate correlate ({len(tbl_catalogate_corr)} oggetti)')
plt.gca().invert_xaxis()
plt.legend()


# --- SISTEMAZIONE ASSE X (MODIFICATA) ---
# Calcola tutti i centri dei bin
bin_centers = (bins[:-1] + bins[1:]) / 2

# Crea tutte le etichette formattate
tick_labels = [f'{c:.2f}' for c in bin_centers]

# Seleziona solo 1 tick ogni 4 (start:stop:step)
step = 4
subset_ticks = bin_centers[::step]
subset_labels = tick_labels[::step]

# Applica i tick filtrati
plt.gca().set_xticks(subset_ticks)
plt.gca().set_xticklabels(subset_labels, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# --- CODICE AGGIUNTIVO PER VISUALIZZARE LE SORGENTI MANCANTI ---

# plt.figure(figsize=(20, 10))

# 1. Ricalcoliamo i conteggi negli stessi bin usati prima
counts_cat, _ = np.histogram(mag_cat_data, bins=bins)
counts_corr, _ = np.histogram(mag_data, bins=bins)

# 2. Calcoliamo la differenza: (Stelle nel Catalogo) - (Stelle Trovate)
# Queste sono le stelle perse (False Negative)
counts_missing = counts_cat - counts_corr

# Evitiamo valori negativi (non dovrebbero esserci se i dati sono coerenti, ma per sicurezza)
counts_missing = np.maximum(counts_missing, 0)

# Calcolo dei centri dei bin per il plot a barre
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

# 3. Plot delle stelle MANCANTI
plt.bar(bin_centers, counts_missing, width=bin_width,
        color='tab:blue', edgecolor='black', alpha=0.7, label='Sorgenti NON Rilevate (Missing)')

# 4. Evidenziamo l'area critica intorno a 8.80
plt.axvspan(8.5, 10.01, color='orange', alpha=0.05, label='Zona Anomalie')

plt.yscale('log')
plt.xlabel('Magnitudine')
plt.ylabel('Numero di Stelle Mancanti')
plt.title('Distribuzione delle stelle presenti nel catalogo ma NON rilevate dal software')
plt.gca().invert_xaxis() # Magnitudini luminose a destra, deboli a sinistra (o viceversa in base al tuo standard)

# Applico la stessa formattazione dell'asse X del tuo grafico originale
plt.gca().set_xticks(subset_ticks)
plt.gca().set_xticklabels(subset_labels, rotation=45, ha='right')

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# --- STAMPA DI DEBUG PER L'INTERVALLO 8.80 ---
# Identifichiamo i bin vicini a 8.80 per vedere quante ne mancano esattamente
idx_interest = np.where((bin_centers >= 8.5) & (bin_centers <= 10.01))[0]
print("\n--- Analisi Stelle Mancanti intorno a Mag 8.80 ---")
for i in idx_interest:
    print(f"Mag {bin_centers[i]:.2f}: Perse {int(counts_missing[i])} su {int(counts_cat[i])} catalogate "
          f"({(counts_missing[i]/counts_cat[i])*100:.1f}% perse)")