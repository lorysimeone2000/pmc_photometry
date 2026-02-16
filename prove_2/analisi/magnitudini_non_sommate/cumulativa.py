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

def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # Cerco una cartella specifica ricorsivamente
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]

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
# Imposto la cartella base in modo dinamico
BASE_DIR = trova_cartella_base("pmc_photometry")

run = 1

# Cerco dinamicamente la cartella tabelle_unite_run_X
nome_cartella_csv = f"tabelle_unite_run_{run}"
cartella_csv_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_csv)
if cartella_csv_path is None:
    print(f"ERRORE CRITICO: Cartella '{nome_cartella_csv}' non trovata.")
    exit()
lista_percorsi_csv = sorted([str(f) for f in cartella_csv_path.glob('*.csv')])

# Cerco dinamicamente la cartella sorgenti_catalogate_run_X
nome_cartella_csv_cat = f"sorgenti_catalogate_run_{run}"
cartella_csv_cat_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_csv_cat)
if cartella_csv_cat_path is None:
    print(f"ERRORE CRITICO: Cartella '{nome_cartella_csv_cat}' non trovata.")
    exit()
lista_percorsi_csv_cat = sorted([str(f) for f in cartella_csv_cat_path.glob('*.csv')])

n_immagine = 35

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

# Preparazione dati
mask_si = np.char.startswith(tbl['Corrispondenza'].astype(str), 'SI')
ids_trovati_e_correlati = set(tbl[mask_si]['ID'])

# 2. Conto quante di queste NON sono nella lista delle trovate
num_perse = 0
for star_id in tbl_cat['ID']:
    if star_id not in ids_trovati_e_correlati:
        num_perse += 1

print(f"Stelle di catalogo NON correlate/trovate: {num_perse}")

tbl_catalogate_corr = tbl[mask_si]
magnitudini = tbl_catalogate_corr['Mag']
magnitudini_cat = tbl_cat['Mag']

# Converto in array puliti
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
# dati_totali = np.concatenate((mag_data, mag_cat_data))

dati_totali = mag_cat_data
n_bin = freedman_diaconis_bins(dati_totali)
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

# Disegno le Sorgenti Catalogate (Totali)
plt.plot(bin_centers, counts_cat,
         color='purple',
         linestyle='-',    # Linea continua
         linewidth=.5,
         label='Sorgenti Catalogate (Totali)')

# Disegno le Sorgenti Correlate (Filtrate)
plt.plot(bin_centers, counts_corr,
         color='red',
         linestyle='-',    # Linea continua
         linewidth=.5,
         label='Sorgenti Correlate (Filtrate)')

# Impostazioni Assi
plt.yscale('log')
plt.xlabel('Magnitudine (Centri dei Bin)')
plt.ylabel('Frequenza (Conteggi)')
plt.title(f'Distribuzione delle magnitudini: Confronto Catalogate vs Correlate\n({len(tbl_catalogate_corr)} match su {len(tbl_cat)} catalogate totali)')

# Invertiamo l'asse X (magnitudini astronomiche)
plt.gca().invert_xaxis()

# Griglia e Legenda
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend()

# Formattazione Tick Asse X: mostro un'etichetta ogni 4 bin per non sovrapporle
tick_labels = [f'{c:.2f}' for c in bin_centers]
step = 4
subset_ticks = bin_centers[::step]
subset_labels = tick_labels[::step]
plt.gca().set_xticks(subset_ticks)
plt.gca().set_xticklabels(subset_labels, rotation=45, ha='right')

plt.tight_layout()
plt.show()