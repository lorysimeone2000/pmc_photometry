import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import warnings
from astropy.wcs import FITSFixedWarning
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

warnings.filterwarnings('ignore', category=FITSFixedWarning)
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from reproject import reproject_interp  # Mi assicuro di aver installato reproject: pip install reproject
import warnings
from astropy.wcs import FITSFixedWarning
from tqdm import tqdm
import os

warnings.filterwarnings('ignore', category=FITSFixedWarning)

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
from astropy.visualization.wcsaxes import SphericalCircle

# Gestisco i warning ignorandoli per mantenere pulito il mio output
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

# Importo i moduli per il salvataggio in parquet e la relativa utilità
from funzioni.utilita_parquet import *
from funzioni.astrometria_parquet import *

# --- SETUP ---
output_dir = cerca_cartella_nel_progetto(BASE_DIR, "grafici/stacking")

# Definisco direttamente la lista dei 3 file salvati dallo step precedente
file_list = [
    os.path.join(output_dir, 'run_1_stacked_sum.fits'),
    os.path.join(output_dir, 'run_2_stacked_sum.fits'),
    os.path.join(output_dir, 'run_3_stacked_sum.fits')
]

for filepath in file_list:
    if not os.path.exists(filepath):
        raise ValueError(f"File non trovato: {filepath}")

# --- PASSO 1: DEFINIRE IL SISTEMA DI RIFERIMENTO (CANVAS) ---
# Uso la seconda immagine della lista (run_1) come riferimento per il WCS e la dimensione finale.
immagine_di_riferimento = 1
print(f"Caricamento riferimento: {file_list[immagine_di_riferimento]}")
hdu_ref = fits.open(file_list[immagine_di_riferimento])[0]
target_header = hdu_ref.header.copy()
target_wcs = WCS(target_header)
target_shape = hdu_ref.data.shape

# Creo la matrice finale (accumulatore) piena di zeri
final_image_sum = np.zeros(target_shape)

# Matrice di copertura
coverage_map = np.zeros(target_shape)

print(f"Inizio stacking complessivo di {len(file_list)} immagini...")
print(f"Dimensioni target: {target_shape}")

# --- PASSO 2: LOOP E RIPROIEZIONE ---
for percorso_file_fits in tqdm(file_list, desc="Stacking Complessivo", unit="img"):
    try:
        with fits.open(percorso_file_fits) as hdu_list:
            # Caricamento dati
            data = hdu_list[0].data
            header = hdu_list[0].header
            wcs_input = WCS(header)

            data_sub = data

            # --- RIPROIEZIONE ---
            # Riproietto l'immagine corrente sul sistema di coordinate dell'immagine di riferimento
            array_reprojected, footprint = reproject_interp(
                (data_sub, wcs_input),
                target_wcs,
                shape_out=target_shape
            )

            # Converto i NaN in 0 per poter sommare.
            array_reprojected = np.nan_to_num(array_reprojected, nan=0.0)

            # Sommo alla matrice finale
            final_image_sum += array_reprojected

            # Aggiorno la mappa di copertura
            coverage_map += np.nan_to_num(footprint, nan=0.0)

    except Exception as e:
        tqdm.write(f"Errore nel file {percorso_file_fits}: {e}")

max_coverage = np.max(coverage_map)

# 2. Inizializzo l'immagine scalata
final_image_scaled = final_image_sum.copy()

# 3. Identifico i pixel da scalare (copertura < massima ma > 0)
# Calcolo l'array dei fattori di scala: max_coverage / coverage_map
scale_factor_map = np.zeros_like(coverage_map, dtype=float)

np.divide(max_coverage, coverage_map,
          out=scale_factor_map,
          where=coverage_map > 0)

# 4. Applico il fattore di scala all'immagine somma
final_image_sum = final_image_sum * scale_factor_map

# --- PASSO 3: SALVATAGGIO ---

# 1. Salvataggio Immagine Sommata Complessiva
output_filename = os.path.join(output_dir, 'master_stacked_sum.fits')
header_finale = target_header.copy()
header_finale['HISTORY'] = 'Immagine ottenuta dallo stacking complessivo delle 3 run.'

fits.writeto(output_filename, final_image_sum, header_finale, overwrite=True)
print(f"Fatto! Immagine complessiva salvata come: {output_filename}")

# 2. Salvataggio Coverage Map Complessiva
coverage_filename = os.path.join(output_dir, 'master_coverage_map.fits')
header_coverage = target_header.copy()
header_coverage['HISTORY'] = 'Mappa di copertura complessiva'

fits.writeto(coverage_filename, coverage_map, header_coverage, overwrite=True)
print(f"Fatto! Coverage map complessiva salvata come: {coverage_filename}")

# --- PASSO 4: VISUALIZZAZIONE VELOCE ---
norm = simple_norm(final_image_sum, 'sqrt')
plt.figure(figsize=(10, 10))
ax = plt.subplot(projection=target_wcs)
ax.imshow(final_image_sum, origin='lower', norm=norm, cmap='viridis')

# Ricavo le coordinate della Nebulosa del Granchio
crab_coord = SkyCoord.from_name("Crab Nebula")

# Creo un cerchio con raggio di 2.5 arcmin (per ottenere un'ampiezza totale di 5 arcmin)
cerchio = SphericalCircle((crab_coord.ra, crab_coord.dec), 2.5 * u.arcmin,
                          edgecolor='red', facecolor='none', transform=ax.get_transform('icrs'))

# Aggiungo il cerchio al grafico
ax.add_patch(cerchio)

plt.colorbar(label='Counts (Sum)')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('Stacking Complessivo Run 1, 2 e 3')
#plt.show()