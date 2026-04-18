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
# Eseguo il ciclo per le run 1, 2 e 3
for run in [1, 2, 3]:
    # Percorso del file lista
    #path_lista = f'/home/lorysimeone/tesi_magistrale/prove_2/liste_percorsi_run/lista_immagini_run_{run}.txt'

    print(f"\n==================== ELABORAZIONE RUN {run} ====================")
    nome_cartella_run = f"20250120_run{run}"
    found_folders = list(BASE_DIR.rglob(nome_cartella_run))
    if not found_folders:
        continue
    run_folder = found_folders[0]

    estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
    path_lista = []
    for ext in estensioni_valide:
        path_lista.extend(run_folder.glob(ext))

    # Assegno direttamente a file_list la lista dei percorsi trovata
    file_list = sorted([str(f) for f in path_lista])

    if not file_list:
        raise ValueError(f"La lista dei file per la run {run} è vuota!")

    # --- PASSO 1: DEFINIRE IL SISTEMA DI RIFERIMENTO (CANVAS) ---
    # Uso la prima immagine come riferimento per il WCS e la dimensione finale.
    immagine_di_riferimento = 12
    print(f"Caricamento riferimento: {file_list[immagine_di_riferimento]}") # Prendo come riferimento un'immagine più stabile
    hdu_ref = fits.open(file_list[immagine_di_riferimento])[0]
    target_header = hdu_ref.header.copy()
    target_wcs = WCS(target_header)
    target_shape = hdu_ref.data.shape

    # Creo la matrice finale (accumulatore) piena di zeri
    final_image_sum = np.zeros(target_shape)

    # (Opzionale) Matrice di copertura se volessi fare la media invece della somma
    coverage_map = np.zeros(target_shape)

    print(f"Inizio stacking di {len(file_list)} immagini...")
    print(f"Dimensioni target: {target_shape}")

    i = 0

    # --- PASSO 2: LOOP E RIPROIEZIONE ---
    for percorso_file_fits in tqdm(file_list, desc="Stacking", unit="img"):
        i = i + 1
        if i == 1 or i == len(file_list) - 2 or i == len(file_list) - 1: continue
        try:
            with fits.open(percorso_file_fits) as hdu_list:
                # Caricamento dati
                data = hdu_list[0].data
                header = hdu_list[0].header
                wcs_input = WCS(header)

                # Non sottraggo il fondo
                data_sub = data

                # --- RIPROIEZIONE ---
                # Riproietto l'immagine corrente (data_sub) sul sistema di coordinate dell'immagine di riferimento (target_wcs)
                array_reprojected, footprint = reproject_interp(
                    (data_sub, wcs_input),
                    target_wcs,
                    shape_out=target_shape
                )

                # reproject mette NaN dove l'immagine non si sovrappone.
                # Converto i NaN in 0 per poter sommare.
                array_reprojected = np.nan_to_num(array_reprojected, nan=0.0)

                # Sommo alla matrice finale
                final_image_sum += array_reprojected

                # (Opzionale) Aggiorno la mappa di copertura
                coverage_map += np.nan_to_num(footprint, nan=0.0)

        except Exception as e:
            tqdm.write(f"Errore nel file {percorso_file_fits}: {e}")

    max_coverage = np.max(coverage_map)

    # 2. Inizializzo l'immagine scalata
    final_image_scaled = final_image_sum.copy()

    # 3. Identifico i pixel da scalare (copertura < massima ma > 0)
    # Copertura Massima: coverage_map == max_coverage (il fattore sarà 1, quindi invariati)
    # Copertura Parziale: 0 < coverage_map < max_coverage
    # Copertura Zero: coverage_map == 0 (non li tocco)

    # Calcolo l'array dei fattori di scala: max_coverage / coverage_map
    # Uso np.divide con 'where' per evitare la divisione per zero e i calcoli non necessari
    scale_factor_map = np.zeros_like(coverage_map, dtype=float)

    np.divide(max_coverage, coverage_map,
              out=scale_factor_map,
              where=coverage_map > 0)

    # 4. Applico il fattore di scala all'immagine somma
    # Moltiplico l'immagine somma per la mappa dei fattori di scala.
    # Dove coverage_map == max_coverage, il fattore è 1, quindi final_image_scaled rimane invariato.
    # Dove coverage_map < max_coverage (ma > 0), il fattore è > 1 e l'intensità viene aumentata.
    # Dove coverage_map == 0, il fattore è 0, quindi il valore rimane 0 (come nell'originale).
    final_image_sum = final_image_sum * scale_factor_map

    # --- PASSO 3: SALVATAGGIO ---

    output_dir = f'/home/lorysimeone/tesi_magistrale/prove_2/stacking/'

    # 1. Salvataggio Immagine Sommata
    output_filename = os.path.join(output_dir, f'run_{run}_stacked_sum.fits')
    header_finale = target_header.copy() # Copio l'header per non modificare l'originale
    header_finale['HISTORY'] = f'Immagine ottenuta sommando N esposizioni riproiettate con reproject usando come riferimento l\' immagine {immagine_di_riferimento}'

    fits.writeto(output_filename, final_image_sum, header_finale, overwrite=True)
    print(f"Fatto! Immagine salvata come: {output_filename}")

    # 2. Salvataggio Coverage Map
    coverage_filename = os.path.join(output_dir, f'run_{run}_coverage_map.fits')
    header_coverage = target_header.copy()
    header_coverage['HISTORY'] = 'Mappa di copertura (numero di immagini per pixel)'

    # La coverage map usa lo stesso WCS dell'immagine, così posso sovrapporle in DS9
    fits.writeto(coverage_filename, coverage_map, header_coverage, overwrite=True)
    print(f"Fatto! Coverage map salvata come: {coverage_filename}")

    # --- PASSO 4: VISUALIZZAZIONE VELOCE ---
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    norm = simple_norm(final_image_sum, 'sqrt')
    plt.figure(figsize=(10, 10))
    plt.subplot(projection=target_wcs)
    plt.imshow(final_image_sum, origin='lower', norm=norm, cmap='viridis')
    plt.colorbar(label='Counts (Sum)')
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title(f'Stacking Run {run}')
    #plt.show()