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

# =============================================================================
# 1. CARICAMENTO DEL FILE CSV
# =============================================================================

# cerco il file csv all'interno della mia cartella base
percorso_csv = None
for file_trovato in BASE_DIR.rglob("oggetti_con_presenza_consecutiva.csv"):
    percorso_csv = file_trovato
    break

if percorso_csv is None:
    print("Errore: Non ho trovato il file 'oggetti_con_presenza_consecutiva.csv'.")
    sys.exit()

# carico i dati nel mio dataframe
df = pd.read_csv(percorso_csv)
print(f"Ho caricato {len(df)} oggetti dal file CSV.")

# definisco le mie due variabili principali per comodità
col_occorrenze = 'occorrenze'
col_flusso = 'media_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE'

if col_occorrenze not in df.columns or col_flusso not in df.columns:
    print("Errore: Le colonne richieste non sono presenti nel file CSV.")
    sys.exit()

# =============================================================================
# 2. CALCOLO DEL NUMERO DI RUN
# =============================================================================

# definisco la cartella dove cercare i file dei non catalogati
cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito"
file_parquet_non_cat = list(cartella_tabelle.rglob("*oggetti_non_catalogati.parquet"))

if not file_parquet_non_cat:
    print("Errore: Nessun file 'oggetti_non_catalogati.parquet' trovato.")
    sys.exit()

# inizializzo il mio dizionario per tracciare le run uniche per ogni label
run_per_label = {lbl: set() for lbl in df['label']}

print("Calcolo il numero di run per ogni oggetto...")
# esploro i file parquet dei non catalogati
for file_p in tqdm(file_parquet_non_cat, desc="Scansione file non catalogati"):
    try:
        # leggo solo la colonna label
        tabella_p = pq.read_table(file_p, columns=['label'])
        labels_nel_file = set(tabella_p.column('label').to_pylist())

        # cerco le intersezioni con i miei oggetti bersaglio
        trovati = set(run_per_label.keys()).intersection(labels_nel_file)

        if trovati:
            header = leggi_header_da_parquet(file_p)
            run_id = header.get('RUN_ID')
            if run_id:
                for l in trovati:
                    run_per_label[l].add(str(run_id))
    except Exception:
        continue

# aggiungo la colonna calcolata al mio dataframe
df['numero_di_run'] = df['label'].apply(lambda x: len(run_per_label.get(x, set())))

print("\nFrequenza dei valori nella colonna numero_di_run:")
print(df['numero_di_run'].value_counts())

# =============================================================================
# 3. STUDIO STATISTICO E CORRELAZIONE
# =============================================================================

# calcolo la correlazione di spearman (robusta agli outlier) e di pearson (lineare) per le occorrenze
corr_spearman_occ = df[col_occorrenze].corr(df[col_flusso], method='spearman')
corr_pearson_occ = df[col_occorrenze].corr(df[col_flusso], method='pearson')

# calcolo la correlazione per il numero di run
corr_spearman_run = df['numero_di_run'].corr(df[col_flusso], method='spearman')
corr_pearson_run = df['numero_di_run'].corr(df[col_flusso], method='pearson')

print(f"\n--- Statistiche e Correlazione (Occorrenze) ---")
print(f"Correlazione di Spearman (non lineare): {corr_spearman_occ:.3f}")
print(f"Correlazione di Pearson (lineare): {corr_pearson_occ:.3f}")

print(f"\n--- Statistiche e Correlazione (Numero di Run) ---")
print(f"Correlazione di Spearman (non lineare): {corr_spearman_run:.3f}")
print(f"Correlazione di Pearson (lineare): {corr_pearson_run:.3f}")

# =============================================================================
# 4. CREAZIONE GRAFICI E SALVATAGGIO
# =============================================================================

# individuo o creo la mia cartella di output
cartella_output = BASE_DIR / "studio_colossale"
cartella_output.mkdir(parents=True, exist_ok=True)

# --- GRAFICO 1: Scatter Plot (Occorrenze vs Flusso) ---
plt.figure(figsize=(10, 6))

# genero il mio scatter plot con punti molto piccoli (s=3) e di colore nero
plt.scatter(df[col_occorrenze], df[col_flusso], s=3, color='black', alpha=0.6)

# applico la scala logaritmica all'asse Y per distribuire correttamente gli outlier del flusso
plt.yscale('log')

plt.title('Relazione tra Durata (Occorrenze) e Flusso Medio', fontsize=14)
plt.xlabel('Occorrenze (Numero di Frame)', fontsize=12)
plt.ylabel('Flusso Medio Decorrelato (Scala Log)', fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.4)

# aggiungo un box testuale con i miei valori di correlazione
testo_stats_occ = f"Spearman: {corr_spearman_occ:.2f}\nPearson: {corr_pearson_occ:.2f}"
plt.text(0.05, 0.95, testo_stats_occ, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

percorso_plot1 = cartella_output / "scatter_occorrenze_vs_flusso.png"
plt.savefig(percorso_plot1, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nHo salvato lo scatter plot delle occorrenze in: {percorso_plot1}")

# --- GRAFICO 2: Boxplot per Quartili ---
plt.figure(figsize=(10, 6))

# divido le mie occorrenze in 4 gruppi (quartili) per studiare l'andamento delle distribuzioni
df['gruppo_occorrenze'] = pd.qcut(df[col_occorrenze], q=4, duplicates='drop')

# genero il boxplot raggruppato
df.boxplot(column=col_flusso, by='gruppo_occorrenze', grid=False, figsize=(10, 6),
           boxprops=dict(color='navy', linewidth=1.5),
           medianprops=dict(color='crimson', linewidth=2),
           whiskerprops=dict(linewidth=1.5),
           capprops=dict(linewidth=1.5))

plt.yscale('log')
plt.title('Distribuzione del Flusso per Fasce di Durata (Quartili)', fontsize=14)
plt.suptitle('')
plt.xlabel('Quartili di Occorrenze', fontsize=12)
plt.ylabel('Flusso Medio Decorrelato (Scala Log)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

percorso_plot2 = cartella_output / "boxplot_flusso_per_occorrenze.png"
plt.savefig(percorso_plot2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Ho salvato il boxplot in: {percorso_plot2}")

# --- GRAFICO 3: Scatter Plot (Numero di Run vs Flusso) ---
plt.figure(figsize=(10, 6))

# genero il mio scatter plot per le run
plt.scatter(df['numero_di_run'], df[col_flusso], s=3, color='black', alpha=0.6)

plt.yscale('log')

plt.title('Relazione tra Numero di Run e Flusso Medio', fontsize=14)
plt.xlabel('Numero di Run', fontsize=12)
plt.ylabel('Flusso Medio Decorrelato (Scala Log)', fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.4)

# aggiungo un box testuale con i miei valori di correlazione per le run
testo_stats_run = f"Spearman: {corr_spearman_run:.2f}\nPearson: {corr_pearson_run:.2f}"
plt.text(0.05, 0.95, testo_stats_run, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

percorso_plot3 = cartella_output / "scatter_numero_run_vs_flusso.png"
plt.savefig(percorso_plot3, dpi=300, bbox_inches='tight')
plt.close()
print(f"Ho salvato lo scatter plot del numero di run in: {percorso_plot3}")