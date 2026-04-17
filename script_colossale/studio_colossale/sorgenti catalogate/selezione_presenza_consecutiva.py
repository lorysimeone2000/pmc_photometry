import pandas as pd
import matplotlib
import argparse
import json
import pyarrow.parquet as pq
import shutil
from astropy.config import paths
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
# 1. LETTURA, ORDINAMENTO CRONOLOGICO E UNIONE DEI FILE PARQUET
# =============================================================================

# definisco il percorso esatto in cui cercare i miei file parquet
cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito"

# cerco tutti i file che terminano con il suffisso specifico sostituendo i numeri con dei jolly
file_parquet = list(cartella_tabelle.rglob("*run_*_run_*_immagine_*.parquet"))

if not file_parquet:
    print("Nessun file trovato con il pattern specificato.")
    sys.exit()

lista_dati_file = []

# leggo i metadati di ciascun file usando la mia funzione per estrarre TSTART e preparare l'ordinamento
for file_p in tqdm(file_parquet, desc="Estrazione TSTART dai file"):
    header_dict = leggi_header_da_parquet(file_p)
    tstart = header_dict.get('TSTART')

    # se non riesco a trovare TSTART o non è un numero valido, assegno infinito per mandarlo in fondo alla lista
    if tstart is None:
        tstart = float('inf')
    else:
        try:
            tstart = float(tstart)
        except ValueError:
            tstart = float('inf')

    # salvo esclusivamente il percorso del file per elaborare i dati in seguito
    lista_dati_file.append({
        'tstart': tstart,
        'file_path': file_p
    })

# ordino la mia lista di dizionari in modo strettamente cronologico usando TSTART
lista_dati_file.sort(key=lambda x: x['tstart'])

lista_df_ordinati = []

print("Unione di tutti i dati in un singolo dataframe in corso...")

# assegno un indice cronologico sequenziale (0, 1, 2...) ed estraggo in maniera ottimizzata i miei dati
for indice_cronologico, dato in enumerate(tqdm(lista_dati_file, desc="Lettura e filtraggio Parquet")):
    # sfrutto il predicate pushdown e la column projection per caricare unicamente le colonne necessarie
    # per le righe dove 'Corrispondenza' è True
    tabella_filtrata = pq.read_table(
        dato['file_path'],
        columns=['label', 'media_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE'],
        filters=[('Corrispondenza', '=', True)]
    )

    # converto in pandas esclusivamente i dati che ho prefiltrato
    df_corrente = tabella_filtrata.to_pandas()
    df_corrente['indice_cronologico'] = indice_cronologico
    # mantengo l'informazione del TSTART del file
    df_corrente['TSTART_file'] = dato['tstart']

    lista_df_ordinati.append(df_corrente)

# genero il mio dataframe complessivo che adesso contiene già solo gli oggetti non catalogati
df_falsi = pd.concat(lista_df_ordinati, ignore_index=True)

# =============================================================================
# 2. ISOLAMENTO OGGETTI NON CATALOGATI E ANALISI DELLA CONSECUTIVITÀ
# =============================================================================

print("Analisi consecutività per gli oggetti non catalogati in corso...")

# raggruppo per label ed estraggo il conteggio delle immagini uniche in cui compare,
# assieme al primo e all'ultimo indice cronologico in cui è stato visto,
# e la media del flusso
df_non_cat = df_falsi.groupby('label').agg(
    occorrenze=('indice_cronologico', 'nunique'),
    min_indice=('indice_cronologico', 'min'),
    max_indice=('indice_cronologico', 'max'),
    media_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE=('media_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE', 'mean')
).reset_index()

# ordino i miei dati per etichetta e indice cronologico per poter analizzare le sequenze
df_ordinato = df_falsi.sort_values(['label', 'indice_cronologico'])

# rimuovo eventuali duplicati dello stesso oggetto nella stessa immagine per non sfalsare il conteggio
df_ordinato = df_ordinato.drop_duplicates(subset=['label', 'indice_cronologico'])

# calcolo i blocchi consecutivi: sottraendo una sequenza incrementale (cumcount)
# al mio indice cronologico, ottengo un identificativo univoco per ogni blocco temporale ininterrotto
df_ordinato['id_blocco'] = df_ordinato['indice_cronologico'] - df_ordinato.groupby('label').cumcount()

# conto la lunghezza di ciascun blocco consecutivo che ho isolato
lunghezza_blocchi = df_ordinato.groupby(['label', 'id_blocco']).size().reset_index(name='n_consecutivi')

# estraggo unicamente le label che possiedono almeno un blocco con 10 o più occorrenze consecutive
label_valide = lunghezza_blocchi[lunghezza_blocchi['n_consecutivi'] >= 10]['label'].unique()

# filtro il mio dataframe riassuntivo mantenendo solo gli oggetti che soddisfano la mia nuova condizione
df_non_cat_aggiornato = df_non_cat[df_non_cat['label'].isin(label_valide)].copy()

# rimuovo le colonne di calcolo temporanee per ripulire il mio dataframe in output
df_non_cat_aggiornato = df_non_cat_aggiornato.drop(columns=['min_indice', 'max_indice'])

percorso_salvataggio = cerca_cartella_nel_progetto(BASE_DIR, "sorgenti catalogate") / "oggetti_CATALOGATI_con_presenza_consecutiva.csv"
# salvo il mio dataframe aggiornato
df_non_cat_aggiornato.to_csv(percorso_salvataggio, index=False)

print(f"Salvataggio completato! Trovati {len(df_non_cat_aggiornato)} oggetti. File salvato in: {percorso_salvataggio}")
print("\nColonne presenti nel file salvato:")
print(df_non_cat_aggiornato.columns.tolist())