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


# ridefinisco internamente la funzione per leggere l'header (come richiesto)
def converti_valore(valore):
    valore = str(valore).strip()
    if not valore: return valore
    try:
        return int(valore)
    except ValueError:
        pass
    try:
        return float(valore)
    except ValueError:
        pass
    if valore.upper() in ['T', 'TRUE']: return True
    if valore.upper() in ['F', 'FALSE']: return False
    return valore


def leggi_header_da_parquet(filename):
    # inizializzo il mio dizionario vuoto per ospitare i metadati estratti
    header_dict = {}

    try:
        # leggo esclusivamente lo schema del file Parquet per accedere ai metadati
        schema = pq.read_schema(filename)
        metadati = schema.metadata

        # verifico se i metadati esistono e se contengono la mia chiave personalizzata
        if metadati and b'metadati_intestazione_csv' in metadati:
            # decodifico la stringa da byte a testo normale usando la codifica utf-8
            metadati_testo = metadati[b'metadati_intestazione_csv'].decode('utf-8')

            # analizzo ogni riga della mia stringa di testo
            for riga in metadati_testo.split('\n'):
                if riga.startswith('#'):
                    # pulisco la riga rimuovendo il cancelletto e gli spazi
                    riga_pulita = riga.strip()[1:].strip()

                    # verifico che la riga contenga un separatore valido
                    if riga_pulita and ': ' in riga_pulita:
                        # separo la mia chiave dal valore
                        chiave, valore = riga_pulita.split(': ', 1)
                        # aggiungo l'elemento al mio dizionario usando la funzione di conversione
                        header_dict[chiave] = converti_valore(valore)

    except Exception:
        pass

    return header_dict


# =============================================================================
# 1. IMPOSTAZIONE BERSAGLIO E RICERCA DATI
# =============================================================================

# INSERISCI QUI IL LABEL DELL'OGGETTO CHE VUOI STUDIARE
LABEL_BERSAGLIO = "INSERISCI_QUI_IL_LABEL"

if LABEL_BERSAGLIO == "INSERISCI_QUI_IL_LABEL":
    print("ERRORE: Devi inserire un label valido alla riga 126 del codice (es. 'RA_159.85DEC48.81').")
    sys.exit()

COLONNA_FLUSSO = 'flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE'

cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito"
file_parquet_immagini = list(cartella_tabelle.rglob("*run_*_run_*_immagine_*.parquet"))

if not file_parquet_immagini:
    print("ERRORE: Nessun file immagine parquet trovato in 'tabelle_COLOSSALE_alleggerito'.")
    sys.exit()

# Liste per accumulare i dati temporali e fotometrici
date_osservazione = []
flussi_estratti = []

print(f"Avvio la ricerca della curva di luce per l'oggetto: {LABEL_BERSAGLIO}")

for file_p in tqdm(file_parquet_immagini, desc="Estrazione Flusso e Tempo"):
    try:
        # 1. Controllo veloce se il label esiste in questo file (leggo solo la colonna label)
        tabella_label = pq.read_table(file_p, columns=['label'])
        labels_nel_file = set(tabella_label.column('label').to_pylist())

        if LABEL_BERSAGLIO in labels_nel_file:
            # 2. Se l'oggetto esiste, ricarico la tabella filtrata per prendere solo la riga giusta e il flusso
            tabella_dati = pq.read_table(file_p, columns=['label', COLONNA_FLUSSO],
                                         filters=[('label', '=', LABEL_BERSAGLIO)])

            if tabella_dati.num_rows > 0:
                # 3. Estraggo il valore numerico del flusso
                valore_flusso = tabella_dati.column(COLONNA_FLUSSO)[0].as_py()

                # 4. Leggo l'header per estrarre la data
                header = leggi_header_da_parquet(file_p)
                data_obs_str = header.get('DATE-OBS')

                if data_obs_str is not None and valore_flusso is not None and not np.isnan(valore_flusso):
                    date_osservazione.append(data_obs_str)
                    flussi_estratti.append(valore_flusso)

    except Exception as e:
        # ignoro silenziosamente i file corrotti o mancanti di colonne
        continue

if not date_osservazione:
    print(f"\nNessun dato valido trovato per il label {LABEL_BERSAGLIO}.")
    sys.exit()

# =============================================================================
# 2. ORDINAMENTO TEMPORALE E CREAZIONE CURVA DI LUCE
# =============================================================================

print(f"\nTrovati {len(date_osservazione)} punti fotometrici. Generazione della Curva di Luce...")

# Converto le stringhe temporali in un oggetto Time di Astropy
tempi_astropy = Time(date_osservazione, format='isot', scale='utc')

# Trasformo in un formato continuo per graficare facilmente (Modified Julian Date)
tempi_mjd = tempi_astropy.mjd

# Creo un dataframe temporaneo per ordinare cronologicamente i punti
df_curva = pd.DataFrame({
    'Tempo_MJD': tempi_mjd,
    'Tempo_Stringa': date_osservazione,
    'Flusso': flussi_estratti
})

# Ordino rigorosamente dal più vecchio al più recente
df_curva = df_curva.sort_values(by='Tempo_MJD').reset_index(drop=True)

# =============================================================================
# 3. PLOTTING
# =============================================================================

cartella_output = BASE_DIR / "studio_colossale" / "curve_di_luce"
cartella_output.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 6))

# Plotto i punti collegandoli con una linea
plt.plot(df_curva['Tempo_MJD'], df_curva['Flusso'], marker='o', linestyle='-',
         color='purple', alpha=0.8, markersize=6, linewidth=1.5, markerfacecolor='orange')

# Aggiungo la retta della media mobile o statica per riferimento
media_flusso = df_curva['Flusso'].mean()
plt.axhline(media_flusso, color='red', linestyle='--', alpha=0.6, label=f'Flusso Medio: {media_flusso:.2f}')

plt.title(f"Curva di Luce Decorrelata: {LABEL_BERSAGLIO}", fontsize=15, pad=15)
plt.xlabel("Tempo (Modified Julian Date)", fontsize=12)
plt.ylabel("Flusso Decorrelato Corretto", fontsize=12)

# Estetica della griglia
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Formatto l'asse X per non mostrare numeri esponenziali (comune con le MJD)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

# Aggiungo un box testuale con le info temporali assolute (inizio e fine)
inizio_str = df_curva['Tempo_Stringa'].iloc[0].replace('T', ' ')
fine_str = df_curva['Tempo_Stringa'].iloc[-1].replace('T', ' ')
testo_tempo = f"Inizio Oss: {inizio_str}\nFine Oss: {fine_str}\nTotale Frame: {len(df_curva)}"
plt.text(0.02, 0.95, testo_tempo, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))

nome_plot = cartella_output / f"curva_luce_{LABEL_BERSAGLIO}.png"
plt.tight_layout()
plt.savefig(nome_plot, dpi=300)
plt.close()

print(f"COMPLETATO! Curva di luce salvata in: {nome_plot}")