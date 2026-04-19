import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
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

# Imposto il wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from astroquery.vizier import Vizier
from astropy.coordinates import Angle

from shapely.geometry import Point, Polygon
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)  # sopprimo il warning FITSFixedWarning

from pathlib import Path


# =============================================================================
# FUNZIONI DI GESTIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    # cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # cerco un file ricorsivamente
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


# trovo la cartella base del mio progetto
BASE_DIR = trova_cartella_base("Lorenzo")

# --- CARICAMENTO E SOMMA DI TUTTE LE RUN ---

# inizializzo le matrici vuote che conterranno la somma totale e la proiezione
total_data = None
total_coverage = None
wcs_totale = None

# ciclo su tutte le 3 run (in questo caso analizzo solo la run 1)
runs = [1]

for run in runs:
    # definisco i nomi dei file salvati nello step precedente per la run corrente
    nome_file_c = f"run_{run}_coverage_map.fits"

    # cerco i file dinamicamente
    percorso_c = cerca_file_nel_progetto(BASE_DIR, nome_file_c)

    # 1. carico e sommo la Coverage Map
    with fits.open(str(percorso_c)) as hdu_c:
        image_data_c = hdu_c[0].data
        if total_coverage is None:
            # creo la matrice di zeri basandomi sulla forma della prima immagine trovata
            total_coverage = np.zeros_like(image_data_c, dtype=float)
        # aggiungo i dati della run corrente al totale
        total_coverage += image_data_c

    # 2. carico e sommo l'Immagine Stacked
    with fits.open(str(percorso_c)) as hdu_sum:
        image_data_sum = hdu_sum[0].data

        # estraggo il WCS dalla prima immagine utile per usarlo nel grafico
        if wcs_totale is None:
            wcs_totale = WCS(hdu_sum[0].header)

        if total_data is None:
            # creo la matrice di zeri per l'immagine
            total_data = np.zeros_like(image_data_sum, dtype=float)
        # aggiungo i conteggi della run corrente al totale
        total_data += image_data_sum

if total_data is None or total_coverage is None:
    raise ValueError("Errore: non è stato possibile caricare i dati di nessuna run.")

# estraggo il valore massimo di copertura globale raggiunto unendo le 3 run
full_coverage_value = np.max(total_coverage)
print(f"Copertura massima totale raggiunta: {full_coverage_value} immagini")

# --- ESTRAZIONE E VISUALIZZAZIONE ---

# calcolo le statistiche sull'immagine totale
mean, median, std = sigma_clipped_stats(total_data, sigma=3.0)
print("Mediana totale: ", median)

# sottraggo il fondo mediano
data_finale = total_data

# preparo la visualizzazione
norm = PowerNorm(gamma=3.)

# dimensiono la figura per occupare 0.85\textwidth in un documento A4
fig = plt.figure(figsize=(8.5, 5))

# estraggo l'asse per poterne controllare le dimensioni dei font delle coordinate
ax = fig.add_subplot(111, projection=wcs_totale)

# --- INSERISCI QUESTE DUE RIGHE ---
# forzo la scala decimale per l'asse X (RA) e l'asse Y (Dec)
ax.coords[0].set_major_formatter('d.ddd')
ax.coords[1].set_major_formatter('d.ddd')
# forzo l'ascensione retta e la declinazione in gradi per la coerenza della scala
ax.coords[0].set_format_unit(u.deg)
ax.coords[1].set_format_unit(u.deg)
# ----------------------------------

# genero l'immagine assicurandomi che l'aspect ratio rimanga rigorosamente 1:1
im = ax.imshow(data_finale, cmap="gray", norm=norm, interpolation='nearest', origin='lower', aspect='equal')
ax.set_aspect('equal')

# aggiusto le dimensioni della colorbar in modo che non superi l'altezza del riquadro dati
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# imposto il label aggiornato e modifico il font direttamente sull'asse della colorbar
cbar.set_label('Total coverage (images)', fontsize=14)

# dimensiono i numeri lungo la colorbar
cbar.ax.tick_params(labelsize=12)

# aggiorno le etichette con le coordinate celesti in inglese britannico formale
ax.set_xlabel('DA (deg)', fontsize=14)
ax.set_ylabel('DEC (deg)', fontsize=14)

# dimensiono le etichette numeriche delle coordinate WCS sugli assi principali
ax.tick_params(axis='both', which='major', labelsize=12)

# salvo l'immagine ad alta risoluzione evitando tagli sui bordi
plt.savefig('coverage_map_run_1.png', dpi=300, bbox_inches='tight')
# plt.show()