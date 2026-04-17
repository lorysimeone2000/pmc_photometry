import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from photutils.aperture import CircularAperture
import matplotlib.pyplot as plt
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry
import time
import sys
import gc
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
import warnings
from astropy.wcs import FITSFixedWarning
from photutils.datasets import make_100gaussians_image
from scipy.optimize import curve_fit
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

# gestisco i warning ignorandoli
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")
PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

from funzioni.utilita_parquet import *
from funzioni.astrometria_parquet import *

percorso_somma_pixel = cerca_file_nel_progetto(BASE_DIR, "risultati_somma_pixel.csv")

# Carico il mio file CSV in un DataFrame
df = pd.read_csv(percorso_somma_pixel)

# Converto la mia colonna dei tempi in un formato data/ora gestibile
df['Tempo_UTC'] = pd.to_datetime(df['Tempo_UTC'])

# Trovo il mio tempo 0 ricavando il tempo minimo registrato nella prima immagine
tempo_zero_globale = df['Tempo_UTC'].min()

# Calcolo i minuti trascorsi dal mio primissimo scatto per tutte le righe
df['minuti_trascorsi'] = (df['Tempo_UTC'] - tempo_zero_globale).dt.total_seconds() / 60.0

# Estraggo i miei valori unici per capire quante run ci sono
RUN = df['Run'].unique()

# calcolo i limiti temporali di ciascuna run per poter disegnare le linee divisorie
run_boundaries = []
for run in RUN:
    t_end = df[df['Run'] == run]['minuti_trascorsi'].max()
    run_boundaries.append((run, t_end))

# =============================================================================
# CREAZIONE E SALVATAGGIO DEL GRAFICO
# =============================================================================
print("Generazione del grafico...")

# imposto dimensioni adatte a un testo A4 con width=0.85\textwidth
plt.figure(figsize=(9, 5))

# Imposto il colore blu come unico elemento della lista per uniformare il grafico al file di riferimento
colori = ['blue']

# aggiungo una variabile di controllo per inserire l'etichetta una sola volta nella legenda
etichetta_aggiunta = False

for idx, run in enumerate(RUN):
    # Isolo i dati appartenenti solo alla mia run corrente
    dati_run = df[df['Run'] == run]

    colore = colori[idx % len(colori)]

    # imposto l'etichetta solo se non l'ho già aggiunta in precedenza
    etichetta_legenda = "Non-stellar background per pixel" if not etichetta_aggiunta else ""

    # Uso i miei nuovi dati calcolati per creare la curva della run
    plt.plot(dati_run['minuti_trascorsi'], dati_run['fondo_per_pixel'], marker='o', markersize=1,
             linestyle='-', linewidth=.5, color=colore, label=etichetta_legenda)

    # aggiorno la variabile per non ripetere l'etichetta nei cicli successivi
    etichetta_aggiunta = True

# disegno le linee divisorie per evidenziare le run e aggiungo i testi dimensionati
for r_idx, (run_num, t_end) in enumerate(run_boundaries):
    plt.axvline(x=t_end, color='gray', linestyle='--', alpha=0.6)
    # aggiungo il testo richiesto accanto alle linee divisorie
    plt.text(t_end, 0.95, f'End of run {run_num}', color='gray', ha='right', va='top', rotation=90,
             transform=plt.gca().get_xaxis_transform(), fontsize=12)

# applico la mia formattazione grafica personalizzata
plt.xlabel("Time from the start of run 1 (minutes)", fontsize=14)
plt.ylabel("Mean non-stellar pixel value", fontsize=14)

# dimensiono i valori numerici sugli assi
plt.tick_params(axis='both', which='major', labelsize=12)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# ottimizzo i margini per la figura
plt.tight_layout()

file_grafico = "fondo_non_stellare.png"

# salvo assicurandomi che non vengano tagliate le etichette
plt.savefig(file_grafico, dpi=300, bbox_inches='tight')
print(f"Grafico salvato con successo: {file_grafico}")
# plt.show()