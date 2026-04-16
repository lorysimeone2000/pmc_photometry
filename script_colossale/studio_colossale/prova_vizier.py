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

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"Moduli esterni caricati con successo.")
print(f"------------------------------")

# scarico le 5 bande fondamentali per simulare il mio sensore FLIR
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1,
)

# ottengo le mie coordinate centrali della nebulosa del granchio
centro_granchio = SkyCoord.from_name('Crab Nebula')
ra_c = centro_granchio.ra.deg
dec_c = centro_granchio.dec.deg

# definisco il mio raggio di ricerca
raggio_ricerca = 5.4 * u.deg

# interrogo la mia regione su vizier
riquadro_esterno_vizier = vizier.query_region(
    coord.SkyCoord(ra=ra_c, dec=dec_c, unit=(u.deg, u.deg), frame='icrs'),
    radius=raggio_ricerca
)

# verifico che la mia query abbia restituito dei risultati
if len(riquadro_esterno_vizier) > 0:
    # estraggo la mia tabella
    tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]

    # stampo la lunghezza finale della mia tabella astropy
    print(f"Lunghezza totale della tabella Astropy scaricata: {len(tbl_riquadro_esterno_vizier)}")
else:
    print("Nessun dato trovato per la regione richiesta.")

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
from astropy.table import Table, vstack, unique
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

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"Moduli esterni caricati con successo.")
print(f"------------------------------")

# scarico le 5 bande fondamentali per simulare il mio sensore FLIR
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1,
)

# ottengo le mie coordinate centrali della nebulosa del granchio
centro_granchio = SkyCoord.from_name('Crab Nebula')
ra_c = centro_granchio.ra.deg
dec_c = centro_granchio.dec.deg

# definisco il mio raggio di ricerca globale
raggio_ricerca = 5.4

# interrogo la mia regione su vizier (modalità singola globale)
print("Esecuzione query singola globale...")
riquadro_esterno_vizier = vizier.query_region(
    coord.SkyCoord(ra=ra_c, dec=dec_c, unit=(u.deg, u.deg), frame='icrs'),
    radius=raggio_ricerca * u.deg
)

if len(riquadro_esterno_vizier) > 0:
    tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]
    print(f"Lunghezza totale della tabella Astropy scaricata: {len(tbl_riquadro_esterno_vizier)}")
else:
    print("Nessun dato trovato per la regione richiesta.")

# =============================================================================
# METODO MINIQUERY A NIDO D'APE
# =============================================================================

print("\nPreparazione griglia a nido d'ape per le miniquery...")
raggio_mini = 0.1

# calcolo i passi della griglia esagonale per non lasciare spazi vuoti
passo_dec = 1.5 * raggio_mini
passo_ra_base = np.sqrt(3) * raggio_mini

centri_ra = []
centri_dec = []

riga_griglia = 0
for offset_dec in np.arange(-raggio_ricerca, raggio_ricerca + passo_dec, passo_dec):
    dec_corrente = dec_c + offset_dec

    # correggo il mio passo in Ascensione Retta in base alla declinazione corrente
    cos_dec = np.cos(np.radians(dec_corrente))
    passo_ra = passo_ra_base / cos_dec

    # sfalso di mezzo passo le righe dispari per chiudere l'esagono
    sfalsamento_ra = (passo_ra / 2.0) if riga_griglia % 2 != 0 else 0.0

    range_ra = raggio_ricerca / cos_dec

    for offset_ra in np.arange(-range_ra - sfalsamento_ra, range_ra + passo_ra, passo_ra):
        ra_corrente = ra_c + offset_ra + sfalsamento_ra
        centri_ra.append(ra_corrente)
        centri_dec.append(dec_corrente)

    riga_griglia += 1

# converto tutti i miei centri in SkyCoord e filtro solo quelli interni all'area
coordinate_centri = SkyCoord(ra=centri_ra * u.deg, dec=centri_dec * u.deg)
distanze_dal_centro = centro_granchio.separation(coordinate_centri)

# uso un filtro per mantenere i centri che coprono effettivamente il raggio richiesto
maschera_interni = distanze_dal_centro.deg <= raggio_ricerca
centri_finali = coordinate_centri[maschera_interni]

print(f"Totale miniquery da eseguire: {len(centri_finali)}")

tabelle_parziali = []

# eseguo il loop sulle mie coordinate con barra di avanzamento
for centro_corrente in tqdm(centri_finali, desc="Scaricamento miniquery"):
    try:
        risultato_mini = vizier.query_region(
            centro_corrente,
            radius=raggio_mini * u.deg
        )
        if len(risultato_mini) > 0:
            tabelle_parziali.append(risultato_mini[0])
    except Exception:
        # ignoro silenziosamente i fallimenti isolati per proseguire l'estrazione
        pass

    # metto la pausa richiesta tra una query e l'altra
    time.sleep(0.2)

if len(tabelle_parziali) > 0:
    print("\nUnione e rimozione dei duplicati in corso...")
    # concateno tutte le mie tabelle
    tabella_combinata = vstack(tabelle_parziali)

    # uso la chiave primaria objID di Pan-STARRS per rimuovere le stelle estratte più volte
    tabella_miniquery_unite = unique(tabella_combinata, keys='objID')

    print(f"Lunghezza totale della tabella miniquery unite: {len(tabella_miniquery_unite)}")
else:
    print("Nessun dato ottenuto dalle miniquery a nido d'ape.")