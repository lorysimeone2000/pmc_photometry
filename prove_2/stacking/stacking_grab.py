import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
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

# --- CARICAMENTO DEI FILE GLOBALI ---

# definisco i nomi dei file globali salvati in precedenza
nome_file_c = "coverage_map_mrk421_globale.fits"
nome_file_sum = "stacked_sum_mrk421_globale.fits"

# cerco i file dinamicamente
percorso_c = cerca_file_nel_progetto(BASE_DIR, nome_file_c)
percorso_sum = cerca_file_nel_progetto(BASE_DIR, nome_file_sum)

if not percorso_c or not percorso_sum:
    raise FileNotFoundError("Errore: impossibile trovare i file globali di Markarian 421.")

# 1. carico la Coverage Map globale
with fits.open(str(percorso_c)) as hdu_c:
    total_coverage = hdu_c[0].data

# 2. carico l'Immagine Stacked globale
with fits.open(str(percorso_sum)) as hdu_sum:
    total_data = hdu_sum[0].data

    # estraggo il WCS usando relax=True per leggere senza problemi i coefficienti di distorsione
    wcs_totale = WCS(hdu_sum[0].header, relax=True)

# estraggo il valore massimo di copertura globale
full_coverage_value = np.max(total_coverage)
print(f"Copertura massima totale raggiunta: {full_coverage_value} immagini")

# --- ESTRAZIONE E VISUALIZZAZIONE ---

# calcolo le statistiche sull'immagine totale
mean, median, std = sigma_clipped_stats(total_data, sigma=3.0)
print("Mediana totale: ", median)

# sottraggo il fondo mediano
data_finale = total_data - median

# visualizzazione
norm = simple_norm(data_finale, 'sqrt')
plt.figure(figsize=(10, 8))

# imposto il sistema di riferimento celeste tramite WCS e lo assegno ad ax
ax = plt.subplot(projection=wcs_totale)

# genero l'immagine e la salvo in una variabile 'im'
im = ax.imshow(data_finale, cmap="viridis", norm=norm, interpolation='nearest', origin='lower')

# cerco la prima tabella catalogata in tabelle_blazar/tabelle_cataloghi
dir_tabelle_cat = BASE_DIR / "tabelle_blazar" / "tabelle_cataloghi"
if dir_tabelle_cat.exists():
    tabelle_csv = sorted(list(dir_tabelle_cat.rglob("*.csv")))
    if tabelle_csv:
        prima_tabella = tabelle_csv[0]
        print(f"Sovrappongo lo scatter della tabella: {prima_tabella.name}")

        # leggo i dati del catalogo
        df_cat = pd.read_csv(prima_tabella, comment='#')

        # identifico le colonne delle coordinate astronomiche
        col_ra = 'RAJ2000' if 'RAJ2000' in df_cat.columns else (
            'RA_centroid' if 'RA_centroid' in df_cat.columns else 'RA')
        col_dec = 'DEJ2000' if 'DEJ2000' in df_cat.columns else (
            'DEC_centroid' if 'DEC_centroid' in df_cat.columns else 'DEC')

        if col_ra in df_cat.columns and col_dec in df_cat.columns:
            # creo un array di coordinate celesti dal catalogo
            cat_coords = SkyCoord(ra=df_cat[col_ra].values * u.deg, dec=df_cat[col_dec].values * u.deg, frame='icrs')

            # converto le coordinate in pixel relativi alla mia immagine ritagliata
            x_pix, y_pix = wcs_totale.world_to_pixel(cat_coords)

            # estraggo le dimensioni dell'immagine
            ny, nx = total_data.shape

            # creo una maschera logica per tenere solo le stelle che cadono dentro il riquadro dell'immagine
            mask_inside = (x_pix >= 0) & (x_pix <= nx) & (y_pix >= 0) & (y_pix <= ny)

            # filtro il dataframe usando la mia maschera
            df_cat_filtered = df_cat[mask_inside]

            print(f"Trovate {len(df_cat_filtered)} stelle di catalogo all'interno del riquadro.")

            # applico il transform='world' per allineare direttamente RA/DEC all'immagine (solo sulle stelle filtrate)
            ax.scatter(df_cat_filtered[col_ra], df_cat_filtered[col_dec], transform=ax.get_transform('world'),
                       s=4, color='red', label='Catalogo (nel riquadro)', zorder=10)
            plt.legend()

# passo l'oggetto 'im' alla colorbar
plt.colorbar(im, label='Counts (Somma Totale)')

# aggiorno il titolo con le info sul Blazar
plt.title(f'Stacking Globale Mrk 421 (1.6x1.6 arcmin)\n(Copertura max={int(full_coverage_value)} immagini)')

# aggiorno le etichette con le coordinate celesti
plt.xlabel('RA')
plt.ylabel('Dec')

# salvo e mostro il risultato
plt.savefig('stacking_mrk421_visualizzazione.png')
plt.show()