import pandas as pd
#pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
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
from astropy.table import Table
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
# warning
import warnings
from astropy.io.fits.verify import VerifyWarning
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning) # Sopprime il warning FITSFixedWarning

from pathlib import Path

vizier = Vizier(
            catalog="II/389/ps1_dr2",
            columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
            row_limit=-1
        )

def converti_valore(valore):
    """
    Converte una stringa nel tipo di dato appropriato.
    Prova in ordine: int, float, mantiene stringa se non è convertibile.
    """
    valore = valore.strip()

    # Se è vuoto, restituisci stringa vuota
    if not valore:
        return valore

    # Prova a convertire in int
    try:
        return int(valore)
    except ValueError:
        pass

    # Prova a convertire in float
    try:
        return float(valore)
    except ValueError:
        pass

    # Prova a riconoscere booleani FITS
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False

    # Altrimenti restituisci la stringa originale
    return valore

def leggi_header_da_csv(filename):
    """Legge l'header FITS dal file CSV"""
    header_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                # Rimuovi il '#' e dividi chiave-valore
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':  # Fine dell'header
                break

    return header_dict

i = 0
x = [] # tempo in secondi
y1 = []
y2 = []

# Cartella contenente i file CSV
cartella_csv = "/home/lorysimeone/tesi_magistrale/prove/analisi/sorgenti_run/sorgenti_run_1"

file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')]) # lista di tutti i file CSV ordinati per nome

# Itera su tutti i file CSV
for nome_file in file_csv:
    i += 1
    filename = os.path.join(cartella_csv, nome_file) # nome del file csv
    # print(filename)
    dataframe = pd.read_csv(filename, skiprows=59)
    header_dal_csv = leggi_header_da_csv(filename)
    percorso_file_fits = header_dal_csv['PERCORSO_FILE']
    hdu_list = fits.open(percorso_file_fits)
    w = WCS(hdu_list[0].header) # creo un oggetto WCS usando l'header del file FITS,
    # che contiene le informazioni per le trasformazioni di coordinate
    data = hdu_list[0].data

    '''if i==1:
        print("Calcolo catalogo") 

        ra_centro = header_dal_csv['RA']
        dec_centro = header_dal_csv['DEC']
        centro = SkyCoord(ra_centro, dec_centro, unit=u.deg)


        riquadro = vizier.query_region(coord.SkyCoord(ra=ra_centro, dec=dec_centro,
                                                      unit=(u.deg, u.deg),
                                                      frame='icrs'),
                                       radius=Angle(centro.separation(alto_destra)*1.5, "deg"),
                                       column_filters={'gmag': f'<{14}'},
                                       )

        tbl_catalogo_esteso = riquadro[0]

        print(f"Tabella catalogo esteso:\n{tbl_catalogo_esteso}")'''

    if i == 1:
        x.append(0)
        t0 = header_dal_csv['TSTART']
        print("Tempo iniziale:",t0, " secondi")
    else: x.append((header_dal_csv['TSTART']-t0)/np.float64(1e3))

    ra_centro = header_dal_csv['RA']
    dec_centro = header_dal_csv['DEC']
    centro = SkyCoord(ra_centro, dec_centro, unit=u.deg)

    if i==1:
        print("-----------------------------------------------------------------------------------")
        print("Calcoli con pixel_to_world")

    # calcoli con pixel_to_world

    alto_destra = w.pixel_to_world(3072, 2048)
    alto_sinistra = w.pixel_to_world(3072, 0)
    # print(f"Coordinate in alto a destra: {alto_destra}")
    basso_sinistra = w.pixel_to_world(0, 0)
    basso_destra = w.pixel_to_world(0, 2048)
    if i==1:
        print(f"Coordinate in alto a destra con pixel_to_world: {alto_destra}")
        print(f"Coordinate in basso a sinistra con pixel_to_world: {basso_sinistra}")
        print(f"Coordinate in alto a sinistra con pixel_to_world: {alto_sinistra}")
        print(f"Coordinate in basso a destra con pixel_to_world: {basso_destra}")

    ra_alto_destra = alto_destra.ra.deg
    ra_basso_destra = basso_destra.ra.deg
    ra_basso_sinistra = basso_sinistra.ra.deg
    ra_alto_sinistra = alto_sinistra.ra.deg
    dec_alto_destra = alto_destra.dec.deg
    dec_basso_destra = basso_destra.dec.deg
    dec_basso_sinistra = basso_sinistra.dec.deg
    dec_alto_sinistra = alto_sinistra.dec.deg

    ra_max = np.max(np.array([ra_alto_destra, ra_basso_sinistra, ra_basso_destra, ra_alto_sinistra]))
    ra_min = np.min(np.array([ra_alto_destra, ra_basso_sinistra, ra_basso_destra, ra_alto_sinistra]))
    dec_max = np.max(np.array([dec_alto_destra, dec_basso_sinistra, dec_basso_destra, dec_alto_sinistra]))
    dec_min = np.min(np.array([dec_alto_destra, dec_basso_sinistra, dec_basso_destra, dec_alto_sinistra]))
    raggio1 = Angle(centro.separation(alto_destra), "deg")
    raggio1 = raggio1.degree
    if i==1:
        print(type(raggio1))

        print("Larghezza: ", alto_destra.separation(alto_sinistra).degree)
        print("Altezza: ", alto_destra.separation(basso_destra).degree)
        print("Raggio: ", raggio1)

    y1.append(raggio1)

    # calcoli con wcs_pix2world

    if i == 1:
        print("-----------------------------------------------------------------------------------")
        print("Calcoli con wcs_pix2world")

    ny, nx = data.shape
    xc, yc = nx / 2, ny / 2  # coordinate del centro in pixel

    # definisco i quattro angoli dell'immagine in pixel
    pixels = np.array([
        [0, 0],
        [0, ny - 1],
        [nx - 1, 0],
        [nx - 1, ny - 1]
    ])

    # Definisco i range di RA e DEC (in gradi) a partire dagli estremi in alto a destra e in basso a sinistra

    world = w.wcs_pix2world(pixels, 0)  # converte i pixel in coordinate celesti (RA, Dec)

    ra_vals = world[:, 0]
    dec_vals = world[:, 1]

    # calcolo minimi e massimi
    ra_min_, ra_max_ = np.min(ra_vals), np.max(ra_vals)
    dec_min_, dec_max_ = np.min(dec_vals), np.max(dec_vals)

    alto_destra_ = SkyCoord(ra_max_, dec_max_, unit=u.deg)
    if i==1: print(f"Coordinate in alto a destra con wcs_pix2world: {alto_destra_}")
    basso_sinistra_ = SkyCoord(ra_min_, dec_min_, unit=u.deg)
    if i==1: print(f"Coordinate in basso a sinistra con wcs_pix2world: {basso_sinistra_}")
    alto_sinistra_ = SkyCoord(ra_min_, dec_max_, unit=u.deg)
    if i == 1: print(f"Coordinate in alto a sinistra con wcs_pix2world: {alto_sinistra_}")
    basso_destra_ = SkyCoord(ra_max_, dec_min_, unit=u.deg)
    if i == 1: print(f"Coordinate in basso a destra con wcs_pix2world: {basso_destra_}")
    raggio2 = Angle(centro.separation(alto_destra_), "deg")
    raggio2 = raggio2.degree
    if i==1:
        print("Larghezza: " , alto_destra_.separation(alto_sinistra_).degree)
        print("Altezza: " , alto_destra_.separation(basso_destra_).degree)
        print("Raggio: " , raggio2)

    if i == 1: print(type(raggio2))
    y2.append(raggio2)

    if i==1:
        print("METODO ALTERNATIVO")
        # Definisci una volta i punti di interesse
        punti_interesse = np.array([
            [3072, 2048],  # alto_destra
            [3072, 0],  # alto_sinistra
            [0, 2048],  # basso_destra
            [0, 0]  # basso_sinistra
        ])

        # Metodo 1: pixel_to_world
        coords1 = w.pixel_to_world(punti_interesse[:, 0], punti_interesse[:, 1])

        # Metodo 2: wcs_pix2world
        coords2 = w.wcs_pix2world(punti_interesse, 0)
        coords2_sky = SkyCoord(coords2[:, 0], coords2[:, 1], unit=u.deg)

        # Calcola con gli stessi punti
        ad1, as1, bd1, bs1 = coords1
        ad2, as2, bd2, bs2 = coords2_sky

        print("Larghezza metodo 1:", ad1.separation(as1).degree)
        print("Larghezza metodo 2:", ad2.separation(as2).degree)

        print("Altezza metodo 1:", ad1.separation(bd1).degree)
        print("Altezza metodo 2:", ad2.separation(bd2).degree)

        print("Raggio metodo 1:", ad1.separation(bs1).degree / 2)
        print("Raggio metodo 2:", ad2.separation(bs2).degree / 2)


x = np.array(x)
y1 = np.array(y1)
y2 = np.array(y2)


plt.plot(x , y1, marker='o', linestyle='-', linewidth=2, markersize=6, label='pixel_to_world')
plt.plot(x , y2, marker='o', linestyle='-', linewidth=2, markersize=6, label='wcs_pix2world')
#plt.plot(y1 , y2, marker='o', linestyle='-', linewidth=2, markersize=6, label='DEC max')


plt.xlabel('Secondi')
plt.ylabel('Raggio')
plt.title('Confronto raggio pixel_to_world vs wcs_pix2world')
plt.grid(True, alpha=0.3)
#plt.ylim(np.min(np.array([y1,y2]))-1, np.max(np.array([y1,y2]))+1)
plt.legend()
plt.show()