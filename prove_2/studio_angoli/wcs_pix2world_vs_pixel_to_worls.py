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

#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_215217.fits"
image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_220519.fits"


hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

data = hdu_list[0].data

w = WCS(hdu_list[0].header) # creo un oggetto WCS usando l'header del file FITS,
# che contiene le informazioni per le trasformazioni di coordinate
hdu_list.close()

ny, nx = data.shape
xc, yc = nx / 2, ny / 2 # coordinate del centro in pixel

# definisco i quattro angoli dell'immagine in pixel
pixels = np.array([
    [0, 0],
    [0, ny - 1],
    [nx - 1, 0],
    [nx - 1, ny - 1]
])

# Definisco i range di RA e DEC (in gradi) a partire dagli estremi in alto a destra e in basso a sinistra

world = w.wcs_pix2world(pixels, 0) # converte i pixel in coordinate celesti (RA, Dec)

ra_vals = world[:, 0]
dec_vals = world[:, 1]

# calcolo minimi e massimi
ra_min, ra_max = np.min(ra_vals), np.max(ra_vals)
dec_min, dec_max = np.min(dec_vals), np.max(dec_vals)

alto_destra = SkyCoord(ra_max, dec_max, unit=u.deg)
#print(f"Coordinate in alto a destra con wcs_pix2world: {alto_destra}")
basso_sinistra = SkyCoord(ra_min, dec_min, unit=u.deg)
#print(f"Coordinate in basso a sinistra con wcs_pix2world: {basso_sinistra}")
alto_sinistra = SkyCoord(ra_min, dec_max, unit=u.deg)
basso_destra = SkyCoord(ra_max, dec_min, unit=u.deg)

#larghezza = np.abs(ra_max - ra_min)
larghezza = alto_destra.separation(alto_sinistra)
print(f"Larghezza con wcs_pix2world: {larghezza}")
#altezza = np.abs(dec_max - dec_min)
altezza = basso_sinistra.separation(alto_sinistra)
print(f"Altezza con wcs_pix2world: {altezza}")


# calcoli con pixel_to_world

alto_destra_ = w.pixel_to_world(3072, 2048)
alto_sinistra_ = w.pixel_to_world(3072, 0)
# print(f"Coordinate in alto a destra: {alto_destra}")
basso_sinistra_ = w.pixel_to_world(0,0)
basso_destra_ = w.pixel_to_world(0,2048)
# print(f"Coordinate in basso a sinistra: {basso_sinistra}")
aperture2 = CircularAperture((0,0), r=300)

ra1 = alto_destra_.ra.deg # oppure .hour per avere in ore
#print(f"ra1 col .deg: {ra1}")
ra2 = basso_sinistra_.ra.deg
larghezza_ = alto_sinistra_.separation(basso_sinistra_)
altezza_ = alto_destra_.separation(alto_sinistra_)
ra_min_ = np.min(np.array([ra1, ra2]))
ra_max_ = np.max(np.array([ra1, ra2]))
'''print(f"RA_min: {ra_min}°")
print(f"RA_max: {ra_max}°")'''
dec1 = alto_destra_.dec.deg
#print(f"dec1 col .deg: {dec1}")

dec2 = basso_sinistra_.dec.deg
dec_min_ = np.min(np.array([dec1, dec2]))
dec_max_ = np.max(np.array([dec1, dec2]))
'''print(f"DEC_min: {dec_min}°")
print(f"DEC_max: {dec_max}°")'''

#larghezza_ = np.abs(ra_max_ - ra_min_)
print(f"Larghezza con pixel_to_world: {larghezza_}")
#altezza_ = np.abs(dec_max_ - dec_min_)
print(f"Altezza con pixel_to_world: {altezza_}")

# cerco la stella più luminosa

# Cartella contenente i file CSV
cartella_csv = "/home/lorysimeone/tesi_magistrale/prove/analisi/sorgenti_run/sorgenti_run_1"

file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')]) # lista tutti i file CSV ordinati per nome

print(f"Trovati {len(file_csv)} file CSV:")
'''for file in file_csv:
    print(f"  - {file}")'''

i = 0
j = 0
posizioni_lista = []  # lista che dovrà essere riempita con tutte le poszioni di tutte le tabelle
distanze = []
numero_stelle_catalogate = []
tempo = []
# Itera su tutti i file CSV
for nome_file in file_csv:
    i += 1
    filename = os.path.join(cartella_csv, nome_file) # nome del file csv
    # print(filename)
    dataframe = pd.read_csv(filename, skiprows=59)
    header_dal_csv = leggi_header_da_csv(filename)
    percorso_file_fits = header_dal_csv['PERCORSO_FILE']

    if percorso_file_fits == image_file:
        print("Trovato")

        # in questa immagine cerco la stella del catalogo più luminosa e vedo se funziona la corrispondenza

        vizier = Vizier(
            catalog="II/389/ps1_dr2",
            columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
            row_limit=-1
        )
        ra_centro = header_dal_csv['RA']
        dec_centro = header_dal_csv['DEC']
        centro = SkyCoord(ra_centro, dec_centro, unit=u.deg)

        riquadro = vizier.query_region(coord.SkyCoord(ra=ra_centro, dec=dec_centro,
                                                      unit=(u.deg, u.deg),
                                                      frame='icrs'),
                                       radius=Angle(centro.separation(alto_destra), "deg"),
                                       column_filters={'gmag': f'<{14}'},
                                       )

        tbl_catalogo_esteso = riquadro[0]
        tbl_catalogo = tbl_catalogo_esteso[(tbl_catalogo_esteso['RAJ2000'] >= ra_min) &
                                           (tbl_catalogo_esteso['RAJ2000'] <= ra_max) &
                                           (tbl_catalogo_esteso['DEJ2000'] >= dec_min) &
                                           (tbl_catalogo_esteso['DEJ2000'] <= dec_max)]
        print(f"Tabella catalogo:\n{tbl_catalogo_esteso}")

        dataframe = pd.read_csv(filename, skiprows=59)
        tbl_trovate = Table.from_pandas(dataframe)
        magnitudini = tbl_catalogo['gmag']
        mag_min_del_catalogo = np.min(magnitudini)
        indice_mag_min = np.argmin(magnitudini)
        stella_piu_luminosa = tbl_catalogo[indice_mag_min] # ho l'intera riga della stella con amgnitudine massima
        print(f"Dati della stella più luminosa:")
        print(f"  RA: {stella_piu_luminosa['RAJ2000']}°")
        print(f"  Dec: {stella_piu_luminosa['DEJ2000']}°")
        print(f"  gmag: {stella_piu_luminosa['gmag']}")
        cordinate_stella = SkyCoord(stella_piu_luminosa['RAJ2000'], stella_piu_luminosa['DEJ2000'], unit=u.deg)
        coordinate_pixel = w.world_to_pixel(cordinate_stella)
        #print(coordinate_pixel.shape)
        coordinate_pixel_stack = np.column_stack(coordinate_pixel)
        print(f"Pixel coordinate: {np.array(coordinate_pixel_stack)}")
        coo_pixel_to_world = w.pixel_to_world(coordinate_pixel[0], coordinate_pixel[1])
        print("Coordinate della stella con pixel_to_world: ", coo_pixel_to_world)
        coo_wcs_pix2world = w.wcs_pix2world(np.array(coordinate_pixel_stack), 0)
        print("Coordinate della stella con wcs_pix2world: ", coo_wcs_pix2world)

        print("Prova 2:")
        coordinate = np.array([3072, 2048])
        print(coordinate.shape)
        coordinate_stack = np.column_stack(coordinate)
        print(f"Pixel coordinate: {coordinate}")
        coo_pixel_to_world = w.pixel_to_world(3072, 2048)
        print("Coordinate della stella con pixel_to_world: ", coo_pixel_to_world)
        coo_wcs_pix2world = w.wcs_pix2world(np.array(coordinate_stack), 0)
        print("Coordinate della stella con wcs_pix2world: ", coo_wcs_pix2world)

        break






'''plt.figure(figsize=(8, 8))
plt.imshow(np.log10(data + 1), cmap='gray_r', origin='lower')
plt.colorbar(label='Count rate')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.show()'''