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

# inizializzo Vizier con i suoi parametri di default
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1
)

# questa funzione restituisce la tabella delle sorgenti catalogate nel riquadro della pmc
def tabella_catalogo(image_file_,magnitudine_massima):
        """
        Seleziona le stelle del catalogo che rientrano nel riquadro e che sono sotto una certa magnitudine

        Parameters:
        image_file_ (string): percorso del file da cui estrarre l'header e quindi le coordinate dell'astrometria
        magnitudine_massima (float): magnitudina massima presa dal catalogo

        Returns:
        astropy.table.Table: Tabella delle stelle del catalogo che rientrano nel riquadro e che sono sotto una certa magnitudine
        """

        hdu_list_ = fits.open(image_file_)
        data = hdu_list_[0].data
        # print(data.shape)
        # trovo gli estremi

        w = WCS(hdu_list_[0].header)  # creo un oggetto WCS usando l'header del file FITS,
        # che contiene le informazioni per le trasformazioni di coordinate

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
        ra_min, ra_max = np.min(ra_vals), np.max(ra_vals)
        dec_min, dec_max = np.min(dec_vals), np.max(dec_vals)

        # seleziono le righe del catalogo esterno che soddisfano le condizioni del file e la magnitudine massima
        subset = tbl_riquadro_esterno[(tbl_riquadro_esterno['RAJ2000'] >= ra_min) &
                            (tbl_riquadro_esterno['RAJ2000'] <= ra_max) &
                            (tbl_riquadro_esterno['DEJ2000'] >= dec_min) &
                            (tbl_riquadro_esterno['DEJ2000'] <= dec_max) &
                            (tbl_riquadro_esterno['gmag'] <= magnitudine_massima)]

        return subset

nome = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212815.fits"
# tabella = tabella_catalogo(nome, 14)

def calcolo_distanze(tbl_trovate, tbl_catalogate, image_file):
    """
    Calcola le distanze dei centroidi rispetto alle stelle catalogate più vicine

    Parameters:
    tbl_trovate (Table): Tabella delle sorgenti trovate con image segmentation
    tbl_catalogate (Table): Tabella delle stelle del catalogo
    image_file (str): Percorso del file FITS

    Returns:
    array: elenco delle distanze minime di tutti i centroidi
    """

    # Carica il WCS dall'immagine
    hdu_list = fits.open(image_file)
    w = WCS(hdu_list[0].header)

    # Converti i centroidi pixel in coordinate celesti
    # metodo pixel_to_world
    coords_trovate = w.pixel_to_world(tbl_trovate['xcentroid'], tbl_trovate['ycentroid'])
    # metodo wcs_pix2world
    '''centroidi_combinati = np.column_stack((tbl_trovate['xcentroid'], tbl_trovate['ycentroid']))
    coords_trovate_pixel = w.wcs_pix2world(centroidi_combinati, 0)
    coords_trovate = SkyCoord(ra=coords_trovate_pixel[:, 0] * u.deg,
                              dec=coords_trovate_pixel[:, 1] * u.deg)'''

    # converto le coordinate catalogate in array numpy e gestisco le unità
    try:
        # estraggo i valori come array numpy puri
        if hasattr(tbl_catalogate['RAJ2000'], 'value'):
            # se ho già unità, estraggo solo i valori
            ra_values = tbl_catalogate['RAJ2000'].value
            dec_values = tbl_catalogate['DEJ2000'].value
        else:
            # altrimenti converto direttamente
            ra_values = np.array(tbl_catalogate['RAJ2000'])
            dec_values = np.array(tbl_catalogate['DEJ2000'])

        '''
        print(f"ra_values tipo: {type(ra_values)}, forma: {ra_values.shape}")
        print(f"dec_values tipo: {type(dec_values)}, forma: {dec_values.shape}")
        '''

        # creo SkyCoord con i valori puri specificando le unità
        coords_catalogate = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)

    except Exception as e:
        print(f"Errore nell'approccio principale: {e}")
        # APPROCCIO ALTERNATIVO: uso direttamente i valori senza moltiplicare per unità
        coords_catalogate = SkyCoord(ra=tbl_catalogate['RAJ2000'],
                                     dec=tbl_catalogate['DEJ2000'],
                                     unit=u.deg)

    print("numero di stelle catalogate" , np.shape(coords_catalogate))

    # calcolo le distanze di tutti i centroidi da tutte le stelle catalogate
    distanze_minime = []
    corrispondenze = []
    righe_tabella_combinata = [] # lista per la tabella combinata

    for i, coord_trovata in enumerate(coords_trovate):
        # calcolo la distanza da tutte le stelle catalogate
        distanze_singola = coord_trovata.separation(coords_catalogate) # Calcola la distanza angolare tra la singola stella trovata
        # (coord_trovata) e tutte le stelle del catalogo (coords_catalogate). Restituisce un array di distanze angolari.
        # if i == 1: print(np.shape(distanze_singola))
        # trovo la distanza minima e l'indice della stella più vicina
        distanza_minima = np.min(distanze_singola)

        distanze_minime.append(distanza_minima)

    distanze_gradi = [d.deg for d in distanze_minime]
    hdu_list.close()

    return distanze_gradi

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
    # if i == 5: break

    tbl_trovate = Table.from_pandas(dataframe)
    if i == 1: # chiamo il sito una volta sola su un'immagine più grande per non mandarlo in down
        print("Nome del file fits:" , percorso_file_fits)
        print("Nome del file csv:" , filename)
        print("Tabella astropy ricavata:\n" , tbl_trovate)


        header = leggi_header_da_csv(filename)
        hdu_list_ = fits.open(percorso_file_fits)
        data = hdu_list_[0].data # mi serve giusto per le dimensioni

        # prendo un riquadro globale leggermente più grande del riquadro della pmc

        # coordinate centro
        image_header = hdu_list_[0].header
        ra_centro = image_header["RA"]
        dec_centro = image_header["DEC"]

        w = WCS(hdu_list_[0].header)  # creo un oggetto WCS usando l'header del file FITS

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
        ra_min, ra_max = np.min(ra_vals), np.max(ra_vals)
        dec_min, dec_max = np.min(dec_vals), np.max(dec_vals)

        # Creo la tabella del catalogo

        # coordinate centro
        ra_centro_ = image_header["RA"]
        print("RA centro: ", ra_centro)
        dec_centro_ = image_header["DEC"]
        print("DEC centro: ", dec_centro)

        larghezza = np.abs(ra_max - ra_min)
        print(f"Larghezza: {larghezza}")
        altezza = np.abs(dec_max - dec_min)
        print(f"Altezza: {altezza}")

        alto_destra = SkyCoord(ra_max, dec_max, unit=u.deg)
        print(f"Coordinate in alto a destra: {alto_destra}")
        basso_sinistra = SkyCoord(ra_min, dec_min, unit=u.deg)
        print(f"Coordinate in basso a sinistra: {basso_sinistra}")
        centro = SkyCoord(ra_centro_, dec_centro, unit=u.deg)

        riquadro_esterno = vizier.query_region(coord.SkyCoord(ra=ra_centro_, dec=dec_centro_,
                                                      unit=(u.deg, u.deg),
                                                      frame='icrs'),
                                       radius=Angle(centro.separation(alto_destra)*1.5, "deg"),
                                       column_filters={'gmag': f'<{15}'}) # ho messo un limite di magnitudine per non scaricare milioni di stelle

        tbl_riquadro_esterno = riquadro_esterno[0] # questo è un riquadro leggermente più grande dove operare
        print("-----------------------------")

    mag_max = 14

    tbl_catalogate = tabella_catalogo(percorso_file_fits , mag_max)
    numero_stelle_catalogate.append(len(tbl_catalogate))
    if i == 1:
        tempo.append(0)
        t0 = header_dal_csv['TSTART']
        print("Tempo iniziale:",t0)
    else: tempo.append((header_dal_csv['TSTART']-t0)/np.float64(1e3))
    print(tbl_catalogate)
    distanze_singola_tabella = calcolo_distanze(tbl_trovate, tbl_catalogate, percorso_file_fits)
    distanze.extend(distanze_singola_tabella)


distanze_array = np.array(distanze)
numero_di_distanze = np.size(distanze_array)
tempo_array_ = np.array(tempo)
tempo_array = np.sort(tempo_array_)
print(tempo_array.shape)
numero_stelle_catalogate_array = np.array(numero_stelle_catalogate)
# definisco la cartella di output dell'array
output_dir = "/home/lorysimeone/tesi_magistrale/prove/analisi/distanze"
filename = os.path.join(output_dir, f'array_distanze_cazzata.csv')
np.savetxt(filename, distanze_array, delimiter=',', header='distanze_gradi', comments='')

plt.plot(tempo_array , numero_stelle_catalogate_array, marker='o', linestyle='-', linewidth=2, markersize=6)

plt.xlabel('Secondi')
plt.ylabel('Numero stelle catalogate')
plt.title('Numero di stelle del catalogo nel riquadro in funzione della run')
plt.grid(True, alpha=0.3)
plt.ylim(0, None)
plt.show()