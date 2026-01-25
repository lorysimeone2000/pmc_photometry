import pandas as pd
#pd.set_option('display.show_dimensions', False)
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
# Set up wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
import matplotlib.cm as cm
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
# warning
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning) # Sopprime il warning FITSFixedWarning

# inizializzo Vizier con i suoi parametri di default

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1
)

catalogo = vizier
print(f"Questo è il catalogo: \n{catalogo}")

catalogs = Vizier.get_catalogs("II/389/ps1_dr2")
print(catalogs)
print(f"Descrizione gmag: {catalogs[0]["gmag"].description}")

# Lettura parametri
parametri = {}
with open('/home/lorysimeone/tesi_magistrale/prove/analisi/parametri_image_segmentation.txt', 'r') as file:
    next(file)  # Salta intestazione
    for riga in file:
        riga = riga.strip()
        if riga and not riga.startswith('#'):
            parametro, valore = riga.split()
            parametri[parametro] = float(valore) if '.' in valore else int(valore)


fwhm = parametri['fwhm']
size = parametri['size']
t = parametri['threshold_sigma']
# threshold = t * std # per adesso lascio stare questo metodo
threshold = parametri['threshold_assoluta']
n = parametri['pixel']

# questa funzione restituisce la tabella delle sorgenti trovate
def analisi_image_segmentation(data_):
    """
    Esegue image segmentation su un'immagine FITS e restituisce la tabella filtrata delle sorgenti
    Il fondo deve essere rimosso

    Returns:
    astropy.table.Table: Tabella delle sorgenti filtrate con label riordinati
    """

    mean, median, std = sigma_clipped_stats(data_, sigma=3.0) # nel caso me lo chiedessi, la std prima o dopo aver sottratto il fondo è la stessa

    # Convoluzione
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data_, kernel)
    mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)

    # Sourcefinder
    t = parametri['threshold_sigma']
    # threshold = t * std # per adesso lascio stare questo metodo
    threshold = parametri['threshold_assoluta']
    n = parametri['pixel']

    finder = SourceFinder(npixels=n, progress_bar=True)
    segment_map = finder(convolved_data, threshold)

    # Catalogo sorgenti
    cat = SourceCatalog(data_, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'

    # filtraggio sorgenti
    soglia_assoluta = 2.5
    soglia_relativa = 0.4

    indici_validi = []
    indici_non_validi = []

    for i, sorgente in enumerate(tbl):
        label = sorgente['label']
        mask_sorgente = (segment_map.data == label)
        valori_originali = data[mask_sorgente]

        pixel_sopra_soglia_assoluta = np.sum(valori_originali > soglia_assoluta)
        pixel_sopra_soglia_relativa = np.sum(valori_originali > soglia_relativa * sorgente['max_value'])

        if pixel_sopra_soglia_assoluta >= 2 and pixel_sopra_soglia_relativa >= 2:
            indici_validi.append(i)

    # Creazione tabella filtrata
    tbl_filtrato = tbl[indici_validi]
    new_labels_validi = np.arange(1, len(tbl_filtrato) + 1)
    tbl_filtrato['label'] = new_labels_validi

    # print(f"Sorgenti dopo filtro: {len(tbl_filtrato)} / {len(tbl)}")

    return tbl_filtrato


# questa funzione restituisce la tabella delle sorgenti catalogate nel riquadro
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
        # trovo gli estremi

        w = WCS(hdu_list_[0].header)  # creo un oggetto WCS usando l'header del file FITS,
        # che contiene le informazioni per le trasformazioni di coordinate

        alto_destra = w.pixel_to_world(3072, 2048)
        #print(f"Coordinate in alto a destra: {alto_destra}")
        basso_sinistra = w.pixel_to_world(0, 0)
        # print(f"Coordinate in basso a sinistra: {basso_sinistra}")

        # Definisco i range di RA e DEC (in gradi) a partire dagli estremi in alto a destra e in basso a sinistra

        ra_min = alto_destra.ra.deg  # oppure .hour per avere in ore
        # print(f"RA_min: {ra_min}°")
        ra_max = basso_sinistra.ra.deg
        # print(f"RA_max: {ra_max}°")
        dec_min = alto_destra.dec.deg
        # print(f"DEC_min: {dec_min}°")
        dec_max = basso_sinistra.dec.deg
        # print(f"DEC_max: {dec_max}°")

        # creo la tabella del catalogo

        # coordinate centro
        image_header = hdu_list_[0].header
        ra_centro = image_header["RA"]
        dec_centro = image_header["DEC"]

        larghezza = np.abs(ra_max - ra_min)
        altezza = np.abs(dec_max - dec_min)

        # Vizier è già stato inizializzato come "vizier"

        riquadro = vizier.query_region(coord.SkyCoord(ra=ra_centro, dec=dec_centro,
                                                      unit=(u.deg, u.deg),
                                                      frame='icrs'),
                                                      width=larghezza * u.deg,  # <-- Larghezza in RA
                                                      height=altezza * u.deg,  # <-- Altezza in Dec
                                                      column_filters={'gmag': f'<{magnitudine_massima}'}, )

        tbl_catalogo = riquadro[0]
        return tbl_catalogo


def calcola_distanza_media_centroidi(tbl_trovate, tbl_catalogate, image_file):
    """
    Calcola la distanza media dei centroidi rispetto alle stelle catalogate più vicine

    Parameters:
    tbl_trovate (Table): Tabella delle sorgenti trovate con image segmentation
    tbl_catalogate (Table): Tabella delle stelle del catalogo
    image_file (str): Percorso del file FITS

    Returns:
    float: Distanza media, mediana, deviazione standard in gradi
    """

    # Carica il WCS dall'immagine
    hdu_list = fits.open(image_file)
    w = WCS(hdu_list[0].header)

    # Converti i centroidi pixel in coordinate celesti
    coords_trovate = w.pixel_to_world(tbl_trovate['xcentroid'], tbl_trovate['ycentroid'])

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

        print(f"ra_values tipo: {type(ra_values)}, forma: {ra_values.shape}")
        print(f"dec_values tipo: {type(dec_values)}, forma: {dec_values.shape}")

        # creo SkyCoord con i valori puri specificando le unità
        coords_catalogate = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)

    except Exception as e:
        print(f"Errore nell'approccio principale: {e}")
        # APPROCCIO ALTERNATIVO: uso direttamente i valori senza moltiplicare per unità
        coords_catalogate = SkyCoord(ra=tbl_catalogate['RAJ2000'],
                                     dec=tbl_catalogate['DEJ2000'],
                                     unit=u.deg)

    # calcolo le distanze di tutti i centroidi da tutte le stelle catalogate
    distanze_minime = []
    corrispondenze = []
    righe_tabella_combinata = [] # lista per la tabella combinata

    for i, coord_trovata in enumerate(coords_trovate):
        # calcolo la distanza da tutte le stelle catalogate
        distanze_singola = coord_trovata.separation(coords_catalogate) # Calcola la distanza angolare tra la singola stella trovata
        # (coord_trovata) e tutte le stelle del catalogo (coords_catalogate). Restituisce un array di distanze angolari.

        # trovo la distanza minima e l'indice della stella più vicina
        distanza_minima = np.min(distanze_singola)
        idx_minimo = np.argmin(distanze_singola)

        distanze_minime.append(distanza_minima)
        corrispondenze.append({
            'centroide_idx': i,
            'catalogo_idx': idx_minimo,
            'distanza_gradi': distanza_minima.deg,
            'coord_centroide': coord_trovata,
            'coord_catalogo': coords_catalogate[idx_minimo]
        })

        # Crea una riga per la tabella combinata
        riga_combinata = {
            'idx_trovata': i,
            'idx_catalogo': idx_minimo,
            'distanza': distanza_minima.deg,
            'ra_trovata': coord_trovata.ra.deg,
            'dec_trovata': coord_trovata.dec.deg,
            'ra_catalogo': coords_catalogate[idx_minimo].ra.deg,
            'dec_catalogo': coords_catalogate[idx_minimo].dec.deg,
            'xcentroid': tbl_trovate['xcentroid'][i],
            'ycentroid': tbl_trovate['ycentroid'][i],
            'gmag_catalogo': tbl_catalogate['gmag'][idx_minimo] if 'gmag' in tbl_catalogate.colnames else 0
        }

        righe_tabella_combinata.append(riga_combinata)

        # creo la tabella combinata
        tabella_combinata = Table(rows=righe_tabella_combinata)

            # calcolo statistiche (in gradi, senza unità per compatibilità)
    distanze_gradi = [d.deg for d in distanze_minime]
    distanza_media = np.mean(distanze_gradi)
    distanza_mediana = np.median(distanze_gradi)
    distanza_std = np.std(distanze_gradi)

    return distanza_media, distanza_mediana, distanza_std, tabella_combinata

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"  # prima immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median # tolgo il fondo
data = image_data



# creo la tabella delle stelle TROVATE con image segmentation

tbl_trovate = analisi_image_segmentation(data)
print(tbl_trovate)

# creo la tabella delle stelle CATALOGATE su Vizier sotto una certa magnitudine

mag_max = 14

tbl_catalogate = tabella_catalogo(image_file,mag_max)
print(tbl_catalogate)

distanza_media,  distanza_mediana, distanza_std, tabella_combinata_ = calcola_distanza_media_centroidi(tbl_trovate, tbl_catalogate, image_file)

print(f"Distanza media {distanza_media}")
print(f"Distanza mediana {distanza_mediana}")
print(f"Distanza std {distanza_std}")
print(f"Tabella combinata:\n {tabella_combinata_}")