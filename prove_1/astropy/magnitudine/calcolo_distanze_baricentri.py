import pandas as pd
import time
import random
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

# inizializzo Vizier con i suoi parametri di default

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1
)



catalogo = vizier
print(f"Questo è il catalogo: \n{catalogo}")

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

        # definisco i range di RA e DEC (in gradi) a partire dagli estremi in alto a destra e in basso a sinistra

        alto_destra = w.pixel_to_world(3072, 2048)
        basso_sinistra = w.pixel_to_world(0, 0)

        ra_min_ = alto_destra.ra.deg
        print(ra_min_)
        ra_max_ = basso_sinistra.ra.deg
        print(ra_max_)
        dec_min_ = alto_destra.dec.deg
        print(dec_min_)
        dec_max_ = basso_sinistra.dec.deg
        print(dec_max_)

        # seleziono le righe del catalogo esterno che soddisfano le condizioni del file e la magnitudine massima
        subset = tbl_riquadro_esterno[(tbl_riquadro_esterno['RAJ2000'] >= ra_min_) &
                            (tbl_riquadro_esterno['RAJ2000'] <= ra_max_) &
                            (tbl_riquadro_esterno['DEJ2000'] >= dec_min_) &
                            (tbl_riquadro_esterno['DEJ2000'] <= dec_max_) &
                            (tbl_riquadro_esterno['gmag'] <= magnitudine_massima)]
        print("riquadro esterno:" , tbl_riquadro_esterno)
        print(f"Subset: {subset}")


        return subset


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

    print(np.shape(coords_catalogate))

    # calcolo le distanze di tutti i centroidi da tutte le stelle catalogate
    distanze_minime = []
    corrispondenze = []
    righe_tabella_combinata = [] # lista per la tabella combinata

    for i, coord_trovata in enumerate(coords_trovate):
        # calcolo la distanza da tutte le stelle catalogate
        distanze_singola = coord_trovata.separation(coords_catalogate) # Calcola la distanza angolare tra la singola stella trovata
        # (coord_trovata) e tutte le stelle del catalogo (coords_catalogate). Restituisce un array di distanze angolari.
        if i == 1: print(np.shape(distanze_singola))
        # trovo la distanza minima e l'indice della stella più vicina
        distanza_minima = np.min(distanze_singola)

        distanze_minime.append(distanza_minima)

    distanze_gradi = [d.deg for d in distanze_minime]
    hdu_list.close()

    return distanze_gradi

# questa funzione restituisce la tabella delle sorgenti trovate
def analisi_image_segmentation(data):
    """
    Esegue image segmentation su un'immagine FITS e restituisce la tabella filtrata delle sorgenti

    Returns:
    astropy.table.Table: Tabella delle sorgenti filtrate con label riordinati
    """

    mean, median, std = sigma_clipped_stats(data, sigma=3.0) # nel caso me lo chiedessi, la std prima o dopo aver sottratto il fondo è la stessa

    # Lettura parametri
    parametri = {}
    with open('/home/lorysimeone/tesi_magistrale/prove/analisi/parametri_image_segmentation.txt', 'r') as file:
        next(file)  # Salta intestazione
        for riga in file:
            riga = riga.strip()
            if riga and not riga.startswith('#'):
                parametro, valore = riga.split()
                parametri[parametro] = float(valore) if '.' in valore else int(valore)

    # Convoluzione
    fwhm = parametri['fwhm']
    size = parametri['size']
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)

    # Sourcefinder
    t = parametri['threshold_sigma']
    # threshold = t * std # per adesso lascio stare questo metodo
    threshold = parametri['threshold_assoluta']
    n = parametri['pixel']

    finder = SourceFinder(npixels=n, progress_bar=True)
    segment_map = finder(convolved_data, threshold)

    # Catalogo sorgenti
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
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

def elabora_file_fits(percorso_file_):
    """Elabora un singolo file FITS
    Come parametro si mete la stringa del percorso
    Si ottiene la matrice dei pixel col fondo sottratto"""
    with fits.open(percorso_file_) as hdu_list_:
        image_data_ = hdu_list_[0].data
        # Fai qui le tue elaborazioni
        mean_, median_, std_ = sigma_clipped_stats(image_data_, sigma=3.0)
        image_data_ = image_data_ - median_

        return image_data_

# Leggo la lista
with open('/home/lorysimeone/tesi_magistrale/prove/analisi/lista_immagini_run_1.txt', 'r') as file:
    file_list = file.read().splitlines() # creo una lista di stringhe che sono i percorsi

# definisco la cartella di output dell'array
output_dir = "/home/lorysimeone/tesi_magistrale/prove/analisi/distanze"

n = 0
distanze = []

# Elaboro tutti i file

for percorso_file in file_list:

    data = elabora_file_fits(percorso_file)  # ottengo la matrice col fondo sottratto
    n = n + 1

    if n==1: # ricavo una tabella grande dal primo file per chiamare il sito solo una volta
        hdu_list_ = fits.open(percorso_file)
        # prendo un riquadro globale leggermente più grande del riquadro della pmc

        # coordinate centro
        image_header = hdu_list_[0].header
        ra_centro = image_header["RA"]
        dec_centro = image_header["DEC"]

        w = WCS(hdu_list_[0].header)  # creo un oggetto WCS usando l'header del file FITS
        alto_destra = w.pixel_to_world(3072, 2048)
        basso_sinistra = w.pixel_to_world(0, 0)

        ra_min = alto_destra.ra.deg
        ra_max = basso_sinistra.ra.deg
        dec_min = alto_destra.dec.deg
        dec_max = basso_sinistra.dec.deg

        larghezza = np.abs(ra_max - ra_min)*1.5
        altezza = np.abs(dec_max - dec_min)*1.5

        riquadro_esterno = vizier.query_region(coord.SkyCoord(ra=ra_centro, dec=dec_centro,
                                                              unit=(u.deg, u.deg),
                                                              frame='icrs'),
                                               width=larghezza * u.deg,  # <-- Larghezza in RA
                                               height=altezza * u.deg,  # <-- Altezza in Dec
                                               column_filters={'gmag': f'<{15}'}) # ho messo un limite di magnitudine per non scaricare milioni di stelle
        tbl_riquadro_esterno = riquadro_esterno[0]

    print(n)
    if n<=5:
        print(f"Elaborando: {percorso_file.split('/')[-1]}")  # Mostra solo il nome file
        # mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        std = np.std(data)
        print(f"std: {std}")

    # Chiamata alla funzione
    tbl_trovate = analisi_image_segmentation(data)


    # Ricavo la tabella delle stelle catalogate

    mag_max = 14

    tbl_catalogate = tabella_catalogo(percorso_file, mag_max)
    print(tbl_catalogate)

    distanze_singola_tabella = calcolo_distanze(tbl_trovate, tbl_catalogate, percorso_file)
    distanze.extend(distanze_singola_tabella)

    if n == 3: break


distanze_array = np.array(distanze)
numero_di_distanze = np.size(distanze_array)
filename = os.path.join(output_dir, f'array_distanze.csv')
np.savetxt(filename, distanze_array, delimiter=',', header='distanze_gradi', comments='')

# Calcola le statistiche di base
distanza_media = np.mean(distanze_array)
distanza_mediana = np.median(distanze_array)
distanza_std = np.std(distanze_array)

print("=== STATISTICHE DISTANZE ===")
print(f"Distanza media: {distanza_media:.6f} gradi")
print(f"Distanza mediana: {distanza_mediana:.6f} gradi")
print(f"Deviazione standard: {distanza_std:.6f} gradi")
print(f"Numero totale di stelle: {len(distanze_array)}")

# Statistiche aggiuntive
print(f"\n=== STATISTICHE AGGIUNTIVE ===")
print(f"Minima distanza: {np.min(distanze_array):.6f} gradi")
print(f"Massima distanza: {np.max(distanze_array):.6f} gradi")
print(f"25° percentile: {np.percentile(distanze_array, 25):.6f} gradi")
print(f"75° percentile: {np.percentile(distanze_array, 75):.6f} gradi")


# Funzione gaussiana per il fit
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# Crea l'istogramma
plt.figure(figsize=(12, 8))

# Istogramma principale (in gradi)
n, bins, patches = plt.hist(distanze_array, bins=250, density=True,
                            alpha=0.7, color='skyblue', edgecolor='black',
                            label='Dati osservati')

# Calcola il centro dei bin
bin_centers = (bins[:-1] + bins[1:]) / 2

# Fit gaussiano
try:
    # Stima iniziale dei parametri - CORRETTO
    initial_guess = [np.max(n), distanza_mediana, distanza_std]

    # Esegui il fit - CORRETTO
    popt, pcov = curve_fit(gaussian, bin_centers, n, p0=initial_guess)

    # Parametri del fit - CORRETTO
    amp_fit, mu_fit, sigma_fit = popt

    # Crea la curva gaussiana fitted - CORRETTO
    x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
    y_fit = gaussian(x_fit, *popt)

    # Plot della gaussiana fitted
    plt.plot(x_fit, y_fit, 'r-', linewidth=2,
             label=f'Gaussiana fit\nμ = {mu_fit:.6f} gradi\nσ = {sigma_fit:.6f} gradi')

    # Aggiungi linee verticali per media e mediana
    plt.axvline(distanza_media, color='green', linestyle='--',
                linewidth=2, label=f'Media = {distanza_media:.6f} gradi')
    plt.axvline(distanza_mediana, color='orange', linestyle='--',
                linewidth=2, label=f'Mediana = {distanza_mediana:.6f} gradi')

    print(f"\n=== FIT GAUSSIANO ===")
    print(f"Ampiezza: {amp_fit:.4f}")
    print(f"Media (μ): {mu_fit:.6f} gradi")
    print(f"Deviazione standard (σ): {sigma_fit:.6f} gradi")

except Exception as e:
    print(f"Errore nel fit gaussiano: {e}")
    # Plot solo delle linee verticali se il fit fallisce
    plt.axvline(distanza_media, color='green', linestyle='--',
                linewidth=2, label=f'Media = {distanza_media:.6f} gradi')
    plt.axvline(distanza_mediana, color='orange', linestyle='--',
                linewidth=2, label=f'Mediana = {distanza_mediana:.6f} gradi')

# Personalizza il grafico
plt.xlabel('Distanza [gradi]', fontsize=12)
plt.ylabel('Densità di probabilità', fontsize=12)
plt.title('Distribuzione delle distanze centroidi-catalogo', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Aggiungi testo con le statistiche
textstr = f'N = {len(distanze_array)} stelle\n' \
          f'Media = {distanza_media:.6f}°\n' \
          f'Mediana = {distanza_mediana:.6f}°\n' \
          f'Std = {distanza_std:.6f}°'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.show()

# Istogramma cumulativo (in gradi)
plt.figure(figsize=(10, 6))
counts, bin_edges = np.histogram(distanze_array, bins=50)
cumulative = np.cumsum(counts) / len(distanze_array)

plt.plot(bin_edges[1:], cumulative, 'b-', linewidth=2, label='Cumulativa')
plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50%')
plt.axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90%')

plt.xlabel('Distanza [gradi]', fontsize=12)
plt.ylabel('Frazione cumulativa', fontsize=12)
plt.title('Distribuzione cumulativa delle distanze', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# Calcola alcuni percentili utili
p50 = np.percentile(distanze_array, 50)
p90 = np.percentile(distanze_array, 90)
p95 = np.percentile(distanze_array, 95)

print(f"\n=== PERCENTILI ===")
print(f"50° percentile (mediana): {p50:.6f} gradi")
print(f"90° percentile: {p90:.6f} gradi")
print(f"95° percentile: {p95:.6f} gradi")

plt.tight_layout()
plt.show()