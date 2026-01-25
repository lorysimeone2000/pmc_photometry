import pandas as pd
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import os
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
import warnings
from astropy.wcs import FITSFixedWarning

# pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from matplotlib.colors import LogNorm  # permette di avere la scala logaritmica
from scipy.optimize import curve_fit
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
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
# warning
import warnings
from astropy.io.fits.verify import VerifyWarning
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.utils.exceptions import AstropyUserWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)

from pathlib import Path

# Ignora i warning specifici sui fix automatici degli header FITS
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)

# inizializzo Vizier con i suoi parametri di default
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)


# --- 1. FUNZIONI HELPER ---

def leggi_file_parametri(percorso):
    """Legge il file dei parametri in un dizionario."""
    parametri = {}
    if not os.path.exists(percorso):
        return {}
    with open(percorso, 'r') as file:
        next(file, None)
        for riga in file:
            riga = riga.split('#')[0].strip()
            if riga:
                parts = riga.split()
                if len(parts) >= 2:
                    parametro = parts[0]
                    valore_str = parts[1]
                    try:
                        valore = float(valore_str) if '.' in valore_str else int(valore_str)
                        parametri[parametro] = valore
                    except ValueError:
                        pass
    return parametri


def elabora_file_fits(percorso_file_):
    """Carica il FITS e sottrae il fondo."""
    # memmap=False previene errori con BZERO/BSCALE
    with fits.open(percorso_file_, memmap=False) as hdu_list_:
        image_data_ = hdu_list_[0].data
        w_ = WCS(hdu_list_[0].header)
        mean_, median_, std_ = sigma_clipped_stats(image_data_, sigma=3.0)
        image_data_ = image_data_ - median_
        return image_data_, median_, w_


def calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pixel, k=2.5, r_min=3.5):
    """
    Esegue l'INTERA pipeline Kron per una singola stella:
    1. Calcola il primo momento dai pixel forniti.
    2. Determina il raggio di Kron.
    3. Esegue la fotometria di apertura sull'immagine originale.

    Returns:
    - float: Il flusso finale integrato (Kron Flux).
    """
    # 1. Calcolo del Raggio
    somma_intensita = np.sum(valori_pixel)

    if somma_intensita <= 0:
        return np.nan, np.nan  # Impossibile calcolare raggio su flusso nullo/negativo

    somma_momenti = np.sum(valori_pixel * distanze_pixel)
    r_1 = somma_momenti / somma_intensita

    # Raggio finale con soglia minima
    r_kron_finale = max(k * r_1, r_min)

    # 2. Misura Fotometrica (Integrazione)
    # Creiamo l'apertura circolare con il raggio calcolato
    aper = CircularAperture((xc, yc), r=r_kron_finale)

    # Eseguiamo la fotometria sull'immagine completa 'data'
    phot = aperture_photometry(data, aper)

    return phot['aperture_sum'][0], r_kron_finale  # Ritorna (flusso, raggio)


def tabella_catalogo(image_file_,
                     magnitudine_massima):  # questa funzione restituisce la tabella delle sorgenti catalogate nel riquadro della pmc
    """
    Seleziona le stelle del catalogo che rientrano nel riquadro e che sono sotto una certa magnitudine
    e che non stanno entro 7 pixel dal bordo

    Parameters:
    image_file_ (string): percorso del file da cui estrarre l'header e quindi le coordinate dell'astrometria
    magnitudine_massima (float): magnitudina massima presa dal catalogo

    Returns:
    astropy.table.Table: Tabella delle stelle del catalogo che rientrano nel riquadro e che sono sotto una certa magnitudine
    """

    hdu_list_ = fits.open(image_file_)

    wcs = WCS(hdu_list_[0].header)  # creo un oggetto WCS usando l'header del file FITS,
    # che contiene le informazioni per le trasformazioni di coordinate
    data_ = hdu_list_[0].data

    # Definisco il bordo
    bordo = 7
    h, w = data_.shape[0], data_.shape[1]

    # tolgo le stelle che non vorrei siano prese da Vizier
    # mag_limite_tra_hipparco_e_vizier = 7.
    tbl_catalogo_vizier = tbl_riquadro_esterno_vizier[
        (tbl_riquadro_esterno_vizier['gmag'] >= mag_limite_tra_hipparco_e_vizier)
    ]

    '''# Esploro il catalogo
    print("\n=== INFORMAZIONI DEL CATALOGO ===")
    print(f"Numero di stelle nel catalogo: {len(tbl_catalogo_hipparco)}")
    print(f"Nomi delle colonne: {tbl_catalogo_hipparco.colnames}")'''

    # adesso mi costruisco una mia tabella astropy complessiva

    # ricreo le colonne dei cataloghi

    nome_catalogo_vizier = []
    for i in range(len(tbl_catalogo_vizier)): nome_catalogo_vizier.append("II/389/ps1_dr2")
    colonne_vizier = {
        'ID': tbl_catalogo_vizier['objID'],
        'RAJ2000': tbl_catalogo_vizier['RAJ2000'],
        'DEJ2000': tbl_catalogo_vizier['DEJ2000'],
        'Mag': tbl_catalogo_vizier['gmag'],
        'Catalogo': nome_catalogo_vizier
    }

    nome_catalogo_hipparco = []
    for i in range(len(tbl_catalogo_hipparco)): nome_catalogo_hipparco.append("I/239/hip_main")
    colonne_hipparco = {
        'Catalogo': nome_catalogo_hipparco,
        'ID': tbl_catalogo_hipparco['HIP'],
        'RAJ2000': tbl_catalogo_hipparco['_RAJ2000'],
        'DEJ2000': tbl_catalogo_hipparco['_DEJ2000'],
        'Mag': tbl_catalogo_hipparco['Vmag'],
    }

    t1 = Table(colonne_vizier)
    t2 = Table(colonne_hipparco)

    tbl_unita_estesa = vstack([t1, t2])
    tbl_unita_estesa['Mag'].description = 'Magnitudine AB nel filtro g di Pan-STARRS'
    # print("Tabella estesa:\n", tbl_unita_estesa)

    # --- MODIFICA QUI ---

    # Invece di proiettare gli angoli sul cielo, proiettiamo il catalogo sui pixel

    # 1. Creiamo un oggetto SkyCoord per tutte le stelle della tabella unita
    coords_catalogo = SkyCoord(
        ra=tbl_unita_estesa['RAJ2000'],
        dec=tbl_unita_estesa['DEJ2000'],
        unit=u.deg
    )

    # 2. Convertiamo tutte le coordinate celesti in coordinate pixel (x, y)
    x_pix, y_pix = wcs.world_to_pixel(coords_catalogo)

    # 3. Creiamo la maschera usando le coordinate pixel
    # Vogliamo le stelle che sono DENTRO i bordi:
    # x deve essere > bordo E < (larghezza - bordo)
    # y deve essere > bordo E < (altezza - bordo)
    mask_bordo = (
            (x_pix >= bordo) &
            (x_pix < (w - bordo)) &
            (y_pix >= bordo) &
            (y_pix < (h - bordo))
    )

    # 4. Applichiamo la maschera alla tabella
    tbl_cataloghi_ = tbl_unita_estesa[mask_bordo]
    # tbl_cataloghi_ = tbl_unita_estesa

    # Opzionale: Se vuoi salvare anche le coordinate pixel nella tabella risultante
    # tbl_cataloghi_['x_pix'] = x_pix[mask_bordo]
    # tbl_cataloghi_['y_pix'] = y_pix[mask_bordo]

    hdu_list_.close()  # È buona norma chiudere il file fits

    return tbl_cataloghi_


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

    print("numero di stelle catalogate", np.shape(coords_catalogate))

    # calcolo le distanze di tutti i centroidi da tutte le stelle catalogate
    distanze_minime = []
    corrispondenze = []
    righe_tabella_combinata = []  # lista per la tabella combinata

    for i, coord_trovata in enumerate(coords_trovate):
        # calcolo la distanza da tutte le stelle catalogate
        distanze_singola = coord_trovata.separation(
            coords_catalogate)  # Calcola la distanza angolare tra la singola stella trovata
        # (coord_trovata) e tutte le stelle del catalogo (coords_catalogate). Restituisce un array di distanze angolari.
        # if i == 1: print(np.shape(distanze_singola))
        # trovo la distanza minima e l'indice della stella più vicina
        distanza_minima = np.min(distanze_singola)

        distanze_minime.append(distanza_minima)

    distanze_gradi = [d.deg for d in distanze_minime]
    hdu_list.close()

    return distanze_gradi


def esegui_fotometria_variabile(data, positions, raggi):
    """
    Helper generico per altri tipi di flusso (es. Raggio Max)
    che richiedono raggi variabili.
    """
    flussi = []
    for (xc, yc), r in zip(positions, raggi):
        if r > 0 and not np.isnan(r):
            aper = CircularAperture((xc, yc), r=r)
            phot = aperture_photometry(data, aper)
            flussi.append(phot['aperture_sum'][0])
        else:
            flussi.append(np.nan)
    return flussi


def salva_csv_con_header_fits(dataframe, header_fits, filename, nome_file_fits, parametri_seg=None):
    """Salva CSV con header FITS come commenti."""
    with open(filename, 'w') as f:
        f.write("# Header FITS:\n")
        for key, value in header_fits.items():
            clean_val = str(value).replace('\n', ' ')
            f.write(f"# {key}: {clean_val}\n")
        f.write(f"# PERCORSO_FILE: {nome_file_fits}\n")
        f.write("#\n# PARAMETRI SEGMENTAZIONE:\n")
        if parametri_seg:
            for key, value in parametri_seg.items():
                f.write(f"# {key}: {value}\n")
        f.write("#\n")
        dataframe.to_csv(f, index=False)


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
    return valore


def leggi_header_da_csv(filename):
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            else:
                break
    return header_dict


# --- 2. FUNZIONE DI ANALISI PRINCIPALE ---

def analisi_image_segmentation(percorso_file_, parametri_globali):
    """
    Esegue image segmentation su un'immagine FITS e restituisce la tabella filtrata.
    """
    # A. Setup
    data, fondo_iniziale, w = elabora_file_fits(percorso_file_)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # B. Parametri
    fwhm = parametri_globali.get('fwhm', 3.0)
    size = parametri_globali.get('size', 5)
    threshold = parametri_globali.get('threshold_assoluta', 3.0)
    pixel_n = parametri_globali.get('pixel', 5)

    # C. Segmentazione
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    finder = SourceFinder(npixels=pixel_n, progress_bar=False)
    segment_map = finder(convolved_data, threshold)

    # D. Catalogo Base
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    if len(tbl) == 0:
        return tbl, parametri_globali

    for col in ['xcentroid', 'ycentroid', 'kron_flux']:
        tbl[col].info.format = '.2f'

    livello_saturazione = 255 - fondo_iniziale - median
    tbl['saturazione'] = np.where(tbl['max_value'] >= livello_saturazione, 'SI', 'NO')

    # E. CALCOLO E FILTRAGGIO (Ciclo Unico)
    K_KRON = 2.5
    R_MIN_KRON = 3.5
    soglia_assoluta = 2.5
    soglia_relativa = 0.05
    bordo = 7
    ny, nx = data.shape

    lista_raggi_max = []  # Serve ancora per l'altro flusso
    kron_manuale_seg = []  # qui salverò i kron delle segmentazioni
    kron_manuale_aper = []  # qui salverò i kron delle aperture
    raggi_kron_aper = []
    mask_keep = []

    # Iteriamo sulle proprietà
    for prop in cat:
        xc, yc = prop.xcentroid, prop.ycentroid  # coordinate pixel del centroide

        # 1. Check Bordo
        dentro_riquadro = (xc >= bordo) and (xc < nx - bordo) and \
                          (yc >= bordo) and (yc < ny - bordo)

        if not dentro_riquadro:
            lista_raggi_max.append(0.5)
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan)
            raggi_kron_aper.append(np.nan)
            mask_keep.append(False)
            continue

        # 2. Recupero Pixel (Ottimizzato con Slice)
        slices = prop.slices  # prendo il rettangolo minimo per velocizzare
        cutout_seg = segment_map.data[slices]
        y_loc, x_loc = np.where(cutout_seg == prop.label)  # seleziono SOLO i pixel che appartengono alla segmentazione

        ypix = y_loc + slices[0].start
        xpix = x_loc + slices[1].start

        valori_pixel = data[ypix, xpix]

        if len(valori_pixel) == 0:
            lista_raggi_max.append(0.5)
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan)
            raggi_kron_aper.append(np.nan)
            mask_keep.append(False)
            continue

        # 3. Calcoli Geometrici di Base
        distanze_pix = np.hypot(xpix - xc, ypix - yc)

        # Raggio Massimo (per "somma apertura ultimo pixel")
        r_max_pix = np.max(distanze_pix) if len(distanze_pix) > 0 else 0.5
        r_max_pix = max(r_max_pix, 0.5)
        lista_raggi_max.append(r_max_pix)

        r_int = int(np.ceil(r_max_pix))  # Arrotondiamo per eccesso per il ritaglio array
        y_min_box = int(max(0, yc - r_int))
        y_max_box = int(min(data.shape[0], yc + r_int + 1))
        x_min_box = int(max(0, xc - r_int))
        x_max_box = int(min(data.shape[1], xc + r_int + 1))
        cutout_box = data[y_min_box:y_max_box, x_min_box:x_max_box]

        # Creiamo una griglia di coordinate per il ritaglio
        y_grid, x_grid = np.ogrid[y_min_box:y_max_box, x_min_box:x_max_box]

        # Calcoliamo la distanza di ogni pixel del ritaglio dal centroide reale
        distanze_box = np.hypot(x_grid - xc, y_grid - yc)

        # Selezioniamo solo i pixel dentro il cerchio
        mask_circle = distanze_box <= r_max_pix

        # Questo array contiene i valori di TUTTI i pixel dentro il cerchio (non solo la segmentazione)
        pixels_apertura_completa = cutout_box[mask_circle]
        distanze_apertura_completa = distanze_box[mask_circle]  # Le distanze corrispondenti (distanze_pixel)

        # calcolo flusso kron sull'apertura massima
        flusso_kron_apertura, raggio_usato = calcola_flusso_kron_completo(
            data=data,
            xc=xc,
            yc=yc,
            valori_pixel=pixels_apertura_completa,
            distanze_pixel=distanze_apertura_completa,
            k=K_KRON,
            r_min=R_MIN_KRON
        )
        kron_manuale_aper.append(flusso_kron_apertura)
        raggi_kron_aper.append(raggio_usato)

        # --- FLUSSO KRON COMPLETO (Helper Function) ---
        # Qui calcoliamo TUTTO: raggio e flusso finale
        flusso_kron_seg, raggio_valore = calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pix,
                                                                      k=K_KRON, r_min=R_MIN_KRON)
        kron_manuale_seg.append(flusso_kron_seg)
        # ----------------------------------------------

        # 4. Check Soglie (Filtraggio Qualità)
        pixel_sopra_soglia_assoluta = np.sum(valori_pixel > soglia_assoluta)
        pixel_sopra_soglia_relativa = np.sum(valori_pixel > soglia_relativa * prop.max_value)

        is_good = (pixel_sopra_soglia_assoluta >= 3) and (pixel_sopra_soglia_relativa >= 2)
        mask_keep.append(is_good)

    # F. ASSEGNAZIONE E CALCOLO RIMANENTE

    # 1. Assegnazione Kron (Già calcolato nel ciclo!)
    tbl['kron_manuale_seg'] = kron_manuale_seg
    tbl['kron_manuale_aper'] = kron_manuale_aper
    tbl['raggio_kron_aper'] = raggi_kron_aper

    # 2. Calcolo "Somma Apertura Ultimo Pixel" (Ancora da fare massivamente)
    positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))
    tbl['somma_apertura_ultimo_pixel'] = esegui_fotometria_variabile(data, positions, lista_raggi_max)

    # Formattazione
    tbl['somma_apertura_ultimo_pixel'].info.format = '%.2f'
    tbl['kron_manuale_seg'].info.format = '%.2f'
    tbl['kron_manuale_aper'].info.format = '%.2f'
    tbl['raggio_kron_aper'].info.format = '%.2f'

    # G. FILTRAGGIO FINALE
    tbl_filtrato = tbl[mask_keep]

    if len(tbl_filtrato) > 0:
        tbl_filtrato['label'] = np.arange(1, len(tbl_filtrato) + 1)

    return tbl_filtrato, parametri_globali


# --- 3. BLOCCO DI ESECUZIONE (MAIN) ---

if __name__ == "__main__":

    soglia_correlazione = 0.003349 * u.deg

    try:
        run = int(input("Quale run vuoi elaborare: "))
    except ValueError:
        print("Input non valido.")
        exit()

    file_parametri = '/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt'
    parametri_caricati = leggi_file_parametri(file_parametri)

    with open(f'/home/lorysimeone/tesi_magistrale/prove_2/liste_percorsi_run/lista_immagini_run_{run}.txt',
              'r') as file:
        file_list = file.read().splitlines()

    cartella_base = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle"
    output_dir = os.path.join(cartella_base, f"tabelle_unite/tabelle_unite_run_{run}")

    # Se vuoi sovrascrivere i file originali, metti uguale a output_dir
    # Per sicurezza qui uso la STESSA cartella, ma i file verranno sovrascritti
    # oppure puoi salvare con _updated se preferisci.
    # Per ora salviamo con lo stesso nome per avere un unico set di dati coerente.
    output_dir_final = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    numero_stelle = []
    n = 0

    with open(f'/home/lorysimeone/tesi_magistrale/prove_2/liste_percorsi_run/lista_immagini_run_{run}.txt',
              'r') as file:
        file_list = file.read().splitlines()  # creo una lista di stringhe che sono i percorsi

    i = 0
    # Itera su tutti i file fits
    print(f"Inizio elaborazione Fase 1 (Segmentazione & Unione) per {len(file_list)} files...")

    for percorso_file in tqdm(file_list, desc="Elaborazione Files"):

        # Creazione tabella catalogo

        i += 1
        hdu_list = fits.open(percorso_file)
        image_header = hdu_list[0].header

        # tbl_trovate = Table.from_pandas(dataframe)
        if i == 1:  # chiamo il sito una volta sola su un'immagine più grande per non mandarlo in down
            # header = leggi_header_da_csv(filename)

            # prendo un riquadro globale leggermente più grande del riquadro della pmc

            # coordinate centro
            ra_centro = image_header["RA"]
            dec_centro = image_header["DEC"]
            data = hdu_list[0].data

            w = WCS(hdu_list[0].header)  # creo un oggetto WCS usando l'header del file FITS
            alto_destra = w.pixel_to_world(3071, 2047)
            alto_sinistra = w.pixel_to_world(3071, 0)
            # print(f"Coordinate in alto a destra: {alto_destra}")
            basso_sinistra = w.pixel_to_world(0, 0)
            basso_destra = w.pixel_to_world(0, 2047)

            centro = SkyCoord(ra_centro, dec_centro, unit=u.deg)

            # creo un riquadro esterno leggermente più grande per assicurarmi che anche gli elementi successivi della run
            # rientrino nella query, poi farò il taglio preciso nella tabella astropy
            riquadro_esterno_vizier = vizier.query_region(coord.SkyCoord(ra=ra_centro, dec=dec_centro,
                                                                         unit=(u.deg, u.deg),
                                                                         frame='icrs'),
                                                          radius=Angle(centro.separation(alto_destra) * 1.5, "deg"),
                                                          column_filters={'gmag': f'<{15}'},
                                                          )  # ho messo un limite di magnitudine per non scaricare milioni di stelle
            tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]

            # aggiungo il catalogo Hipparco per le magnitudini inferiori a 7

            file_hipparco = "/home/lorysimeone/tesi_magistrale/prove_2/cataloghi_scaricati/hipparco.fit"

            # Apro il catalogo in formato fit, lo faccio solo una volta

            hdu_list_hipparco = fits.open(file_hipparco)
            # print("Info catalogo Hipparco: \n", hdu_list_hipparco)

            # I dati sono nella seconda estensione (V_SO_catalog), non nella prima
            table_data = Table(hdu_list_hipparco[1].data)  # Uso l'indice 1 per la seconda estensione
            # tolgo le stelle che non vorrei siano prese da Hipparco
            mag_limite_tra_hipparco_e_vizier = 7.
            tbl_catalogo_hipparco = table_data[(table_data['Vmag']) < mag_limite_tra_hipparco_e_vizier]

            # print("-----------------------------")

        # print(f"Elaborando {percorso_file}")
        # print("\n")

        mag_max = 15

        tbl_catalogate = tabella_catalogo(percorso_file,
                                          mag_max)  # tabelle del catalogo all'interno della specifica run

        # print(f"Trovate {len(tbl_catalogate)} stelle dei cataloghi nel riquadro {i}")

        # Creazione tabella segmentazione

        n += 1

        # Analisi
        tbl, _ = analisi_image_segmentation(percorso_file, parametri_caricati)
        numero_stelle.append(len(tbl))

        # Header e Salvataggio
        header_fits = fits.getheader(percorso_file)

        # Selezione colonne dinamica
        all_cols = tbl.colnames
        cols_base = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']
        cols_finali = []
        # 1. Aggiungi le colonne base (se presenti)
        cols_finali.extend([c for c in cols_base if c in all_cols])
        # 2. Aggiungi 'saturazione'
        if 'saturazione' in all_cols:
            cols_finali.append('saturazione')
        # 3. Aggiungi 'kron_flux' (SUBITO DOPO SATURAZIONE)
        if 'kron_flux' in all_cols:
            cols_finali.append('kron_flux')
        # 4. Aggiungi tutte le altre colonne dinamiche (per es. altri flussi calcolati)
        # Cerchiamo di prendere tutto ciò che segue 'saturazione' nel file originale
        try:
            if 'saturazione' in all_cols:
                idx_start = all_cols.index('saturazione') + 1
                cols_extra = all_cols[idx_start:]
            else:
                # Se manca saturazione, controlliamo tutto tranne le base
                cols_extra = [c for c in all_cols if c not in cols_base]
            for c in cols_extra:
                # Aggiungiamo solo se non è già stata inserita (evita duplicati di kron_flux)
                if c not in cols_finali:
                    cols_finali.append(c)
        except ValueError:
            pass
        tbl_save = tbl[cols_finali]

        tbl_segmentazione = tbl_save

        df_trovate = tbl_segmentazione.to_pandas()
        df_catalogate = tbl_catalogate.to_pandas()

        # --- LOGICA DI SELEZIONE COLONNE DINAMICA (Corretta e Completa) ---
        all_cols = df_trovate.columns.tolist()
        cols_base = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']

        try:
            # Cerca l'indice di 'saturazione'
            if 'saturazione' in all_cols:
                idx_satura = all_cols.index('saturazione')
                cols_extra = all_cols[idx_satura:]  # Prende Satura e TUTTO ciò che segue
            else:
                # Se manca Satura, prova a cercare da kron_flux
                idx_start = all_cols.index('kron_flux') if 'kron_flux' in all_cols else len(all_cols)
                cols_extra = all_cols[idx_start:]

            cols_dinamiche = []

            # 1. Aggiungi Satura (se c'è)
            if 'saturazione' in cols_extra:
                cols_dinamiche.append('saturazione')

            # 2. Aggiungi kron_flux (se c'è) subito dopo
            if 'kron_flux' in all_cols:
                cols_dinamiche.append('kron_flux')

            # 3. Aggiungi TUTTO il resto che era in cols_extra
            for c in cols_extra:
                if c not in cols_dinamiche and c not in cols_base:
                    cols_dinamiche.append(c)

        except ValueError:
            # Fallback totale se qualcosa va storto con gli indici
            cols_dinamiche = []
            if 'kron_flux' in all_cols: cols_dinamiche.append('kron_flux')

        cols_finali = cols_base + cols_dinamiche

        # Filtriamo il dataframe mantenendo solo le colonne desiderate nell'ordine corretto
        # Usiamo intersection per evitare errori se per caso una colonna base manca
        cols_presenti = [c for c in cols_finali if c in df_trovate.columns]
        df_trovate = df_trovate[cols_presenti]
        # -----------------------------------------------------------

        # Calcolo coordinate celesti (Vettoriale con WCS)
        try:
            # Usa memmap=False per sicurezza con header complessi
            with fits.open(percorso_file, memmap=False) as hdu_list:
                w = WCS(hdu_list[0].header)
            coords_trovate = w.pixel_to_world(df_trovate['xcentroid'], df_trovate['ycentroid'])

            # Aggiungi le nuove colonne al dataframe filtrato
            # Nota: Pandas avvisa se modifichi una slice, usiamo .copy() implicito o loc se serve,
            # ma qui df_trovate è già un nuovo oggetto dopo il filtro colonne
            df_trovate = df_trovate.copy()
            df_trovate['RA_centroid'] = coords_trovate.ra.deg
            df_trovate['DEC_centroid'] = coords_trovate.dec.deg

        except Exception as e:
            print(f"Errore WCS/FITS: {e}")
            continue

        # Preparazione coordinate catalogate
        if 'RAJ2000' in df_catalogate.columns:
            coords_catalogate = SkyCoord(ra=df_catalogate['RAJ2000'].values * u.deg,
                                         dec=df_catalogate['DEJ2000'].values * u.deg)
        else:
            continue

        # print(f"Cercata correlazione in {len(coords_trovate)} stelle")

        # --- 2. MATCHING VELOCE (search_around_sky) ---
        idx_trovate, idx_catalogate, d2d, _ = coords_catalogate.search_around_sky(coords_trovate, soglia_correlazione)

        matches = pd.DataFrame({
            'idx_t': idx_trovate,
            'idx_c': idx_catalogate,
            'dist': d2d.deg,
            'mag': df_catalogate.iloc[idx_catalogate]['Mag'].values
        })

        # --- 3. LOGICA RANK ---
        matches.sort_values(by=['idx_t', 'mag'], inplace=True)
        matches['rank'] = matches.groupby('idx_t').cumcount() + 1
        matches['Corrispondenza'] = 'SI (Rank ' + matches['rank'].astype(str) + ')'

        # --- 4. COSTRUZIONE TABELLA FINALE ---

        # A. Match SI
        part_trovate = df_trovate.iloc[matches['idx_t']].reset_index(drop=True)
        part_catalogate = df_catalogate.iloc[matches['idx_c']].reset_index(drop=True)
        part_rank = matches[['Corrispondenza']].reset_index(drop=True)

        df_si = pd.concat([part_trovate, part_rank, part_catalogate], axis=1)

        # B. Match NO
        all_indices = set(range(len(df_trovate)))
        matched_indices = set(matches['idx_t'].unique())
        unmatched_indices = list(all_indices - matched_indices)

        if unmatched_indices:
            df_no = df_trovate.iloc[unmatched_indices].copy()
            df_no['Corrispondenza'] = 'NO'

            for col in df_catalogate.columns:
                if col == 'Catalogo':
                    df_no[col] = 'N/A'
                elif pd.api.types.is_integer_dtype(df_catalogate[col]):
                    df_no[col] = -999
                elif pd.api.types.is_float_dtype(df_catalogate[col]):
                    df_no[col] = np.nan
                else:
                    df_no[col] = 'N/A'

            # Allinea colonne
            # Assicuriamoci che df_no abbia tutte le colonne di df_si nello stesso ordine
            # Le colonne mancanti (quelle del catalogo) sono state appena create, quindi dovrebbe combaciare
            df_no = df_no.reindex(columns=df_si.columns, fill_value=np.nan)

        else:
            df_no = pd.DataFrame(columns=df_si.columns)

        # C. Unione
        df_finale = pd.concat([df_si, df_no], ignore_index=True)

        if 'label' in df_finale.columns:
            df_finale.sort_values('label', inplace=True)

        # --- 5. RIORDINAMENTO FINALE (Catalogo prima di ID) ---
        colonne = df_finale.columns.tolist()
        if 'ID' in colonne and 'Catalogo' in colonne:
            colonne.remove('Catalogo')
            pos_id = colonne.index('ID')
            colonne.insert(pos_id, 'Catalogo')
            df_finale = df_finale[colonne]

        df_unite = df_finale

        filename_out = os.path.join(output_dir, f'run_{run}_stelle_trovate_e_catalogate_immagine_{n:03d}.csv')

        # Aggiungiamo il percorso file FITS all'header se non c'è, è importante per la fase 2!
        if 'PERCORSO_FILE' not in header_fits:
            # Header_fits è un oggetto astropy Header, non un dict, ma si comporta similmente
            # Meglio non modificarlo direttamente se non serve, ma salva_csv_con_header_fits si aspetta un dict
            pass

        # Per la fase 2 serve un header dict con PERCORSO_FILE
        # convertiamo header_fits in dict e aggiungiamo il path
        header_dict_save = dict(header_fits)
        header_dict_save['PERCORSO_FILE'] = percorso_file

        salva_csv_con_header_fits(df_unite, header_dict_save, filename_out, percorso_file,
                                  parametri_seg=parametri_caricati)

    print("\n--- FASE 1 COMPLETATA: Tabelle unite create. ---")

    # =============================================================================
    # FASE 2: CALCOLO DEL RAGGIO MASSIMO PER OGNI STELLA SU TUTTA LA RUN
    # =============================================================================
    print("\n--- FASE 2.A: Mappatura Raggi Massimi Globali (Max Run) ---")

    # Usiamo la cartella appena popolata
    base_input_path = output_dir

    file_csv_list = sorted(
        [os.path.join(base_input_path, f) for f in os.listdir(base_input_path) if f.endswith('.csv')])

    # Liste per accumulare i dati necessari al calcolo del max
    all_ids = []
    all_radii = []

    for file_csv in tqdm(file_csv_list, desc="Scansione Raggi"):
        # Leggiamo solo ID e raggio per velocità
        try:
            df_temp = pd.read_csv(file_csv, comment='#', usecols=['ID', 'raggio_kron_aper'])
            # Filtriamo eventuali NaN o ID non validi
            df_temp = df_temp.dropna(subset=['ID', 'raggio_kron_aper'])
            all_ids.append(df_temp['ID'].values)
            all_radii.append(df_temp['raggio_kron_aper'].values)
        except Exception as e:
            # Questo può succedere se raggio_kron_aper non è stato salvato o il file è vuoto
            # print(f"Skipping scan {os.path.basename(file_csv)}: {e}")
            pass

    # Concatenazione veloce
    if len(all_ids) > 0:
        big_ids = np.concatenate(all_ids)
        big_radii = np.concatenate(all_radii)

        # Creiamo un DataFrame temporaneo globale
        df_global = pd.DataFrame({'ID': big_ids, 'R': big_radii})

        # Raggruppiamo per ID e prendiamo il MASSIMO raggio visto nella run
        map_raggi_max = df_global.groupby('ID').quantile(0.95).to_dict()

        print(f"\nMappate {len(map_raggi_max)} stelle uniche per apertura fissa.")
    else:
        print("ATTENZIONE: Nessun raggio trovato. Salto la fase 2.")
        exit()

    # =============================================================================
    # FASE 3: RICALCOLO FLUSSI CON APERTURA FISSA (SPECIFICA PER STELLA)
    # =============================================================================

    print("\n--- FASE 2.B: Fotometria con Apertura Fissa (Max Run) ---")

    # Funzione helper locale per salvare (sovrascrivere) con l'header corretto
    def salva_csv_con_header_aggiornato(df, header_dict, output_file):
        with open(output_file, 'w') as f:
            f.write("# Header FITS:\n")
            for k, v in header_dict.items():
                f.write(f"# {k}: {v}\n")
            f.write("#\n")
            df.to_csv(f, index=False)


    for file_csv in tqdm(file_csv_list, desc="Ricalcolo Flussi"):

        # 1. Caricamento Dati
        df_frame = pd.read_csv(file_csv, comment='#')
        header_info = leggi_header_da_csv(file_csv)
        path_fits = header_info.get('PERCORSO_FILE', '')

        if not os.path.exists(path_fits):
            # Prova a ricostruire il path se è relativo o sbagliato
            # (nel tuo caso sembra essere assoluto, ma per sicurezza)
            continue

        # 2. Caricamento Immagine
        with fits.open(path_fits, memmap=False) as hdu:
            data = hdu[0].data
            _, median_bg, _ = sigma_clipped_stats(data[::10, ::10], sigma=3.0)
            data_sub = data - median_bg

        # 3. Preparazione Raggi
        raggi_fissi = []
        ids_presenti = df_frame['ID'].values

        # 4. Esecuzione Fotometria (CICLO SICURO PER EVITARE ERRORI VETTORIALI)
        # CircularAperture con array può fallire se ci sono NaN o versioni vecchie.
        flussi_calcolati = []

        for i, star_id in enumerate(ids_presenti):
            # A. Recupero Raggio
            r_globale = map_raggi_max.get(star_id, np.nan)

            if np.isnan(r_globale) or r_globale <= 0:
                # Fallback locale
                idx = df_frame.index[i]
                if 'raggio_kron_aper' in df_frame.columns:
                    r_globale = df_frame.at[idx, 'raggio_kron_aper']
                else:
                    r_globale = np.nan

            # Salvataggio raggio usato
            raggi_fissi.append(r_globale)

            # B. Calcolo Fotometria Singola
            if r_globale > 0 and not np.isnan(r_globale):
                # Posizione singola
                pos = (df_frame.at[i, 'xcentroid'], df_frame.at[i, 'ycentroid'])
                aper = CircularAperture(pos, r=r_globale)

                # Aperture photometry su singola stella
                # data_sub è l'immagine con il fondo sottratto
                phot = aperture_photometry(data_sub, aper)
                flusso = phot['aperture_sum'][0]
                flussi_calcolati.append(flusso)
            else:
                flussi_calcolati.append(np.nan)

        # 5. Aggiornamento DataFrame
        df_frame['flusso_fisso_max_run'] = flussi_calcolati
        df_frame['raggio_fisso_max_run'] = raggi_fissi

        # Formattazione
        df_frame['flusso_fisso_max_run'] = df_frame['flusso_fisso_max_run'].map(
            lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')
        df_frame['raggio_fisso_max_run'] = df_frame['raggio_fisso_max_run'].map(
            lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')

        # --- NUOVA LOGICA: RIORDINO RIGHE E COLONNE ---
        # 1. Ordinamento Righe (Label -> Rank)
        if 'label' in df_frame.columns and 'Corrispondenza' in df_frame.columns:
            df_frame.sort_values(by=['label', 'Corrispondenza'], ascending=[True, True], inplace=True)

        # 2. Spostamento Colonne
        cols = df_frame.columns.tolist()
        cols_move = ['flusso_fisso_max_run', 'raggio_fisso_max_run']
        target = 'RA_centroid'

        # Rimuoviamo le colonne nuove dalla loro posizione attuale (fondo)
        for c in cols_move:
            if c in cols: cols.remove(c)

        # Inseriamo prima di RA_centroid
        if target in cols:
            idx = cols.index(target)
            for c in reversed(cols_move):
                cols.insert(idx, c)
        else:
            # Fallback se RA_centroid non c'è
            cols.extend(cols_move)

        df_frame = df_frame[cols]
        # ---------------------------------------------

        # 6. Sovrascrittura file (Aggiornamento)
        salva_csv_con_header_aggiornato(df_frame, header_info, file_csv)

    print(f"\nElaborazione Finale completata. I file in {output_dir} sono stati aggiornati.")