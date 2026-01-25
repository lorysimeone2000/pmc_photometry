import pandas as pd
#pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import os
from tqdm import tqdm
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
# Set up wcs
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture
import warnings
from astropy.wcs import FITSFixedWarning

# Ignora i warning specifici sui fix automatici degli header FITS
warnings.simplefilter('ignore', category=FITSFixedWarning)

from tqdm import tqdm  # <--- Assicurati di avere questo import in cima al file!


def analisi_image_segmentation(percorso_file_):
    """
    Esegue image segmentation su un'immagine FITS e restituisce la tabella filtrata
    """
    # 1. Setup iniziale
    data, fondo_iniziale, w = elabora_file_fits(percorso_file_)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # 2. Lettura parametri (CORRETTO)
    parametri = {}
    with open('/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt', 'r') as file:
        next(file, None)  # Salta intestazione se presente
        for riga in file:
            riga = riga.split('#')[0].strip()  # Rimuove commenti
            if riga:
                parts = riga.split()
                if len(parts) >= 2:
                    parametro = parts[0]  # <--- CORRETTO QUI
                    valore_str = parts[1]
                    try:
                        valore = float(valore_str) if '.' in valore_str else int(valore_str)
                        parametri[parametro] = valore
                    except ValueError:
                        pass

    # 3. Convoluzione e SourceFinder
    # Assicurati che i parametri esistano nel file, altrimenti metti dei default o gestisci l'errore
    fwhm = parametri.get('fwhm', 3.0)
    size = parametri.get('size', 5)

    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)

    threshold = parametri.get('threshold_assoluta', 3.0)
    pixel_n = parametri.get('pixel', 5)

    finder = SourceFinder(npixels=pixel_n, progress_bar=True)
    segment_map = finder(convolved_data, threshold)

    # 4. Creazione Tabella Madre
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    # Formattazione
    for col in ['xcentroid', 'ycentroid', 'kron_flux']:
        tbl[col].info.format = '.2f'

    # Saturazione
    livello_saturazione = 255 - fondo_iniziale - median
    tbl['saturazione'] = np.where(tbl['max_value'] >= livello_saturazione, 'SI', 'NO')

    # ---------------------------------------------------------
    # 5. CALCOLO APERTURE E RAGGI (Iterativo con Progress Bar)
    # ---------------------------------------------------------
    raggi_max_deg = []
    raggi_max_pix_calc = []
    lista_flussi_ultimo_pixel = []
    lista_flussi_kron_manuale = []  # <--- NUOVA LISTA

    # Parametri per Kron manuale
    K_KRON = 2.5  # Costante standard (Kron 1980 / SExtractor)
    R_MIN_KRON = 3.5  # Raggio minimo in pixel (per evitare aperture minuscole su stelle deboli)

    # AVVOLGIAMO 'tbl' CON tqdm PER AVERE LA BARRA DI SCORRIMENTO
    for row in tqdm(tbl, desc="Calcolo aperture", unit="stella"):
        label = row['label']
        xc, yc = row['xcentroid'], row['ycentroid']

        # A. Troviamo i pixel dalla mappa
        # ypix, xpix sono gli indici dei pixel che appartengono alla segmentazione
        ypix, xpix = np.where(segment_map.data == label)

        if len(xpix) == 0:
            raggi_max_deg.append(np.nan)
            raggi_max_pix_calc.append(np.nan)
            lista_flussi_ultimo_pixel.append(np.nan)
            lista_flussi_kron_manuale.append(np.nan)
            continue

        # B. Distanze Euclidee dal centroide per ogni pixel
        distanze_pix = np.hypot(xpix - xc, ypix - yc)

        # --- CALCOLO FLUSSO RAGGIO MASSIMO (Tuo vecchio metodo) ---
        r_max_pix = np.max(distanze_pix)
        r_max_pix = max(r_max_pix, 0.5)
        raggi_max_pix_calc.append(r_max_pix)

        aper_pix = CircularAperture((xc, yc), r=r_max_pix)
        phot_single = aperture_photometry(data, aper_pix)
        lista_flussi_ultimo_pixel.append(phot_single['aperture_sum'][0])

        # --- CALCOLO KRON FLUX MANUALE (Nuovo metodo) ---
        # 1. Recuperiamo i valori di intensità (I_i) dei pixel della segmentazione
        valori_pixel = data[ypix, xpix]  # data[y, x] in numpy

        # 2. Calcolo Primo Momento (r_1)
        # r1 = sum(Valore * Distanza) / sum(Valore)
        somma_intensita = np.sum(valori_pixel)

        if somma_intensita > 0:
            somma_momenti = np.sum(valori_pixel * distanze_pix)
            r_1 = somma_momenti / somma_intensita
        else:
            r_1 = 0.0

        # 3. Calcolo Raggio di Kron (R_k)
        r_kron_calc = K_KRON * r_1

        # Applichiamo un raggio minimo (fondamentale per sorgenti piccole)
        r_kron_finale = max(r_kron_calc, R_MIN_KRON)

        # 4. Fotometria sull'apertura di Kron
        aper_kron = CircularAperture((xc, yc), r=r_kron_finale)
        phot_kron = aperture_photometry(data, aper_kron)
        flusso_kron_man = phot_kron['aperture_sum'][0]

        lista_flussi_kron_manuale.append(flusso_kron_man)

        # D. Conversione in Cielo (opzionale)
        try:
            aper_sky = aper_pix.to_sky(w)
            raggi_max_deg.append(aper_sky.r)
        except Exception:
            raggi_max_deg.append(np.nan)

    # ---------------------------------------------------------
    # 6. ASSEGNAZIONE COLONNE
    # ---------------------------------------------------------

    tbl['somma_apertura_ultimo_pixel'] = lista_flussi_ultimo_pixel
    tbl['somma_apertura_ultimo_pixel'].info.format = '%.2f'

    # Nuova colonna
    tbl['kron_flux_manuale'] = lista_flussi_kron_manuale
    tbl['kron_flux_manuale'].info.format = '%.2f'

    # ---------------------------------------------------------
    # 7. FILTRAGGIO FINALE
    # ---------------------------------------------------------
    soglia_assoluta = 2.5
    soglia_relativa = 0.05
    bordo = 7
    ny, nx = data.shape
    indici_validi = []

    for i, row in enumerate(tbl):
        label = row['label']
        mask_sorgente = (segment_map.data == label)
        valori_originali = data[mask_sorgente]

        xc, yc = row['xcentroid'], row['ycentroid']

        dentro_riquadro = (xc >= bordo) and (xc < nx - bordo) and \
                          (yc >= bordo) and (yc < ny - bordo)

        if not dentro_riquadro:
            continue

        pixel_sopra_soglia_assoluta = np.sum(valori_originali > soglia_assoluta)
        pixel_sopra_soglia_relativa = np.sum(valori_originali > soglia_relativa * row['max_value'])

        if pixel_sopra_soglia_assoluta >= 3 and pixel_sopra_soglia_relativa >= 2:
            indici_validi.append(i)

    # 8. Creazione Tabella Filtrata
    tbl_filtrato = tbl[indici_validi]

    # 9. Riordino Label
    if len(tbl_filtrato) > 0:
        tbl_filtrato['label'] = np.arange(1, len(tbl_filtrato) + 1)

    return tbl_filtrato, parametri
def elabora_file_fits(percorso_file_):
    """Elabora un singolo file FITS
    Come parametro si mete la stringa del percorso
    Si ottiene la matrice dei pixel col fondo sottratto"""
    with fits.open(percorso_file_) as hdu_list_:
        image_data_ = hdu_list_[0].data
        w_ = WCS(hdu_list_[0].header)
        # Fai qui le tue elaborazioni
        mean_, median_, std_ = sigma_clipped_stats(image_data_, sigma=3.0)
        image_data_ = image_data_ - median_

        return image_data_, median_, w_

def salva_csv_con_header_fits(dataframe, header_fits, filename, nome_file_fits, parametri_seg=None):
    """Salva il DataFrame in CSV includendo l'header FITS come commenti"""
    with open(filename, 'w') as f:
        # Scrivi l'header FITS come commenti
        f.write("# Header FITS:\n")
        for key, value in header_fits.items():
            f.write(f"# {key}: {value}\n")
        f.write(f"# PERCORSO_FILE: {nome_file_fits}\n")
        f.write("#\n# PARAMETRI SEGMENTAZIONE:\n")
        for key, value in parametri_seg.items():
            f.write(f"# {key}: {value}\n")
        f.write("#\n")  # Linea vuota per separare header dai dati
        # Scrivi il DataFrame
        dataframe.to_csv(f, index=False)

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
                    header_dict[key] = value
            elif line.strip() == '#':  # Fine dell'header
                break

    return header_dict

run = int(input("Quale run vuoi elaborare: ")) # numero run: 1, 2 o 3

# Leggo la lista
with open(f'/home/lorysimeone/tesi_magistrale/prove_2/liste_percorsi_run/lista_immagini_run_{run}.txt', 'r') as file:
    file_list = file.read().splitlines() # creo una lista di stringhe che sono i percorsi
# print(file_list)


n = 0
numero_stelle = []

# Elaboro tutti i file

# Definisci la cartella di output
output_dir = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_trovate_run/sorgenti_trovate_run_{run}"

for percorso_file in tqdm(file_list, desc="Elaborazione Files"):

    n = n + 1
    print(n)
    print(f"Elaborando: {percorso_file.split('/')[-1]}")  # Mostra solo il nome file
    # mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    # std = np.std(data)
    # print(f"std: {std}")

    tbl, parametri_usati = analisi_image_segmentation(percorso_file)
    '''if n <= 1:
        # print(f"Prime 5 tabelle: \n {tbl}")
        plt.imshow(data, cmap="grey_r", norm=LogNorm(),
                   interpolation='nearest')  # genero l'immagine con scala di colori bianco e nero
        plt.title('Dati', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # inverto asse y
        plt.colorbar()
        plt.show()'''
    numero_stelle.append(len(tbl))

    # ricavo l'header
    hdu_list = fits.open(percorso_file)
    header_fits = hdu_list[0].header

    '''# salvataggio in più file .csv
    dataframe = tbl.to_pandas()
    filename = os.path.join(output_dir, f'run_2_immagine_{n}.csv')
    dataframe.to_csv(filename, index=False)'''

    # accedo ai valori dell'header e li metto nel file
    dataframe = tbl.to_pandas()
    filename = os.path.join(output_dir, f'run_{run}_stelle_trovate_immagine_{n:03d}.csv')
    salva_csv_con_header_fits(dataframe, header_fits, filename, percorso_file, parametri_seg=parametri_usati)
    # if n == 1: break


print(numero_stelle)
array_finale = np.array(numero_stelle)

plt.plot(array_finale, marker='o', linestyle='-', linewidth=2, markersize=6)

plt.xlabel('Indice')
plt.ylabel('Numero stelle')
plt.title('Numero di stelle trovate in funzione della run')
plt.grid(True, alpha=0.3)
plt.ylim(0, None)
# plt.show()


# image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits" # prima immagine
# image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine

'''# Apertura e preparazione immagine
hdu_list = fits.open(image_file)
image_data = hdu_list[0].data
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data'''



# Mostra risultati
#print("\nTabella finale:")
# print(tbl)

#dataframe = tbl.to_pandas()
#print(dataframe) # .to_string(index=False) lo uso per non rappresentare la prima colonna di default di pandas

# Salvo in un file csv

#dataframe.to_csv('prova_prima_immagine.csv', index=False)