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
import numpy as np
import numpy.ma as ma
import os
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture

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
    with open('/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt', 'r') as file:
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

def salva_csv_con_header_fits(dataframe, header_fits, filename, nome_file_fits):
    """Salva il DataFrame in CSV includendo l'header FITS come commenti"""
    with open(filename, 'w') as f:
        # Scrivi l'header FITS come commenti
        f.write("# Header FITS:\n")
        for key, value in header_fits.items():
            f.write(f"# {key}: {value}\n")
        f.write(f"# PERCORSO_FILE: {nome_file_fits}\n")
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

#run = int(input("Quale run vuoi elaborare: ")) # numero run: 1, 2 o 3
run = 1

image_file = f"/home/lorysimeone/tesi_magistrale/prove_2/stacking/run_{run}_stacked_sum.fits"
image_file_c = "/home/lorysimeone/tesi_magistrale/prove_2/stacking/run_1_coverage_map.fits"

# --- CARICAMENTO E CREAZIONE MASCHERA ---

# 1. Caricamento Coverage Map
hdu_list_c = fits.open(image_file_c)
print("Informazioni Coverage Map:")
hdu_list_c.info()
image_data_c = hdu_list_c[0].data
# La maschera è True dove la copertura è massima (115)
full_coverage_value = np.max(image_data_c) # Il valore che vuoi mascherare
mask_max_coverage = image_data_c == full_coverage_value
hdu_list_c.close()

# 2. Caricamento Immagine Sommata
hdu_list = fits.open(image_file)
print("\nInformazioni Immagine Sommata:")
hdu_list.info()
image_data = hdu_list[0].data
header = hdu_list[0].header
hdu_list.close()

# --- ESTRAZIONE E VISUALIZZAZIONE ---

# Applichiamo la maschera:
# Usiamo ~mask_max_coverage per nascondere tutti i pixel NON UGUALI a 115.
data = ma.masked_array(image_data, mask=~mask_max_coverage)
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)

data = data - median
print(data)
print("Media: ", mean)
print("Mediana: ", median)
print("Deviazione standard: ", std)
# quit()
data = data.filled(0)

plt.imshow(data, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.colorbar()

plt.show()

#quit()

tbl = analisi_image_segmentation(data)
dataframe = tbl.to_pandas()

# Definisci la cartella di output
output_dir = f"/home/lorysimeone/tesi_magistrale/prove_2/stacking"

filename = os.path.join(output_dir, f'run_{run}_stelle_trovate.csv')
salva_csv_con_header_fits(dataframe, header, filename, image_file)

