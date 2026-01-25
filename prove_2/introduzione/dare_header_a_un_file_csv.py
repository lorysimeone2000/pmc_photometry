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
from astropy.table import Table

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

def salva_csv_con_header_fits(dataframe, header_fits, filename):
    """Salva il DataFrame in CSV includendo l'header FITS come commenti"""
    with open(filename, 'w') as f:
        # Scrivi l'header FITS come commenti
        f.write("# Header FITS:\n")
        for key, value in header_fits.items():
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

output_dir = "/home/lorysimeone/tesi_magistrale/prove/automatizzazione" # definisco la cartella di output

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits" # primo elemento della run
data = elabora_file_fits(image_file)
hdu_list = fits.open(image_file)
tbl = analisi_image_segmentation(data)
dataframe = tbl.to_pandas()
filename = os.path.join(output_dir, f'run_1_immagine_{1}.csv')
dataframe.to_csv(filename, index=False)
header_fits = hdu_list[0].header
salva_csv_con_header_fits(dataframe, header_fits, filename)

# accedo ai valori dell'header
header_dal_csv = leggi_header_da_csv("/home/lorysimeone/tesi_magistrale/prove/automatizzazione/run_2_immagine_1.csv")
data_osservazione = header_dal_csv['DATE-OBS']
print(f"Data osservazione: {data_osservazione}")

# accedo alla tabella astropy
print(filename)
dataframe = pd.read_csv(filename, comment="#")
tabella_astropy = Table.from_pandas(dataframe)
print(tabella_astropy)
