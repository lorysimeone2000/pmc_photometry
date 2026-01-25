import pandas as pd
#pd.set_option('display.show_dimensions', False)
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

parametri = {}
with open('/home/lorysimeone/tesi_magistrale/prove/analisi/parametri_image_segmentation.txt', 'r') as file:
    # Salta la prima riga (intestazione)
    next(file)

    # Legge le righe vuota e successive
    for riga in file:
        riga = riga.strip()
        if riga and not riga.startswith('#'):  # Ignora righe vuote
            parametro, valore = riga.split()
            print(f"{parametro} = {valore}")
            # AGGIUNGI al dizionario
            parametri[parametro] = float(valore) if '.' in valore else int(valore)


image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212855.fits"

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data

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

# print("Colonne della tabella:")
# print(tbl.info())

aree = np.array(tbl['area'])

# Ora creo l'array delle magnitudini