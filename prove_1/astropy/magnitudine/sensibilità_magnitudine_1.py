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
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture

from automatizzazione.creazione_tabelle_delle_run import leggi_header_da_csv

# Cartella contenente i file CSV
cartella_csv = "/home/lorysimeone/tesi_magistrale/prove/analisi/sorgenti_run/sorgenti_run_1"

# Lista tutti i file CSV e ordinali
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])

print(f"Trovati {len(file_csv)} file CSV:")
'''for file in file_csv:
    print(f"  - {file}")'''

i=0
j=0
posizioni_lista = [] # lista che dovrà essere riempita con tutte le poszioni di tutte le tabelle
# Itera su tutti i file CSV
for nome_file in file_csv:
    i += 1
    filename = os.path.join(cartella_csv, nome_file)
    # print(filename)
    dataframe = pd.read_csv(filename, skiprows=58)
    tbl = Table.from_pandas(dataframe)
    if i == 1:
        print(filename)
        print(tbl)
    header = leggi_header_da_csv(filename)