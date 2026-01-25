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
from astropy.table import Table
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture

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

# Cartella contenente i file CSV
cartella_csv = "/home/lorysimeone/tesi_magistrale/prove/analisi/sorgenti_run/sorgenti_run_1"

# Lista tutti i file CSV e ordinali
# file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
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
    percorso_completo = os.path.join(cartella_csv, nome_file)

    if i <=2:
        print(f"\n{'=' * 50}")
        print(f"Elaborazione: {nome_file}")
        print(f"{'=' * 50}")
    # Leggi il file CSV
    try:
        df = pd.read_csv(percorso_completo)
        tbl = Table.from_pandas(df)
        j = j + len(tbl)
        posizioni_file = np.transpose((tbl['xcentroid'], tbl['ycentroid'])) # creo l'array di posizioni per questo file
        posizioni_lista.append(posizioni_file) # lo aggiungo alla lista totale

    except Exception as e:
        print(f"Errore nella lettura di {nome_file}: {e}")



posizioni_array = np.vstack(posizioni_lista)
print(f"\n{'='*60}")
print(f"ARRAY FINALE CREATO")
print(f"{'='*60}")
print(f"Dimensioni array posizioni: {posizioni_array.shape}")

print(f"massimo x: {np.max(posizioni_array[:, 0])}")
print(f"massimo y: {np.max(posizioni_array[:, 1])}")
print("Ho verificato che l'asse x e l'asse y sono corrispondenti a quelli dell'immagine")

plt.scatter(posizioni_array[:, 0], posizioni_array[:, 1],
            s=2,
            alpha=0.3,
            color='blue',
            edgecolors='none', # ⬅ Nessun bordo (era 'black')
            linewidth=0)       # ⬅ Nessuna linea di bordo)

plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.title(f'Posizioni di tutte le sorgenti\nTotale: {len(posizioni_array)} sorgenti da {len(file_csv)} file della run')
plt.grid(True, alpha=0.3)
plt.gca().set_aspect('equal')

plt.tight_layout()
plt.show()