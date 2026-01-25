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

df = pd.read_csv("/home/lorysimeone/tesi_magistrale/prove/analisi/distanze/array_distanze.csv")
print(df)

distanze_array = np.array(df['distanze_gradi'])

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
n, bins, patches = plt.hist(distanze_array, bins=25000, density=True,
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

    '''# Plot della gaussiana fitted
    plt.plot(x_fit, y_fit, 'r-', linewidth=2,
             label=f'Gaussiana fit\nμ = {mu_fit:.6f} gradi\nσ = {sigma_fit:.6f} gradi')'''

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