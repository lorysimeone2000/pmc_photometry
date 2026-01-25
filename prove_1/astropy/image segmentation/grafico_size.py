from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
import numpy as np
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"  # prima immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data

fwhm = float(input("FWHM = " ))
threshold = float(input('Threshold (n. sigma) = ')) # threshold
t = threshold

'''def numero_stelle(dati , bs):
    tbl = find_peaks(dati, threshold, box_size=bs)  # trovo i picchi
    tbl['peak_value'].info.format = '%.8g'  # per rappresentare meglio la tabella

    return len(tbl)'''

def numero_stelle(dati , sz, np):

    # convoluzione
    kernel = make_2dgaussian_kernel(fwhm, size=sz)  # inserisco FWHM e dimensione
    convolved_dati = convolve(dati, kernel)

    # SourceFinder
    finder = SourceFinder(npixels=np, progress_bar=False)  # inserisco n. minimo di pixel
    segment_map = finder(convolved_dati, threshold)

    # tabella sorgenti
    cat = SourceCatalog(dati, segment_map, convolved_data=convolved_dati)
    tbl = cat.to_table()

    return len(tbl)

numero_di_pixel = 31
x = np.arange(1, numero_di_pixel, 2) # asse x fatta dal numero di pixel (dispari) che scorre
y = []
for size in x:
    num_stars = numero_stelle(data,size,n)  # chiama la funzione per ogni singolo valore
    y.append(num_stars)
    #print(f"FWHM={fwhm_value:.2f}, Stelle trovate: {num_stars}")

y = np.array(y)

# Crea il grafico
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'r-', linewidth=2, markersize=4, label=f'Stelle vs dimensioni kernel\n(Threshold = {t}σ, FWHM = {fwhm}, n. pixel min = {n})')
plt.xlabel('Dimensione kernel (pixel)', fontsize=12)
plt.ylabel('Numero di stelle trovate', fontsize=12)
plt.title('Stelle vs dimensioni kernel', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'{yi}', (xi, yi), textcoords="offset points",
                 xytext=(0,10), ha='center', fontsize=9)

plt.show()

'''size = int(input("Size = " ))
kernel = make_2dgaussian_kernel(fwhm, size=size)  # inserisco FWHM e dimensione
convolved_data = convolve(data, kernel)
mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)
threshold = t * std
print(f"Threshold convoluzione = {threshold}")

numero_di_pixel_contigui_max = 150

x = np.arange(1, numero_di_pixel_contigui_max) # asse x fatta dal numero di pixel contigui che scorre
y = []
for n_p in x:
    num_stars = numero_stelle(data,size,n_p)  # chiama la funzione per ogni singolo valore
    y.append(num_stars)
    #print(f"FWHM={fwhm_value:.2f}, Stelle trovate: {num_stars}")

y = np.array(y)

# Crea il grafico
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'r-', linewidth=2, markersize=4, label=f'Stelle vs n. pixel contigui\n(Threshold = {t}σ, FWHM = {fwhm}, size = {size})')
plt.xlabel('n. pixel contigui', fontsize=12)
plt.ylabel('Numero di stelle trovate', fontsize=12)
plt.title('Stelle vs n. pixel contigui', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'{yi}', (xi, yi), textcoords="offset points",
                 xytext=(0,10), ha='center', fontsize=9)

plt.show()'''