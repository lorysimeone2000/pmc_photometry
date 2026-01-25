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

def azzera_pixel_isolati_veloce(dataf, min_neighbors=2, threshold_ratio=0.3):
    """
    Versione ottimizzata per azzerare pixel isolati
    """
    data_pulita = dataf.copy()

    # Soglia per considerare un pixel significativo
    meanf, medianf, stdf = sigma_clipped_stats(dataf, sigma=3.0)
    soglia_minima = 5 * stdf
    mask_significativi = dataf > soglia_minima

    # Crea una mappa dei pixel da azzerare
    mask_da_azzera = np.zeros_like(dataf, dtype=bool)

    y_coords, x_coords = np.where(mask_significativi)
    for y, x in zip(y_coords, x_coords):
        pixel_val = dataf[y, x]
        threshold_locale = threshold_ratio * pixel_val

        # Controlla regione 3x3
        y_min, y_max = max(0, y - 1), min(dataf.shape[0], y + 2)
        x_min, x_max = max(0, x - 1), min(dataf.shape[1], x + 2)

        region = dataf[y_min:y_max, x_min:x_max]
        mask_forti = region > threshold_locale

        # Conta vicini forti (escludendo il centro)
        center_y, center_x = 1 if y_min < y else 0, 1 if x_min < x else 0
        mask_forti[center_y, center_x] = False  # Esclude il centro

        vicini_forti = np.sum(mask_forti)

        if vicini_forti < min_neighbors:
            mask_da_azzera[y, x] = True

    # Crea l'immagine azzerata mantenendo i pixel isolati
    data_azzerata = np.zeros_like(dataf)
    data_azzerata[mask_da_azzera] = data_pulita[mask_da_azzera]
    data_pulita[mask_da_azzera] = 0  # Azzera i pixel isolati nell'immagine pulita

    return data_pulita, data_azzerata

# Applica la funzione
data_pulita, data_azzerata = azzera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)

# Visualizza l'immagine con pixel isolati
plt.figure(figsize=(10, 8))
plt.imshow(data_azzerata, cmap="grey_r", norm=LogNorm(), interpolation='nearest')
plt.gca().invert_yaxis()
plt.title("Pixel isolati identificati")
plt.colorbar()
plt.show()

# convoluzione
fwhm = float(input("FWHM = " ))
size = int(input("Size = " ))

kernel = make_2dgaussian_kernel(fwhm, size=size)
convolved_data = convolve(data_azzerata, kernel)
mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)

# Sourcefinder
t = float(input('Threshold (n. sigma) = ')) # threshold
print(f"Deviazione standard convoluzione = {std_c}")
threshold = t * std_c  # CORRETTO: usa std_c invece di std
print(f"Threshold convoluzione = {threshold}")
n = int(input("Numero minimo pixel contigui = " ))

finder = SourceFinder(npixels=n, progress_bar=True)
segment_map = finder(convolved_data, threshold)

print(segment_map)

plt.figure(figsize=(10, 8))
plt.imshow(data_azzerata, cmap="grey_r", norm=LogNorm(), interpolation='nearest')
plt.gca().invert_yaxis()

# tabella sorgenti
cat = SourceCatalog(data_azzerata, segment_map, convolved_data=convolved_data)
print(cat)

tbl = cat.to_table()
tbl['xcentroid'].info.format = '.2f'  # optional format
tbl['ycentroid'].info.format = '.2f'
if 'kron_flux' in tbl.colnames:
    tbl['kron_flux'].info.format = '.2f'
print(tbl)

positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))
apertures = CircularAperture(positions, r=5.0)
apertures.plot(color='#0547f9', lw=0.7)
plt.title("Pixel isolati trovati con l'image segmentation")

plt.show()