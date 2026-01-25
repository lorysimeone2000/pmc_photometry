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

# VERSIONE OTTIMIZZATA CON ARRAY OPERATIONS
def azzera_pixel_isolati_veloce(dati, min_neighbors=2, threshold_ratio=0.3):
    """
    Versione ottimizzata per azzerare pixel isolati
    """
    from scipy import ndimage

    data_pulita = dati.copy()

    # Soglia per considerare un pixel significativo
    soglia_minima = 5 * std
    mask_significativi = dati > soglia_minima

    # Crea una mappa dei vicini forti
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Esclude il pixel centrale

    # Per ogni pixel, conta quanti vicini sono sopra threshold_ratio del pixel centrale
    mask_vicini_forti = np.zeros_like(dati, dtype=bool)

    y_coords, x_coords = np.where(mask_significativi)
    for y, x in zip(y_coords, x_coords):
        pixel_val = dati[y, x]
        threshold_locale = threshold_ratio * pixel_val

        # Controlla regione 3x3
        y_min, y_max = max(0, y - 1), min(dati.shape[0], y + 2)
        x_min, x_max = max(0, x - 1), min(dati.shape[1], x + 2)

        region = dati[y_min:y_max, x_min:x_max]
        mask_forti = region > threshold_locale

        # Conta vicini forti (escludendo il centro)
        center_y, center_x = 1 if y_min < y else 0, 1 if x_min < x else 0
        mask_forti[center_y, center_x] = False  # Esclude il centro

        vicini_forti = np.sum(mask_forti)

        if vicini_forti < min_neighbors:
            data_pulita[y, x] = 0  # Azzera il pixel isolato

    return data_pulita


# Uso la versione pulita

data_pulita = azzera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)

'''plt.imshow(image_data, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.title('Dati originali', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis() # inverto asse y
plt.colorbar()

plt.show()'''

# convoluzione

fwhm = float(input("FWHM = " ))
size = int(input("Size = " ))

kernel = make_2dgaussian_kernel(fwhm, size=size)  # inserisco FWHM e dimensione
convolved_data = convolve(data_pulita, kernel)
mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)

'''plt.imshow(convolved_data, cmap='grey_r', origin='lower', norm=LogNorm(), interpolation='nearest')
plt.title(f'Dati Convoluti con Kernel Gaussiano\n(FWHM = {fwhm} pixel e size = {size})', fontsize=14, fontweight='bold')
plt.colorbar()

plt.show()'''

# Sourcefinder

t = float(input('Threshold (n. sigma) = ')) # threshold
print(f"Deviazione standard convoluzione = {std_c}")
threshold = t * std
print(f"Threshold convoluzione = {threshold}")
n = int(input("Numero minimo pixel contigui = " ))

finder = SourceFinder(npixels=n, progress_bar=True) # inserisco n. minimo di pixel
segment_map = finder(convolved_data, threshold)
print(segment_map)

plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
plt.title(f'Sorgenti Rilevate con pixel azzerati\n(Threshold = {t} σ, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)', fontsize=14, fontweight='bold')

#plt.show()

# tabella sorgenti

cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
print(cat)

tbl = cat.to_table()
tbl['xcentroid'].info.format = '.2f'  # optional format
tbl['ycentroid'].info.format = '.2f'
tbl['kron_flux'].info.format = '.2f'
print(tbl)

positions = np.transpose((tbl['xcentroid'], tbl['ycentroid'])) # creo un array di posizioni
apertures = CircularAperture(positions, r=5.0) # creo le aperture per ogni posizione
apertures.plot(color='#0547f9', lw=0.7)

plt.show()

plt.imshow(data_pulita, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.title(f'Sorgenti Rilevate con pixel azzerati'
          f'\n(Threshold = {t} σ, n. pixel min = {n}, 'f' FWHM = {fwhm}, dimensioni kernel = {size} pixel)',
          fontsize=14, fontweight='bold')
plt.colorbar()
apertures.plot(color='#0547f9', lw=0.7)

plt.show()

# provo con mask

# mask booleano

def maschera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3):
    """
    Versione ottimizzata per mascherare pixel isolati
    """

    matrice_booleana = np.full_like(data, False, dtype=bool)

    from scipy import ndimage

    data_pulita = data.copy()

    # Soglia per considerare un pixel significativo
    soglia_minima = 5 * std
    mask_significativi = data > soglia_minima

    # Crea una mappa dei vicini forti
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Esclude il pixel centrale

    # Per ogni pixel, conta quanti vicini sono sopra threshold_ratio del pixel centrale
    mask_vicini_forti = np.zeros_like(data, dtype=bool)

    y_coords, x_coords = np.where(mask_significativi)
    for y, x in zip(y_coords, x_coords):
        pixel_val = data[y, x]
        threshold_locale = threshold_ratio * pixel_val

        # Controlla regione 3x3
        y_min, y_max = max(0, y - 1), min(data.shape[0], y + 2)
        x_min, x_max = max(0, x - 1), min(data.shape[1], x + 2)

        region = data[y_min:y_max, x_min:x_max]
        mask_forti = region > threshold_locale

        # Conta vicini forti (escludendo il centro)
        center_y, center_x = 1 if y_min < y else 0, 1 if x_min < x else 0
        mask_forti[center_y, center_x] = False  # Esclude il centro

        vicini_forti = np.sum(mask_forti)

        if vicini_forti < min_neighbors:
            matrice_booleana[y, x] = True  # rendi True il pixel isolato

    return matrice_booleana

pixel_isolati_bool = maschera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)

convolved_data = convolve(data, kernel, mask = pixel_isolati_bool)
mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)

finder = SourceFinder(npixels=n, progress_bar=True) # inserisco n. minimo di pixel
segment_map = finder(convolved_data, threshold)
print(segment_map)

plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
plt.title(f'Sorgenti Rilevate con pixel mascherati\n(Threshold = {t} σ, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)', fontsize=14, fontweight='bold')

#plt.show()

# tabella sorgenti

cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
print(cat)

tbl = cat.to_table()
tbl['xcentroid'].info.format = '.2f'  # optional format
tbl['ycentroid'].info.format = '.2f'
tbl['kron_flux'].info.format = '.2f'
print(tbl)

positions = np.transpose((tbl['xcentroid'], tbl['ycentroid'])) # creo un array di posizioni
apertures = CircularAperture(positions, r=5.0) # creo le aperture per ogni posizione
apertures.plot(color='#0547f9', lw=0.7)

plt.show()

plt.imshow(data_pulita, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.title(f'Sorgenti Rilevate versione pulita'
          f'\n(Threshold = {t} σ, n. pixel min = {n}, 'f' FWHM = {fwhm}, dimensioni kernel = {size} pixel)',
          fontsize=14, fontweight='bold')
plt.colorbar()
apertures.plot(color='#0547f9', lw=0.7)

plt.show()
