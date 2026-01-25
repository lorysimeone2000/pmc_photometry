from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # permette di avere la scala logaritmica
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
# image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine

hdu_list = fits.open(image_file)
hdu_list.info()  # dà le informazioni del file

image_data = hdu_list[0].data  # creo la matrice dei valori dei pixel

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data

data_azzerata = np.zeros_like(data)

'''plt.imshow(data_azzerata, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.show()'''


def azzera_pixel_isolati_veloce(dataf, min_neighbors=2, threshold_ratio=0.3):
    """
    Versione ottimizzata per azzerare pixel isolati
    """

    # azzero
    data_azzerataf = np.zeros_like(dataf)

    from scipy import ndimage

    data_pulita = dataf.copy()

    # Soglia per considerare un pixel significativo
    meanf, medianf, stdf = sigma_clipped_stats(dataf, sigma=3.0)
    soglia_minima = 5 * stdf
    mask_significativi = dataf > soglia_minima

    # Crea una mappa dei vicini forti
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Esclude il pixel centrale

    # Per ogni pixel, conta quanti vicini sono sopra threshold_ratio del pixel centrale
    mask_vicini_forti = np.zeros_like(dataf, dtype=bool)

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
            data_azzerataf[y, x] = data_pulita[y, x]
            data_azzerata = data_pulita[y, x]
            data_pulita[y, x] = 0  # Azzera il pixel isolato

    # return data_pulita


azzera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)

# convoluzione

fwhm = float(input("FWHM = "))
size = int(input("Size = "))

kernel = make_2dgaussian_kernel(fwhm, size=size)  # inserisco FWHM e dimensione
convolved_data = convolve(data_azzerata, kernel)
mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)

# Sourcefinder

t = float(input('Threshold (n. sigma) = '))  # threshold
print(f"Deviazione standard convoluzione = {std_c}")
threshold = t * std
print(f"Threshold convoluzione = {threshold}")
n = int(input("Numero minimo pixel contigui = "))

finder = SourceFinder(npixels=n, progress_bar=True)  # inserisco n. minimo di pixel
segment_map = finder(convolved_data, threshold)

# CORREZIONE: Controlla se sono state trovate sorgenti
if segment_map is None:
    print("Nessuna sorgente trovata con i parametri attuali!")
    print("Prova a diminuire la threshold o il numero minimo di pixel")

    # Crea una segment_map vuota per evitare errori
    from photutils.segmentation import SegmentationImage

    segment_map = SegmentationImage(np.zeros_like(data, dtype=int))
else:
    print(f"Trovate {segment_map.nlabels} sorgenti")
    print(segment_map)

plt.imshow(data_azzerata, cmap="grey_r", norm=LogNorm(),
           interpolation='nearest')  # genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis()  # inverto asse y

# tabella sorgenti - SOLO SE CI SONO SORGENTI
if segment_map.nlabels > 0:
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    print(cat)

    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'
    print(tbl)

    positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))  # creo un array di posizioni
    apertures = CircularAperture(positions, r=5.0)  # creo le aperture per ogni posizione
    apertures.plot(color='#0547f9', lw=0.7)
    plt.title("Pixel isolati trovati con l'image segmentation")
else:
    print("Nessuna sorgente da visualizzare")
    tbl = None

plt.show()

# con local peak detection - SOLO SE CI SONO DATI
if data_azzerata is not None and np.any(data_azzerata > 0):
    threshold = 9 * std  # threshold
    print(f"Threshold= {threshold}")
    box_size = int(input("Inserisci il numero di pixel del lato della regione minima: "))  # lato regione minima

    tbl = find_peaks(data_azzerata, threshold, box_size=box_size)  # trovo i picchi
    if len(tbl) > 0:
        tbl['peak_value'].info.format = '%.8g'  # per rappresentare meglio la tabella
        print(tbl[:10])  # rappresento i primi 10 picchi
        print(len(tbl))

        positions = np.transpose((tbl['x_peak'], tbl['y_peak']))  # creo un array di posizioni
        apertures = CircularAperture(positions, r=5.0)  # creo le aperture per ogni posizione
        plt.imshow(data_azzerata, cmap='grey_r', origin='lower', norm=LogNorm(), interpolation='nearest')
        apertures.plot(color='#0547f9', lw=1.5)
        plt.title(f'Pixel isolati trovati con la local peak detection', fontsize=14)
        plt.show()
    else:
        print("Nessun picco trovato con la local peak detection")
else:
    print("Nessun dato disponibile per la local peak detection")


# Il resto del tuo codice per la maschera booleana...

# mask booleano

def maschera_pixel_isolati_veloce(dataf, min_neighbors=2, threshold_ratio=0.3):
    """
    Versione ottimizzata per mascherare pixel isolati
    """

    # azzero
    data_azzerataf = np.zeros_like(dataf)

    matrice_booleana = np.full_like(dataf, False, dtype=bool)

    from scipy import ndimage

    data_pulita = dataf.copy()

    # Soglia per considerare un pixel significativo
    meanf, medianf, stdf = sigma_clipped_stats(dataf, sigma=3.0)
    soglia_minima = 5 * stdf
    mask_significativi = dataf > soglia_minima

    # Crea una mappa dei vicini forti
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Esclude il pixel centrale

    # Per ogni pixel, conta quanti vicini sono sopra threshold_ratio del pixel centrale
    mask_vicini_forti = np.zeros_like(dataf, dtype=bool)

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
            data_azzerataf[y, x] = data_pulita[y, x]
            matrice_booleana[y, x] = True  # rendi True il pixel isolato

    return matrice_booleana

pixel_isolati_bool = maschera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)
plt.imshow(pixel_isolati_bool, cmap='grey', origin='lower', interpolation='nearest')
plt.title("Pixel isolati booleani")
#plt.show()

#print(pixel_isolati_bool)

numero_pixel_true = np.sum(pixel_isolati_bool)
print(f"Numero di pixel isolati immagine vecchia: {numero_pixel_true}")

'''#plt.show()

# conto i pixel isolati nella run

def numero_pixel_isolati(file):
    hdu_listf = fits.open(file)
    #hdu_list.info() # dà le informazioni del file

    image_dataf = hdu_listf[0].data # creo la matrice dei valori dei pixel

    meanf, medianf, stdf = sigma_clipped_stats(image_dataf, sigma=3.0)
    image_dataf = image_dataf - medianf
    dataf = image_dataf

    pixel_isolati_boolf = maschera_pixel_isolati_veloce(dataf, min_neighbors=2, threshold_ratio=0.3)

    numero_pixel_truef = np.sum(pixel_isolati_boolf)

    return numero_pixel_truef

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"
print(f"Numero di pixel isolati immagine vecchia: {numero_pixel_isolati(image_file)}")
image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"
print(f"Numero di pixel isolati immagine vecchia: {numero_pixel_isolati(image_file)}")

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212815.fits"
print(f"Numero di pixel isolati 1: {numero_pixel_isolati(image_file)}")

hdu_list = fits.open(image_file)
#hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data

pixel_isolati_bool = maschera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)
plt.imshow(pixel_isolati_bool, cmap='grey', origin='lower', interpolation='nearest')
plt.title("Pixel isolati booleani 1")
plt.show()


image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212835.fits"
print(f"Numero di pixel isolati 2: {numero_pixel_isolati(image_file)}")

hdu_list = fits.open(image_file)
#hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data

pixel_isolati_bool = maschera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)
plt.imshow(pixel_isolati_bool, cmap='grey', origin='lower', interpolation='nearest')
plt.title("Pixel isolati booleani 2")
plt.show()

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212855.fits"
print(f"Numero di pixel isolati 3: {numero_pixel_isolati(image_file)}")

hdu_list = fits.open(image_file)
#hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data

pixel_isolati_bool = maschera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)
plt.imshow(pixel_isolati_bool, cmap='grey', origin='lower', interpolation='nearest')
plt.title("Pixel isolati booleani 3")
plt.show()

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212915.fits"
print(f"Numero di pixel isolati 4: {numero_pixel_isolati(image_file)}")
pixel_isolati_bool = maschera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)
plt.imshow(pixel_isolati_bool, cmap='grey', origin='lower', interpolation='nearest')
plt.title("Pixel isolati booleani 4")


image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212936.fits"
print(f"Numero di pixel isolati 5: {numero_pixel_isolati(image_file)}")
pixel_isolati_bool = maschera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)
plt.imshow(pixel_isolati_bool, cmap='grey', origin='lower', interpolation='nearest')
plt.title("Pixel isolati booleani 5")

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_220833.fits"
print(f"Numero di pixel isolati penultima immagine: {numero_pixel_isolati(image_file)}")

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_220853.fits"
print(f"Numero di pixel isolati ultima immagine: {numero_pixel_isolati(image_file)}")'''

