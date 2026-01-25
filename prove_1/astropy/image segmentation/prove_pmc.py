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

'''plt.imshow(image_data, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.title('Dati originali', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis() # inverto asse y
plt.colorbar()

plt.show()'''

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

print("--------------------------------------- Inizio analisi ---------------------------------------")

# convoluzione

# fwhm = float(input("FWHM = " ))
fwhm = parametri['fwhm']
# size = int(input("Size = " ))
size = parametri['size']

kernel = make_2dgaussian_kernel(fwhm, size=size)  # inserisco FWHM e dimensione
convolved_data = convolve(data, kernel)
mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)

plt.imshow(convolved_data, cmap='grey_r', origin='lower', norm=LogNorm(), interpolation='nearest')
plt.title(f'Dati Convoluti con Kernel Gaussiano\n(FWHM = {fwhm} pixel e size = {size})', fontsize=14, fontweight='bold')
plt.colorbar()

plt.show()

# Sourcefinder

# per adesso uso una soglia assoluta (3.610293688717009)

# t = float(input('Threshold (n. sigma) = ')) # threshold
t = parametri['threshold_sigma']
print(f"Deviazione standard convoluzione = {std_c}")
print(f"Threshold (n. sigma) = {t}")
# threshold = t * std per adesso lascio stare questo metodo
threshold = parametri['threshold_assoluta']
print(f"Threshold convoluzione = {threshold}")
# n = int(input("Numero minimo pixel contigui = " ))
n = parametri['pixel']
print(f"Numero minimo di pixel contigui = {n}")

finder = SourceFinder(npixels=n, progress_bar=True) # inserisco n. minimo di pixel
segment_map = finder(convolved_data, threshold)
print(segment_map)

plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
plt.title(f'Sorgenti Rilevate\n(Threshold = {t} σ, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)', fontsize=14, fontweight='bold')

plt.show()

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

plt.imshow(image_data, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.title(f'Sorgenti Rilevate'
          f'\n(Threshold = {t} σ, n. pixel min = {n}, 'f' FWHM = {fwhm}, dimensioni kernel = {size} pixel)',
          fontsize=14, fontweight='bold')
plt.colorbar()
apertures.plot(color='#0547f9', lw=0.7)

plt.show()

# prov