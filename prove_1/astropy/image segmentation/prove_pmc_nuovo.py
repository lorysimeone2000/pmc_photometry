# In questo codice prendo i parametri da un file a parte e ci applico l'image segmentation su un'immagine sola,
# filtrando i pixel isolati che vengono scambiati per sorgenti

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

#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"  # prima immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212815.fits"
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212835.fits"
image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212855.fits"

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
print(f"std prima: {std}")
image_data = image_data - median
data = image_data
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
print(f"std dopo: {std}")



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

'''
plt.imshow(convolved_data, cmap='grey_r', origin='lower', norm=LogNorm(), interpolation='nearest')
plt.title(f'Dati Convoluti con Kernel Gaussiano\n(FWHM = {fwhm} pixel e size = {size})', fontsize=14, fontweight='bold')
plt.colorbar()

plt.show()'''

# Sourcefinder

# t = float(input('Threshold (n. sigma) = ')) # threshold
t = parametri['threshold_sigma']
print(f"Deviazione standard convoluzione = {std_c}")
print(f"Threshold (n. sigma) = {t}")
# threshold = t * std # per adesso lascio stare questo metodo
threshold = parametri['threshold_assoluta']
print(f"Threshold convoluzione = {threshold}")
# n = int(input("Numero minimo pixel contigui = " ))
n = parametri['pixel']
print(f"Numero minimo di pixel contigui = {n}")

finder = SourceFinder(npixels=n, progress_bar=True) # inserisco n. minimo di pixel
segment_map = finder(convolved_data, threshold)
'''print(segment_map)

plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
plt.title(f'Sorgenti Rilevate\n(Threshold = {t} σ, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)', fontsize=14, fontweight='bold')

plt.show()'''

# tabella sorgenti con alche alcuni pixel di errore

cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
print(cat)

tbl = cat.to_table()
tbl['xcentroid'].info.format = '.2f'  # optional format
tbl['ycentroid'].info.format = '.2f'
tbl['kron_flux'].info.format = '.2f'
# print(tbl)

# soglia da verificare
soglia = 2.5 # soglia assoluta usata per i pixel isolati
soglia_relativa = 0.4 # rispetto al massimo

# creo il catalogo COMPLETO
tbl_completo = tbl

print(f"Sorgenti totali iniziali: {len(tbl_completo)}")

# lista per tenere gli indici delle righe valide
indici_validi = []
indici_non_validi = []
indici_in_aloni = []

# per ogni sorgente nel catalogo, verifica la condizione
for i, sorgente in enumerate(tbl_completo):
    #label = i + 1  # Le righe della tabella corrispondono alle labels 1, 2, 3, ...
    label = sorgente['label']

    # Ottieni i pixel della sorgente
    mask_sorgente = (segment_map.data == label)
    valori_originali = data[mask_sorgente]

    # Conta i pixel sopra soglia
    pixel_sopra_soglia_assoluta = np.sum(valori_originali > soglia)
    pixel_sopra_soglia_relativa = np.sum(valori_originali >  soglia_relativa*sorgente['max_value']) # conto tutti i pixel che superano il 40% del massimo

    if i <3:
        print(f"Sorgente {label}: {pixel_sopra_soglia_assoluta} pixel sopra soglia")

    if pixel_sopra_soglia_assoluta >= 2 and pixel_sopra_soglia_relativa >= 2:
        indici_validi.append(i)
        if i <= 2: print(f"Sorgente {label}  ✅ MANTENUTA")

    else:
        indici_non_validi.append(i)
        if i <= 2: print(f"Sorgente {label}  ❌ RIMOSSA")
    if pixel_sopra_soglia_assoluta >= 2 and pixel_sopra_soglia_relativa < 2:
        indici_in_aloni.append(i)


# creo il catalogo FILTRATO
tbl_filtrato = tbl_completo[indici_validi]

new_labels_validi = np.arange(1, len(tbl_filtrato) + 1) # creo una nuova colonna con i label progressivi da 1 a N
tbl_filtrato['label'] = new_labels_validi # sostituisco la colonna 'label' con i nuovi label progressivi

tbl_non_validi = tbl_completo[indici_non_validi]

new_labels_non_validi = np.arange(1, len(tbl_non_validi) + 1) # creo una nuova colonna con i label progressivi da 1 a N
tbl_non_validi['label'] = new_labels_non_validi # sostituisco la colonna 'label' con i nuovi label progressivi

print(f"\nSorgenti dopo filtro: {len(tbl_filtrato)} / {len(tbl_completo)}")
print(f"Tabella filtrata:\n {tbl_filtrato}")

positions = np.transpose((tbl_filtrato['xcentroid'], tbl_filtrato['ycentroid'])) # creo un array di posizioni
apertures = CircularAperture(positions, r=5.0) # creo le aperture per ogni posizione

plt.imshow(image_data, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.title(f'Sorgenti Rilevate'
          f'\n(Threshold = {t} σ, n. pixel min = {n}, 'f' FWHM = {fwhm}, dimensioni kernel = {size} pixel)',
          fontsize=14, fontweight='bold')
plt.colorbar()
apertures.plot(color='#0547f9', lw=0.7)

plt.show()

'''# pixel in aloni
tbl_pixel_in_aloni = tbl_completo[indici_in_aloni]
print(f"Tabella pixel in aloni: \n {tbl_pixel_in_aloni}")'''

'''valori_pixel__non_validi = tbl_non_validi['max_value']
print(f"Tabella pixel di errore contati comunque:\n {tbl_non_validi}")'''

'''
plt.bar(range(len(valori_pixel__non_validi)), valori_pixel__non_validi, color='skyblue', edgecolor='navy', alpha=0.7, width=0.8, label='Valori pixel isolati scambiati per stelle con image segmentation')
plt.title('Valori pixel isolati scambiati per stelle con image segmentation')

plt.show()'''

