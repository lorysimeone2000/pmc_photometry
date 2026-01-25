import numpy as np
import os

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

# Set up astropy
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve

# Set up wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table

# Sopprimo tutti i warning non rilevanti
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning) # cancello gli avvertimenti non rilevanti dovuti alle modifiche di astropy

#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212815.fits"  # prima immagine
# image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"  # prima immagine
# image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine
image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212815.fits"


hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'headerimport numpy as np

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median # tolgo il fondo
data = image_data

print(image_data.shape)

# trovo gli estremi

w = WCS(hdu_list[0].header) # creo un oggetto WCS usando l'header del file FITS,
# che contiene le informazioni per le trasformazioni di coordinate

alto_destra = w.pixel_to_world(3072, 2048)
print(f"Coordinate in alto a destra: {alto_destra}")
aperture1 = CircularAperture((3072,2048), r=300)
basso_sinistra = w.pixel_to_world(0,0)
print(f"Coordinate in basso a sinistra: {basso_sinistra}")
aperture2 = CircularAperture((0,0), r=300)

# Definisco i range di RA e DEC (in gradi) a partire dagli estremi in alto a destra e in basso a sinistra

ra1 = alto_destra.ra.deg  # oppure .hour per avere in ore
ra2 = basso_sinistra.ra.deg
ra_min = np.min(np.array([ra1,ra2]))
ra_max = np.max(np.array([ra1,ra2]))
print(f"RA_min: {ra_min}°")
print(f"RA_max: {ra_max}°")
dec1 = alto_destra.dec.deg
dec2 = basso_sinistra.dec.deg
dec_min = np.min(np.array([dec1,dec2]))
dec_max = np.max(np.array([dec1,dec2]))
print(f"DEC_min: {dec_min}°")
print(f"DEC_max: {dec_max}°")

larghezza = np.abs(ra_max - ra_min)
altezza = np.abs(dec_max - dec_min)
area = larghezza * altezza
print(f"Area: {area} gradi quadrati")

data_pmc = image_data # mi serve per dopo

# rappresento i dati della PMC

'''plt.imshow(image_data, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')
aperture1.plot(color='blue', lw=0.8, alpha=0.5)
aperture2.plot(color='blue', lw=0.8, alpha=0.5)

plt.show()'''

'''# Definisco i range di RA e DEC (in gradi)
ra_min, ra_max = 100.0, 120.0  # esempio range RA
dec_min, dec_max = -10.0, 10.0  # esempio range DEC'''

#image_file = "/home/lorysimeone/tesi_magistrale/prove/astropy/catalog matching/bright star catalogue/asu.fit"
image_file = "/home/lorysimeone/tesi_magistrale/prove/astropy/catalog matching/hipparco/hipparco.fit"

# Apro il catalogo in formato fit

hdu_list = fits.open(image_file)
hdu_list.info()

# I dati sono nella seconda estensione (V_SO_catalog), non nella prima
table_data = Table(hdu_list[1].data)  # Uso l'indice 1 per la seconda estensione

# Esploro il catalogo
print("\n=== INFORMAZIONI DEL CATALOGO ===")
print(f"Numero di stelle nel catalogo: {len(table_data)}")
print(f"Nomi delle colonne: {table_data.colnames}")

# Mostra le prime righe della tabella
print("\n=== PRIME 5 RIGHE DEL CATALOGO ===")
print(table_data[:5])

# Seleziono le righe che soddisfano entrambe le condizioni
subset = table_data[(table_data['_RAJ2000'] >= ra_min) &
                    (table_data['_RAJ2000'] <= ra_max) &
                    (table_data['_DEJ2000'] >= dec_min) &
                    (table_data['_DEJ2000'] <= dec_max)]

print(f"Trovate {len(subset)} stelle nel range specificato")

# Rappresento la mappa RA/DEC delle stelle nel range

plt.figure(figsize=(12, 8))

# Usa la magnitudine per la dimensione e colore dei punti
# Stelle più brillanti (magnitudine minore) = punti più grandi e gialli
#sizes = 50 * (8 - subset['Vmag'])  # Scala le dimensioni
sizes = 15 * (8 - subset['Vmag'])  # Scala le dimensioni
sizes = np.clip(sizes, 10, 200)  # Limita dimensioni min/max

# Colori basati sulla magnitudine
colors = subset['Vmag']

#scatter = plt.scatter(subset['_RAJ2000'], subset['_DEJ2000'],
#                     c=colors, s=sizes, alpha=0.7, cmap='viridis_r')

scatter = plt.scatter(subset['_RAJ2000'], subset['_DEJ2000'],
                      c=colors, s = sizes, alpha=0.7, cmap='viridis_r')

plt.colorbar(scatter, label='Magnitudine Visuale (Vmag)')
plt.xlabel('Ascensione Retta (RA J2000) [gradi]')
plt.ylabel('Declinazione (DEC J2000) [gradi]')
plt.title(f'Mappa del Catalogo Stellare Hipparco ({len(subset)} stelle), catturata dalla pmc')
plt.gca().invert_xaxis()  # RA aumenta verso est (convenzione astronomica)
plt.grid(True, alpha=0.3)
plt.xlim(ra_max, ra_min)  # Nota: invertito perché RA diminuisce verso destra
plt.ylim(dec_min, dec_max)

plt.show()

# Faccio il matching

posizioni_vere_celesti = SkyCoord(ra=subset['_RAJ2000']*u.degree,
                                 dec=subset['_DEJ2000']*u.degree,
                                 frame='icrs')
posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti) # converto da celesti a pixel
posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

apertures = CircularAperture(posizioni_vere_pixel, r=10.0) # crea cerchi di raggio r pixel

# Rappresento il matching

plt.figure(figsize=(14, 8))
plt.imshow(data_pmc, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')
apertures.plot(color='blue', lw=1.5, alpha=0.3, fill=True)

posizioni_vere_celesti = SkyCoord(ra=subset['_RAJ2000']*u.degree,
                                 dec=subset['_DEJ2000']*u.degree,
                                 frame='icrs')
posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti)
posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

magnitudini = subset['Vmag']

# Parametri per i raggi
raggio_min = 4.0
raggio_max = 20.0
raggi = raggio_max - (magnitudini - magnitudini.min()) * (raggio_max - raggio_min) / (magnitudini.max() - magnitudini.min())

# Crea una scala di colori
cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())

# Creo una figura e un asse esplicitamente
#fig, ax = plt.subplots(figsize=(14, 8))
#plt.imshow(data_pmc, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')

# Disegno ogni apertura
for i, (position, radius) in enumerate(zip(posizioni_vere_pixel, raggi)):
    color = cmap(norm(magnitudini[i]))
    aperture = CircularAperture(position, r=radius)
    aperture.plot(color=color, lw=1.0, alpha=0.7, fill=True)

# Aggiungo la legenda - ORA specificando l'asse
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label='Magnitudine V')  # AGGIUNGI ax=plt.gca()

# Rappresento il matching aggiungendoci l'image segmentation

# convoluzione

fwhm = float(input("FWHM = " ))
size = int(input("Size = " ))

kernel = make_2dgaussian_kernel(fwhm, size=size)  # inserisco FWHM e dimensione
convolved_data = convolve(data, kernel)
mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)
t = float(input('Threshold (n. sigma) = ')) # threshold
print(f"Deviazione standard convoluzione = {std_c}")
threshold = t * std
print(f"Threshold convoluzione = {threshold}")
n = int(input("Numero minimo pixel contigui = " ))

finder = SourceFinder(npixels=n, progress_bar=True) # inserisco n. minimo di pixel
segment_map = finder(convolved_data, threshold)

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
apertures.plot(color='red', lw=1.)

plt.title(f'Matching: {len(subset)} stelle - Cerchi dimensionati per magnitudine\n(Threshold = {t} σ, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')

plt.show()

print(np.max(subset['Vmag']))