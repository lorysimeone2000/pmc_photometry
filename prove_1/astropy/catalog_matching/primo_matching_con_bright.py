import numpy as np
import os

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

from astropy.io import fits
from astropy.utils.data import download_file

from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture

# Set up wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.table import Table

warnings.filterwarnings('ignore', category=FITSFixedWarning) # cancello gli avvertimenti non rilevanti dovuti alle modifiche di astropy

image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"  # prima immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine
# image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212815.fits"

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'headerimport numpy as np

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'header

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

alto_destra = w.pixel_to_world(3072, 2048)
alto_sinistra = w.pixel_to_world(3072, 0)
# print(f"Coordinate in alto a destra: {alto_destra}")
basso_sinistra = w.pixel_to_world(0, 0)
basso_destra = w.pixel_to_world(0, 2048)

print(f"Coordinate in alto a destra: {alto_destra}")
print(f"Coordinate in basso a sinistra: {basso_sinistra}")
print(f"Coordinate in alto a sinistra: {alto_sinistra}")
print(f"Coordinate in basso a destra con wcs_pix2world: {basso_destra}")

ra_alto_destra = alto_destra.ra.deg
ra_basso_destra = basso_destra.ra.deg
ra_basso_sinistra = basso_sinistra.ra.deg
ra_alto_sinistra = alto_sinistra.ra.deg
dec_alto_destra = alto_destra.dec.deg
dec_basso_destra = basso_destra.dec.deg
dec_basso_sinistra = basso_sinistra.dec.deg
dec_alto_sinistra = alto_sinistra.dec.deg

ra_max = np.max(np.array([ra_alto_destra, ra_basso_sinistra, ra_basso_destra, ra_alto_sinistra]))
ra_min = np.min(np.array([ra_alto_destra, ra_basso_sinistra, ra_basso_destra, ra_alto_sinistra]))
dec_max = np.max(np.array([dec_alto_destra, dec_basso_sinistra, dec_basso_destra, dec_alto_sinistra]))
dec_min = np.min(np.array([dec_alto_destra, dec_basso_sinistra, dec_basso_destra, dec_alto_sinistra]))

data_pmc = image_data # mi serve per dopo

# Creo la tabella del catalogo

# coordinate centro
image_header = hdu_list[0].header
ra_centro = image_header["RA"]
print("RA centro: ", ra_centro)
dec_centro = image_header["DEC"]
print("DEC centro: ", dec_centro)

larghezza = alto_destra.separation(alto_sinistra).degree
print(f"Larghezza: {larghezza}")
altezza = alto_destra.separation(basso_destra).degree
print(f"Altezza: {altezza}")

data_pmc = image_data # mi serve per dopo

# rappresento i dati della PMC

'''plt.imshow(image_data, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')
aperture1.plot(color='blue', lw=0.8, alpha=0.5)
aperture2.plot(color='blue', lw=0.8, alpha=0.5)

plt.show()'''

'''# Definisco i range di RA e DEC (in gradi)
ra_min, ra_max = 100.0, 120.0  # esempio range RA
dec_min, dec_max = -10.0, 10.0  # esempio range DEC'''

image_file = "/home/lorysimeone/tesi_magistrale/prove/astropy/catalog matching/bright star catalogue/asu.fit"
#image_file = "/home/lorysimeone/tesi_magistrale/prove/astropy/catalog matching/hipparco/hipparco.fit"

# Apri il file FITS
hdu_list = fits.open(image_file)
hdu_list.info()

# I dati sono nella seconda estensione (V_SO_catalog), non nella prima
table_data = Table(hdu_list[1].data)  # Uso l'indice 1 per la seconda estensione

# Seleziono le righe che soddisfano entrambe le condizioni
subset = table_data[(table_data['_RAJ2000'] >= ra_min) &
                    (table_data['_RAJ2000'] <= ra_max) &
                    (table_data['_DEJ2000'] >= dec_min) &
                    (table_data['_DEJ2000'] <= dec_max)]
tbl_catalogo = subset

print(f"Trovate {len(subset)} stelle nel range specificato")

magnitudini = tbl_catalogo['Vmag']
mag_min_del_catalogo = np.min(magnitudini)
indice_mag_min = np.argmin(magnitudini)
stella_piu_luminosa = tbl_catalogo[indice_mag_min] # ho l'intera riga della stella con amgnitudine massima
coo = SkyCoord(stella_piu_luminosa['_RAJ2000'], stella_piu_luminosa['_DEJ2000'], unit=u.deg)
coordinate_pixel_stella = w.world_to_pixel(coo)
print(f"Dati della stella più luminosa:")
print(f"  RA: {stella_piu_luminosa['_RAJ2000']}°")
print(f"  Dec: {stella_piu_luminosa['_DEJ2000']}°")
print(f"  Vmag: {stella_piu_luminosa['Vmag']}")
print(f"  coordinate pixel: {coordinate_pixel_stella}")

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
plt.title(f'Mappa del Catalogo Stellare Bright Star ({len(subset)} stelle), una parte')
plt.gca().invert_xaxis()  # RA aumenta verso est (convenzione astronomica)
plt.grid(True, alpha=0.3)
plt.xlim(ra_max, ra_min)  # Nota: invertito perché RA diminuisce verso destra
plt.ylim(dec_min, dec_max)

plt.show()

# Rappresento il matching

plt.imshow(data_pmc, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')

posizioni_vere_celesti = SkyCoord(ra=subset['_RAJ2000']*u.degree,
                                 dec=subset['_DEJ2000']*u.degree,
                                 frame='icrs')
posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti) # converto da celesti a pixel
posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

apertures = CircularAperture(posizioni_vere_pixel, r=10.0) # crea cerchi di raggio r pixel
apertures.plot(color='blue', lw=1.5, alpha=0.3, fill=True)

# Rappresento il matching con legenda

plt.figure(figsize=(14, 8))
plt.imshow(data_pmc, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')

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

# Crea una figura e un asse esplicitamente
fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(data_pmc, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')

# Disegna ogni apertura
for i, (position, radius) in enumerate(zip(posizioni_vere_pixel, raggi)):
    color = cmap(norm(magnitudini[i]))
    aperture = CircularAperture(position, r=radius)
    aperture.plot(color=color, lw=1.0, alpha=0.7, fill=True, ax=ax)

# Aggiungi legenda - ORA specificando l'asse
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Magnitudine V')
cbar.set_label('Magnitudine Visuale (Vmag)', rotation=270, labelpad=15)

ax.set_title(f'Matching: {len(subset)} stelle - Cerchi dimensionati per magnitudine')

plt.show()