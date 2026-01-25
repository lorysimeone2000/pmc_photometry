import numpy as np
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
import os

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

# Set up astropy
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from astropy.visualization import SqrtStretch
from astropy.table import Table
from astropy.table import MaskedColumn, QTable
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
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning

mag_max = 14

# Lettura parametri
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

# questa funzione restituisce la tabella delle sorgenti trovate
def analisi_image_segmentation(data):
    """
    Esegue image segmentation su un'immagine FITS e restituisce la tabella filtrata delle sorgenti
    Il fondo deve essere rimosso

    Returns:
    astropy.table.Table: Tabella delle sorgenti filtrate con label riordinati
    """

    mean, median, std = sigma_clipped_stats(data, sigma=3.0) # nel caso me lo chiedessi, la std prima o dopo aver sottratto il fondo è la stessa

    # Convoluzione
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)

    # Sourcefinder
    t = parametri['threshold_sigma']
    # threshold = t * std # per adesso lascio stare questo metodo
    threshold = parametri['threshold_assoluta']
    n = parametri['pixel']

    finder = SourceFinder(npixels=n, progress_bar=True)
    segment_map = finder(convolved_data, threshold)

    # Catalogo sorgenti
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'

    # filtraggio sorgenti
    soglia_assoluta = 2.5
    soglia_relativa = 0.4

    indici_validi = []
    indici_non_validi = []

    for i, sorgente in enumerate(tbl):
        label = sorgente['label']
        mask_sorgente = (segment_map.data == label)
        valori_originali = data[mask_sorgente]

        pixel_sopra_soglia_assoluta = np.sum(valori_originali > soglia_assoluta)
        pixel_sopra_soglia_relativa = np.sum(valori_originali > soglia_relativa * sorgente['max_value'])

        if pixel_sopra_soglia_assoluta >= 2 and pixel_sopra_soglia_relativa >= 2:
            indici_validi.append(i)

    # Creazione tabella filtrata
    tbl_filtrato = tbl[indici_validi]
    new_labels_validi = np.arange(1, len(tbl_filtrato) + 1)
    tbl_filtrato['label'] = new_labels_validi

    # print(f"Sorgenti dopo filtro: {len(tbl_filtrato)} / {len(tbl)}")

    return tbl_filtrato

# inizializzo Vizier con i suoi parametri di default

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1
)

catalogo = vizier
print(f"Questo è il catalogo: \n{catalogo}")
catalogs = Vizier.get_catalogs("II/389/ps1_dr2")
print(catalogs)

# image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"  # prima immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine
# image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212855.fits"
image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_215217.fits"

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

image_data = hdu_list[0].data # creo la matrice dei valori dei pixel
#image_data = hdu_list[0].data[961:1086 , 2276:2438] # Ritaglia un'area tot x tot pixel
#print(hdu_list[0].header) #mette tutti i dati dell'headerimport numpy as np

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median # tolgo il fondo
data = image_data

'''plt.imshow(image_data, cmap="grey_r", norm=LogNorm(), interpolation='nearest') #genero l'immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.colorbar()

plt.show()'''

print(image_data.shape)

# trovo gli estremi

w = WCS(hdu_list[0].header) # creo un oggetto WCS usando l'header del file FITS,
# che contiene le informazioni per le trasformazioni di coordinate
hdu_list.close()

ny, nx = data.shape
xc, yc = nx / 2, ny / 2 # coordinate del centro in pixel

# definisco i quattro angoli dell'immagine in pixel
pixels = np.array([
    [0, 0],
    [0, ny - 1],
    [nx - 1, 0],
    [nx - 1, ny - 1]
])

print("Estremi: ",pixels)

# Definisco i range di RA e DEC (in gradi) a partire dagli estremi in alto a destra e in basso a sinistra

world = w.wcs_pix2world(pixels, 0) # converte i pixel in coordinate celesti (RA, Dec)

ra_vals = world[:, 0]
dec_vals = world[:, 1]

# calcolo minimi e massimi
ra_min, ra_max = np.min(ra_vals), np.max(ra_vals)
dec_min, dec_max = np.min(dec_vals), np.max(dec_vals)


'''
ra1 = alto_destra.ra.deg  # oppure .hour per avere in ore
ra2 = basso_sinistra.ra.deg
ra_min = np.min(np.array([ra1, ra2]))
ra_max = np.max(np.array([ra1, ra2]))
print(f"RA_min: {ra_min}°")
print(f"RA_max: {ra_max}°")
dec1 = alto_destra.dec.deg
dec2 = basso_sinistra.dec.deg
dec_min = np.min(np.array([dec1, dec2]))
dec_max = np.max(np.array([dec1, dec2]))
print(f"DEC_min: {dec_min}°")
print(f"DEC_max: {dec_max}°")
'''

data_pmc = image_data # mi serve per dopo

# Creo la tabella del catalogo

# coordinate centro
image_header = hdu_list[0].header
ra_centro = image_header["RA"]
print("RA centro: ", ra_centro)
dec_centro = image_header["DEC"]
print("DEC centro: ", dec_centro)

larghezza = np.abs(ra_max - ra_min)
print(f"Larghezza: {larghezza}")
altezza = np.abs(dec_max - dec_min)
print(f"Altezza: {altezza}")

alto_destra = SkyCoord(ra_max, dec_max, unit=u.deg)
print(f"Coordinate in alto a destra: {alto_destra}")
basso_sinistra = SkyCoord(ra_min, dec_min, unit=u.deg)
print(f"Coordinate in basso a sinistra: {basso_sinistra}")

# inizializzo Vizier con i suoi parametri di default

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1
)

catalogo = vizier
print(f"Questo è il catalogo: \n{catalogo}")

catalogs = Vizier.get_catalogs("II/389/ps1_dr2")
print(catalogs)
print(f"Descrizione gmag: {catalogs[0]["gmag"].description}")

centro = SkyCoord(ra_centro, dec_centro, unit=u.deg)

'''riquadro = vizier.query_region(coord.SkyCoord(ra=ra_centro, dec=dec_centro,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                        radius= Angle(centro.separation(alto_destra), "deg"),
                        column_filters={'gmag': f'<{mag_max}'},)'''


riquadro = vizier.query_region(coord.SkyCoord(ra=ra_centro, dec=dec_centro,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                        radius= Angle(centro.separation(alto_destra), "deg"),
                        )
tbl_cazzata = riquadro[0]
df = tbl_cazzata.to_pandas()
df_filtrate = df[(df['gmag'] < mag_max) | (df['gmag'].isnull())]
tbl_catalogo_esteso = QTable.from_pandas(df_filtrate)

print(f"Riquadro esteso funzionante: \n{riquadro}")

# tbl_catalogo_esteso = riquadro[0]
tbl_catalogo = tbl_catalogo_esteso[(tbl_catalogo_esteso['RAJ2000'] >= ra_min) &
                            (tbl_catalogo_esteso['RAJ2000'] <= ra_max) &
                            (tbl_catalogo_esteso['DEJ2000'] >= dec_min) &
                            (tbl_catalogo_esteso['DEJ2000'] <= dec_max)]
magnitudini = tbl_catalogo['gmag']
mag_min_del_catalogo = np.min(magnitudini)
indice_mag_min = np.argmin(magnitudini)
stella_piu_luminosa = tbl_catalogo[indice_mag_min] # ho l'intera riga della stella con amgnitudine massima
print(f"Dati della stella più luminosa:")
print(f"  RA: {stella_piu_luminosa['RAJ2000']}°")
print(f"  Dec: {stella_piu_luminosa['DEJ2000']}°")
print(f"  gmag: {stella_piu_luminosa['gmag']}")
print(tbl_catalogo)

# Usa la magnitudine per la dimensione e colore dei punti
# Stelle più brillanti (magnitudine minore) = punti più grandi e gialli
#sizes = 50 * (8 - tbl_catalogo['Vmag'])  # Scala le dimensioni
sizes = 15 * (8 - tbl_catalogo['gmag'])  # Scala le dimensioni
sizes = np.clip(sizes, 10, 200)  # Limita dimensioni min/max

# Colori basati sulla magnitudine
colors = tbl_catalogo['gmag']

posizioni_vere_celesti = SkyCoord(ra=tbl_catalogo['RAJ2000'],
                                 dec=tbl_catalogo['DEJ2000'],
                                 frame='icrs')
scatter = plt.scatter(tbl_catalogo['RAJ2000'], tbl_catalogo['DEJ2000'],
                      c=colors, s = sizes, alpha=0.7, cmap='viridis_r')
plt.colorbar(scatter, label='Magnitudine Visuale (Vmag)')


plt.gca().invert_xaxis()  # RA aumenta verso est (convenzione astronomica)
plt.grid(True, alpha=0.3)
plt.xlim(ra_max, ra_min)  # Nota: invertito perché RA diminuisce verso destra
plt.ylim(dec_min, dec_max)
plt.xlabel('Ascensione Retta (RA J2000) [gradi]')
plt.ylabel('Declinazione (DEC J2000) [gradi]')
plt.title(f'Mappa del catalogo Vizier ({len(tbl_catalogo)} stelle), catturata dalla pmc')

plt.show()

# Matching con l'immagine della PMC

posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti) # converto da celesti a pixel
posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

magnitudini = tbl_catalogo['gmag']

# Parametri per i raggi
raggio_min = 4.0
raggio_max = 20.0
raggi = raggio_max - (magnitudini - magnitudini.min()) * (raggio_max - raggio_min) / (magnitudini.max() - magnitudini.min())

# Crea una scala di colori
cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())

fig = plt.figure(figsize=(12, 8))
ax = plt.subplot()

# Disegno cerchi colorati delle stelle catalogate
for i, (position, radius) in enumerate(zip(posizioni_vere_pixel, raggi)):
    color = cmap(norm(magnitudini[i]))
    aperture = CircularAperture(position, r=radius)
    aperture.plot(color=color, lw=1.0, alpha=0.6, fill=True)

# Aggiungo la legenda - ORA specificando l'asse
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label='Magnitudine V')

# Rappresento il matching aggiungendoci l'image segmentation

ax.imshow(data_pmc, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')

tbl = analisi_image_segmentation(data)

tbl['xcentroid'].info.format = '.2f'  # optional format
tbl['ycentroid'].info.format = '.2f'
tbl['kron_flux'].info.format = '.2f'
print(tbl)

positions = np.transpose((tbl['xcentroid'], tbl['ycentroid'])) # creo un array di posizioni
apertures = CircularAperture(positions, r=5.0) # creo le aperture per ogni posizione
apertures.plot(color='red', lw=1.)

# MODIFICA: Imposto i limiti degli assi con ra_vals e dec_vals
ax.set_xlim(ra_min, ra_max)  # Usa ra_vals come limite X
ax.set_ylim(dec_min, dec_max)  # Usa dec_vals come limite Y

# MODIFICA: Imposto i label degli assi per RA/DEC in gradi
ax.set_xlabel('Ascensione Retta (gradi)')
ax.set_ylabel('Declinazione (gradi)')
ax.set_title(f'Matching: {len(tbl_catalogo)} stelle del catalogo II/389/ps1_dr2 \n Cerchi dimensionati per magnitudine (<{mag_max})\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')

# MODIFICA: Creo funzioni per convertire pixel in coordinate
def pixel_to_ra(x):
    """Converte coordinate pixel X in RA (gradi)"""
    coord = w.pixel_to_world(x, 0)  # y=0 per semplicità
    return coord.ra.deg

def pixel_to_dec(y):
    """Converte coordinate pixel Y in DEC (gradi)"""
    coord = w.pixel_to_world(0, y)  # x=0 per semplicità
    return coord.dec.deg

# MODIFICA: Imposto i formatter per convertire automaticamente pixel→gradi
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{pixel_to_ra(x):.3f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{pixel_to_dec(x):.3f}'))

# MODIFICA: Calcolo i tick appropriati per le coordinate
x_ticks = np.linspace(0, data_pmc.shape[1], 6)  # 6 tick lungo X
y_ticks = np.linspace(0, data_pmc.shape[0], 6)  # 6 tick lungo Y

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# MODIFICA: Aggiungo griglia per le coordinate celesti
ax.grid(True, color='white', alpha=0.3)

'''plt.title(f'Matching: {len(tbl_catalogo)} stelle del catalogo II/389/ps1_dr2 \n Cerchi dimensionati per magnitudine (<{mag_max})\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')'''

# legenda
legend_elements = [
    # Stelle catalogate (cerchi colorati)
    Circle((0.5, 0.5), 0.4,facecolor='blue', alpha=0.7, edgecolor='black', linewidth=1,
          label=f'Stelle catalogo ({len(tbl_catalogo)} oggetti)'),

    # Sorgenti rilevate (aperture rosse)
    Line2D([0], [0], marker='o', color='red', linestyle='None',
           markersize=8, markerfacecolor='none', markeredgewidth=1,
           label=f'Sorgenti rilevate ({len(tbl)} oggetti)')
]

# Aggiungi la legenda
ax.legend(handles=legend_elements, loc='upper right',
           framealpha=0.85, fancybox=True, shadow=True)

plt.show()