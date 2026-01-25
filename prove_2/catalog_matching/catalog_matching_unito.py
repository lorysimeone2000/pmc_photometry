import numpy as np
import pandas as pd
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
from astropy.table import Table, vstack
import warnings
from astropy.wcs import FITSFixedWarning

# Soppressione warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)

# Lettura parametri
parametri = {}
with open('/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt', 'r') as file:
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


def converti_valore(valore):
    """
    Converte una stringa nel tipo di dato appropriato.
    Prova in ordine: int, float, mantiene stringa se non è convertibile.
    """
    valore = valore.strip()

    # Se è vuoto, restituisci stringa vuota
    if not valore:
        return valore

    # Prova a convertire in int
    try:
        return int(valore)
    except ValueError:
        pass

    # Prova a convertire in float
    try:
        return float(valore)
    except ValueError:
        pass

    # Prova a riconoscere booleani FITS
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False

    # Altrimenti restituisci la stringa originale
    return valore

def leggi_header_da_csv(filename):
    """Legge l'header FITS dal file CSV"""
    header_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                # Rimuovi il '#' e dividi chiave-valore
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':  # Fine dell'header
                break

    return header_dict

# run = int(input("Quale run vuoi elaborare: ")) # numero run: 1, 2 o 3
run = 1

# cartella contenente i file CSV delle stelle catalogate
# cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
file_csv =  "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_1/run_1_stelle_trovate_e_catalogate_immagine_035.csv"
dataframe = pd.read_csv(file_csv, comment='#')
# tbl = Table.from_pandas(dataframe)
header = leggi_header_da_csv(file_csv)
image_file = header['PERCORSO_FILE']

# image_file = "/home/lorysimeone/tesi_magistrale/prove_1/20250106_231255.fits"  # prima immagine
# image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine
# image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212855.fits"
# image_file = "/home/lorysimeone/tesi_magistrale/20250120_run1/20250120_212815.fits"

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

# inizializzo Vizier con i suoi parametri di default

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID','RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)

catalogo = vizier
print(f"Questo è il catalogo: \n{catalogo}")

catalogs = Vizier.get_catalogs("II/389/ps1_dr2")
print(catalogs)
print(f"Descrizione gmag: {catalogs[0]["objID"].description}")

centro = SkyCoord(ra_centro, dec_centro, unit=u.deg)

mag_max = 14
mag_limite_tra_hipparco_e_vizier = 7.

riquadro_vizier = vizier.query_region(coord.SkyCoord(ra=ra_centro, dec=dec_centro,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                        radius= Angle(centro.separation(alto_destra), "deg"),
                        column_filters={'gmag': f'<{mag_max}'},

                        )
tbl_catalogo_vizier = riquadro_vizier[0]

print(f"Query circolare ricavata dalla mezza diagonale: \n{riquadro_vizier}")

# tolgo le stelle che non vorrei siano prese da Vizier
tbl_catalogo_vizier = tbl_catalogo_vizier[(tbl_catalogo_vizier['gmag'] >= mag_limite_tra_hipparco_e_vizier)]

# aggiungo il catalogo Hipparco per le magnitudini inferiori a 7

file_hipparco = "/home/lorysimeone/tesi_magistrale/prove_2/cataloghi_scaricati/hipparco.fit"

# Apro il catalogo in formato fit

hdu_list_hipparco = fits.open(file_hipparco)
print("Info catalogo Hipparco: \n",hdu_list_hipparco)

# I dati sono nella seconda estensione (V_SO_catalog), non nella prima
table_data = Table(hdu_list_hipparco[1].data)  # Uso l'indice 1 per la seconda estensione
# tolgo le stelle che non vorrei siano prese da Hipparco
tbl_catalogo_hipparco = table_data[(table_data['Vmag']) < mag_limite_tra_hipparco_e_vizier]

# Esploro il catalogo
print("\n=== INFORMAZIONI DEL CATALOGO ===")
print(f"Numero di stelle nel catalogo: {len(tbl_catalogo_hipparco)}")
print(f"Nomi delle colonne: {tbl_catalogo_hipparco.colnames}")

# adesso mi costruisco una mia tabella astropy complessiva

# creco la colonna del catalogo di riferimento
nome_catalogo_vizier = []
for i in range(len(tbl_catalogo_vizier)): nome_catalogo_vizier.append("II/389/ps1_dr2")
nome_catalogo_hipparco = []
for i in range(len(tbl_catalogo_hipparco)): nome_catalogo_hipparco.append("I/239/hip_main")

colonne_vizier = {
    'ID': tbl_catalogo_vizier['objID'],
    'RAJ2000': tbl_catalogo_vizier['RAJ2000'],
    'DEJ2000': tbl_catalogo_vizier['DEJ2000'],
    'Mag': tbl_catalogo_vizier['gmag'],
    'Catalogo': nome_catalogo_vizier
}

colonne_hipparco = {
    'ID': tbl_catalogo_hipparco['HIP'],
    'RAJ2000': tbl_catalogo_hipparco['_RAJ2000'],
    'DEJ2000': tbl_catalogo_hipparco['_DEJ2000'],
    'Mag': tbl_catalogo_hipparco['Vmag'],
    'Catalogo': nome_catalogo_hipparco
}

t1 = Table(colonne_vizier)
t2 = Table(colonne_hipparco)

tbl_unita_estesa = vstack([t1, t2])
tbl_unita_estesa['Mag'].description = 'Magnitudine AB nel filtro g di Pan-STARRS'
print("Tabella estesa:\n", tbl_unita_estesa)

# questa parte va fatta solo per ritagliare correttamente il riquadro
corners_pix = np.array([
    [0, 0],
    [0, data.shape[0]],
    [data.shape[1], data.shape[0]],
    [data.shape[1], 0]
])
corners_world = w.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
from shapely.geometry import Point, Polygon

polygon_coords = np.column_stack((corners_world.ra.deg, corners_world.dec.deg))
poly = Polygon(polygon_coords)
# creo una liste degli elementi del catalogo che rientrano nel riquadro in coordinate celesti
mask = [
    poly.contains(Point(ra, dec))
    for ra, dec in zip(tbl_unita_estesa['RAJ2000'], tbl_unita_estesa['DEJ2000'])
]
tbl_cataloghi = tbl_unita_estesa[mask]

# Fine creazione tabella complessiva

magnitudini = tbl_cataloghi['Mag']
mag_min_del_catalogo = np.min(magnitudini)
indice_mag_min = np.argmin(magnitudini)
stella_piu_luminosa = tbl_cataloghi[indice_mag_min] # ho l'intera riga della stella con amgnitudine massima
coo = SkyCoord(stella_piu_luminosa['RAJ2000'], stella_piu_luminosa['DEJ2000'], unit=u.deg)
coordinate_pixel_stella = w.world_to_pixel(coo)
print(f"Dati della stella più luminosa:")
print(f"  RA: {stella_piu_luminosa['RAJ2000']}°")
print(f"  Dec: {stella_piu_luminosa['DEJ2000']}°")
print(f"  Mag: {stella_piu_luminosa['Mag']}")
print(f"  Catalogo: {stella_piu_luminosa['Catalogo']}")
print(f"  Coordinate pixel: {coordinate_pixel_stella}")

print("Tabella finale:\n",tbl_cataloghi)

# Usa la magnitudine per la dimensione e colore dei punti
# Stelle più brillanti (magnitudine minore) = punti più grandi e gialli
#sizes = 50 * (8 - tbl_cataloghi['Vmag'])  # Scala le dimensioni
sizes = 15 * (8 - tbl_cataloghi['Mag'])  # Scala le dimensioni
sizes = np.clip(sizes, 10, 200)  # Limita dimensioni min/max

# Colori basati sulla magnitudine
colors = tbl_cataloghi['Mag']

posizioni_vere_celesti = SkyCoord(ra=tbl_cataloghi['RAJ2000'],
                                 dec=tbl_cataloghi['DEJ2000'],
                                 frame='icrs')
scatter = plt.scatter(tbl_cataloghi['RAJ2000'], tbl_cataloghi['DEJ2000'],
                      c=colors, s = sizes, alpha=0.7, cmap='viridis_r')
plt.colorbar(scatter, label='Magnitudine Visuale (Vmag)')


plt.gca().invert_xaxis()  # RA aumenta verso est (convenzione astronomica)
plt.grid(True, alpha=0.3)
plt.xlim(ra_max, ra_min)  # Nota: invertito perché RA diminuisce verso destra
plt.ylim(dec_min, dec_max)
plt.xlabel('Ascensione Retta (RA J2000) [gradi]')
plt.ylabel('Declinazione (DEC J2000) [gradi]')
plt.title(f'Mappa del catalogo Vizier ({len(tbl_cataloghi)} stelle), catturata dalla pmc')

plt.show()

# Matching con l'immagine della PMC

posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti) # converto da celesti a pixel
posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

magnitudini = tbl_cataloghi['Mag']

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
# print(tbl)

positions = np.transpose((tbl['xcentroid'], tbl['ycentroid'])) # creo un array di posizioni
# positions_sky = SkyCoord(positions, unit=u.deg, frame='icrs')
posizioni_celesti_segmentation = w.pixel_to_world(positions)
posizioni_celesti_segmentation_ra = np.array(posizioni_celesti_segmentation.ra)
posizioni_celesti_segmentation_dec = np.array(posizioni_celesti_segmentation.dec)
ra_segmentation_max = np.max(posizioni_celesti_segmentation_ra)
'''print("RA max segmentazione: ", ra_segmentation_max)
print("RA max catalogo: ", ra_max)'''

apertures = CircularAperture(positions, r=5.0) # creo le aperture per ogni posizione
apertures.plot(color='red', lw=1.)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Matching: {len(tbl_cataloghi)} stelle del catalogo II/389/ps1_dr2 + Hipparco\n Cerchi dimensionati per magnitudine (<{mag_max})\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')


'''plt.title(f'Matching: {len(tbl_cataloghi)} stelle del catalogo II/389/ps1_dr2 \n Cerchi dimensionati per magnitudine (<{mag_max})\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')'''

# legenda
legend_elements = [
    # Stelle catalogate (cerchi colorati)
    Circle((0.5, 0.5), 0.4,facecolor='blue', alpha=0.7, edgecolor='black', linewidth=1,
          label=f'Stelle catalogo ({len(tbl_cataloghi)} oggetti)'),

    # Sorgenti rilevate (aperture rosse)
    Line2D([0], [0], marker='o', color='red', linestyle='None',
           markersize=8, markerfacecolor='none', markeredgewidth=1,
           label=f'Sorgenti rilevate ({len(tbl)} oggetti)')
]

# Aggiungi la legenda
ax.legend(handles=legend_elements, loc='upper right',
           framealpha=0.85, fancybox=True, shadow=True)

plt.show()