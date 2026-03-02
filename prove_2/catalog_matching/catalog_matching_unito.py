import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
import os

# Imposto matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # Permetto di avere la scala logaritmica
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

# Imposto astropy
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

# Imposto wcs
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

# Importo Path per la gestione dinamica dei percorsi
from pathlib import Path
from astropy.wcs.utils import proj_plane_pixel_scales

# Sopprimo i warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# FUNZIONI DI GESTIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    # Cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # Cerco un file ricorsivamente
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # Cerco una cartella specifica ricorsivamente
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]

# Trovo la cartella base del mio progetto
BASE_DIR = trova_cartella_base("Lorenzo")

# Leggo i parametri cercandoli dinamicamente
parametri = {}
file_parametri_path = cerca_file_nel_progetto(BASE_DIR, 'parametri_image_segmentation.txt')

if file_parametri_path is not None:
    with open(file_parametri_path, 'r') as file:
        next(file)  # Salto l'intestazione
        for riga in file:
            riga = riga.strip()
            if riga and not riga.startswith('#'):
                # Divido la riga usando solo il primo spazio come separatore
                parti = riga.split(maxsplit=1)
                if len(parti) == 2:
                    parametro = parti[0]
                    valore = parti[1]
                    try:
                        parametri[parametro] = float(valore) if '.' in valore else int(valore)
                    except ValueError:
                        # Se non riesco a convertire in numero puro, lo lascio come stringa
                        parametri[parametro] = valore
else:
    print("ERRORE: File dei parametri non trovato.")
    exit()

fwhm = parametri['fwhm']
size = parametri['size']
t = parametri['threshold_sigma']
# threshold = t * std # Per adesso lascio stare questo metodo
threshold = parametri['threshold_assoluta']
n = parametri['pixel']

# Questa funzione mi restituisce la tabella delle sorgenti trovate
def analisi_image_segmentation(data):
    """
    Esegue image segmentation su un'immagine FITS e restituisce la tabella filtrata delle sorgenti
    Il fondo deve essere rimosso
    """
    # Estraggo media, mediana e deviazione standard
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # Faccio la convoluzione
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)

    # Imposto il Sourcefinder
    t = parametri['threshold_sigma']
    threshold = parametri['threshold_assoluta']
    n = parametri['pixel']

    finder = SourceFinder(npixels=n, progress_bar=True)
    segment_map = finder(convolved_data, threshold)

    # Creo il catalogo delle sorgenti
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'

    # Filtro le sorgenti
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

    # Creo la tabella filtrata
    tbl_filtrato = tbl[indici_validi]
    new_labels_validi = np.arange(1, len(tbl_filtrato) + 1)
    tbl_filtrato['label'] = new_labels_validi

    return tbl_filtrato


def converti_valore(valore):
    valore = valore.strip()
    if not valore: return valore
    try: return int(valore)
    except ValueError: pass
    try: return float(valore)
    except ValueError: pass
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']: return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']: return False
    return valore

def leggi_header_da_csv(filename):
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':
                break
    return header_dict

# Imposto la run
run = 1

# Cerco la cartella in cui trovo i file CSV delle stelle catalogate
nome_cartella_csv = f"tabelle/tabelle_unite/tabelle_unite_run_{run}"
cartella_csv_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_csv)

if cartella_csv_path is None:
    print(f"ERRORE: Cartella '{nome_cartella_csv}' non trovata.")
    exit()

cartella_csv = str(cartella_csv_path)

# Cerco il file CSV specifico
nome_file_csv = f"run_{run}_stelle_trovate_e_catalogate_immagine_035.csv"
file_csv_path = cerca_file_nel_progetto(BASE_DIR, nome_file_csv)

if file_csv_path is None:
    print(f"ERRORE: File '{nome_file_csv}' non trovato in pmc_photometry.")
    exit()

file_csv = str(file_csv_path)

# Leggo il dataframe e l'header
dataframe = pd.read_csv(file_csv, comment='#')
header = leggi_header_da_csv(file_csv)

# Ricavo il nome del file FITS dall'header e cerco il suo percorso completo
nome_file_fits = header.get('PERCORSO_FILE_FITS', header.get('NOME_FILE_FITS', header.get('PERCORSO_FILE')))
nome_solo_fits = os.path.basename(str(nome_file_fits).strip())
file_trovato = cerca_file_nel_progetto(BASE_DIR, nome_solo_fits)

if file_trovato is None:
    print(f"ERRORE: File '{nome_solo_fits}' non trovato all'interno di {BASE_DIR}.")
    exit()

image_file = str(file_trovato)

hdu_list = fits.open(image_file)
hdu_list.info() # Mi dà le informazioni del file

image_data = hdu_list[0].data # Creo la matrice dei valori dei pixel

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median # Tolgo il fondo
data = image_data

print(image_data.shape)

# Trovo gli estremi
w = WCS(hdu_list[0].header) # Creo un oggetto WCS che mi contiene le info sulle coordinate
hdu_list.close()

alto_destra = w.pixel_to_world(3072, 2048)
alto_sinistra = w.pixel_to_world(3072, 0)
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

data_pmc = image_data # Mi serve per dopo

# Creo la tabella del catalogo

# Trovo le coordinate del centro
image_header = header
ra_centro = image_header["RA"]
print("RA centro: ", ra_centro)
dec_centro = image_header["DEC"]
print("DEC centro: ", dec_centro)

larghezza = alto_destra.separation(alto_sinistra).degree
print(f"Larghezza: {larghezza}")
altezza = alto_destra.separation(basso_destra).degree
print(f"Altezza: {altezza}")

# Inizializzo Vizier con i suoi parametri di default
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID','RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)

catalogo = vizier
print(f"Questo è il catalogo: \n{catalogo}")

catalogs = Vizier.get_catalogs("II/389/ps1_dr2")
print(catalogs)
print(f"Descrizione gmag: {catalogs[0]['objID'].description}")

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

# Tolgo le stelle che non vorrei siano prese da Vizier
tbl_catalogo_vizier = tbl_catalogo_vizier[(tbl_catalogo_vizier['gmag'] >= mag_limite_tra_hipparco_e_vizier)]

# Aggiungo il catalogo Hipparco cercandolo in modo dinamico
file_hipparco_path = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
if file_hipparco_path is None:
    print("ERRORE: Catalogo Hipparco non trovato.")
    exit()
file_hipparco = str(file_hipparco_path)

# Apro il catalogo in formato fit
hdu_list_hipparco = fits.open(file_hipparco)
print("Info catalogo Hipparco: \n",hdu_list_hipparco)

# I dati sono nella seconda estensione
table_data = Table(hdu_list_hipparco[1].data)

# Tolgo le stelle che non vorrei siano prese da Hipparco
tbl_catalogo_hipparco = table_data[(table_data['Vmag']) < mag_limite_tra_hipparco_e_vizier]

# Esploro il catalogo
print("\n=== INFORMAZIONI DEL CATALOGO ===")
print(f"Numero di stelle nel catalogo: {len(tbl_catalogo_hipparco)}")
print(f"Nomi delle colonne: {tbl_catalogo_hipparco.colnames}")

# Adesso mi costruisco una mia tabella astropy complessiva

# Creo la colonna del catalogo di riferimento
nome_catalogo_vizier = ["II/389/ps1_dr2"] * len(tbl_catalogo_vizier)
nome_catalogo_hipparco = ["I/239/hip_main"] * len(tbl_catalogo_hipparco)

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

# Faccio questa parte solo per ritagliare correttamente il riquadro
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

# Creo una lista degli elementi che rientrano nel riquadro
mask = [
    poly.contains(Point(ra, dec))
    for ra, dec in zip(tbl_unita_estesa['RAJ2000'], tbl_unita_estesa['DEJ2000'])
]
tbl_cataloghi = tbl_unita_estesa[mask]

# Ho finito di creare la tabella complessiva

magnitudini = tbl_cataloghi['Mag']
mag_min_del_catalogo = np.min(magnitudini)
indice_mag_min = np.argmin(magnitudini)

# Ho l'intera riga della stella con magnitudine massima
stella_piu_luminosa = tbl_cataloghi[indice_mag_min]
coo = SkyCoord(stella_piu_luminosa['RAJ2000'], stella_piu_luminosa['DEJ2000'], unit=u.deg)
coordinate_pixel_stella = w.world_to_pixel(coo)

print(f"Dati della stella più luminosa:")
print(f"  RA: {stella_piu_luminosa['RAJ2000']}°")
print(f"  Dec: {stella_piu_luminosa['DEJ2000']}°")
print(f"  Mag: {stella_piu_luminosa['Mag']}")
print(f"  Catalogo: {stella_piu_luminosa['Catalogo']}")
print(f"  Coordinate pixel: {coordinate_pixel_stella}")

print("Tabella finale:\n",tbl_cataloghi)

# Uso la magnitudine per la dimensione e il colore dei punti
# Scalo le dimensioni
sizes = 15 * (8 - tbl_cataloghi['Mag'])
# Limito le dimensioni min/max
sizes = np.clip(sizes, 10, 200)

# Baso i colori sulla magnitudine
colors = tbl_cataloghi['Mag']

posizioni_vere_celesti = SkyCoord(ra=tbl_cataloghi['RAJ2000'],
                                 dec=tbl_cataloghi['DEJ2000'],
                                 frame='icrs')
scatter = plt.scatter(tbl_cataloghi['RAJ2000'], tbl_cataloghi['DEJ2000'],
                      c=colors, s = sizes, alpha=0.7, cmap='viridis_r')
plt.colorbar(scatter, label='Magnitudine Visuale (Vmag)')


plt.gca().invert_xaxis()  # RA aumenta verso est
plt.grid(True, alpha=0.3)
plt.xlim(ra_max, ra_min)
plt.ylim(dec_min, dec_max)
plt.xlabel('Ascensione Retta (RA J2000) [gradi]')
plt.ylabel('Declinazione (DEC J2000) [gradi]')
plt.title(f'Mappa del catalogo Vizier ({len(tbl_cataloghi)} stelle), catturata dalla pmc')

plt.savefig('catalog_matching_unito_cazzata.png')
# plt.show()

# Faccio il matching con l'immagine della PMC

# Converto da celesti a pixel
posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti)
posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

magnitudini = tbl_cataloghi['Mag']

# Imposto i parametri per i raggi
raggio_min = 4.0
raggio_max = 20.0
raggi = raggio_max - (magnitudini - magnitudini.min()) * (raggio_max - raggio_min) / (magnitudini.max() - magnitudini.min())

# Creo una scala di colori
cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())

fig = plt.figure(figsize=(12, 8))
ax = plt.subplot()

# Disegno i cerchi colorati delle stelle catalogate
for i, (position, radius) in enumerate(zip(posizioni_vere_pixel, raggi)):
    color = cmap(norm(magnitudini[i]))
    aperture = CircularAperture(position, r=radius)
    aperture.plot(color=color, lw=1.0, alpha=0.6, fill=True)

# Aggiungo la legenda specificando l'asse
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label='Magnitudine V')

# Rappresento il matching aggiungendoci l'image segmentation
ax.imshow(data_pmc, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')

tbl = analisi_image_segmentation(data)

tbl['xcentroid'].info.format = '.2f'
tbl['ycentroid'].info.format = '.2f'
tbl['kron_flux'].info.format = '.2f'

# Creo un array di posizioni
positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))
posizioni_celesti_segmentation = w.pixel_to_world(positions)
posizioni_celesti_segmentation_ra = np.array(posizioni_celesti_segmentation.ra)
posizioni_celesti_segmentation_dec = np.array(posizioni_celesti_segmentation.dec)
ra_segmentation_max = np.max(posizioni_celesti_segmentation_ra)


# Calcolo la scala di piastra media dell'immagine in gradi per pixel
pixel_scales = proj_plane_pixel_scales(w)
pixel_scale_mean = np.mean(pixel_scales)

# Imposto la mia tolleranza di 35 arcosecondi e la converto in gradi
tolleranza_arcsec = 35.0
tolleranza_gradi = tolleranza_arcsec / 3600.0

# Calcolo il raggio effettivo in pixel
raggio_pixel = tolleranza_gradi / pixel_scale_mean
print(f"Raggio di {tolleranza_arcsec} arcsec convertito in {raggio_pixel:.2f} pixel")

# Creo le aperture per ogni posizione usando il raggio dinamico
apertures = CircularAperture(positions, r=raggio_pixel)
apertures.plot(color='red', lw=1.)


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Matching: {len(tbl_cataloghi)} stelle del catalogo II/389/ps1_dr2 + Hipparco\n Cerchi dimensionati per magnitudine (<{mag_max})\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')


# Aggiungo la legenda
legend_elements = [
    Circle((0.5, 0.5), 0.4,facecolor='blue', alpha=0.7, edgecolor='black', linewidth=1,
          label=f'Stelle catalogo ({len(tbl_cataloghi)} oggetti)'),

    Line2D([0], [0], marker='o', color='red', linestyle='None',
           markersize=8, markerfacecolor='none', markeredgewidth=1,
           label=f'Sorgenti rilevate ({len(tbl)} oggetti)')
]

ax.legend(handles=legend_elements, loc='upper right',
           framealpha=0.85, fancybox=True, shadow=True)

plt.savefig('catalog_matching_unito.png', bbox_inches='tight')
# plt.show()