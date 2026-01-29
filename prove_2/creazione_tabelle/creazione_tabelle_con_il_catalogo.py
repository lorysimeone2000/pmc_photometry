import pandas as pd
# pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # permette di avere la scala logaritmica
from scipy.optimize import curve_fit
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
import numpy as np
import os
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture

# Set up wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from astroquery.vizier import Vizier
from astropy.coordinates import Angle

from shapely.geometry import Point, Polygon
# warning
import warnings
from astropy.io.fits.verify import VerifyWarning
import warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)  # Sopprime il warning FITSFixedWarning

# --- IMPORT FONDAMENTALE PER LA PORTABILITÀ ---
from pathlib import Path

# sopprimo i warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI DINAMICA (PORTABILITÀ TOTALE)
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    """
    Risale la directory partendo dalla posizione dello script fino a trovare
    la cartella target (es. 'pmc_photometry').
    """
    path_corrente = Path(__file__).resolve()

    # Risaliamo fino a trovare la cartella target
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent

    # Fallback: se non la trova, usa la cartella dello script
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    """
    Cerca un file ricorsivamente in tutte le sottocartelle di base_dir.
    """
    files_trovati = list(base_dir.rglob(nome_file_esatto))

    if not files_trovati:
        return None

    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
        print(
            f"INFO: Trovati {len(files_trovati)} file '{nome_file_esatto}'. Uso il primo: {files_trovati[0].relative_to(base_dir)}")

    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    """
    Cerca una CARTELLA ricorsivamente in tutte le sottocartelle di base_dir.
    """
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]

    if not cartelle_trovate:
        return None

    cartelle_trovate.sort(key=lambda p: len(str(p)))

    if len(cartelle_trovate) > 1:
        print(
            f"INFO: Trovate {len(cartelle_trovate)} cartelle '{nome_cartella_esatto}'. Uso la prima: {cartelle_trovate[0].relative_to(base_dir)}")

    return cartelle_trovate[0]


# Definisco la BASE_DIR dinamicamente
BASE_DIR = trova_cartella_base("pmc_photometry")

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")

# Inizializzo Vizier
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)


def salva_csv_con_header_fits(dataframe, header_fits, filename, nome_file_fits):
    """Salva il DataFrame in CSV includendo l'header FITS come commenti"""
    with open(filename, 'w') as f:
        # Scrivi l'header FITS come commenti
        f.write("# Header FITS:\n")
        for key, value in header_fits.items():
            f.write(f"# {key}: {value}\n")
        f.write(f"# PERCORSO_FILE: {nome_file_fits}\n")
        f.write("#\n")  # Linea vuota per separare header dai dati
        # Scrivi il DataFrame
        dataframe.to_csv(f, index=False)


def tabella_catalogo(image_file_,
                     magnitudine_massima):  # questa funzione restituisce la tabella delle sorgenti catalogate nel riquadro della pmc
    """
    Seleziona le stelle del catalogo che rientrano nel riquadro e che sono sotto una certa magnitudine
    e che non stanno entro 7 pixel dal bordo

    Parameters:
    image_file_ (string): percorso del file da cui estrarre l'header e quindi le coordinate dell'astrometria
    magnitudine_massima (float): magnitudina massima presa dal catalogo

    Returns:
    astropy.table.Table: Tabella delle stelle del catalogo che rientrano nel riquadro e che sono sotto una certa magnitudine
    """
    #
    # This image helps visualize how a FITS file stores the Header (metadata/WCS) separately from the Data (pixel array).
    hdu_list_ = fits.open(image_file_)

    wcs = WCS(hdu_list_[0].header)  # creo un oggetto WCS usando l'header del file FITS,
    # che contiene le informazioni per le trasformazioni di coordinate
    data_ = hdu_list_[0].data

    # Definisco il bordo
    bordo = 7
    h, w = data_.shape[0], data_.shape[1]

    # tolgo le stelle che non vorrei siano prese da Vizier
    # mag_limite_tra_hipparco_e_vizier = 7.
    tbl_catalogo_vizier = tbl_riquadro_esterno_vizier[
        (tbl_riquadro_esterno_vizier['gmag'] >= mag_limite_tra_hipparco_e_vizier)
    ]

    '''# Esploro il catalogo
    print("\n=== INFORMAZIONI DEL CATALOGO ===")
    print(f"Numero di stelle nel catalogo: {len(tbl_catalogo_hipparco)}")
    print(f"Nomi delle colonne: {tbl_catalogo_hipparco.colnames}")'''

    # adesso mi costruisco una mia tabella astropy complessiva

    # ricreo le colonne dei cataloghi

    nome_catalogo_vizier = []
    for i in range(len(tbl_catalogo_vizier)): nome_catalogo_vizier.append("II/389/ps1_dr2")
    colonne_vizier = {
        'ID': tbl_catalogo_vizier['objID'],
        'RAJ2000': tbl_catalogo_vizier['RAJ2000'],
        'DEJ2000': tbl_catalogo_vizier['DEJ2000'],
        'Mag': tbl_catalogo_vizier['gmag'],
        'Catalogo': nome_catalogo_vizier
    }

    nome_catalogo_hipparco = []
    for i in range(len(tbl_catalogo_hipparco)): nome_catalogo_hipparco.append("I/239/hip_main")
    colonne_hipparco = {
        'Catalogo': nome_catalogo_hipparco,
        'ID': tbl_catalogo_hipparco['HIP'],
        'RAJ2000': tbl_catalogo_hipparco['_RAJ2000'],
        'DEJ2000': tbl_catalogo_hipparco['_DEJ2000'],
        'Mag': tbl_catalogo_hipparco['Vmag'],
    }

    t1 = Table(colonne_vizier)
    t2 = Table(colonne_hipparco)

    tbl_unita_estesa = vstack([t1, t2])
    tbl_unita_estesa['Mag'].description = 'Magnitudine AB nel filtro g di Pan-STARRS'
    # print("Tabella estesa:\n", tbl_unita_estesa)

    # --- MODIFICA QUI ---

    # Invece di proiettare gli angoli sul cielo, proiettiamo il catalogo sui pixel

    # 1. Creiamo un oggetto SkyCoord per tutte le stelle della tabella unita
    coords_catalogo = SkyCoord(
        ra=tbl_unita_estesa['RAJ2000'],
        dec=tbl_unita_estesa['DEJ2000'],
        unit=u.deg
    )

    #
    # This diagram illustrates how WCS maps the curved celestial sphere (RA/Dec) onto a flat 2D pixel grid.
    # 2. Convertiamo tutte le coordinate celesti in coordinate pixel (x, y)
    x_pix, y_pix = wcs.world_to_pixel(coords_catalogo)

    # 3. Creiamo la maschera usando le coordinate pixel
    # Vogliamo le stelle che sono DENTRO i bordi:
    # x deve essere > bordo E < (larghezza - bordo)
    # y deve essere > bordo E < (altezza - bordo)
    mask_bordo = (
            (x_pix >= bordo) &
            (x_pix < (w - bordo)) &
            (y_pix >= bordo) &
            (y_pix < (h - bordo))
    )

    # 4. Applichiamo la maschera alla tabella
    tbl_cataloghi_ = tbl_unita_estesa[mask_bordo]

    # Opzionale: Se vuoi salvare anche le coordinate pixel nella tabella risultante
    # tbl_cataloghi_['x_pix'] = x_pix[mask_bordo]
    # tbl_cataloghi_['y_pix'] = y_pix[mask_bordo]

    hdu_list_.close()  # È buona norma chiudere il file fits

    return tbl_cataloghi_


def calcolo_distanze(tbl_trovate, tbl_catalogate, image_file):
    """
    Calcola le distanze dei centroidi rispetto alle stelle catalogate più vicine

    Parameters:
    tbl_trovate (Table): Tabella delle sorgenti trovate con image segmentation
    tbl_catalogate (Table): Tabella delle stelle del catalogo
    image_file (str): Percorso del file FITS

    Returns:
    array: elenco delle distanze minime di tutti i centroidi
    """

    # Carica il WCS dall'immagine
    hdu_list = fits.open(image_file)
    w = WCS(hdu_list[0].header)

    # Converti i centroidi pixel in coordinate celesti
    coords_trovate = w.pixel_to_world(tbl_trovate['xcentroid'], tbl_trovate['ycentroid'])

    # converto le coordinate catalogate in array numpy e gestisco le unità
    try:
        # estraggo i valori come array numpy puri
        if hasattr(tbl_catalogate['RAJ2000'], 'value'):
            # se ho già unità, estraggo solo i valori
            ra_values = tbl_catalogate['RAJ2000'].value
            dec_values = tbl_catalogate['DEJ2000'].value
        else:
            # altrimenti converto direttamente
            ra_values = np.array(tbl_catalogate['RAJ2000'])
            dec_values = np.array(tbl_catalogate['DEJ2000'])

        '''
        print(f"ra_values tipo: {type(ra_values)}, forma: {ra_values.shape}")
        print(f"dec_values tipo: {type(dec_values)}, forma: {dec_values.shape}")
        '''

        # creo SkyCoord con i valori puri specificando le unità
        coords_catalogate = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)

    except Exception as e:
        print(f"Errore nell'approccio principale: {e}")
        # APPROCCIO ALTERNATIVO: uso direttamente i valori senza moltiplicare per unità
        coords_catalogate = SkyCoord(ra=tbl_catalogate['RAJ2000'],
                                     dec=tbl_catalogate['DEJ2000'],
                                     unit=u.deg)

    print("numero di stelle catalogate", np.shape(coords_catalogate))

    # calcolo le distanze di tutti i centroidi da tutte le stelle catalogate
    distanze_minime = []
    corrispondenze = []
    righe_tabella_combinata = []  # lista per la tabella combinata

    #
    # Useful to understand that 'separation' calculates the Great Circle distance, not linear Euclidean distance.
    for i, coord_trovata in enumerate(coords_trovate):
        # calcolo la distanza da tutte le stelle catalogate
        distanze_singola = coord_trovata.separation(
            coords_catalogate)  # Calcola la distanza angolare tra la singola stella trovata
        # (coord_trovata) e tutte le stelle del catalogo (coords_catalogate). Restituisce un array di distanze angolari.
        # if i == 1: print(np.shape(distanze_singola))
        # trovo la distanza minima e l'indice della stella più vicina
        distanza_minima = np.min(distanze_singola)

        distanze_minime.append(distanza_minima)

    distanze_gradi = [d.deg for d in distanze_minime]
    hdu_list.close()

    return distanze_gradi


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


run = int(input("Quale run vuoi elaborare: "))  # numero run: 1, 2 o 3

# =============================================================================
# RICERCA FILE DALLA CARTELLA RUN (NUOVA LOGICA)
# =============================================================================

# --- RICERCA CARTELLE RUN ---
nome_cartella_run = f"20250120_run{run}"
found_folders = list(BASE_DIR.rglob(nome_cartella_run))

if not found_folders:
    print(f"ERRORE: Cartella '{nome_cartella_run}' non trovata in nessuna sottocartella di {BASE_DIR}")
    exit()

run_folder = found_folders[0]
if len(found_folders) > 1:
    print(f"AVVISO: Trovate {len(found_folders)} cartelle. Uso la prima: {run_folder}")
else:
    print(f"Cartella dati trovata: {run_folder.relative_to(BASE_DIR)}")

# Cerca i file FITS
estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
file_list = []
for ext in estensioni_valide:
    file_list.extend(run_folder.glob(ext))

file_list = sorted(file_list, key=lambda x: x.name)
file_list = [str(f) for f in file_list]

if not file_list:
    print(f"ERRORE: Nessun file FITS trovato in {run_folder}")
    exit()

print(f"Trovati {len(file_list)} file da elaborare.")

# =============================================================================
# FINE RICERCA FILE
# =============================================================================


# --- GESTIONE DINAMICA CARTELLA OUTPUT ---
cartella_tabelle = cerca_cartella_nel_progetto(BASE_DIR, "tabelle")
if cartella_tabelle is None:
    # Crea se non esiste in base_dir
    cartella_tabelle = BASE_DIR / "tabelle"
    cartella_tabelle.mkdir(exist_ok=True)

# Crea sottocartella specifica
output_dir = cartella_tabelle / "sorgenti_catalogate_run" / f"sorgenti_catalogate_run_{run}"
output_dir.mkdir(parents=True, exist_ok=True)
output_dir_str = str(output_dir)

print(f"Cartella Output: {output_dir.relative_to(BASE_DIR)}")

i = 0
j = 0
posizioni_lista = []  # lista che dovrà essere riempita con tutte le poszioni di tutte le tabelle
distanze = []
numero_stelle_catalogate = []
tempo = []
# Itera su tutti i file fits
for percorso_file_fits in file_list:
    i += 1
    # Check esistenza file
    if not os.path.exists(percorso_file_fits):
        print(f"AVVISO: File non trovato, salto: {percorso_file_fits}")
        continue

    hdu_list = fits.open(percorso_file_fits)
    image_header = hdu_list[0].header

    # tbl_trovate = Table.from_pandas(dataframe)
    if i == 1:  # chiamo il sito una volta sola su un'immagine più grande per non mandarlo in down

        # header = leggi_header_da_csv(filename)

        # prendo un riquadro globale leggermente più grande del riquadro della pmc

        # coordinate centro
        ra_centro = image_header["RA"]
        dec_centro = image_header["DEC"]
        data = hdu_list[0].data

        w = WCS(hdu_list[0].header)  # creo un oggetto WCS usando l'header del file FITS
        alto_destra = w.pixel_to_world(3071, 2047)
        alto_sinistra = w.pixel_to_world(3071, 0)
        # print(f"Coordinate in alto a destra: {alto_destra}")
        basso_sinistra = w.pixel_to_world(0, 0)
        basso_destra = w.pixel_to_world(0, 2047)

        centro = SkyCoord(ra_centro, dec_centro, unit=u.deg)

        #
        # Illustrates the "query_region" operation: selecting stars within a specific angular radius around a central coordinate.
        # creo un riquadro esterno leggermente più grande per assicurarmi che anche gli elementi successivi della run
        # rientrino nella query, poi farò il taglio preciso nella tabella astropy
        riquadro_esterno_vizier = vizier.query_region(coord.SkyCoord(ra=ra_centro, dec=dec_centro,
                                                                     unit=(u.deg, u.deg),
                                                                     frame='icrs'),
                                                      radius=Angle(centro.separation(alto_destra) * 1.5, "deg"),
                                                      column_filters={'gmag': f'<{15}'},
                                                      )  # ho messo un limite di magnitudine per non scaricare milioni di stelle
        tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]

        # aggiungo il catalogo Hipparco per le magnitudini inferiori a 7

        # --- RICERCA DINAMICA HIPPARCO ---
        file_hipparco_path = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
        if file_hipparco_path is None:
            print("ERRORE CRITICO: Catalogo 'hipparco.fit' non trovato.")
            exit()
        file_hipparco = str(file_hipparco_path)

        # Apro il catalogo in formato fit, lo faccio solo una volta

        hdu_list_hipparco = fits.open(file_hipparco)
        # print("Info catalogo Hipparco: \n", hdu_list_hipparco)

        # I dati sono nella seconda estensione (V_SO_catalog), non nella prima
        table_data = Table(hdu_list_hipparco[1].data)  # Uso l'indice 1 per la seconda estensione
        # tolgo le stelle che non vorrei siano prese da Hipparco
        mag_limite_tra_hipparco_e_vizier = 7.
        tbl_catalogo_hipparco = table_data[(table_data['Vmag']) < mag_limite_tra_hipparco_e_vizier]

        print("-----------------------------")

    print(f"Elaborando {percorso_file_fits}")
    print("\n")

    mag_max = 15

    tbl_catalogate = tabella_catalogo(percorso_file_fits,
                                      mag_max)  # tabelle del catalogo all'interno della specifica run
    numero_stelle_catalogate.append(len(tbl_catalogate))
    print(f"Trovate {len(tbl_catalogate)} stelle dei cataloghi nel riquadro {i}")

    # creo i file csv
    dataframe = tbl_catalogate.to_pandas()
    # Uso output_dir (Path object)
    filename = output_dir / f'run_{run}_stelle_catalogate_immagine_{i:03d}.csv'
    salva_csv_con_header_fits(dataframe, image_header, str(filename), percorso_file_fits)

    # if i == 10: break