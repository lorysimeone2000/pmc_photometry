import pandas as pd
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
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
from astropy.coordinates import match_coordinates_sky
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
from astropy.wcs import FITSFixedWarning
from pathlib import Path

# sopprimo i warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)

# --- FUNZIONI UTILITY ---

# 2. Lettura parametri
parametri = {}
with open('/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt', 'r') as file:
    next(file)
    for riga in file:
        riga = riga.split('#')[0].strip()
        if riga:
            parts = riga.split()
            if len(parts) >= 2:
                parametro = parts[0]
                valore_str = parts[1]
                try:
                    valore = float(valore_str) if '.' in valore_str else int(valore_str)
                    parametri[parametro] = valore
                except ValueError:
                    pass


fwhm = parametri.get('fwhm', 3.0)
size = parametri.get('size', 5)
t = parametri.get('threshold_sigma', 1.5)
threshold = parametri.get('threshold_assoluta', 0.5)
n = parametri.get('pixel', 5)


def calcolo_distanze(tbl_trovate, tbl_catalogate, image_file):
    """
    Calcola le distanze dei centroidi rispetto alle stelle catalogate più vicine.
    (Non strettamente usata nel loop principale, ma mantenuta per completezza)
    """

    # Carica il WCS dall'immagine
    hdu_list = fits.open(image_file)
    w = WCS(hdu_list[0].header)

    # Converti i centroidi pixel in coordinate celesti
    coords_trovate = w.pixel_to_world(tbl_trovate['xcentroid'], tbl_trovate['ycentroid'])

    try:
        if hasattr(tbl_catalogate['RAJ2000'], 'value'):
            ra_values = tbl_catalogate['RAJ2000'].value
            dec_values = tbl_catalogate['DEJ2000'].value
        else:
            ra_values = np.array(tbl_catalogate['RAJ2000'])
            dec_values = np.array(tbl_catalogate['DEJ2000'])

        coords_catalogate = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)

    except Exception as e:
        # print(f"Errore nell'approccio principale: {e}")
        coords_catalogate = SkyCoord(ra=tbl_catalogate['RAJ2000'],
                                     dec=tbl_catalogate['DEJ2000'],
                                     unit=u.deg)

    distanze_minime = []

    for i, coord_trovata in enumerate(coords_trovate):
        distanze_singola = coord_trovata.separation(coords_catalogate)
        distanza_minima = np.min(distanze_singola)
        distanze_minime.append(distanza_minima)

    distanze_gradi = [d.deg for d in distanze_minime]
    hdu_list.close()

    return distanze_gradi


def salva_csv_con_header_fits(dataframe, header_fits, filename, nome_file_fits):
    """Salva il DataFrame in CSV includendo l'header FITS come commenti"""
    with open(filename, 'w') as f:
        # Scrivi l'header FITS come commenti
        f.write("# Header FITS:\n")
        f.write(
            f"# DESCRIZIONE: Questo file csv contiene la tabella di tutte le sorgenti trovate con image segmentation insieme alle informazioni dell'eventuale stella corrispondente dei cataloghi (Rank 1, 2, 3 per luminosita')\n")
        for key, value in header_fits.items():
            f.write(f"# {key}: {value}\n")
        # f.write(f"# PERCORSO_FILE: {nome_file_fits}\n")
        f.write("#\n")  # Linea vuota per separare header dai dati
        # Scrivi il DataFrame
        dataframe.to_csv(f, index=False)


def converti_valore(valore):
    """
    Converte una stringa nel tipo di dato appropriato.
    """
    valore = valore.strip()
    if not valore: return valore
    try:
        return int(valore)
    except ValueError:
        pass
    try:
        return float(valore)
    except ValueError:
        pass
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False
    return valore


def leggi_header_da_csv(filename):
    """Legge l'intero header (FITS + Parametri) dal file CSV"""
    header_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            # Legge tutte le righe che iniziano con #
            if line.startswith('#'):
                # Rimuove il '#' iniziale e gli spazi
                clean_line = line.strip()[1:].strip()

                # Se la riga contiene un valore (es. "KEY: VALUE")
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            else:
                # Se la riga NON inizia con #, siamo arrivati ai dati. Interrompi.
                break

    return header_dict


# --- INPUT UTENTE E INIZIALIZZAZIONE ---

# RICHIESTA INPUT UTENTE
try:
    run = int(input("Quale run vuoi elaborare: "))  # numero run: 1, 2 o 3
except ValueError:
    print("Input non valido. Si prega di inserire un numero intero (1, 2 o 3).")
    exit()

# Definizione dei percorsi
cartella_base = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle"
cartella_csv = os.path.join(cartella_base, f"sorgenti_catalogate_run/sorgenti_catalogate_run_{run}")
cartella_csv_ = os.path.join(cartella_base, f"sorgenti_trovate_run/sorgenti_trovate_run_{run}")
output_dir = os.path.join(cartella_base, f"tabelle_unite/tabelle_unite_run_{run}")

# Assicurati che le directory di output esistano
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Lista dei file CSV
file_csv_catalogate = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv_stelle_catalogate = [os.path.join(cartella_csv, file) for file in file_csv_catalogate]

file_csv_trovate = sorted([f for f in os.listdir(cartella_csv_) if f.endswith('.csv')])
lista_percorsi_csv_stelle_trovate = [os.path.join(cartella_csv_, file) for file in file_csv_trovate]

if len(lista_percorsi_csv_stelle_trovate) != len(lista_percorsi_csv_stelle_catalogate):
    print("ATTENZIONE: Il numero di file trovati e catalogati non corrisponde. Interruzione.")
    exit()

soglia_correlazione = 0.003349 * u.deg  # soglia fissa (in gradi)
MAG_CUTOFF = 10.4  # Nuova soglia di magnitudine

# --- LOOP PRINCIPALE DI ELABORAZIONE ---

i = 0
for n in range(len(lista_percorsi_csv_stelle_catalogate)):

    percorso_csv_stelle_trovate = lista_percorsi_csv_stelle_trovate[n]
    percorso_csv_stelle_catalogate = lista_percorsi_csv_stelle_catalogate[n]

    # 1. Lettura dati e Header
    header_dal_csv = leggi_header_da_csv(percorso_csv_stelle_trovate)
    percorso_file_fits = header_dal_csv['PERCORSO_FILE']
    i = i + 1
    print(f"\nElaborando file {i} di {len(lista_percorsi_csv_stelle_catalogate)}:")
    print(percorso_file_fits)

    # Leggi i DataFrame, saltando l'header commentato
    dataframe1 = pd.read_csv(percorso_csv_stelle_trovate, comment='#')
    dataframe2 = pd.read_csv(percorso_csv_stelle_catalogate, comment='#')
    tbl_trovate = Table.from_pandas(dataframe1)
    tbl_catalogate = Table.from_pandas(dataframe2)

    # Rimuovo colonne non necessarie e aggiungo coordinate celesti

    all_cols = tbl_trovate.colnames
    cols_base = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value'] # 2. Definisci le colonne base, kron_flux lo aggiungo dopo
    try:
        idx_satura = all_cols.index('Satura')
        cols_extra = all_cols[idx_satura:]
        cols_dinamiche = []
        if 'Satura' in cols_extra:
            cols_dinamiche.append('Satura')
        if 'kron_flux' in all_cols:
            cols_dinamiche.append('kron_flux')
        for c in cols_extra:
            if c not in cols_dinamiche and c not in cols_base:
                cols_dinamiche.append(c)
    except ValueError:
        # Se 'Satura' non c'è, cerchiamo almeno di salvare kron_flux se c'è
        cols_dinamiche = []
        if 'kron_flux' in all_cols:
            cols_dinamiche.append('kron_flux')
    cols_finali = cols_base + cols_dinamiche
    tbl_trovate = tbl_trovate[cols_finali]

    with fits.open(percorso_file_fits) as hdu_list:
        header = hdu_list[0].header
        w = WCS(header)

    coords_trovate = w.pixel_to_world(tbl_trovate['xcentroid'], tbl_trovate['ycentroid'])
    tbl_trovate['RA_centroid'] = coords_trovate.ra
    tbl_trovate['DEC_centroid'] = coords_trovate.dec

    # 2. Preparazione Tabella Finale Vuota
    tbl_finale = Table()

    # Aggiungi colonne dalla prima tabella preservando i tipi
    for colname in tbl_trovate.colnames:
        tbl_finale[colname] = tbl_trovate[colname][:0]

    # Colonna chiave per il matching (tipo stringa per Rank)
    tbl_finale['Corrispondenza'] = 'SI (Rank 1)'

    # Aggiungi colonne dalla seconda tabella preservando i tipi
    for colname in tbl_catalogate.colnames:
        tbl_finale[colname] = tbl_catalogate[colname][:0]

    # Metto la colonna "Catalogo" prima di "ID"
    colonne = tbl_finale.colnames
    pos_id = colonne.index('ID')
    nuovo_ordine = colonne[:pos_id] + ['Catalogo'] + [col for col in colonne if
                                                      col != 'Catalogo' and col not in colonne[:pos_id]]
    tbl_finale = tbl_finale[nuovo_ordine]

    # 3. Conversione Coordinate Catalogate
    try:
        if hasattr(tbl_catalogate['RAJ2000'], 'value'):
            ra_values = tbl_catalogate['RAJ2000'].value
            dec_values = tbl_catalogate['DEJ2000'].value
        else:
            ra_values = np.array(tbl_catalogate['RAJ2000'])
            dec_values = np.array(tbl_catalogate['DEJ2000'])
        coords_catalogate = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)
    except Exception:
        coords_catalogate = SkyCoord(ra=tbl_catalogate['RAJ2000'],
                                     dec=tbl_catalogate['DEJ2000'],
                                     unit=u.deg)

    print(f"Cercata correlazione in {len(coords_trovate)} stelle")

    # 4. Logica di Correlazione (Rank 1, 2, 3 per luminosità)
    colonna_magnitudine = 'Mag'
    righe_da_aggiungere = []

    for idx_trovato in range(len(coords_trovate)):
        coord_trovata = coords_trovate[idx_trovato]
        distanza_da_trovata = coords_catalogate.separation(coord_trovata)

        # Criterio 1: Stelle entro la soglia di distanza
        mask_distanza = distanza_da_trovata <= soglia_correlazione

        '''# Criterio 2: Stelle più luminose di Mag < MAG_CUTOFF (10.4)
        magnitudini_candidate = tbl_catalogate[colonna_magnitudine]
        mask_luminosita = magnitudini_candidate < MAG_CUTOFF'''

        # Maschera finale: deve soddisfare entrambi i criteri
        # maschera_finale = mask_distanza & mask_luminosita
        maschera_finale = mask_distanza
        indici_catalogate_vicine = np.where(maschera_finale)[0]

        ha_corrispondenza = len(indici_catalogate_vicine) > 0

        # Prepara la riga base (dati della sorgente trovata)
        riga_base = {}
        for colname in tbl_trovate.colnames:
            riga_base[colname] = tbl_trovate[colname][idx_trovato]

        if ha_corrispondenza:
            # 1. Ordina le stelle corrispondenti per luminosità
            magnitudini_vicine = tbl_catalogate[colonna_magnitudine][indici_catalogate_vicine]

            # Ottieni gli indici SORTATI per luminosità
            indici_luminosita_sort = np.argsort(magnitudini_vicine)

            # Ottieni gli indici originali (nella tbl_catalogate) delle stelle più luminose, ordinate
            indici_originali_ordinati = indici_catalogate_vicine[indici_luminosita_sort]

            indici_da_considerare = indici_originali_ordinati

            # 2. Crea una riga per OGNI corrispondenza valida trovata (Rank 1, 2, o 3)
            for rank, idx_catalogata in enumerate(indici_da_considerare, 1):
                nuova_riga = riga_base.copy()

                # Dati dalla stella catalogata corrispondente
                for colname in tbl_catalogate.colnames:
                    nuova_riga[colname] = tbl_catalogate[colname][idx_catalogata]

                # Aggiungi informazioni sul "rank"
                nuova_riga['Corrispondenza'] = f'SI (Rank {rank})'

                righe_da_aggiungere.append(nuova_riga)

        else:
            # Nessuna corrispondenza valida trovata -> UNA SOLA riga NO
            nuova_riga_no_match = riga_base.copy()
            nuova_riga_no_match['Corrispondenza'] = 'NO'

            # Imposta i valori sentinella per i dati del catalogo
            for colname in tbl_catalogate.colnames:
                if colname == 'Catalogo':
                    nuova_riga_no_match[colname] = 'N/A'
                elif tbl_catalogate[colname].dtype.kind in ['i', 'u']:
                    nuova_riga_no_match[colname] = -999
                elif tbl_catalogate[colname].dtype.kind == 'f':
                    nuova_riga_no_match[colname] = np.nan
                elif tbl_catalogate[colname].dtype.kind == 'O':
                    nuova_riga_no_match[colname] = 'N/A'
                else:
                    nuova_riga_no_match[colname] = None

            righe_da_aggiungere.append(nuova_riga_no_match)

    # 5. Aggiungi tutte le righe raccolte alla tabella finale
    for riga in righe_da_aggiungere:
        tbl_finale.add_row(riga)

    # 6. Salvataggio
    tbl_correlate = tbl_finale[tbl_finale['Corrispondenza'] != 'NO']

    # creo il file csv
    dataframe = tbl_finale.to_pandas()
    filename = os.path.join(output_dir, f'run_{run}_stelle_trovate_e_catalogate_immagine_{i:03d}.csv')
    salva_csv_con_header_fits(dataframe, header_dal_csv, filename, percorso_file_fits)

    # Rimuovi il codice di visualizzazione per eseguire il loop completo più velocemente
    # Se vuoi visualizzare un'immagine per debug, de-commenta il blocco print(f"File {i}:...")