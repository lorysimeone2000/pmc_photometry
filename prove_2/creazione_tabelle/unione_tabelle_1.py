import pandas as pd
#pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
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
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning) # Sopprime il warning FITSFixedWarning

from pathlib import Path

# sopprimo i warning non critici
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

    print("numero di stelle catalogate" , np.shape(coords_catalogate))

    # calcolo le distanze di tutti i centroidi da tutte le stelle catalogate
    distanze_minime = []
    corrispondenze = []
    righe_tabella_combinata = [] # lista per la tabella combinata

    for i, coord_trovata in enumerate(coords_trovate):
        # calcolo la distanza da tutte le stelle catalogate
        distanze_singola = coord_trovata.separation(coords_catalogate) # Calcola la distanza angolare tra la singola stella trovata
        # (coord_trovata) e tutte le stelle del catalogo (coords_catalogate). Restituisce un array di distanze angolari.
        # if i == 1: print(np.shape(distanze_singola))
        # trovo la distanza minima e l'indice della stella più vicina
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
        f.write(f"# DESCRIZIONE: Questo file csv contiene la tabella di tutte le sorgenti trovate con image segmentation insieme alle informazioni dell'eventuale stella corrispondente dei cataloghi\n")
        for key, value in header_fits.items():
            f.write(f"# {key}: {value}\n")
        f.write(f"# PERCORSO_FILE: {nome_file_fits}\n")
        f.write("#\n")  # Linea vuota per separare header dai dati
        # Scrivi il DataFrame
        dataframe.to_csv(f, index=False)

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

run = int(input("Quale run vuoi elaborare: ")) # numero run: 1, 2 o 3

# cartella contenente i file CSV delle stelle catalogate
cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}"

file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])

# creo i percorsi per ogni file
lista_percorsi_csv_stelle_catalogate = [os.path.join(cartella_csv, file) for file in file_csv]

# cartella contenente i file CSV delle stelle trovate
cartella_csv_ = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_trovate_run/sorgenti_trovate_run_{run}"

file_csv_ = sorted([f for f in os.listdir(cartella_csv_) if f.endswith('.csv')])

# creo i percorsi per ogni file
lista_percorsi_csv_stelle_trovate = [os.path.join(cartella_csv_, file) for file in file_csv_]

print(range(len(lista_percorsi_csv_stelle_trovate)))
soglia_correlazione = 0.003349 * u.deg # soglia fissa che uso come limite di distanza accettabile per la correlazione con la stella più vicina

# definisco la cartella di output
output_dir = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"

i=0
for n in range(len(lista_percorsi_csv_stelle_catalogate)):

    percorso_csv_stelle_trovate = lista_percorsi_csv_stelle_trovate[n]
    percorso_csv_stelle_catalogate = lista_percorsi_csv_stelle_catalogate[n]
    header_dal_csv = leggi_header_da_csv(percorso_csv_stelle_catalogate)
    percorso_file_fits = header_dal_csv['PERCORSO_FILE']
    i = i+1
    print("Elaborando file \n", i)

    dataframe1 = pd.read_csv(percorso_csv_stelle_trovate, skiprows=59)
    dataframe2 = pd.read_csv(percorso_csv_stelle_catalogate, skiprows=59)
    tbl_trovate_completa = Table.from_pandas(dataframe1)
    tbl_catalogate = Table.from_pandas(dataframe2)
    tbl_trovate = tbl_trovate_completa
    tbl_trovate.keep_columns(['label','xcentroid', 'ycentroid', 'area', 'max_value', 'kron_flux'])
    hdu_list = fits.open(percorso_file_fits)
    header = hdu_list[0].header
    w = WCS(header)

    # converto i centroidi pixel in coordinate celesti
    coords_trovate = w.pixel_to_world(tbl_trovate['xcentroid'], tbl_trovate['ycentroid'])
    coords_celesti = coords_trovate
    tbl_trovate['RA_centroid'] = coords_celesti.ra
    tbl_trovate['DEC_centroid'] = coords_celesti.dec

    # creo una tabella vuota con la struttura combinata preservando i tipi
    tbl_finale = Table()

    # Aggiungi colonne dalla prima tabella preservando i tipi
    for colname in tbl_trovate.colnames:
            tbl_finale[colname] = tbl_trovate[colname][:0]  # Prende solo la struttura (0 righe) ma preserva il tipo

    tbl_finale['Corrispondenza'] = 'SI'

    # Aggiungi colonne dalla seconda tabella preservando i tipi
    for colname in tbl_catalogate.colnames:
        tbl_finale[colname] = tbl_catalogate[colname][:0]  # Prende solo la struttura (0 righe) ma preserva il tipo

    # metto la colonna "Catalogo" a prima di "ID"
    colonne = tbl_finale.colnames
    pos_id = colonne.index('ID')
    nuovo_ordine = colonne[:pos_id] + ['Catalogo'] + [col for col in colonne if
                                                      col != 'Catalogo' and col not in colonne[:pos_id]]
    tbl_finale = tbl_finale[nuovo_ordine]


    '''if i==1:
        print("Colonne stelle catalogate:\n",tbl_catalogate.colnames)
        print("Colonne stelle trovate:\n", tbl_trovate.colnames)
        # print("Tabella stelle catalogate:\n", tbl_catalogate)
        print("Tabella combinata (da costruire):\n", tbl_finale)'''

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

        # creo SkyCoord con i valori puri specificando le unità
        coords_catalogate = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)

    except Exception as e:
        print(f"Errore nell'approccio principale: {e}")
        # APPROCCIO ALTERNATIVO: uso direttamente i valori senza moltiplicare per unità
        coords_catalogate = SkyCoord(ra=tbl_catalogate['RAJ2000'],
                                     dec=tbl_catalogate['DEJ2000'],
                                     unit=u.deg)

    idx, distanze_2d, _ = match_coordinates_sky(coords_trovate, coords_catalogate)

    # filtro solo le stelle sotto soglia
    mask = distanze_2d <= soglia_correlazione
    indici_correlati = np.where(mask)[0] #trovo gli indici delle stelle trovate che soddisfano la condizione della maschera

    print(f"Trovate {len(indici_correlati)} corrispondenze sotto soglia")

    for idx_trovato in range(len(coords_trovate)):
        # Verifica se questa stella ha una corrispondenza catalogata
        ha_corrispondenza = idx_trovato in indici_correlati

        nuova_riga = {}  # creo un dizionario vuoto per contenere i dati della nuova riga combinata

        # Aggiungo al dizionario il valore delle colonne per la stella trovata corrente

        # dati dalla stella trovata (SEMPRE presenti)
        for colname in tbl_trovate.colnames:
            nuova_riga[colname] = tbl_trovate[colname][idx_trovato]

        nuova_riga['Corrispondenza'] = 'SI' if ha_corrispondenza else 'NO'

        # dati dalla stella catalogata corrispondente (SOLO se esiste corrispondenza)
        for colname in tbl_catalogate.colnames:
            if ha_corrispondenza:
                # Se ha corrispondenza, prendi i dati dalla stella catalogata
                idx_catalogato = idx[idx_trovato]
                nuova_riga[colname] = tbl_catalogate[colname][idx_catalogato]
            else:
                # Se NON ha corrispondenza, lascia la colonna vuota
                # Gestione specifica per ogni tipo di colonna
                if colname == 'Catalogo':  # colonna di stringhe
                    nuova_riga[colname] = 'N/A'  # stringa speciale per "non disponibile"
                elif tbl_catalogate[colname].dtype.kind in ['i', 'u']:  # interi
                    nuova_riga[colname] = -999  # valore sentinella per interi
                elif tbl_catalogate[colname].dtype.kind == 'f':  # float
                    nuova_riga[colname] = np.nan  # NaN per float
                elif tbl_catalogate[colname].dtype.kind == 'O':  # oggetti (stringhe)
                    nuova_riga[colname] = 'N/A'  # stringa vuota
                else:
                    nuova_riga[colname] = None  # valore generico

        tbl_finale.add_row(nuova_riga)


    tbl_correlate = tbl_finale[tbl_finale['Corrispondenza'] == 'SI']

    # creo sottocataloghi giusto per analisi
    tbl_vizier_correlate = tbl_finale[(tbl_finale['Corrispondenza'] == 'SI') & (tbl_finale['Catalogo'] == 'II/389/ps1_dr2')]
    tbl_hipparco_correlate = tbl_finale[(tbl_finale['Corrispondenza'] == 'SI') & (tbl_finale['Catalogo'] == 'I/239/hip_main')]

    num_correlate = len(tbl_correlate)
    num_non_correlate = len(tbl_finale) - len(tbl_correlate)
    num_vizier_correlate = len(tbl_vizier_correlate)
    num_hipparco_correlate = len(tbl_hipparco_correlate)

    '''if i == 1:
        print(f"File {i}:\n")
        print(f"Tabella stelle trovate:\n")
        print(tbl_trovate)
        print(f"Tabella stelle correlate del file {i}:")
        print(tbl_correlate)

        tbl = tbl_correlate

        # Matching con l'immagine della PMC

        posizioni_vere_celesti = SkyCoord(ra=tbl['RAJ2000'],
                                          dec=tbl['DEJ2000'],
                                          unit ='deg',
                                          frame='icrs')

        posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti)  # converto da celesti a pixel
        posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

        magnitudini = tbl['Mag']

        # Parametri per i raggi
        raggio_min = 4.0
        raggio_max = 20.0
        raggi = raggio_max - (magnitudini - magnitudini.min()) * (raggio_max - raggio_min) / (
                    magnitudini.max() - magnitudini.min())

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

        image_data = hdu_list[0].data
        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
        data_pmc = image_data - median  # tolgo il fondo

        ax.imshow(data_pmc, cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')

        tbl_trovate['xcentroid'].info.format = '.2f'  # optional format
        tbl_trovate['ycentroid'].info.format = '.2f'
        tbl_trovate['kron_flux'].info.format = '.2f'
        # print(tbl)

        positions = np.transpose((tbl_trovate['xcentroid'], tbl_trovate['ycentroid']))  # creo un array di posizioni
        # positions_sky = SkyCoord(positions, unit=u.deg, frame='icrs')
        posizioni_celesti_segmentation = w.pixel_to_world(positions)
        posizioni_celesti_segmentation_ra = np.array(posizioni_celesti_segmentation.ra)
        posizioni_celesti_segmentation_dec = np.array(posizioni_celesti_segmentation.dec)
        ra_segmentation_max = np.max(posizioni_celesti_segmentation_ra)
        ''''''print("RA max segmentazione: ", ra_segmentation_max)
        print("RA max catalogo: ", ra_max)''''''

        apertures = CircularAperture(positions, r=5.0)  # creo le aperture per ogni posizione
        apertures.plot(color='red', lw=1.)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(
            f'Matching: {len(tbl)} stelle del catalogo II/389/ps1_dr2 + Hipparco (se magnitudine < 7)\n Cerchi dimensionati per magnitudine (<{15})\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')

        ''''''plt.title(f'Matching: {len(tbl)} stelle del catalogo II/389/ps1_dr2 \n Cerchi dimensionati per magnitudine (<{mag_max})\n(Threshold = {threshold}, n. pixel min = {n}, FWHM = {fwhm}, dimensioni kernel = {size} pixel)')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')''''''

        # legenda
        legend_elements = [

            # Sorgenti rilevate (aperture rosse)
            Line2D([0], [0], marker='o', color='red', linestyle='None',
                   markersize=8, markerfacecolor='none', markeredgewidth=1,
                   label=f'Sorgenti rilevate: {len(tbl_trovate)} oggetti'),

            # Stelle catalogate (cerchi colorati)
            Circle((0.5, 0.5), 0.4, facecolor='blue', alpha=0.7, edgecolor='black', linewidth=1,
                   label=f'Oggetti cataloghi con corrispondenza: {len(tbl)}'),

            Line2D([0], [0], marker='', color='green', linestyle='None',
                   markersize=8, markerfacecolor='green', markeredgewidth=1,
                   label=f'\n----------------------------------------------\nCorrispondenze trovate: {num_correlate}'
                         f' di cui\n- {num_vizier_correlate} di II/389/ps1_dr2\n- {num_hipparco_correlate} di Hipparco\n'),

            Line2D([0], [0], marker='', color='orange', linestyle='None',
                   markersize=8, markerfacecolor='orange', markeredgewidth=1,
                   label=f'Corrispondenze non trovate: {num_non_correlate}')
        ]

        # Aggiungi la legenda
        ax.legend(handles=legend_elements, loc='upper right',
                  framealpha=0.85, fancybox=True, shadow=True)

        plt.show()

        hdu_list.close()
        break'''


    # creo i file csv
    dataframe = tbl_finale.to_pandas()
    filename = os.path.join(output_dir, f'run_{run}_stelle_trovate_e_catalogate_immagine_{i:03d}.csv')
    salva_csv_con_header_fits(dataframe, header, filename, percorso_file_fits)

    # if i == 15: break



