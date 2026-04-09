import pandas as pd
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
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

warnings.filterwarnings('ignore', category=FITSFixedWarning)

from pathlib import Path
from tqdm import tqdm


# =============================================================================
# FUNZIONI DI GESTIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # Cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # Cerco una cartella specifica ricorsivamente
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


def converti_valore(valore):
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


def freedman_diaconis_bins(data, num_images=1, max_bins=60):
    # Riduco N al numero medio di stelle per immagine per non far esplodere la formula
    data_clean = data.compressed() if hasattr(data, 'mask') else data
    n_effettivo = len(data_clean) / max(num_images, 1)
    if n_effettivo < 2: return 1

    iqr = np.percentile(data_clean, 75) - np.percentile(data_clean, 25)
    if iqr == 0: return 1

    bin_width = 2 * iqr / (n_effettivo ** (1 / 3))
    data_range = np.max(data_clean) - np.min(data_clean)
    bins = int(np.ceil(data_range / bin_width))

    # Impongo un tetto massimo e un minimo di 1
    return min(max(bins, 1), max_bins)


# --- INIZIO CODICE ---
# Imposto la cartella base in modo dinamico
BASE_DIR = trova_cartella_base("Lorenzo")

RUNS = [1, 2, 3]

# Imposto le liste globali per accumulare i dati di tutte le immagini di tutte le run
tutti_mag_data = []
tutti_mag_cat_data = []
totale_perse = 0
totale_catalogate = 0
totale_correlate = 0

# Preparo le strutture per salvare i parametri per il titolo
fwhm_usato = None
size_usato = None

print("Inizio scansione Run...")

# Aggiungo tqdm al ciclo principale delle Run
for run in tqdm(RUNS, desc="Avanzamento Globale Runs"):
    # Cerco dinamicamente la cartella tabelle_unite_run_X
    nome_cartella_csv = f"tabelle/tabelle_unite/tabelle_unite_run_{run}"
    cartella_csv_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_csv)
    if cartella_csv_path is None:
        print(f"AVVISO: Cartella '{nome_cartella_csv}' non trovata. Salto la run {run}.")
        continue
    lista_percorsi_csv = sorted([str(f) for f in cartella_csv_path.glob('run_*_stelle_trovate_e_catalogate_immagine_*.csv')])

    # Cerco dinamicamente la cartella sorgenti_catalogate_run_X
    nome_cartella_csv_cat = f"tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}"
    cartella_csv_cat_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_csv_cat)
    if cartella_csv_cat_path is None:
        print(f"AVVISO: Cartella '{nome_cartella_csv_cat}' non trovata. Salto la run {run}.")
        continue
    lista_percorsi_csv_cat = sorted([str(f) for f in cartella_csv_cat_path.glob('*.csv')])

    # Mi assicuro che il numero di file corrisponda
    num_file = min(len(lista_percorsi_csv), len(lista_percorsi_csv_cat))
    if num_file == 0:
        print(f"AVVISO: Nessun file trovato per la run {run}.")
        continue

    print(f"\nElaborazione Run {run} ({num_file} immagini)...")

    for n_immagine in tqdm(range(num_file), desc=f"Run {run} Dettaglio", leave=False):
        percorso_file_csv = lista_percorsi_csv[n_immagine]
        dataframe = pd.read_csv(percorso_file_csv, comment='#')
        tbl = Table.from_pandas(dataframe)

        percorso_file_csv_cat = lista_percorsi_csv_cat[n_immagine]
        dataframe_cat = pd.read_csv(percorso_file_csv_cat, comment='#')
        tbl_cat = Table.from_pandas(dataframe_cat)

        # Prendo i parametri (basta prenderli dalla prima immagine valida)
        if fwhm_usato is None:
            header_dal_csv = leggi_header_da_csv(percorso_file_csv)
            fwhm_usato = header_dal_csv.get('seg_fwhm', header_dal_csv.get('SEG_FWHM'))
            size_usato = header_dal_csv.get('seg_size', header_dal_csv.get('SEG_SIZE'))

        # Prendo i dati di corrispondenza
        mask_si = np.char.startswith(tbl['Corrispondenza'].astype(str), 'SI')
        ids_trovati_e_correlati = set(tbl[mask_si]['ID'])

        # Conto le perse
        for star_id in tbl_cat['ID']:
            if star_id not in ids_trovati_e_correlati:
                totale_perse += 1

        # Rimuovo duplicati per il multiple matching
        df_catalogate_corr = tbl[mask_si].to_pandas()
        df_uniche = df_catalogate_corr.drop_duplicates(subset=['ID'])

        totale_catalogate += len(tbl_cat)
        totale_correlate += len(df_uniche)

        magnitudini = df_uniche['Mag']
        magnitudini_cat = tbl_cat['Mag']

        # Converto in array puliti
        if hasattr(magnitudini, 'compressed'):
            mag_data = magnitudini.compressed()
        else:
            mag_data = np.array(magnitudini)
            mag_data = mag_data[~np.isnan(mag_data)]

        if hasattr(magnitudini_cat, 'compressed'):
            mag_cat_data = magnitudini_cat.compressed()
        else:
            mag_cat_data = np.array(magnitudini_cat)
            mag_cat_data = mag_cat_data[~np.isnan(mag_cat_data)]

        tutti_mag_data.extend(mag_data)
        tutti_mag_cat_data.extend(mag_cat_data)

import time

# Fine ciclo, verifico di avere dati
if len(tutti_mag_cat_data) == 0:
    print("ERRORE: Nessun dato valido caricato.")
    exit()

print("\n--- Inizio monitoraggio tempi post-ciclo ---")

# Avvio il cronometro per le immagini totali
t0_immagini = time.perf_counter()
# Calcolo quante immagini totali ho processato per la formula dei bin e per le medie
immagini_totali = sum(
    len(list(cerca_cartella_nel_progetto(BASE_DIR, f"tabelle_unite_run_{r}").glob('*.csv'))) for r in RUNS if
    cerca_cartella_nel_progetto(BASE_DIR, f"tabelle_unite_run_{r}"))
print(f"Tempo calcolo immagini_totali (rglob): {time.perf_counter() - t0_immagini:.3f} s")

print(f"\nRiepilogo Globale:")
print(f"Stelle totali di catalogo (tutte le run): {totale_catalogate}")
print(f"Stelle correlate uniche (tutte le run): {totale_correlate}")
print(f"Stelle di catalogo NON correlate/trovate: {totale_perse}")

# Avvio il cronometro per la concatenazione
t0_concat = time.perf_counter()
# 1. Calcolo i bin comuni GLOBALI
dati_totali = np.concatenate((tutti_mag_data, tutti_mag_cat_data))
print(f"Tempo concatenazione dati totali: {time.perf_counter() - t0_concat:.3f} s")

# Avvio il cronometro per i percentili
t0_fd = time.perf_counter()
# Passo il numero di immagini alla funzione e mi assicuro che i bin non esplodano
n_bin = freedman_diaconis_bins(dati_totali, num_images=immagini_totali)
print(f"Tempo Freedman-Diaconis (np.percentile su {len(dati_totali)} elementi): {time.perf_counter() - t0_fd:.3f} s")

# Avvio il cronometro per i bin edges
t0_edges = time.perf_counter()
hist_range = (np.min(dati_totali), np.max(dati_totali))
bins = np.histogram_bin_edges(dati_totali, bins=n_bin, range=hist_range)
print(f"Tempo calcolo Bin Edges: {time.perf_counter() - t0_edges:.3f} s")

# Avvio il cronometro per gli istogrammi
t0_hist = time.perf_counter()
# 2. Calcolo i conteggi GLOBALI
counts_cat, bin_edges = np.histogram(tutti_mag_cat_data, bins=bins)
counts_corr, _ = np.histogram(tutti_mag_data, bins=bins)
print(f"Tempo np.histogram: {time.perf_counter() - t0_hist:.3f} s")

# 3. Calcolo le Medie (divido per il numero totale di immagini analizzate)
media_counts_cat = counts_cat / immagini_totali
media_counts_corr = counts_corr / immagini_totali

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Avvio il cronometro per il plot
t0_plot = time.perf_counter()
# 4. Creo il grafico a linee

# Impostazione dimensioni testo ottimali per un foglio A4
plt.rc('font', size=12)          # Dimensione testo base
plt.rc('axes', titlesize=13)     # Titolo del grafico
plt.rc('axes', labelsize=12)     # Etichette assi X e Y
plt.rc('xtick', labelsize=10)    # Numeri asse X
plt.rc('ytick', labelsize=10)    # Numeri asse Y
plt.rc('legend', fontsize=10)    # Testo della legenda

# Dimensioni del grafico (7x5 pollici è ideale per riempire la larghezza in un A4)
plt.figure(figsize=(7, 5))

# Disegno le Sorgenti Catalogate (Media)
plt.plot(bin_centers, media_counts_cat,
         color='purple',
         linestyle='-',
         linewidth=1.5,
         label='Catalogued Sources (Mean)')

# Disegno le Sorgenti Correlate (Media)
plt.plot(bin_centers, media_counts_corr,
         color='red',
         linestyle='-',
         linewidth=1.5,
         label='Correlated Sources (Mean)')

# Imposto gli Assi
plt.yscale('log')
plt.xlabel('Magnitude (Bin Centres)')
plt.ylabel('Mean Frequency (Counts / Image)')

# Costruisco il titolo
titolo = f'Mean Magnitude Distribution: Catalogued vs Correlated (Runs {RUNS})\n'
titolo += f'Mean of {totale_correlate / immagini_totali:.1f} matches out of {totale_catalogate / immagini_totali:.1f} catalogued per image'
if fwhm_usato and size_usato:
    titolo += f' (FWHM = {fwhm_usato}, size = {size_usato})'
plt.title(titolo, pad=12) # Aggiungo padding per staccare il titolo dal riquadro

# Inverto l'asse X (magnitudini astronomiche)
plt.gca().invert_xaxis()

# Aggiungo Griglia e Legenda
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
print(f"Tempo preparazione grafico Matplotlib: {time.perf_counter() - t0_plot:.3f} s")
print("--- Fine monitoraggio, avvio render grafico ---")

# Suggerimento: salvalo come PDF vettoriale per la tesi in LaTeX
plt.savefig('magnitude_distribution.pdf', format='pdf', bbox_inches='tight')
plt.savefig('magnitude_distribution.png', format='png', bbox_inches='tight')

plt.show()