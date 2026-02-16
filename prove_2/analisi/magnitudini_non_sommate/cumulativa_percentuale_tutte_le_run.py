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
BASE_DIR = trova_cartella_base("pmc_photometry")

RUNS = [1, 2, 3]

# Inizializzo le liste globali per accumulare i dati di tutte le immagini
tutti_mag_data = []
tutti_mag_cat_data = []
totale_perse = 0
totale_catalogate = 0
totale_correlate = 0

# Variabili per tenere traccia dei parametri
fwhm_usato = None
size_usato = None
immagini_totali = 0

print("Inizio scansione Run...")

for run in RUNS:
    # Cerco dinamicamente la cartella tabelle_unite_run_X
    nome_cartella_csv = f"tabelle_unite_run_{run}"
    cartella_csv_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_csv)
    if cartella_csv_path is None:
        print(f"AVVISO: Cartella '{nome_cartella_csv}' non trovata. Salto la run {run}.")
        continue
    lista_percorsi_csv = sorted([str(f) for f in cartella_csv_path.glob('*.csv')])

    # Cerco dinamicamente la cartella sorgenti_catalogate_run_X
    nome_cartella_csv_cat = f"sorgenti_catalogate_run_{run}"
    cartella_csv_cat_path = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_csv_cat)
    if cartella_csv_cat_path is None:
        print(f"AVVISO: Cartella '{nome_cartella_csv_cat}' non trovata. Salto la run {run}.")
        continue
    lista_percorsi_csv_cat = sorted([str(f) for f in cartella_csv_cat_path.glob('*.csv')])

    # Mi assicuro che il numero di file corrisponda tra le due cartelle
    num_file = min(len(lista_percorsi_csv), len(lista_percorsi_csv_cat))
    if num_file == 0:
        print(f"AVVISO: Nessun file trovato per la run {run}.")
        continue

    print(f"Elaborazione Run {run} ({num_file} immagini)...")

    for n_immagine in tqdm(range(num_file), desc=f"Run {run}"):
        percorso_file_csv = lista_percorsi_csv[n_immagine]
        dataframe = pd.read_csv(percorso_file_csv, comment='#')
        tbl = Table.from_pandas(dataframe)

        percorso_file_csv_cat = lista_percorsi_csv_cat[n_immagine]
        dataframe_cat = pd.read_csv(percorso_file_csv_cat, comment='#')
        tbl_cat = Table.from_pandas(dataframe_cat)

        # Salvo i parametri dalla prima immagine valida per scriverli nel titolo
        if fwhm_usato is None:
            header_dal_csv = leggi_header_da_csv(percorso_file_csv)
            fwhm_usato = header_dal_csv.get('fwhm', header_dal_csv.get('FWHM'))
            size_usato = header_dal_csv.get('size', header_dal_csv.get('SIZE'))

        # Preparo i dati
        mask_si = np.char.startswith(tbl['Corrispondenza'].astype(str), 'SI')
        ids_trovati_e_correlati = set(tbl[mask_si]['ID'])

        # Conto le stelle non correlate (perse)
        for star_id in tbl_cat['ID']:
            if star_id not in ids_trovati_e_correlati:
                totale_perse += 1

        # Rimuovo i duplicati per evitare conteggi superiori al 100%
        df_catalogate_corr = tbl[mask_si].to_pandas()
        df_uniche = df_catalogate_corr.drop_duplicates(subset=['ID'])

        totale_catalogate += len(tbl_cat)
        totale_correlate += len(df_uniche)

        magnitudini = df_uniche['Mag']
        magnitudini_cat = tbl_cat['Mag']

        # Converto in array puliti rimuovendo i NaN
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

        # Aggiungo i dati dell'immagine corrente alle liste globali
        tutti_mag_data.extend(mag_data)
        tutti_mag_cat_data.extend(mag_cat_data)
        immagini_totali += 1

# Blocco l'esecuzione se non ho caricato dati
if len(tutti_mag_cat_data) == 0:
    print("ERRORE: Nessun dato valido caricato.")
    exit()

print(f"\nRiepilogo Globale:")
print(f"Stelle totali di catalogo (tutte le run): {totale_catalogate}")
print(f"Stelle correlate uniche (tutte le run): {totale_correlate}")
print(f"Stelle di catalogo NON correlate/trovate: {totale_perse}")

# 1. Calcolo i bin comuni GLOBALI
dati_totali = np.concatenate((tutti_mag_data, tutti_mag_cat_data))
n_bin = freedman_diaconis_bins(dati_totali, num_images=immagini_totali)
hist_range = (np.min(dati_totali), np.max(dati_totali))
bins = np.histogram_bin_edges(dati_totali, bins=n_bin, range=hist_range)

# 2. Calcolo i conteggi GLOBALI
counts_cat, bin_edges = np.histogram(tutti_mag_cat_data, bins=bins)
counts_corr, _ = np.histogram(tutti_mag_data, bins=bins)

# 3. Calcolo LA PERCENTUALE GLOBALE (COMPLETEZZA)
# Dividendo i totali in ogni bin, ottengo la media pesata di completezza dell'intero dataset
with np.errstate(divide='ignore', invalid='ignore'):
    percentuale_completezza = (counts_corr / counts_cat) * 100

# Sostituisco i NaN e gli Inf (che derivano dalle divisioni per zero) con 0
percentuale_completezza = np.nan_to_num(percentuale_completezza)

# Calcolo i centri dei bin
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 4. Creo il grafico a linee
plt.figure(figsize=(14, 8))

plt.plot(bin_centers, percentuale_completezza,
         color='green',
         marker='o',
         markersize=4,
         linestyle='-',
         linewidth=2,
         label='Percentuale di rilevamento Globale')

# Traccio le linee di riferimento al 100% e al 50%
plt.axhline(100, color='gray', linestyle='--', alpha=0.5, label='100% Completezza')
plt.axhline(50, color='red', linestyle=':', alpha=0.5, label='50% Completezza')

# Imposto gli assi
plt.xlabel('Magnitudine (Centri dei Bin)')
plt.ylabel('Percentuale stelle correlate / totali (%)')

titolo = f'Funzione di Completezza Globale: Efficienza per Magnitudine (Run {RUNS})\n'
titolo += f'Media globale di {totale_correlate / immagini_totali:.1f} match su {totale_catalogate / immagini_totali:.1f} catalogate per immagine'
if fwhm_usato and size_usato:
    titolo += f' (FWHM = {fwhm_usato}, size = {size_usato})'
plt.title(titolo)

# Inverto l'asse X (magnitudini astronomiche decrescenti)
plt.gca().invert_xaxis()

# Imposto il limite Y da 0 a poco più di 100 per mantenere il grafico pulito
plt.ylim(0, 110)

# Aggiungo la griglia
plt.grid(True, which="both", linestyle='--', alpha=0.6)

# Formatto i Tick sull'Asse X mostrando solo un tot di etichette
tick_labels = [f'{c:.2f}' for c in bin_centers]
step = 4
subset_ticks = bin_centers[::step]
subset_labels = tick_labels[::step]
plt.gca().set_xticks(subset_ticks)
plt.gca().set_xticklabels(subset_labels, rotation=45, ha='right')

plt.legend()
plt.tight_layout()
plt.show()