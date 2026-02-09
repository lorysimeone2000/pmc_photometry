import pandas as pd
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import os
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
import warnings
from astropy.wcs import FITSFixedWarning
from photutils.datasets import make_100gaussians_image
from scipy.optimize import curve_fit
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.utils.data import download_file
from astropy.table import Table, vstack
from photutils.detection import find_peaks
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from shapely.geometry import Point, Polygon
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning

# --- IMPORT FONDAMENTALE PER LA PORTABILITÀ ---
from pathlib import Path

# --- GESTIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI DINAMICA (PORTABILITÀ TOTALE)
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
        print(
            f"INFO: Trovati {len(files_trovati)} file '{nome_file_esatto}'. Uso il primo: {files_trovati[0].relative_to(base_dir)}")
    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    if len(cartelle_trovate) > 1:
        print(
            f"INFO: Trovate {len(cartelle_trovate)} cartelle '{nome_cartella_esatto}'. Uso la prima: {cartelle_trovate[0].relative_to(base_dir)}")
    return cartelle_trovate[0]


def leggi_file_parametri(percorso):
    parametri = {}
    if not os.path.exists(percorso): return {}
    with open(percorso, 'r') as file:
        next(file, None)
        for riga in file:
            riga = riga.split('#')[0].strip()
            if riga:
                parts = riga.split()
                if len(parts) >= 2:
                    try:
                        valore = float(parts[1]) if '.' in parts[1] else int(parts[1])
                        parametri[parts[0]] = valore
                    except ValueError:
                        pass
    return parametri


def elabora_file_fits(percorso_file_):
    with fits.open(percorso_file_, memmap=False) as hdu_list_:
        image_data_ = hdu_list_[0].data
        w_ = WCS(hdu_list_[0].header)
        mean_, median_, std_ = sigma_clipped_stats(image_data_, sigma=3.0)
        image_data_ = image_data_ - median_
        return image_data_, median_, w_


def converti_valore(valore):
    valore = str(valore).strip()
    if not valore: return valore
    try:
        return int(valore)
    except ValueError:
        pass
    try:
        return float(valore)
    except ValueError:
        pass
    return valore


def leggi_header_da_csv(filename):
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            else:
                break
    return header_dict


# Definisco la BASE_DIR dinamicamente
BASE_DIR = trova_cartella_base("pmc_photometry")
print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")

RUN = [1, 2, 3]
files_per_run = {}

print("\n--- RICERCA FILE CSV ---")
for r in RUN:
    nome_cartella_target = f"tabelle_unite_run_{r}"
    path_cartella_run = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_target)
    if path_cartella_run is None:
        print(f"[ATTENZIONE] Cartella '{nome_cartella_target}' non trovata. Salto la run {r}.")
        files_per_run[r] = []
        continue

    print(f"Run {r}: Cartella trovata in -> {path_cartella_run.relative_to(BASE_DIR)}")
    lista_csv = sorted(list(path_cartella_run.glob("*.csv")))
    files_per_run[r] = lista_csv
    if not lista_csv:
        print(f"   [AVVISO] Nessun file .csv trovato.")
    else:
        print(f"   Trovati {len(lista_csv)} file CSV.")

# =============================================================================
# ANALISI GLOBALE SU TUTTE LE RUN E ISTOGRAMMA GENERALIZZATO
# =============================================================================

totale_immagini_processate = 0
lista_totale_dfs = []

# 2. CICLO DI RACCOLTA DATI
for r in RUN:
    files_correnti = files_per_run.get(r, [])
    if not files_correnti: continue

    # Incrementiamo il massimo possibile asse X
    totale_immagini_processate += len(files_correnti)

    descrizione_bar = f"Lettura Run {r} ({len(files_correnti)} file)"
    for nome_csv in tqdm(files_correnti, desc=descrizione_bar):
        try:
            # Leggiamo il CSV
            df_temp = pd.read_csv(nome_csv, comment='#')

            # Verifichiamo che ci siano le colonne necessarie
            if 'Corrispondenza' not in df_temp.columns: continue
            if 'ripetizioni' not in df_temp.columns: continue

            # Filtriamo solo quelli NON catalogati
            df_no = df_temp[df_temp['Corrispondenza'] == 'NO'].copy()

            if not df_no.empty:
                # Teniamo solo le colonne utili per alleggerire
                cols_to_keep = ['ID', 'ripetizioni']
                # Se esiste run_unique_id usiamo quello, altrimenti ID (che dovrebbe essere INT_X)
                if 'run_unique_id' in df_no.columns:
                    cols_to_keep.append('run_unique_id')

                lista_totale_dfs.append(df_no[[c for c in cols_to_keep if c in df_no.columns]])

        except Exception as e:
            print(f"Errore lettura {nome_csv}: {e}")
            continue

# 3. ELABORAZIONE E DEDUPLICAZIONE
print("\n--- ELABORAZIONE DATI ---")

valori_finali_ripetizioni = []

if lista_totale_dfs:
    # Concateniamo tutto in un unico DataFrame gigante
    df_global = pd.concat(lista_totale_dfs, ignore_index=True)

    # Determiniamo quale colonna usare come ID univoco
    col_id = 'run_unique_id' if 'run_unique_id' in df_global.columns else 'ID'
    print(f"Uso la colonna '{col_id}' come identificativo univoco.")

    # Rimuoviamo i duplicati.
    # Poiché 'ripetizioni' è già il totale calcolato nel passaggio precedente,
    # ogni riga dello stesso oggetto ha lo stesso valore di 'ripetizioni'.
    # Ne basta una per oggetto.
    df_unique = df_global.drop_duplicates(subset=[col_id])

    # Estraiamo i valori per il plot
    valori_finali_ripetizioni = df_unique['ripetizioni'].dropna().astype(int).values

    print(f"Totale righe grezze lette: {len(df_global)}")
    print(f"Totale oggetti UNICI identificati: {len(valori_finali_ripetizioni)}")
else:
    print("Nessun dato trovato per oggetti 'NO'.")

# 4. PLOTTING
print(f"\n--- GENERAZIONE GRAFICO ---")
print(f"Max asse X (Totale Immagini): {totale_immagini_processate}")

if len(valori_finali_ripetizioni) > 0:
    plt.figure(figsize=(15, 7))

    # Binning: da 1 al totale immagini (+2 per margine destro ed estetico)
    # Esempio: se ho 100 immagini, voglio bin da 1 a 100.
    bins = np.arange(1, totale_immagini_processate + 2) - 0.5

    plt.hist(valori_finali_ripetizioni,
             bins=bins,
             color='royalblue',
             edgecolor='black',
             alpha=0.8,
             log=True)  # Scala logaritmica attiva

    # Setup Assi
    plt.xlim(0.5, totale_immagini_processate + 0.5)

    # IMPORTANTE PER SCALA LOG: Impostiamo il fondo a un valore > 0
    plt.ylim(bottom=0.8)

    plt.xlabel(f"Numero Totale di Apparizioni (Su {totale_immagini_processate} immagini totali)", fontsize=12)
    plt.ylabel("Numero di Oggetti 'NO' Unici (Scala Log)", fontsize=12)
    plt.title(
        f"Distribuzione Ripetizioni Oggetti Non Catalogati\n(Totale oggetti unici: {len(valori_finali_ripetizioni)})",
        fontsize=14)

    # Grid e Ticks intelligenti
    plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
    plt.minorticks_on()

    # Gestione dinamica dei tick sull'asse X per evitare affollamento
    if totale_immagini_processate > 50:
        step = int(np.ceil(totale_immagini_processate / 20))
        plt.xticks(np.arange(0, totale_immagini_processate + 1, step))
    else:
        plt.xticks(np.arange(0, totale_immagini_processate + 1, 1))

    output_plot = "istogramma_ripetizioni_globale_log.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {output_plot}")
    plt.show()

else:
    print("Nessun dato da graficare (lista valori vuota).")