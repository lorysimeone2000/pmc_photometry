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
    Restituisce il Path del primo file trovato o None se non esiste.
    """
    # rglob cerca in modo ricorsivo (*)
    files_trovati = list(base_dir.rglob(nome_file_esatto))

    if not files_trovati:
        return None

    if len(files_trovati) > 1:
        # Ordina per lunghezza del percorso per prendere magari quello più "vicino" o specifico
        files_trovati.sort(key=lambda p: len(str(p)))
        print(
            f"INFO: Trovati {len(files_trovati)} file '{nome_file_esatto}'. Uso il primo: {files_trovati[0].relative_to(base_dir)}")

    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    """
    Cerca una CARTELLA ricorsivamente in tutte le sottocartelle di base_dir.
    Restituisce il Path della prima cartella trovata o None se non esiste.
    """
    # Cerchiamo directory che matchano il nome
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]

    if not cartelle_trovate:
        return None

    # Ordiniamo per lunghezza percorso (preferiamo quelle nella root o più in alto)
    cartelle_trovate.sort(key=lambda p: len(str(p)))

    if len(cartelle_trovate) > 1:
        print(
            f"INFO: Trovate {len(cartelle_trovate)} cartelle '{nome_cartella_esatto}'. Uso la prima: {cartelle_trovate[0].relative_to(base_dir)}")

    return cartelle_trovate[0]


def leggi_file_parametri(percorso):
    """Legge il file dei parametri in un dizionario."""
    parametri = {}
    if not os.path.exists(percorso):
        print(f"Warning: File parametri non trovato in {percorso}")
        return {}
    with open(percorso, 'r') as file:
        next(file, None)
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
    return parametri


def elabora_file_fits(percorso_file_):
    """Carica il FITS e sottrae il fondo."""
    # memmap=False previene errori con BZERO/BSCALE
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

# Dizionario per salvare i percorsi dei file: { numero_run: [lista_percorsi_csv] }
files_per_run = {}

print("\n--- RICERCA FILE CSV ---")

for r in RUN:
    # Nome della cartella che ci aspettiamo di trovare (es. tabelle_unite_run_1)
    nome_cartella_target = f"tabelle_unite_run_{r}"

    # 1. Cerchiamo la cartella specifica per questa run in tutto il progetto
    path_cartella_run = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_target)

    if path_cartella_run is None:
        print(f"[ATTENZIONE] Cartella '{nome_cartella_target}' non trovata. Salto la run {r}.")
        files_per_run[r] = []
        continue

    print(f"Run {r}: Cartella trovata in -> {path_cartella_run.relative_to(BASE_DIR)}")

    # 2. Troviamo tutti i CSV all'interno di quella cartella
    # Usiamo sorted() per averli in ordine (es. immagine_001, immagine_002...)
    lista_csv = sorted(list(path_cartella_run.glob("*.csv")))

    if not lista_csv:
        print(f"   [AVVISO] Nessun file .csv trovato in questa cartella.")
    else:
        print(f"   Trovati {len(lista_csv)} file CSV.")

    files_per_run[r] = lista_csv

# =============================================================================
# ANALISI GLOBALE SU TUTTE LE RUN E ISTOGRAMMA GENERALIZZATO
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- MODIFICA 1: Dizionario per accumulare le ripetizioni per ID invece di lista piatta ---
mappa_ripetizioni_globali = {}

# --- MODIFICA 2: Contatore totale delle immagini processate per definire il range X ---
totale_immagini_processate = 0

# 2. CICLO SU TUTTE LE RUN
for r in RUN:
    files_correnti = files_per_run.get(r, [])

    if not files_correnti:
        continue

    # Aggiorniamo il contatore totale delle immagini (asse X massimo potenziale)
    totale_immagini_processate += len(files_correnti)

    lista_dfs_run_corrente = []

    # -- A. Estrazione e Filtraggio per la Run corrente --
    descrizione_bar = f"Elaborazione Run {r} ({len(files_correnti)} file)"
    for nome_csv in tqdm(files_correnti, desc=descrizione_bar):

        # Leggi CSV
        try:
            df_temp = pd.read_csv(nome_csv, comment='#')
        except Exception as e:
            print(f"Errore lettura {nome_csv}: {e}")
            continue

        # Filtra 'NO'
        if 'Corrispondenza' in df_temp.columns:
            df_no = df_temp[df_temp['Corrispondenza'] == 'NO'].copy()
            if not df_no.empty:
                lista_dfs_run_corrente.append(df_no)

    # -- B. Concatenazione e Deduplicazione per la Run corrente --
    if lista_dfs_run_corrente:
        # Concateniamo tutti gli oggetti della run r
        df_concat_run = pd.concat(lista_dfs_run_corrente, ignore_index=True)

        # Riduciamo a oggetti unici (una riga per ID).
        # IMPORTANTE: L'ID 'INT_X' è univoco tra le run grazie allo script creazione_tabelle.py
        if 'ID' in df_concat_run.columns:
            df_unici_run = df_concat_run.drop_duplicates(subset=['ID'], keep='first')

            # Nome colonna specifico per questa run
            nome_colonna_ripetizioni = f"ripetizioni_run_{r}"

            if nome_colonna_ripetizioni in df_unici_run.columns:
                # Estraiamo ID e valori di ripetizione
                ids_oggetti = df_unici_run['ID'].values
                ripetizioni = df_unici_run[nome_colonna_ripetizioni].values

                # --- MODIFICA 3: Somma accumulativa per ID ---
                count_nuovi = 0
                for obj_id, n_rep in zip(ids_oggetti, ripetizioni):
                    # Convertiamo a int per sicurezza (gestione NaN o float)
                    try:
                        val = int(float(n_rep))
                    except:
                        val = 0

                    if val > 0:
                        # Se l'ID esiste già (trovato in run precedenti), sommiamo
                        if obj_id in mappa_ripetizioni_globali:
                            mappa_ripetizioni_globali[obj_id] += val
                        else:
                            # Altrimenti inizializziamo
                            mappa_ripetizioni_globali[obj_id] = val
                        count_nuovi += 1

                print(f" -> Run {r}: Aggiornati/Aggiunti {count_nuovi} oggetti unici nella mappa globale.")
            else:
                print(f" -> [ERRORE] Run {r}: La colonna '{nome_colonna_ripetizioni}' non esiste nel DataFrame!")

# Estraiamo la lista finale dei valori (sommati) dal dizionario
valori_ripetizione_globali = list(mappa_ripetizioni_globali.values())
oggetti_con_ripetizioni = sum(1 for x in valori_ripetizione_globali if x > 1)

# Definizione del massimo asse X basata sulla somma totale delle immagini
max_immagini_globali = totale_immagini_processate
print(f"\nTotale immagini analizzate (tutte le run): {max_immagini_globali}")
print(f"Totale oggetti unici trovati: {len(valori_ripetizione_globali)}")
print(f"Totale oggetti unici che si ripetono almeno una volta: {oggetti_con_ripetizioni}")

# 3. GENERAZIONE ISTOGRAMMA GLOBALE
if valori_ripetizione_globali:

    plt.figure(figsize=(16, 8))

    # Definizione bin dinamici fino al totale delle immagini processate
    # max_immagini_globali + 2 per assicurare che l'ultimo bin sia incluso
    bins = np.arange(1, max_immagini_globali + 2) - 0.5

    plt.hist(valori_ripetizione_globali,
             bins=bins,
             color='forestgreen',
             edgecolor='black',
             alpha=0.75,
             zorder=3)

    # Configurazione Assi Dinamica
    plt.xlim(0.5, max_immagini_globali + 0.5)

    # Gestione etichette asse X:
    # Se abbiamo molte immagini, sfoltiamo le etichette per leggibilità
    if max_immagini_globali < 50:
        step_xticks = 1
    elif max_immagini_globali < 150:
        step_xticks = 5
    else:
        step_xticks = 20

    plt.xticks(np.arange(0, max_immagini_globali + 1, step_xticks))

    plt.title(
        f"Distribuzione Globale Ripetizioni (SOMMATE su tutte le Run)\nTotale Oggetti Unici: {len(valori_ripetizione_globali)}",
        fontsize=16)
    plt.xlabel("Numero Totale di Apparizioni (Somma Run 1 + 2 + 3)", fontsize=14)
    plt.ylabel("Numero di Oggetti Unici", fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.grid(axis='x', linestyle=':', alpha=0.3, zorder=0)

    # plt.yscale('log') # Scommenta se vuoi la scala logaritmica

    plt.tight_layout()
    plt.show()

    # =============================================================================
    # PLOTTING
    # =============================================================================

    valori_totali = list(mappa_ripetizioni_globali.values())

    print(f"\n--- STATISTICHE FINALI ---")
    print(f"Totale Immagini (Run {RUN}): {totale_immagini_processate}")
    print(f"Oggetti 'NO' Unici Totali: {len(valori_totali)}")

    if valori_totali:
        plt.figure(figsize=(15, 7))

        # Binning: da 1 al totale immagini (+2 per margine destro)
        bins = np.arange(1, totale_immagini_processate + 2) - 0.5

        # --- MODIFICA QUI: log=True ---
        plt.hist(valori_totali,
                 bins=bins,
                 color='royalblue',
                 edgecolor='black',
                 alpha=0.8,
                 log=True)  # <--- Questo attiva la scala logaritmica sulle barre

        # Setup Assi
        plt.xlim(0, totale_immagini_processate + 1)

        # IMPORTANTE PER SCALA LOG: Impostiamo il fondo a un valore > 0 (es. 0.5 o 0.8)
        # altrimenti matplotlib potrebbe cercare di disegnare fino a -infinito
        plt.ylim(bottom=0.8)

        plt.xlabel(f"Numero Totale di Apparizioni (Somma su Run {RUN})", fontsize=12)
        plt.ylabel("Numero di Oggetti 'NO' Unici (Scala Log)", fontsize=12)
        plt.title(
            f"Istogramma Cumulativo Oggetti Senza Corrispondenza\n(Totale immagini analizzate: {totale_immagini_processate})",
            fontsize=14)

        # Grid e Ticks intelligenti
        plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')  # which='both' mostra la griglia logaritmica fine
        plt.minorticks_on()

        # Se l'asse X è molto lungo, sfoltiamo i tick
        if totale_immagini_processate > 50:
            step = int(totale_immagini_processate / 20)
            plt.xticks(np.arange(0, totale_immagini_processate + 1, step))
        else:
            plt.xticks(np.arange(0, totale_immagini_processate + 1, 1))

        output_plot = "istogramma_ripetizioni_globale_log.png"
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato in: {output_plot}")
        plt.show()

    else:
        print("Nessun dato da graficare.")

else:
    print("\n[AVVISO] Nessun oggetto non catalogato trovato in nessuna delle run.")



