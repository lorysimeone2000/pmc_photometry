import numpy as np
import os
import sys
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import warnings
from astropy.wcs import FITSFixedWarning
from tqdm import tqdm
import pandas as pd

# importo i moduli necessari per la statistica e la visualizzazione
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats

warnings.filterwarnings('ignore', category=FITSFixedWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    # cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


# trovo la cartella base del mio progetto
BASE_DIR = trova_cartella_base("Lorenzo")

PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")
if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

# =============================================================================
# 1. RICERCA E RACCOLTA DATI
# =============================================================================
print("--- INIZIO RICERCA IMMAGINI BLAZAR ---")

# imposto la cartella radice dei dati specifica
dir_dati = Path("/home/lorysimeone/tesi_magistrale/Lorenzo/PMC_DATA_BLAZAR")

if not dir_dati.exists():
    print(f"ERRORE: Impossibile trovare la cartella dati {dir_dati}")
    exit()

tutti_file_fits = []
nomi_run_processate = []

# trovo tutte le cartelle che contengono almeno un file fits
estensioni_valide = ['.fit', '.fits']
cartelle_con_fits = set(f.parent for f in dir_dati.rglob('*') if f.suffix.lower() in estensioni_valide)

# esploro ogni cartella trovata per estrarre le immagini
for cartella in sorted(cartelle_con_fits):
    # estraggo e ordino alfabeticamente i file fits nella cartella corrente
    file_run = sorted([str(f) for f in cartella.glob('*') if f.suffix.lower() in estensioni_valide and f.is_file()])

    # salto la prima e le ultime due immagini della singola run per evitare scarti
    if len(file_run) > 3:
        file_run_validi = file_run[1:-2]
        tutti_file_fits.extend(file_run_validi)
        nomi_run_processate.append(cartella.name)

if not tutti_file_fits:
    print("ERRORE: Nessun file FITS valido trovato nelle cartelle specificate!")
    exit()

print(f"Trovate {len(tutti_file_fits)} immagini valide in {len(nomi_run_processate)} run totali.")