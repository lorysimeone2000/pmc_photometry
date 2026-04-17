import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.optimize import curve_fit
import warnings
from pathlib import Path
from tqdm import tqdm
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


def trova_cartella_base(nome_target="pmc_photometry"):
    # cerco la mia cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")

PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

from funzioni.utilita_parquet import *
from funzioni.astrometria_parquet import *

# =============================================================================
# 1. LETTURA DEL FILE ORIGINALE E PREPARAZIONE TARGET
# =============================================================================

# individuo il file CSV originale
percorso_csv = cerca_file_nel_progetto(BASE_DIR, "oggetti_presenza_multirun.csv")

# estraggo la tabella pandas dal file
df = pd.read_csv(percorso_csv)

# filtro il dataframe estraendo unicamente gli oggetti con Mag_estratta_max minore di 8
df_filtrato = df[df['Mag_estratta_max'] < 8].copy()

# =============================================================================
# 2. SALVATAGGIO DEI RISULTATI
# =============================================================================

# cerco la cartella di output
cartella_output = cerca_cartella_nel_progetto(BASE_DIR, "presenza_consecutiva_multirun")

# definisco il percorso completo per il nuovo file
percorso_candidati = cartella_output / "candidati.csv"

# salvo il dataframe in formato csv escludendo la colonna degli indici
df_filtrato.to_csv(percorso_candidati, index=False)

print(f"\nSalvataggio completato! Trovati {len(df_filtrato)} candidati.")
print(f"File salvato in: {percorso_candidati}")