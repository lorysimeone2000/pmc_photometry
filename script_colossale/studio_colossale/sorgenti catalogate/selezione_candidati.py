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
percorso_csv = cerca_file_nel_progetto(BASE_DIR, "oggetti_presenza_CATALOGATI_multirun.csv")

# estraggo la tabella pandas dal file
df = pd.read_csv(percorso_csv)

# filtro il mio dataframe mantenendo solo le righe con numero di run maggiore di 4
df = df[df['numero_di_run'] > 4]

# estraggo casualmente una delle stelle più luminose (valore minimo di magnitudine)
stella_bright = df[df['Mag_estratta_max'] == df['Mag_estratta_max'].min()].sample(n=1)

# escludo la stella più luminosa per le prossime estrazioni per assicurarmi di avere 4 stelle uniche
df_rimanente = df.drop(stella_bright.index)

# estraggo casualmente una stella con magnitudine attorno a 6 (intervallo tra 5.5 e 6.5)
stella_mag6 = df_rimanente[(df_rimanente['Mag_estratta_max'] >= 5.5) & (df_rimanente['Mag_estratta_max'] <= 6.5)].sample(n=1)

# estraggo casualmente una stella con magnitudine attorno a 8 (intervallo tra 7.5 e 8.5)
stella_mag8 = df_rimanente[(df_rimanente['Mag_estratta_max'] >= 7.5) & (df_rimanente['Mag_estratta_max'] <= 8.5)].sample(n=1)

# estraggo casualmente una stella con magnitudine attorno a 10 (intervallo tra 9.5 e 10.5)
stella_mag10 = df_rimanente[(df_rimanente['Mag_estratta_max'] >= 9.5) & (df_rimanente['Mag_estratta_max'] <= 10.5)].sample(n=1)

# concateno le mie 4 selezioni in un unico dataframe
df_filtrato = pd.concat([stella_bright, stella_mag6, stella_mag8, stella_mag10])

# =============================================================================
# 2. SALVATAGGIO DEI RISULTATI
# =============================================================================

# cerco la mia cartella di output
cartella_output = cerca_cartella_nel_progetto(BASE_DIR, "sorgenti catalogate")

# definisco il percorso completo per il nuovo file
percorso_candidati = cartella_output / "candidati.csv"

# salvo il mio dataframe in formato csv escludendo la colonna degli indici
df_filtrato.to_csv(percorso_candidati, index=False)

print(f"\nSalvataggio completato! Trovati {len(df_filtrato)} candidati.")
print(f"File salvato in: {percorso_candidati}")