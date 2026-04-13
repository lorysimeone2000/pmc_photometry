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
# 1. LETTURA DEL FILE CANDIDATI E PREPARAZIONE TARGET
# =============================================================================

# individuo il file csv dei candidati
percorso_csv = cerca_file_nel_progetto(BASE_DIR, "candidati.csv")

# fermo l'esecuzione se non trovo il file
if percorso_csv is None:
    print("ERRORE: Non ho trovato il file 'candidati.csv'.")
    sys.exit()

# estraggo i dati ed isolo i label di interesse
df_candidati = pd.read_csv(percorso_csv)

# Converto in lista perché il filtro 'in' di PyArrow richiede una lista (o tupla)
lista_candidati = list(df_candidati['label'].unique())

print(f"Caricati {len(lista_candidati)} candidati unici dal file.")

# =============================================================================
# 2. RICERCA ED ESTRAZIONE DAI FILE PARQUET TRAMITE METADATI
# =============================================================================

# individuo la cartella contenente i file parquet
cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito"
file_parquet = list(cartella_tabelle.rglob("*run_*_run_*_immagine_*.parquet"))

if not file_parquet:
    print("ERRORE: Nessun file Parquet trovato nella cartella specificata.")
    sys.exit()

# preparo una lista vuota per immagazzinare i dataframe filtrati
dati_estratti = []

print("Inizio la scansione ottimizzata sfruttando le proprietà dei Parquet...")

# analizzo un file parquet alla volta
for file_p in tqdm(file_parquet, desc="Analisi file"):
    try:
        # Utilizzo i filtri nativi di PyArrow (pushdown predicates)
        tabella_p = pq.read_table(
            file_p,
            columns=['label', 'Mag_estratta', 'err_Mag_estratta'],
            filters=[('label', 'in', lista_candidati)]
        )

        # se la tabella non è vuota (ovvero ho trovato almeno un candidato in questo file)
        if tabella_p.num_rows > 0:
            df_trovati = tabella_p.to_pandas()

            # Leggo l'header del file parquet corrente
            header = leggi_header_da_parquet(file_p)

            # Estraggo la data di osservazione dal dizionario
            date_obs = header.get('DATE-OBS')

            # Aggiungo la colonna della data al dataframe temporaneo
            df_trovati['DATE-OBS'] = date_obs

            dati_estratti.append(df_trovati)

    except Exception:
        # ignoro eventuali file corrotti o illeggibili
        continue

# =============================================================================
# 3. UNIONE E SALVATAGGIO DEI RISULTATI
# =============================================================================

# assemblo il dataframe finale se ho estratto qualcosa
if dati_estratti:
    df_risultati = pd.concat(dati_estratti, ignore_index=True)

    # Cerco la cartella dove salvare i risultati
    cartella_output = cerca_cartella_nel_progetto(BASE_DIR, "presenza_consecutiva_multirun")

    # Definisco il percorso per il nuovo file
    percorso_finale = cartella_output / "candidati_frame.csv"

    # Salvo il dataframe finale in formato csv escludendo l'indice
    df_risultati.to_csv(percorso_finale, index=False)

    print(f"\nOperazione completata! Estratti {len(df_risultati)} record.")
    print(f"File salvato correttamente in: {percorso_finale}")
else:
    print("\nNessun dato utile trovato nei file Parquet analizzati.")