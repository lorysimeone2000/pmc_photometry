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
percorso_csv = cerca_file_nel_progetto(BASE_DIR, "oggetti_CATALOGATI_con_presenza_consecutiva.csv")

if percorso_csv is None:
    print("ERRORE: Non ho trovato il file 'oggetti_CATALOGATI_con_presenza_consecutiva.csv'.")
    sys.exit()

# carico i dati originali
df_originale = pd.read_csv(percorso_csv)
labels_da_cercare = set(df_originale['label'].values)

print(f"Caricati {len(labels_da_cercare)} oggetti dal file CSV originale.")

# inizializzo un dizionario con set vuoti per tracciare i run_id unici per ogni label
run_per_label = {label: set() for label in labels_da_cercare}

# inizializzo un dizionario per tracciare la magnitudine massima per ogni label
mag_max_per_label = {label: None for label in labels_da_cercare}

# =============================================================================
# 2. RICERCA DELLE RUN NEI FILE PARQUET
# =============================================================================

cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito"
file_parquet = list(cartella_tabelle.rglob("*run_*_run_*_immagine_*.parquet"))

if not file_parquet:
    print("ERRORE: Nessun file Parquet trovato nella cartella specificata.")
    sys.exit()

print("Inizio la scansione per identificare le run in cui compare ciascun oggetto...")

# ciclo su ogni file Parquet
for file_p in tqdm(file_parquet, desc="Analisi file"):
    try:
        # leggo le colonne 'label' e 'Mag_estratta'
        tabella_p = pq.read_table(file_p, columns=['label', 'Mag_estratta'])

        # converto in dataframe per agevolare il calcolo del massimo
        df_p = tabella_p.to_pandas()

        # estraggo le etichette uniche nel file
        labels_nel_file = set(df_p['label'].unique())

        # cerco le intersezioni tra gli oggetti nel file e i miei target
        trovati = labels_da_cercare.intersection(labels_nel_file)

        # se trovo almeno un target, leggo i metadati ed estraggo le magnitudini
        if trovati:
            header = leggi_header_da_parquet(file_p)
            run_id = header.get('RUN_ID')

            # filtro il dataframe solo per i label individuati
            df_trovati = df_p[df_p['label'].isin(trovati)]

            # raggruppo per label ed estraggo il valore minimo di Mag_estratta in questo file
            max_mag_file = df_trovati.groupby('label')['Mag_estratta'].min()

            for l in trovati:
                # ricavo il valore massimo per il label 'l' e aggiorno il dizionario principale
                valore_max = max_mag_file.get(l)
                if pd.notna(valore_max):
                    if mag_max_per_label[l] is None or valore_max > mag_max_per_label[l]:
                        mag_max_per_label[l] = valore_max

                # se ho trovato correttamente il RUN_ID, lo assegno ai set degli oggetti presenti
                if run_id:
                    run_per_label[l].add(str(run_id))
    except Exception:
        continue

# =============================================================================
# 3. AGGIORNAMENTO DEL DATAFRAME E FILTRAGGIO
# =============================================================================

print("\nCalcolo del numero di run e filtraggio degli eventi a singola run...")

# creo la nuova colonna calcolando la lunghezza del set per ogni label
df_originale['numero_di_run'] = df_originale['label'].apply(lambda x: len(run_per_label.get(x, set())))

# creo la nuova colonna per la magnitudine massima aggiungendo i valori estratti
df_originale['Mag_estratta_max'] = df_originale['label'].apply(lambda x: mag_max_per_label.get(x))

# filtro eliminando gli oggetti che compaiono in una sola run (o nessuna)
df_multirun = df_originale[df_originale['numero_di_run'] > 1].copy()

# =============================================================================
# 4. SALVATAGGIO DEI RISULTATI
# =============================================================================

cartella_output = cerca_cartella_nel_progetto(BASE_DIR, "sorgenti catalogate")
percorso = cartella_output / "oggetti_presenza_CATALOGATI_multirun.csv"

df_multirun.to_csv(percorso, index=False)

print(f"\nOperazione completata con successo!")
print(f"Oggetti originali: {len(df_originale)}")
print(f"Oggetti mantenuti (multirun): {len(df_multirun)}")
print(f"File salvato in: {percorso}")