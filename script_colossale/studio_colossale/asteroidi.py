import pandas as pd
import argparse
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np
import os
import sys
import re
from tqdm import tqdm
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
import warnings

# gestisco i warning ignorandoli per mantenere pulito il mio output
warnings.filterwarnings('ignore')


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E FUNZIONI BASE
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory del mio script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")
PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

# importo la mia utilità per leggere l'header
from funzioni.utilita_parquet import leggi_header_da_parquet, cerca_cartella_nel_progetto

# definisco la mia tolleranza massima per considerare due oggetti come "lo stesso che si muove"
MAX_SEPARAZIONE_ARCSEC = 2.0
# definisco il gap massimo di frame tollerabile tra la scomparsa di A e l'apparizione di B
MAX_GAP_FRAME = 2

# =============================================================================
# 1. LETTURA, ORDINAMENTO CRONOLOGICO E RAGGRUPPAMENTO
# =============================================================================

cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito"
file_parquet = list(cartella_tabelle.rglob("*run_*_run_*_immagine_*.parquet"))

if not file_parquet:
    print("Nessun file trovato con il pattern specificato.")
    sys.exit()

lista_dati_file = []

for file_p in tqdm(file_parquet, desc="Estrazione TSTART dai file"):
    header_dict = leggi_header_da_parquet(file_p)
    tstart = header_dict.get('TSTART')
    if tstart is None:
        tstart = float('inf')
    else:
        try:
            tstart = float(tstart)
        except ValueError:
            tstart = float('inf')

    # salvo esclusivamente il percorso del file
    lista_dati_file.append({
        'tstart': tstart,
        'file_path': file_p
    })

# ordino cronologicamente
lista_dati_file.sort(key=lambda x: x['tstart'])

lista_df_ordinati = []
print("Caricamento ed estrazione dati temporali in corso...")

for indice_cronologico, dato in enumerate(tqdm(lista_dati_file, desc="Estrazione dati dai Parquet")):
    # leggo solo le label dei falsi positivi per risparmiare ram
    tabella_filtrata = pq.read_table(
        dato['file_path'],
        columns=['label'],
        filters=[('Corrispondenza', '=', False)]
    )

    df_corrente = tabella_filtrata.to_pandas()
    df_corrente['indice_cronologico'] = indice_cronologico
    lista_df_ordinati.append(df_corrente)

df_falsi = pd.concat(lista_df_ordinati, ignore_index=True)

# raggruppo ogni label per trovare il primo e l'ultimo momento in cui l'ho visto
print("Calcolo il ciclo di vita (min/max frame) per ogni oggetto...")
df_stats = df_falsi.groupby('label').agg(
    min_idx=('indice_cronologico', 'min'),
    max_idx=('indice_cronologico', 'max'),
    occorrenze=('indice_cronologico', 'count')
).reset_index()


# =============================================================================
# 2. ESTRAZIONE COORDINATE E RICERCA SPAZIALE (CROSS-MATCH)
# =============================================================================

def estrai_ra_dec(label):
    pattern = r'RA_([\d\.]+)DEC([\-]?\d+\.?\d*)'
    match = re.match(pattern, label)
    if match:
        return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan


print("Estraggo le coordinate fisiche dai label...")
# applico la mia funzione di estrazione
coordinate = df_stats['label'].apply(estrai_ra_dec)
df_stats['RA'] = [c[0] for c in coordinate]
df_stats['DEC'] = [c[1] for c in coordinate]

# rimuovo eventuali label che non sono riuscito a formattare correttamente
df_stats = df_stats.dropna(subset=['RA', 'DEC']).reset_index(drop=True)

print("Inizio la ricerca spaziale degli oggetti vicini (< 2 arcsec)...")
# costruisco il mio catalogo spaziale
coords = SkyCoord(ra=df_stats['RA'].values * u.deg, dec=df_stats['DEC'].values * u.deg)

# identifico tutte le coppie di oggetti che distano meno della mia soglia
idx1, idx2, sep2d, _ = search_around_sky(coords, coords, MAX_SEPARAZIONE_ARCSEC * u.arcsec)

# =============================================================================
# 3. VERIFICA DEI "PASSAGGI DI TESTIMONE" CRONOLOGICI
# =============================================================================

print("Analizzo le relazioni temporali per scovare gli asteroidi in movimento...")
passaggi_trovati = []

# analizzo ogni coppia vicina
for i, j, sep in zip(idx1, idx2, sep2d):
    # ignoro il match dell'oggetto con se stesso
    if i == j:
        continue

    # calcolo quanti frame passano tra l'ultima apparizione di A (i) e la prima di B (j)
    gap_cronologico = df_stats.at[j, 'min_idx'] - df_stats.at[i, 'max_idx']

    # se l'oggetto B appare esattamente dopo la scomparsa di A (o con un gap piccolo)
    if 0 <= gap_cronologico <= MAX_GAP_FRAME:
        # ho trovato un probabile oggetto in movimento
        passaggi_trovati.append({
            'Label_A (Scomparso)': df_stats.at[i, 'label'],
            'Label_B (Apparso)': df_stats.at[j, 'label'],
            'Separazione_Arcsec': round(sep.arcsec, 3),
            'Frame_Scomparsa_A': df_stats.at[i, 'max_idx'],
            'Frame_Apparizione_B': df_stats.at[j, 'min_idx'],
            'Gap_Frame': gap_cronologico,
            'Occorrenze_A': df_stats.at[i, 'occorrenze'],
            'Occorrenze_B': df_stats.at[j, 'occorrenze']
        })

df_risultati = pd.DataFrame(passaggi_trovati)

# =============================================================================
# 4. SALVATAGGIO RISULTATI
# =============================================================================

cartella_output = cerca_cartella_nel_progetto(BASE_DIR, "studio_colossale")
if cartella_output is None:
    cartella_output = BASE_DIR / "pmc_photometry" / "script_colossale" / "studio_colossale"
    cartella_output.mkdir(parents=True, exist_ok=True)

percorso_salvataggio = Path(cartella_output) / "candidati_asteroidi_passaggio_testimone.csv"

if not df_risultati.empty:
    # rimuovo eventuali duplicati speculari (anche se l'ordinamento temporale A->B lo previene naturalmente)
    df_risultati = df_risultati.drop_duplicates()
    df_risultati.to_csv(percorso_salvataggio, index=False)
    print(f"\nCOMPLETATO! Trovati {len(df_risultati)} potenziali salti. Salvato in:\n{percorso_salvataggio}")
else:
    print("\nNessun passaggio di testimone trovato con questi parametri.")

    # salvo un csv vuoto con le colonne corrette per coerenza
    df_vuoto = pd.DataFrame(columns=['Label_A (Scomparso)', 'Label_B (Apparso)', 'Separazione_Arcsec',
                                     'Frame_Scomparsa_A', 'Frame_Apparizione_B', 'Gap_Frame',
                                     'Occorrenze_A', 'Occorrenze_B'])
    df_vuoto.to_csv(percorso_salvataggio, index=False)