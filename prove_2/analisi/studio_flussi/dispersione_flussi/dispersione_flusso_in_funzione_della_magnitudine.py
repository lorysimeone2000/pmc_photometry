import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from astropy.table import Table
import warnings
from astropy.wcs import FITSFixedWarning

# Sopprime il warning FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)


# --- FUNZIONI DI UTILITÀ ---

def converti_valore(valore):
    """Converte una stringa nel tipo di dato appropriato."""
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
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False
    return valore


def somma_magnitudini(series_mags):
    """Integra le magnitudini sommando i flussi."""
    mags = series_mags.dropna()
    if len(mags) == 0: return np.nan
    flussi = 10 ** (-0.4 * np.array(mags))
    flusso_totale = np.sum(flussi)
    if flusso_totale <= 0: return np.nan
    return -2.5 * np.log10(flusso_totale)


# --- PARAMETRI CONFIGURAZIONE ---

run = 1
INDICE_IMMAGINE_RIFERIMENTO = 35

# Percorsi
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

if not os.path.exists(cartella_csv):
    print(f"Errore: La cartella {cartella_csv} non esiste.")
    exit()

# Lista file ordinata
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

if not lista_percorsi_csv:
    print("Nessun file CSV trovato.")
    exit()

if INDICE_IMMAGINE_RIFERIMENTO >= len(lista_percorsi_csv):
    INDICE_IMMAGINE_RIFERIMENTO = 0
    print("Indice riferimento fuori range, uso il primo file.")

# --- RILEVAMENTO NOMI COLONNE FLUSSO ---
# Leggiamo solo l'header del primo file per capire quali colonne sono flussi
df_header = pd.read_csv(lista_percorsi_csv[0], comment='#', nrows=0)
colonne = df_header.columns.tolist()
try:
    idx_start = colonne.index('saturazione') + 1
    idx_end = colonne.index('RA_centroid')
    tipi_flusso = colonne[idx_start:idx_end]
except ValueError:
    # Fallback se le colonne non sono esattamente quelle previste, cerchiamo parole chiave
    tipi_flusso = [c for c in colonne if 'flux' in c or 'flusso' in c or 'somma' in c]

print(f"Tipi di flusso rilevati: {tipi_flusso}")

# --- FASE 1: CARICAMENTO RIFERIMENTO E SEPARAZIONE ---

print(
    f"--- Caricamento riferimento dal file #{INDICE_IMMAGINE_RIFERIMENTO} ({os.path.basename(lista_percorsi_csv[INDICE_IMMAGINE_RIFERIMENTO])}) ---")

path_ref = lista_percorsi_csv[INDICE_IMMAGINE_RIFERIMENTO]
df_ref = pd.read_csv(path_ref, comment='#')

# Aggiungiamo colonne Mag aggregate se mancano (per coerenza col codice precedente)
# Qui facciamo un'aggregazione locale rapida per label se ci fossero duplicati, ma su 1 file dovrebbero essere unici per label
df_ref_agg = df_ref.groupby('label').agg(
    Corrispondenza=('Corrispondenza', 'first'),
    saturazione=('saturazione', 'first'),
    ID=('ID', 'first'),
    Mag_Integrata=('Mag', somma_magnitudini),
    Mag_Brightest=('Mag', 'min'),
    # Manteniamo tutti i flussi originali
    **{col: (col, 'first') for col in tipi_flusso}
).reset_index()

# SEPARAZIONE: MATCH vs NO MATCH
mask_no = df_ref_agg['Corrispondenza'] == 'NO'

# 1. Gruppo MATCH (Corrispondenza != NO) -> Calcoleremo le statistiche su tutti i file
df_targets_match = df_ref_agg[~mask_no].copy()

# 2. Gruppo NO MATCH -> Prenderemo i valori fissi, ID NaN, Mag NaN
df_targets_nomatch = df_ref_agg[mask_no].copy()

print(f"Sorgenti con MATCH (da analizzare su tutti i file): {len(df_targets_match)}")
print(f"Sorgenti NO MATCH (da fissare con valori frame 35): {len(df_targets_nomatch)}")

# --- FASE 2: ANALISI SORGENTI MATCHATE (Su tutti i file) ---

ids_interessanti = set(df_targets_match['ID'].dropna().unique())
df_stats_match = pd.DataFrame()

if len(ids_interessanti) > 0:
    print("Lettura file CSV per le sorgenti matchate...")
    lista_frames = []
    colonne_da_leggere = ['ID'] + tipi_flusso

    for percorso_csv in tqdm(lista_percorsi_csv, desc="Analisi temporale"):
        try:
            df_temp = pd.read_csv(percorso_csv, comment='#', usecols=lambda x: x in colonne_da_leggere)
            # Teniamo solo righe con ID presenti nel gruppo MATCH
            df_temp = df_temp[df_temp['ID'].isin(ids_interessanti)]
            lista_frames.append(df_temp)
        except Exception as e:
            pass

    # Unione e Calcolo Statistiche MATCH
    if lista_frames:
        df_totale = pd.concat(lista_frames, ignore_index=True)

        agg_rules = {}
        for flusso in tipi_flusso:
            agg_rules[flusso] = ['count', 'mean', 'std']

        stats = df_totale.groupby('ID').agg(agg_rules)
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]

        rename_map = {}
        for flusso in tipi_flusso:
            rename_map[f'{flusso}_mean'] = f'media_{flusso}'
            rename_map[f'{flusso}_std'] = f'std_{flusso}'
            rename_map[f'{flusso}_count'] = f'count_{flusso}'

        df_stats_match = stats.rename(columns=rename_map).reset_index()

        # Uniamo le info statiche (Mag, etc) dal riferimento ai dati calcolati
        # Usiamo df_targets_match per recuperare Mag_Brightest etc.
        df_final_match = pd.merge(df_targets_match[['ID', 'Mag_Integrata', 'Mag_Brightest', 'label', 'Corrispondenza', 'saturazione']],
                                  df_stats_match, on='ID', how='inner')
    else:
        df_final_match = pd.DataFrame()
else:
    df_final_match = pd.DataFrame()

# --- FASE 3: PREPARAZIONE SORGENTI NO MATCH (Solo frame riferimento) ---

print("Elaborazione sorgenti non matchate...")

if len(df_targets_nomatch) > 0:
    # Impostiamo ID e Magnitudini a NaN come richiesto
    df_targets_nomatch['ID'] = np.nan
    df_targets_nomatch['Mag_Integrata'] = np.nan
    df_targets_nomatch['Mag_Brightest'] = np.nan

    # Per ogni tipo di flusso, mappiamo il valore singolo come "media" e NaN come "std"
    for flusso in tipi_flusso:
        # La colonna 'media_X' prende il valore di 'X' (che viene dal frame 35)
        df_targets_nomatch[f'media_{flusso}'] = df_targets_nomatch[flusso]
        # La colonna 'std_X' diventa NaN
        df_targets_nomatch[f'std_{flusso}'] = np.nan
        # La colonna 'count_X' diventa 1 (visto una volta)
        df_targets_nomatch[f'count_{flusso}'] = 1

        # Rimuoviamo la colonna originale 'flusso' per non avere duplicati o confusione dopo
        df_targets_nomatch.drop(columns=[flusso], inplace=True)

    # Colonne finali per NO MATCH
    df_final_nomatch = df_targets_nomatch
else:
    df_final_nomatch = pd.DataFrame()

# --- FASE 4: UNIONE E SALVATAGGIO ---

print("Unione dei risultati...")

# Concateniamo i due dataframe.
# Pandas allineerà le colonne (es. media_flux esiste in entrambi).
df_completo = pd.concat([df_final_match, df_final_nomatch], ignore_index=True)

# Ordinamento opzionale (es. per label)
if 'label' in df_completo.columns:
    df_completo = df_completo.sort_values('label')

tbl_finale = Table.from_pandas(df_completo)

nome_file_csv = f"risultati_analisi_run_{run}.csv"
percorso_output = os.path.join(os.getcwd(), nome_file_csv)

# Scrittura
tbl_finale.write(percorso_output, format='ascii.csv', overwrite=True)

print(f"✅ Finito! File salvato in: {percorso_output}")
print(f"Totale righe: {len(df_completo)} (Match: {len(df_final_match)}, No Match: {len(df_final_nomatch)})")

# --- DEBUG PLOT (Opzionale) ---
# Mostriamo che i "No Match" ci sono (saranno quelli con std=NaN, quindi non appariranno nel plot standard x vs std,
# ma possiamo verificare se ci sono i dati)
try:
    col_test = f"media_{tipi_flusso[0]}"
    nan_count = df_completo[col_test].isna().sum()
    print(f"Controllo: Ci sono {nan_count} righe con media flusso NaN (dovrebbe essere 0).")

    # Esempio primi 3 NO MATCH
    if len(df_final_nomatch) > 0:
        print("\nEsempio righe NO MATCH elaborate:")
        cols_show = ['label', 'ID', 'Mag_Brightest', f'media_{tipi_flusso[0]}', f'std_{tipi_flusso[0]}']
        print(df_final_nomatch[cols_show].head(3).to_string())
except Exception as e:
    print(f"Skip debug print: {e}")