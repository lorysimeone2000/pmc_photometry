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


def leggi_header_da_csv(filename):
    """Legge l'header FITS salvato nelle prime righe del file CSV."""
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


# --- PARAMETRI CONFIGURAZIONE ---

run = 1
# Indice del file nella lista da usare come riferimento
INDICE_IMMAGINE_RIFERIMENTO = 35

# Percorsi
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

# Verifica esistenza cartella
if not os.path.exists(cartella_csv):
    print(f"Errore: La cartella {cartella_csv} non esiste.")
    exit()

# Lista file ordinata
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

if not lista_percorsi_csv:
    print("Nessun file CSV trovato.")
    exit()

# --- FASE 1: PREPARAZIONE TABELLA DI RIFERIMENTO (INCLUSI PALLOCCHI) ---

print(f"--- FASE 1: Caricamento riferimento (inclusi match 'NO') dal file #{INDICE_IMMAGINE_RIFERIMENTO} ---")

if INDICE_IMMAGINE_RIFERIMENTO >= len(lista_percorsi_csv):
    INDICE_IMMAGINE_RIFERIMENTO = 0
    print("Indice riferimento fuori range, uso il primo file.")

path_ref = lista_percorsi_csv[INDICE_IMMAGINE_RIFERIMENTO]
df_ref = pd.read_csv(path_ref, comment='#')

# --- MODIFICA CHIAVE: Gestione dei match 'NO' ---
# 1. Identifichiamo le righe senza match
mask_no = df_ref['Corrispondenza'] == 'NO'

# 2. Assegniamo un ID UNIVOCO FITTIZIO a queste righe
# (Altrimenti il groupby('ID') successivo le fonderebbe tutte insieme!)
# Usiamo: "NOMATCH_label_{numero_label}"
if mask_no.any():
    print(f"Trovate {mask_no.sum()} sorgenti senza match nel frame di riferimento.")
    # Sostituiamo l'ID (spesso NaN o -999) con una stringa univoca
    df_ref.loc[mask_no, 'ID'] = 'NOMATCH_lbl_' + df_ref.loc[mask_no, 'label'].astype(str)

    # 3. Impostiamo a NaN le colonne di catalogo per sicurezza
    cols_catalog = ['Mag', 'RAJ2000', 'DEJ2000', 'Catalogo']
    for col in cols_catalog:
        if col in df_ref.columns:
            # Per le colonne numeriche usiamo np.nan
            if pd.api.types.is_numeric_dtype(df_ref[col]):
                df_ref.loc[mask_no, col] = np.nan
            else:
                # Per colonne stringa (es. Catalogo) usiamo None o stringa vuota
                df_ref.loc[mask_no, col] = None

# Ora df_ref contiene TUTTO: Match SI (con ID catalogo) e Match NO (con ID fittizio e Mag=NaN)

# Raggruppamento (uguale a prima, ma ora include tutti)
df_raggruppato = df_ref.groupby('label').agg(
    Corrispondenza=('Corrispondenza', 'first'),
    ID=('ID', 'first'),
    Mag_Integrata=('Mag', somma_magnitudini),
    Mag_Brightest=('Mag', 'min'),
    kron_flux=('kron_flux', 'first'),
    area=('area', 'first'),
    max_value=('max_value', 'first'),
    num_stelle=('Corrispondenza', 'count')
).reset_index()

tbl_raggruppato = Table.from_pandas(df_raggruppato)
print(f"Numero totale sorgenti tracciate: {len(tbl_raggruppato)}")

# --- RILEVAMENTO FLUSSI ---
df_header = pd.read_csv(lista_percorsi_csv[0], comment='#', nrows=0)
colonne = df_header.columns.tolist()
idx_start = colonne.index('saturazione') + 1
idx_end = colonne.index('RA_centroid')
tipi_flusso = colonne[idx_start:idx_end]
print(f"Tipi di flusso: {tipi_flusso}")

# --- FASE 2: RACCOLTA DATI OTTIMIZZATA ---

print("Lettura di tutti i file CSV in corso...")

lista_frames = []
colonne_da_leggere = ['ID'] + tipi_flusso

# Set di ID da cercare
ids_interessanti = set(tbl_raggruppato['ID'])

for percorso_csv in tqdm(lista_percorsi_csv, desc="Caricamento files"):
    try:
        # Leggiamo solo le colonne necessarie
        df_temp = pd.read_csv(percorso_csv, comment='#', usecols=lambda x: x in colonne_da_leggere)

        # Filtriamo per ID.
        # NOTA: Gli ID 'NOMATCH_lbl_...' non verranno trovati negli altri file
        # (perché l'ID non è persistente senza catalogo).
        # Questo è corretto: avranno count=1 (solo frame rif).
        df_temp = df_temp[df_temp['ID'].isin(ids_interessanti)]

        lista_frames.append(df_temp)
    except Exception as e:
        print(f"Errore lettura file {percorso_csv}: {e}")

# Unione
print("Concatenazione dei dati...")
df_totale = pd.concat(lista_frames, ignore_index=True)

# --- FASE 3: CALCOLO STATISTICHE ---

print("Calcolo statistiche raggruppate...")

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

stats = stats.rename(columns=rename_map)

# --- FASE 4: MERGE E OUTPUT ---

df_raggruppato_finale = tbl_raggruppato.to_pandas()
df_finale = pd.merge(df_raggruppato_finale, stats, on='ID', how='left')

# Filtro Count: Qui devi decidere.
# Se vuoi mantenere i "Pallocchi NO match", essi avranno count=1 (o pochi se per caso l'ID collide).
# Se filtri count >= 25, i pallocchi spariranno di nuovo.
# PER ORA: Rimuovo il filtro count per mostrarti tutto, oppure metto un filtro lasco.
flusso_principale = 'flusso_fisso_max_run'
colonna_count = f'count_{flusso_principale}'

# Manteniamo tutto per ora, anche chi appare 1 volta sola (i NO match)
# Se vuoi filtrare solo i matchati stabili, scommenta la riga sotto:
# df_finale = df_finale[df_finale[colonna_count] >= 25]

tbl_finale = Table.from_pandas(df_finale)

nome_file_csv = f"risultati_analisi_run_{run}.csv"
percorso_output = os.path.join(os.getcwd(), nome_file_csv)
tbl_finale.write(percorso_output, format='ascii.csv', overwrite=True)

print(f"✅ Finito! File salvato in: {percorso_output}")

# Plot (opzionale, sulle stelle che hanno dati)
df_plot = df_finale.dropna(subset=[f'media_{flusso_principale}', f'std_{flusso_principale}'])
if not df_plot.empty:
    x_sorted = df_plot[f'media_{flusso_principale}'].values
    y_sorted = df_plot[f'std_{flusso_principale}'].values
    sort_idx = np.argsort(x_sorted)

    plt.figure(figsize=(10, 6))
    plt.plot(x_sorted[sort_idx], y_sorted[sort_idx], marker='o', linestyle='', markersize=2, alpha=0.5)
    plt.title(f'Errore vs {flusso_principale} (Inclusi match NO)')
    plt.xlabel(f'Media {flusso_principale} (Log)')
    plt.ylabel('Deviazione standard')
    plt.xscale('log')
    plt.yscale('log')  # Utile per vedere i pallocchi deboli
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()