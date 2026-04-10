import pandas as pd
import pyarrow.parquet as pq
import os
import sys
from pathlib import Path
from tqdm import tqdm
import warnings

# gestisco i warning ignorandoli per mantenere pulito il mio output
warnings.filterwarnings('ignore')


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E FUNZIONI DI SUPPORTO
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"WARNING: '{nome_target}' folder not found in the tree. Using script directory.")
    return path_corrente.parent


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
    if valore.upper() in ['T', 'TRUE']: return True
    if valore.upper() in ['F', 'FALSE']: return False
    return valore


def leggi_header_da_parquet(filename):
    # inizializzo il mio dizionario vuoto per ospitare i metadati estratti
    header_dict = {}

    try:
        # leggo esclusivamente lo schema del file Parquet per accedere ai metadati
        schema = pq.read_schema(filename)
        metadati = schema.metadata

        # verifico se i metadati esistono e se contengono la mia chiave personalizzata
        if metadati and b'metadati_intestazione_csv' in metadati:
            # decodifico la stringa da byte a testo normale usando la codifica utf-8
            metadati_testo = metadati[b'metadati_intestazione_csv'].decode('utf-8')

            # analizzo ogni riga della mia stringa di testo
            for riga in metadati_testo.split('\n'):
                if riga.startswith('#'):
                    # pulisco la riga rimuovendo il cancelletto e gli spazi
                    riga_pulita = riga.strip()[1:].strip()

                    # verifico che la riga contenga un separatore valido
                    if riga_pulita and ': ' in riga_pulita:
                        # separo la mia chiave dal valore
                        chiave, valore = riga_pulita.split(': ', 1)
                        # aggiungo l'elemento al mio dizionario usando la funzione di conversione
                        header_dict[chiave] = converti_valore(valore)

    except Exception:
        pass

    return header_dict


BASE_DIR = trova_cartella_base("Lorenzo")

# =============================================================================
# 1. LETTURA DEL FILE ORIGINALE E PREPARAZIONE TARGET
# =============================================================================

# individuo il file CSV originale
percorso_csv = None
for file in BASE_DIR.rglob("oggetti_con_presenza_consecutiva.csv"):
    percorso_csv = file
    break

if percorso_csv is None:
    print("ERRORE: Non ho trovato il file 'oggetti_con_presenza_consecutiva.csv'.")
    sys.exit()

# carico i dati originali
df_originale = pd.read_csv(percorso_csv)
labels_da_cercare = set(df_originale['label'].values)

print(f"Caricati {len(labels_da_cercare)} oggetti dal file CSV originale.")

# inizializzo un dizionario con set vuoti per tracciare i run_id unici per ogni label
run_per_label = {label: set() for label in labels_da_cercare}

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
        # leggo esclusivamente la colonna 'label'
        tabella_p = pq.read_table(file_p, columns=['label'])
        labels_nel_file = set(tabella_p.column('label').to_pylist())

        # cerco le intersezioni tra gli oggetti nel file e i miei target
        trovati = labels_da_cercare.intersection(labels_nel_file)

        # se trovo almeno un target, leggo i metadati per estrarre la run
        if trovati:
            header = leggi_header_da_parquet(file_p)
            run_id = header.get('RUN_ID')

            # se ho trovato correttamente il RUN_ID, lo assegno ai set degli oggetti presenti
            if run_id:
                for l in trovati:
                    run_per_label[l].add(str(run_id))
    except Exception:
        continue

# =============================================================================
# 3. AGGIORNAMENTO DEL DATAFRAME E FILTRAGGIO
# =============================================================================

print("\nCalcolo del numero di run e filtraggio degli eventi a singola run...")

# creo la nuova colonna calcolando la lunghezza del set per ogni label
df_originale['numero_di_run'] = df_originale['label'].apply(lambda x: len(run_per_label.get(x, set())))

# filtro eliminando gli oggetti che compaiono in una sola run (o nessuna)
df_multirun = df_originale[df_originale['numero_di_run'] > 1].copy()

# =============================================================================
# 4. SALVATAGGIO DEI RISULTATI
# =============================================================================

cartella_output = percorso_csv.parent
nuovo_percorso = cartella_output / "oggetti_presenza_multirun.csv"

df_multirun.to_csv(nuovo_percorso, index=False)

print(f"\nOperazione completata con successo!")
print(f"Oggetti originali: {len(df_originale)}")
print(f"Oggetti mantenuti (multirun): {len(df_multirun)}")
print(f"File salvato in: {nuovo_percorso}")