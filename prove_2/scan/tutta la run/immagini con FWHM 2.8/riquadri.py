import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import json


def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory del mio script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")

# definisco il percorso esatto in cui cercare i miei file parquet
cartella_tabelle = BASE_DIR / "tabelle_alleggerite"

# cerco tutti i file che terminano con il suffisso degli oggetti non catalogati
file_parquet = list(cartella_tabelle.rglob("*_oggetti_non_catalogati.parquet"))

lista_df = []
i = 0
# leggo i file uno per uno
for file in file_parquet:

    i += 1

    parquet_file = pq.ParquetFile(file)

    # accedo al dizionario dei metadati a livello di file
    metadati_grezzi = parquet_file.metadata.metadata

    # verifico se la mia chiave personalizzata esiste all'interno dei metadati
    if metadati_grezzi and b"astro_metadata" in metadati_grezzi:

        # decodifico i byte e parso la stringa JSON per ricreare il mio dizionario Python
        dizionario_meta = json.loads(metadati_grezzi[b"astro_metadata"].decode("utf-8"))

        # estraggo il mio header FITS dal dizionario
        header_fits = dizionario_meta.get("FITS_HEADER", {})

        print("--- HEADER FITS ESTRATTO ---")
        for chiave, valore in header_fits.items():
            print(f"{chiave}: {valore}")

        # estraggo e stampo tutte le altre chiavi escludendo l'header appena stampato
        print("--- ALTRE CHIAVI DEI METADATI ---")
        for chiave, valore in dizionario_meta.items():
            if chiave != "FITS_HEADER":
                print(f"{chiave}: {valore}")

    else:
        print("Nessun 'astro_metadata' salvato in questo file.")

    print("--------------------------------------\n")

    # leggo la mia tabella parquet come dataframe e la stampo
    df = pd.read_parquet(file)
    print(df)