import os
import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import argparse


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # creo il mio parser per catturare la directory passata da terminale
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, help="Il percorso della mia cartella base")
    args, _ = parser.parse_known_args()

    # se ho passato un percorso specifico da terminale, do la priorità a quello
    cartella_effettiva = Path(args.base_dir).resolve() if args.base_dir else Path(base_dir)

    files_trovati = list(cartella_effettiva.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # creo il mio parser per catturare la directory passata da terminale
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, help="Il percorso della mia cartella base")
    args, _ = parser.parse_known_args()

    # se ho passato un percorso specifico da terminale, do la priorità a quello
    cartella_effettiva = Path(args.base_dir).resolve() if args.base_dir else Path(base_dir)

    cartelle_trovate = [p for p in cartella_effettiva.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


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


def leggi_header_da_csv(filename):
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            else:
                break
    return header_dict


def leggi_file_parametri(percorso):
    # creo il mio parser per catturare il percorso passato da terminale
    parser = argparse.ArgumentParser()
    parser.add_argument('--percorso_parametri', type=str, help="Il percorso del mio file parametri")
    args, _ = parser.parse_known_args()

    # se ho passato un percorso specifico da terminale, do la priorità a quello
    percorso_effettivo = args.percorso_parametri if args.percorso_parametri else percorso

    parametri = {}
    if not os.path.exists(percorso_effettivo): return {}
    with open(percorso_effettivo, 'r') as file:
        next(file, None)
        for riga in file:
            riga = riga.split('#')[0].strip()
            if riga:
                parts = riga.split()
                if len(parts) >= 2:
                    try:
                        valore = float(parts[1]) if '.' in parts[1] else int(parts[1])
                        parametri[parts[0]] = valore
                    except ValueError:
                        pass
    return parametri


def salva_csv_con_header_fits(dataframe, header_fits, filename, nome_file_fits, parametri_seg=None):
    nome_solo = os.path.basename(str(nome_file_fits))
    with open(filename, 'w') as f:
        f.write("# Header FITS:\n")
        for key, value in header_fits.items():
            clean_val = str(value).replace('\n', ' ')
            f.write(f"# {key}: {clean_val}\n")
        f.write(f"# NOME_FILE_FITS: {nome_solo}\n")
        f.write("#\n# PARAMETRI SEGMENTAZIONE:\n")
        if parametri_seg:
            for key, value in parametri_seg.items():
                f.write(f"# {key}: {value}\n")
        f.write("#\n")
        dataframe.to_csv(f, index=False)


def salva_tabella_parquet(dataframe, header_fits, filename, nome_file_fits, parametri_seg=None, num_run=None,
                          num_immagine=None):
    # 1. Preparo il mio dizionario dei metadati
    meta_dict = {
        "NOME_FILE_FITS": os.path.basename(str(nome_file_fits)),
        "FITS_HEADER": {str(k): str(v).replace('\n', ' ') for k, v in header_fits.items()},
        "NOTE": "Numero di falsi positivi esclusi sicuramente: 0"
    }

    # inserisco i miei numeri di run e immagine nei metadati se forniti
    if num_run is not None:
        meta_dict["NUMERO_RUN"] = num_run
    if num_immagine is not None:
        meta_dict["NUMERO_IMMAGINE"] = num_immagine

    if parametri_seg:
        meta_dict["PARAMETRI_SEGMENTAZIONE"] = {str(k): v for k, v in parametri_seg.items()}

    # 2. Converto il DataFrame in una Tabella PyArrow
    table = pa.Table.from_pandas(dataframe)

    # 3. Serializzo il dizionario in JSON e lo aggiungo ai miei metadati della tabella
    custom_metadata = table.schema.metadata or {}
    custom_metadata.update({b"astro_metadata": json.dumps(meta_dict).encode("utf-8")})
    table = table.replace_schema_metadata(custom_metadata)

    # 4. Scrivo su disco
    pq.write_table(table, filename)


def leggi_tabella_parquet(filename):
    table = pq.read_table(filename)
    df = table.to_pandas()
    meta_raw = table.schema.metadata.get(b"astro_metadata")
    metadata = json.loads(meta_raw.decode("utf-8")) if meta_raw else {}
    return df, metadata


def arrotonda_dinamico(valore, incertezza):
    if pd.isna(valore) or pd.isna(incertezza) or incertezza <= 0:
        return valore
    ordine_grandezza = int(np.floor(np.log10(incertezza)))
    decimali_da_tenere = max(0, -(ordine_grandezza - 2))
    return round(valore, decimali_da_tenere)

def cerca_cartella_intero_pc(nome_cartella=""):
    # creo il mio parser per catturare il nome della cartella da cercare nell'intero pc
    parser = argparse.ArgumentParser()
    parser.add_argument('--nome_cartella_globale', type=str,
                        help="Il nome della mia cartella da cercare nell'intero pc")
    args, _ = parser.parse_known_args()

    # se passo il nome da terminale do la priorità a quello
    cartella_da_cercare = args.nome_cartella_globale if args.nome_cartella_globale else nome_cartella

    if not cartella_da_cercare:
        return None

    # individuo i miei percorsi radice in base al sistema operativo
    if os.name == 'nt':
        import string
        radici = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    else:
        radici = ['/']

    # avvio la mia ricerca globale esplorando l'intero file system
    for radice in radici:
        for root, dirs, files in os.walk(radice):
            if cartella_da_cercare in dirs:
                # restituisco la stringa del mio percorso esatto non appena trovo la cartella
                return os.path.join(root, cartella_da_cercare)

    return None