import os
import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import argparse
import json
import sys


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


def converti_csv_in_parquet(percorso_csv):
    # genero in automatico il nome del file Parquet sostituendo l'estensione
    file_parquet = percorso_csv.replace('.csv', '.parquet')

    # estraggo i metadati aprendo il file in lettura e isolando le righe che iniziano con "#"
    metadati_estratti = []
    with open(percorso_csv, 'r') as f:
        for riga in f:
            if riga.startswith('#'):
                metadati_estratti.append(riga.strip())
            else:
                break

    # unisco tutte le righe dei metadati in un'unica stringa
    metadati_str = "\n".join(metadati_estratti)

    # carico i dati dal file CSV istruendo pandas di ignorare le righe commentate con "#"
    df = pd.read_csv(percorso_csv, comment='#')

    # converto il DataFrame in una tabella PyArrow senza includere l'indice
    tabella = pa.Table.from_pandas(df, preserve_index=False)

    # definisco i miei metadati personalizzati codificandoli in formato byte
    miei_metadati = {'metadati_intestazione_csv': metadati_str.encode('utf-8')}

    # recupero eventuali metadati preesistenti creati da pandas e li unisco ai miei
    metadati_esistenti = tabella.schema.metadata
    if metadati_esistenti:
        miei_metadati.update(metadati_esistenti)

    # applico i metadati aggiornati allo schema della tabella
    tabella_con_metadati = tabella.replace_schema_metadata(miei_metadati)

    # salvo i dati convertendoli nel formato Parquet
    pq.write_table(tabella_con_metadati, file_parquet)

    # elimino definitivamente il file CSV originale
    os.remove(percorso_csv)


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


import pandas as pd
import matplotlib
import argparse
import json
import pyarrow.parquet as pq
import shutil
from astropy.config import paths
import matplotlib.pyplot as plt
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import time
import os
import sys
import gc
from scipy.optimize import curve_fit
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
import warnings
from astropy.wcs import FITSFixedWarning
from photutils.datasets import make_100gaussians_image
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.utils.data import download_file
from astropy.table import Table, vstack
from photutils.detection import find_peaks
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.coordinates import search_around_sky
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from shapely.geometry import Point, Polygon
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import label
import re
from pathlib import Path
from astropy.time import Time

# gestisco i warning ignorandoli per mantenere pulito il mio output
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='.*deblending mode.*')


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
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

# importo i moduli per il salvataggio in parquet e la relativa utilità
from funzioni.utilita_parquet import *
from funzioni.astrometria_parquet import *

# =============================================================================
# 1. LETTURA, ORDINAMENTO CRONOLOGICO E UNIONE DEI FILE PARQUET
# =============================================================================

# definisco il percorso esatto in cui cercare i miei file parquet
cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito"

# cerco tutti i file che terminano con il suffisso specifico sostituendo i numeri con dei jolly
file_parquet = list(cartella_tabelle.rglob("*run_*_run_*_immagine_*.parquet"))

if not file_parquet:
    print("Nessun file trovato con il pattern specificato.")
    sys.exit()

lista_dati_file = []

# leggo i metadati di ciascun file usando la mia funzione per estrarre TSTART e preparare l'ordinamento
for file_p in tqdm(file_parquet, desc="Estrazione TSTART dai file"):
    header_dict = leggi_header_da_parquet(file_p)
    tstart = header_dict.get('TSTART')

    # se non riesco a trovare TSTART o non è un numero valido, assegno infinito per mandarlo in fondo alla lista
    if tstart is None:
        tstart = float('inf')
    else:
        try:
            tstart = float(tstart)
        except ValueError:
            tstart = float('inf')

    # salvo esclusivamente il percorso del file per elaborare i dati in seguito
    lista_dati_file.append({
        'tstart': tstart,
        'file_path': file_p
    })

# ordino la mia lista di dizionari in modo strettamente cronologico usando TSTART
lista_dati_file.sort(key=lambda x: x['tstart'])

lista_df_ordinati = []

print("Unione di tutti i dati in un singolo dataframe in corso...")

# assegno un indice cronologico sequenziale (0, 1, 2...) ed estraggo in maniera ottimizzata i miei dati
for indice_cronologico, dato in enumerate(tqdm(lista_dati_file, desc="Lettura e filtraggio Parquet")):
    # sfrutto il predicate pushdown e la column projection per caricare unicamente la 'label' per le righe dove 'Corrispondenza' è False
    tabella_filtrata = pq.read_table(
        dato['file_path'],
        columns=['label'],
        filters=[('Corrispondenza', '=', False)]
    )

    # converto in pandas esclusivamente i dati che ho prefiltrato
    df_corrente = tabella_filtrata.to_pandas()
    df_corrente['indice_cronologico'] = indice_cronologico
    # mantengo l'informazione del TSTART del file
    df_corrente['TSTART_file'] = dato['tstart']

    lista_df_ordinati.append(df_corrente)

# genero il mio dataframe complessivo che adesso contiene già solo gli oggetti non catalogati
df_falsi = pd.concat(lista_df_ordinati, ignore_index=True)

# =============================================================================
# 2. ISOLAMENTO OGGETTI NON CATALOGATI E ANALISI DELLA CONSECUTIVITÀ
# =============================================================================

print("Analisi consecutività per gli oggetti non catalogati in corso...")

# raggruppo per label ed estraggo il conteggio delle immagini uniche in cui compare,
# assieme al primo e all'ultimo indice cronologico in cui è stato visto
df_non_cat = df_falsi.groupby('label').agg(
    occorrenze=('indice_cronologico', 'nunique'),
    min_indice=('indice_cronologico', 'min'),
    max_indice=('indice_cronologico', 'max')
).reset_index()

# calcolo quanti indici consecutivi mi aspetto ci siano tra la prima e l'ultima apparizione
df_non_cat['consecutivi_attesi'] = df_non_cat['max_indice'] - df_non_cat['min_indice'] + 1

# filtro il dataframe secondo le condizioni richieste: occorrenze > 10 E occorrenze uguali ai consecutivi attesi (nessun buco cronologico)
maschera_consecutivi = (df_non_cat['occorrenze'] > 10) & (df_non_cat['occorrenze'] == df_non_cat['consecutivi_attesi'])
df_non_cat_aggiornato = df_non_cat[maschera_consecutivi].copy()

# rimuovo le colonne di calcolo temporanee per ripulire il dataframe in output
df_non_cat_aggiornato = df_non_cat_aggiornato.drop(columns=['min_indice', 'max_indice', 'consecutivi_attesi'])

# cerco la cartella di output "studio_colossale"
cartella_output = cerca_cartella_nel_progetto(BASE_DIR, "studio_colossale")

# se la cartella non esiste, la creo partendo dalla directory del mio script principale
if cartella_output is None:
    cartella_output = BASE_DIR / "pmc_photometry" / "script_colossale" / "studio_colossale"
    cartella_output.mkdir(parents=True, exist_ok=True)

percorso_salvataggio = Path(cartella_output) / "oggetti_con_presenza_consecutiva.csv"

# salvo il dataframe aggiornato
df_non_cat_aggiornato.to_csv(percorso_salvataggio, index=False)

print(f"Salvataggio completato! Trovati {len(df_non_cat_aggiornato)} oggetti. File salvato in: {percorso_salvataggio}")