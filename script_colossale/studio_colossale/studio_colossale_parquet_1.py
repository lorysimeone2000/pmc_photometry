import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import json
import numpy as np
import warnings
import re
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from tqdm import tqdm

from prove_2.analisi.andamento_anomalie.path_non_correlate_tutte import cerca_cartella_nel_progetto

# ignoro i warning per mantenere l'output pulito
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)



def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory del mio script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")

# definisco il percorso esatto in cui cercare i miei file parquet
cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito"

# cerco tutti i file che terminano con il suffisso degli oggetti non catalogati
file_parquet = list(cartella_tabelle.rglob("*_oggetti_non_catalogati.parquet"))

lista_df = []

# leggo i file uno per uno
for file in file_parquet:
    df_temp = pd.read_parquet(file)
    # salvo il nome del file come identificatore della run di provenienza
    df_temp['run_provenienza'] = file.name
    lista_df.append(df_temp)

# procedo all'analisi solo se ho trovato e caricato dei file
if lista_df:
    # unisco tutti i dataframe in un unico blocco dati
    df_globale_non_cat = pd.concat(lista_df, ignore_index=True)

    # raggruppo per label e calcolo il numero di run uniche in cui appare
    frequenza_oggetti = df_globale_non_cat.groupby('label').agg(
        numero_di_run_in_cui_appare=('run_provenienza', 'nunique'),
        somma_ripetizioni_intra_run=('ripetizioni', 'sum'),
        RA_medio=('RA_centroid', 'mean'),
        DE_medio=('DEC_centroid', 'mean')
    ).reset_index()

    # ordino la mia tabella dalla label più frequente a quella meno frequente
    frequenza_oggetti = frequenza_oggetti.sort_values('numero_di_run_in_cui_appare', ascending=False)

    # imposto una soglia arbitraria per definire un oggetto come "persistente"
    soglia_alta_frequenza = 5

    # separo i miei oggetti in base alla frequenza di apparizione
    oggetti_persistenti = frequenza_oggetti[frequenza_oggetti['numero_di_run_in_cui_appare'] >= soglia_alta_frequenza]
    eventi_singoli = frequenza_oggetti[frequenza_oggetti['numero_di_run_in_cui_appare'] == 1]

    print(f"Ho analizzato un totale di {len(frequenza_oggetti)} oggetti non catalogati unici.")
    print(f" -> Oggetti Persistenti (apparsi in >= {soglia_alta_frequenza} run): {len(oggetti_persistenti)}")
    print(f" -> Eventi Singoli (apparsi in esattamente 1 run): {len(eventi_singoli)}")

    cartella_output = cerca_cartella_nel_progetto(BASE_DIR , "studio_colossale")

    # salvo i risultati in file csv per poterli ispezionare
    output_persistenti = cartella_output / "analisi_oggetti_persistenti.csv"
    output_eventi = cartella_output / "analisi_eventi_singoli.csv"

    oggetti_persistenti.to_csv(output_persistenti, index=False)
    eventi_singoli.to_csv(output_eventi, index=False)

    print("Ho salvato i file CSV con i risultati dell'analisi nella cartella tabelle_COLOSSALE_alleggerito.")

else:
    print("Non ho trovato nessun file *_oggetti_non_catalogati.parquet nella directory specificata.")


def pulisci_e_converti_header(header_raw):
    # converto i valori dell'header nei tipi corretti ed elimino eventuali commenti
    header_pulito = {}
    for k, v in header_raw.items():
        if isinstance(v, str):
            # estraggo solo la parte prima dell'eventuale commento '/' o '='
            val_str = re.split(r'[=/]', v)[0].strip().replace("'", "")
            try:
                if '.' in val_str or 'E' in val_str.upper():
                    header_pulito[k] = float(val_str)
                else:
                    header_pulito[k] = int(val_str)
            except ValueError:
                header_pulito[k] = val_str  # rimane stringa se non è numerico
        else:
            header_pulito[k] = v
    return header_pulito


file_riassunti = list(cartella_tabelle.rglob("*_oggetti_non_catalogati.parquet"))
file_immagini = list(cartella_tabelle.rglob("*_immagine_*.parquet"))

if not file_riassunti or not file_immagini:
    print("ERRORE: File parquet non trovati.")
    exit()

print(f"Caricamento oggetti non catalogati...")
lista_df_non_cat = [pd.read_parquet(f) for f in file_riassunti]
df_labels = pd.concat(lista_df_non_cat, ignore_index=True)

stat_obj = df_labels.groupby('label').agg({
    'RA_centroid': 'mean',
    'DEC_centroid': 'mean'
}).reset_index()

coords_oggetti = SkyCoord(ra=stat_obj['RA_centroid'].values * u.deg,
                          dec=stat_obj['DEC_centroid'].values * u.deg)

copertura_totale = np.zeros(len(stat_obj), dtype=int)
presenza_reale = np.zeros(len(stat_obj), dtype=int)

print(f"Analisi dinamica su {len(file_immagini)} immagini...")

for f_img in tqdm(file_immagini):
    schema = pq.read_schema(f_img)

    # verifico la presenza della mia nuova chiave dei metadati
    if not schema.metadata or b'metadati_intestazione_csv' not in schema.metadata:
        continue

    # estraggo il mio testo puro dei metadati
    metadati_testo = schema.metadata[b'metadati_intestazione_csv'].decode('utf-8')

    # costruisco il mio dizionario estrapolando i dati riga per riga
    header_raw = {}
    for riga in metadati_testo.split('\n'):
        if riga.startswith('#'):
            riga_pulita = riga.strip()[1:].strip()
            if riga_pulita and ': ' in riga_pulita:
                chiave, valore = riga_pulita.split(': ', 1)
                header_raw[chiave] = valore

    if not header_raw: continue

    # applico la pulizia per evitare il TypeError sulle stringhe
    header_final = pulisci_e_converti_header(header_raw)

    try:
        w = WCS(fits.Header(header_final))

        # uso i valori già convertiti per RA e DEC
        ra_c = header_final.get("RA")
        dec_c = header_final.get("DEC")

        if ra_c is not None and dec_c is not None:
            centro_img = SkyCoord(ra_c, dec_c, unit=u.deg)
            alto_destra = w.pixel_to_world(3071, 2047)
            raggio_fov = Angle(centro_img.separation(alto_destra) * 1.5, "deg")

            distanze = centro_img.separation(coords_oggetti)
            copertura_totale += (distanze <= raggio_fov).astype(int)

            # controllo la mia presenza effettiva nel file
            df_img = pd.read_parquet(f_img)
            presenza_reale += stat_obj['label'].isin(df_img['label'].values).astype(int)

    except Exception as e:
        continue

stat_obj['Copertura_Immagini'] = copertura_totale
stat_obj['Presenza_Effettiva'] = presenza_reale

with np.errstate(divide='ignore', invalid='ignore'):
    stat_obj['Costanza_Percentuale'] = (stat_obj['Presenza_Effettiva'] / stat_obj['Copertura_Immagini'] * 100).round(2)

stat_obj.loc[stat_obj['Costanza_Percentuale'] > 100, 'Costanza_Percentuale'] = 100.0
stat_obj['Costanza_Percentuale'] = stat_obj['Costanza_Percentuale'].fillna(0.0)

stat_obj.sort_values('Costanza_Percentuale', ascending=False).to_csv("analisi_costanza_dinamica_ULTRA_PRECISA.csv",
                                                                     index=False)

print(f"\nAnalisi completata. File: analisi_costanza_dinamica_ULTRA_PRECISA.csv")