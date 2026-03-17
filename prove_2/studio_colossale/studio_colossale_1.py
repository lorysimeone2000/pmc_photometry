import pandas as pd
from pathlib import Path
import os
import sys
import warnings
import numpy as np
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import WCS
from astropy.io import fits
from shapely.geometry import Point, Polygon
from tqdm import tqdm

# definisco le costanti dimensionali del sensore lette dall'header FITS
BORDO = 7
NAXIS1 = 3072
NAXIS2 = 2048

# gestisco i warning ignorandoli
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
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

from funzioni.utilita import *
from funzioni.astrometria import *

# --- PARAMETRI CONFIGURAZIONE ---

base_path = BASE_DIR / "tabelle_COLOSSALE" / "tabelle_unite"

# verifico esistenza cartella base
if not base_path.exists():
    print(f"Errore: La cartella {base_path} non esiste.")
    exit()

# inizializzo una lista vuota in cui memorizzo i DataFrame parziali degli oggetti senza corrispondenza
dati_non_catalogati = []

# inizializzo una lista per salvare i poligoni di ogni singola immagine analizzata
poligoni_immagini = []

# inizializzo un contatore per capire quanti file vengono effettivamente letti
file_letti = 0

# cerco tutti i file csv ricorsivamente all'interno della cartella base e di tutte le sue sottocartelle
lista_csv = list(base_path.rglob("*.csv"))

for file_csv in tqdm(lista_csv, desc="Elaborazione CSV e Creazione Poligoni FOV"):
    file_letti += 1

    try:
        # leggo il csv ignorando l'header FITS iniziale
        df = pd.read_csv(file_csv, comment='#')

        # leggo l'header dal csv per definire il poligono non euclideo
        header_info = leggi_header_da_csv(file_csv)
        nome_fits = header_info.get('NOME_FILE_FITS', '')

        # estraggo il nome se la chiave principale è vuota
        if not nome_fits:
            percorso_raw = header_info.get('PERCORSO_FILE', '')
            nome_fits = os.path.basename(str(percorso_raw))

        nome_fits = str(nome_fits).strip()

        # cerco il file FITS nel progetto per estrarne il WCS corretto
        file_trovato = cerca_file_nel_progetto(BASE_DIR, nome_fits)

        if file_trovato is not None:
            # estraggo il WCS reale dal FITS
            with fits.open(str(file_trovato), memmap=False) as hdu:
                w = WCS(hdu[0].header)

            # calcolo i 4 vertici dell'immagine sul piano celeste
            c1 = w.pixel_to_world(0, 0)
            c2 = w.pixel_to_world(NAXIS1 - 1, 0)
            c3 = w.pixel_to_world(NAXIS1 - 1, NAXIS2 - 1)
            c4 = w.pixel_to_world(0, NAXIS2 - 1)

            # creo il mio poligono non euclideo con shapely
            poligono = Polygon([
                (c1.ra.deg, c1.dec.deg),
                (c2.ra.deg, c2.dec.deg),
                (c3.ra.deg, c3.dec.deg),
                (c4.ra.deg, c4.dec.deg)
            ])

            # salvo il poligono e i suoi confini minimi e massimi per scremare velocemente i punti
            minx, miny, maxx, maxy = poligono.bounds
            poligoni_immagini.append({
                'poly': poligono,
                'minx': minx, 'miny': miny,
                'maxx': maxx, 'maxy': maxy
            })
        else:
            print(
                f"ATTENZIONE: File FITS originale '{nome_fits}' non trovato all'interno del progetto. Impossibile creare il poligono per questo frame.")
            continue

    except Exception as e:
        print(f"Errore nella lettura o elaborazione di {file_csv}: {e}")
        continue

    # controllo che il dataframe non sia vuoto e contenga le coordinate necessarie
    if not df.empty and 'Corrispondenza' in df.columns and 'RA_centroid' in df.columns:

        # estraggo esclusivamente le sorgenti senza alcun match nel catalogo
        df_no_corr = df[df['Corrispondenza'] == 'NO'].copy()

        # creo la maschera booleana per scartare gli oggetti entro 7 pixel dal bordo
        maschera_bordo = (
                (df_no_corr['xcentroid'] > BORDO) &
                (df_no_corr['xcentroid'] < (NAXIS1 - BORDO)) &
                (df_no_corr['ycentroid'] > BORDO) &
                (df_no_corr['ycentroid'] < (NAXIS2 - BORDO))
        )

        # applico il filtro spaziale
        df_no_corr_filtrato = df_no_corr[maschera_bordo]

        # aggiungo i dati validi alla mia lista, limitando alle colonne essenziali
        if not df_no_corr_filtrato.empty:
            colonne_utili = ['label', 'run_id', 'img_index', 'RA_centroid', 'DEC_centroid']
            dati_non_catalogati.append(df_no_corr_filtrato[colonne_utili])

print(
    f"\nElaborazione base completata. Ho analizzato un totale di {file_letti} file CSV e creato {len(poligoni_immagini)} poligoni.")

# unisco tutti i risultati in un unico set di dati globale
if dati_non_catalogati:
    df_totale = pd.concat(dati_non_catalogati, ignore_index=True)

    # raggruppo il dataset in base al label univoco calcolando anche le coordinate medie
    statistiche_oggetti = df_totale.groupby('label').agg(
        rilevamenti_totali=('label', 'count'),
        lista_run=('run_id', lambda x: list(set(x))),
        run_uniche=('run_id', 'nunique'),
        ra_medio=('RA_centroid', 'mean'),
        dec_medio=('DEC_centroid', 'mean')
    ).reset_index()

    # calcolo dinamicamente in quante immagini l'oggetto DOVEVA essere presente usando i poligoni
    osservazioni_attese = []

    for ra, dec in tqdm(zip(statistiche_oggetti['ra_medio'], statistiche_oggetti['dec_medio']),
                        total=len(statistiche_oggetti), desc="Controllo Presenza nei Poligoni"):
        punto = Point(ra, dec)
        conteggio = 0

        # itero sui poligoni usando i bounds per scremare ed evitare di controllare punti fuori area
        for p_info in poligoni_immagini:
            if p_info['minx'] <= ra <= p_info['maxx'] and p_info['miny'] <= dec <= p_info['maxy']:
                if p_info['poly'].contains(punto):
                    conteggio += 1

        osservazioni_attese.append(conteggio)

    # inserisco i risultati nel dataframe
    statistiche_oggetti['osservazioni_attese'] = osservazioni_attese

    # calcolo il tasso di persistenza reale
    # sostituisco gli zeri con 1 per evitare errori di divisione
    statistiche_oggetti['tasso_persistenza'] = statistiche_oggetti['rilevamenti_totali'] / statistiche_oggetti[
        'osservazioni_attese'].replace(0, 1)

    # stabilisco una soglia di persistenza basata sulla percentuale
    soglia_tasso = 0.80

    # separo i target sempre presenti da quelli specifici in base al tasso reale
    oggetti_persistenti = statistiche_oggetti[statistiche_oggetti['tasso_persistenza'] >= soglia_tasso]
    oggetti_transitori = statistiche_oggetti[statistiche_oggetti['tasso_persistenza'] < soglia_tasso]

    # ordino i risultati per tasso decrescente
    oggetti_persistenti = oggetti_persistenti.sort_values(by='tasso_persistenza', ascending=False)

    print(f"\nOggetti persistenti trovati: {len(oggetti_persistenti)}")
    print(f"Oggetti transitori trovati: {len(oggetti_transitori)}")
else:
    print("Nessun oggetto senza corrispondenza trovato nei CSV elaborati (oppure tutti sono vicini ai bordi).")