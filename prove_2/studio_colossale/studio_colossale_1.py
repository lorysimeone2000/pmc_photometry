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
# 0. CONFIGURAZIONE PERCORSI E IDENTIFICAZIONE RADICE ASTRI
# =============================================================================

def trova_radice_astri(nome_target="ASTRI"):
    """Risale il path dallo script corrente fino a trovare la cartella ASTRI."""
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    # Se non la trova, usa la cartella 'Lorenzo' come ripiego o la home
    return Path.home() / "ASTRI"


# Individuo la cartella ASTRI per la ricerca dei FITS
ASTRI_DIR = trova_radice_astri("ASTRI")
print(f"Radice ASTRI individuata per ricerca FITS: {ASTRI_DIR}")


# Individuo la cartella Lorenzo per le funzioni e le tabelle
def trova_cartella_lorenzo(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent


BASE_DIR = trova_cartella_lorenzo("Lorenzo")
PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

from funzioni.utilita import *
from funzioni.astrometria import *

# --- PARAMETRI CONFIGURAZIONE ---

base_path = BASE_DIR / "tabelle_COLOSSALE" / "tabelle_unite"

if not base_path.exists():
    print(f"Errore: La cartella {base_path} non esiste.")
    exit()

# Inizializzazione strutture dati
dati_non_catalogati = []
poligoni_immagini = []
file_letti = 0

# Cerco tutti i file csv ricorsivamente
lista_csv = list(base_path.rglob("*.csv"))

print(f"Inizio analisi di {len(lista_csv)} file CSV...")

for file_csv in tqdm(lista_csv, desc="Analisi CSV e Ricostruzione WCS"):
    file_letti += 1

    try:
        # Leggo il csv
        df = pd.read_csv(file_csv, comment='#')

        # Recupero il nome del file FITS dall'header del CSV
        header_info = leggi_header_da_csv(file_csv)
        nome_fits = header_info.get('NOME_FILE_FITS', '')
        if not nome_fits:
            nome_fits = os.path.basename(str(header_info.get('PERCORSO_FILE', '')))

        nome_fits = str(nome_fits).strip()

        # RICERCA FITS PARTENDO DA ASTRI: salto i symlink rotti
        file_trovato = None
        for p in ASTRI_DIR.rglob(nome_fits):
            if p.exists() and p.is_file():
                file_trovato = p
                break

        if file_trovato:
            with fits.open(str(file_trovato), memmap=False) as hdu:
                w = WCS(hdu[0].header)

            # Calcolo i vertici per il poligono FOV
            c1 = w.pixel_to_world(0, 0)
            c2 = w.pixel_to_world(NAXIS1 - 1, 0)
            c3 = w.pixel_to_world(NAXIS1 - 1, NAXIS2 - 1)
            c4 = w.pixel_to_world(0, NAXIS2 - 1)

            poligono = Polygon([
                (c1.ra.deg, c1.dec.deg),
                (c2.ra.deg, c2.dec.deg),
                (c3.ra.deg, c3.dec.deg),
                (c4.ra.deg, c4.dec.deg)
            ])

            minx, miny, maxx, maxy = poligono.bounds
            poligoni_immagini.append({
                'poly': poligono,
                'minx': minx, 'miny': miny,
                'maxx': maxx, 'maxy': maxy
            })
        else:
            # Se il FITS è introvabile, non posso calcolare le osservazioni attese correttamente
            continue

        # Estraggo gli oggetti NO catalogati
        if not df.empty and 'Corrispondenza' in df.columns:
            df_no = df[df['Corrispondenza'] == 'NO'].copy()

            # Filtro bordo 7 pixel
            mask_bordo = (
                    (df_no['xcentroid'] > BORDO) & (df_no['xcentroid'] < (NAXIS1 - BORDO)) &
                    (df_no['ycentroid'] > BORDO) & (df_no['ycentroid'] < (NAXIS2 - BORDO))
            )
            df_no = df_no[mask_bordo]

            if not df_no.empty:
                dati_non_catalogati.append(df_no[['label', 'run_id', 'RA_centroid', 'DEC_centroid']])

    except Exception:
        continue

# --- AGGREGAZIONE E CLASSIFICAZIONE ---

if dati_non_catalogati:
    df_totale = pd.concat(dati_non_catalogati, ignore_index=True)

    # Raggruppo per label
    stats = df_totale.groupby('label').agg(
        n_rilevati=('label', 'count'),
        ra=('RA_centroid', 'mean'),
        dec=('DEC_centroid', 'mean')
    ).reset_index()

    print("\nCalcolo osservazioni attese tramite poligoni non euclidei...")
    attese = []

    for ra, dec in tqdm(zip(stats['ra'], stats['dec']), total=len(stats)):
        punto = Point(ra, dec)
        visto = 0
        for p_info in poligoni_immagini:
            # Pre-filtro rapido prima del test topologico
            if p_info['minx'] <= ra <= p_info['maxx'] and p_info['miny'] <= dec <= p_info['maxy']:
                if p_info['poly'].contains(punto):
                    visto += 1
        attese.append(visto)

    stats['n_attesi'] = attese
    stats['tasso_persistenza'] = stats['n_rilevati'] / stats['n_attesi'].replace(0, 1)

    # Filtro finale (soglia 80%)
    persistenti = stats[stats['tasso_persistenza'] >= 0.8].sort_values('tasso_persistenza', ascending=False)
    transienti = stats[stats['tasso_persistenza'] < 0.8]

    print(f"\nRISULTATI FINALI:")
    print(f"Oggetti Persistenti (stelle mancanti/fisse): {len(persistenti)}")
    print(f"Oggetti Transitori (rumore/raggi cosmici): {len(transienti)}")
else:
    print("Nessun dato trovato.")