import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.optimize import curve_fit
import warnings
from pathlib import Path
from tqdm import tqdm
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import FITSFixedWarning
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.aperture import CircularAperture, aperture_photometry
from datetime import datetime

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


def trova_cartella_base(nome_target="pmc_photometry"):
    # cerco la mia cartella base risalendo l'albero delle directory
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

from funzioni.utilita_parquet import *
from funzioni.astrometria_parquet import *

# =============================================================================
# 1. LETTURA DEL FILE CANDIDATI E PREPARAZIONE TARGET
# =============================================================================

# individuo il file csv dei candidati
percorso_csv = cerca_file_nel_progetto(BASE_DIR, "candidati.csv")

# fermo l'esecuzione se non trovo il file
if percorso_csv is None:
    print("ERRORE: Non ho trovato il file 'candidati.csv'.")
    sys.exit()

# estraggo i dati ed isolo i label di interesse
df_candidati = pd.read_csv(percorso_csv)

# Converto in lista perché il filtro 'in' di PyArrow richiede una lista (o tupla)
lista_candidati = list(df_candidati['label'].unique())

print(f"Caricati {len(lista_candidati)} candidati unici dal file.")

# =============================================================================
# 2. RICERCA ED ESTRAZIONE DAI FILE PARQUET TRAMITE METADATI
# =============================================================================

# individuo la cartella contenente i file parquet
cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito"
file_parquet = list(cartella_tabelle.rglob("*run_*_run_*_immagine_*.parquet"))

if not file_parquet:
    print("ERRORE: Nessun file Parquet trovato nella cartella specificata.")
    sys.exit()

# preparo una lista vuota per immagazzinare i dataframe filtrati
dati_estratti = []

# preparo un dizionario per memorizzare i dati temporali e i parametri estratti da ogni label
parquet_info_per_label = {label: [] for label in lista_candidati}

print("Inizio la scansione ottimizzata sfruttando le proprietà dei Parquet...")

# analizzo un file parquet alla volta
for file_p in tqdm(file_parquet, desc="Analisi file"):
    try:
        # Utilizzo i filtri nativi di PyArrow (pushdown predicates)
        tabella_p = pq.read_table(
            file_p,
            columns=['label', 'Mag_estratta', 'err_Mag_estratta', 'xcentroid', 'ycentroid', 'RA_centroid',
                     'DEC_centroid'],
            filters=[('label', 'in', lista_candidati)]
        )

        # se la tabella non è vuota (ovvero ho trovato almeno un candidato in questo file)
        if tabella_p.num_rows > 0:
            df_trovati = tabella_p.to_pandas()

            # Leggo l'header del file parquet corrente
            header = leggi_header_da_parquet(file_p)

            # Estraggo la data di osservazione, nome del file e i parametri di fit
            date_obs = header.get('DATE-OBS')
            nome_file_fits = header.get('NOME_FILE_FITS')
            fit_m = header.get('FIT_M', 1.0)
            fit_q = header.get('FIT_Q', 0.0)

            # Ricavo il nome della cartella dalle prime 8 cifre del nome del file FITS
            nome_cartella = str(nome_file_fits)[:8] if nome_file_fits is not None else None

            # Aggiungo le colonne al dataframe temporaneo e rendo True la provenienza dal Parquet
            df_trovati['DATE-OBS'] = date_obs
            df_trovati['nome_file_fits'] = nome_file_fits
            df_trovati['nome_cartella'] = nome_cartella
            df_trovati['segmentazione_trovata'] = True

            dati_estratti.append(df_trovati)

            # conservo i dati utili per ogni label di questo parquet
            for label in df_trovati['label'].unique():
                parquet_info_per_label[label].append({
                    'NOME_FILE_FITS': nome_file_fits,
                    'DATE-OBS': date_obs,
                    'FIT_M': float(fit_m) if fit_m is not None else 1.0,
                    'FIT_Q': float(fit_q) if fit_q is not None else 0.0
                })

    except Exception:
        # ignoro eventuali file corrotti o illeggibili
        continue

# assemblo il dataframe parziale
if dati_estratti:
    df_risultati = pd.concat(dati_estratti, ignore_index=True)
else:
    df_risultati = pd.DataFrame(
        columns=['label', 'Mag_estratta', 'err_Mag_estratta', 'xcentroid', 'ycentroid', 'RA_centroid', 'DEC_centroid',
                 'DATE-OBS', 'nome_file_fits', 'nome_cartella', 'segmentazione_trovata'])

# =============================================================================
# 3. INTEGRAZIONE CON IMMAGINI FITS FOTOGRAFATE MA SENZA PARQUET
# =============================================================================

# precalcolo le coordinate di tutti i candidati dal loro label
coords_candidati = {}
for label in lista_candidati:
    try:
        parti = label.replace("RA_", "").split("DEC")
        ra_label = float(parti[0])
        dec_label = float(parti[1])
        coords_candidati[label] = (ra_label, dec_label, SkyCoord(ra=ra_label, dec=dec_label, unit='deg'))
    except Exception:
        continue

# cerco la cartella ASTRI1 in tutto il pc
print("\nCerco la cartella 'ASTRI1' nell'intero PC per espandere i rilevamenti...")
percorso_astri1 = cerca_cartella_intero_pc("ASTRI1")

if percorso_astri1:
    cartella_astri1 = Path(percorso_astri1)
    file_fits_astri1 = list(cartella_astri1.rglob("*.fits"))

    nuove_righe = []

    for file_f in tqdm(file_fits_astri1, desc="Scansione FITS in ASTRI1"):
        try:
            with fits.open(file_f) as hdul:
                header_fits = hdul[0].header
                data_fits = hdul[0].data

                if data_fits is None:
                    continue

                wcs = WCS(header_fits)
                naxis2, naxis1 = data_fits.shape
                nome_fits = file_f.name
                data_fits_str = header_fits.get('DATE-OBS', None)

                for label, (ra_label, dec_label, coord_label) in coords_candidati.items():
                    x_pix, y_pix = wcs.world_to_pixel(coord_label)

                    # verifico se le coordinate in pixel rientrano nell'immagine
                    if 0 <= x_pix <= naxis1 and 0 <= y_pix <= naxis2:
                        fits_gia_presenti = [p['NOME_FILE_FITS'] for p in parquet_info_per_label.get(label, []) if
                                             p['NOME_FILE_FITS']]

                        # se il file fits non è già tra quelli da cui ho estratto un parquet per quel label
                        if nome_fits not in fits_gia_presenti:
                            fit_m_scelto = 1.0
                            fit_q_scelto = 0.0

                            # cerco il parquet temporalmente più vicino alla data di osservazione del fits
                            if data_fits_str and parquet_info_per_label.get(label):
                                try:
                                    data_corrente = datetime.strptime(data_fits_str, "%Y-%m-%dT%H:%M:%S.%f")
                                except ValueError:
                                    data_corrente = datetime.strptime(data_fits_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")

                                min_diff = None
                                for p in parquet_info_per_label[label]:
                                    try:
                                        p_date_str = p['DATE-OBS']
                                        try:
                                            p_date = datetime.strptime(p_date_str, "%Y-%m-%dT%H:%M:%S.%f")
                                        except ValueError:
                                            p_date = datetime.strptime(p_date_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")

                                        diff = abs((data_corrente - p_date).total_seconds())
                                        if min_diff is None or diff < min_diff:
                                            min_diff = diff
                                            fit_m_scelto = p['FIT_M']
                                            fit_q_scelto = p['FIT_Q']
                                    except Exception:
                                        continue

                            # calcolo il flusso con apertura di raggio 5
                            apertura = CircularAperture((x_pix, y_pix), r=3.6)
                            phot_table = aperture_photometry(data_fits, apertura)
                            flusso = phot_table['aperture_sum'][0]

                            # converto in magnitudine con la formula lineare inversa usando M e Q
                            if flusso > 0:
                                mag_strumentale = -2.5 * np.log10(flusso)
                                mag_estratta = (mag_strumentale - fit_q_scelto) / fit_m_scelto
                            else:
                                mag_estratta = np.nan

                            nuove_righe.append({
                                'label': label,
                                'Mag_estratta': mag_estratta,
                                'err_Mag_estratta': 0.0,
                                'xcentroid': float(x_pix),
                                'ycentroid': float(y_pix),
                                'RA_centroid': ra_label,
                                'DEC_centroid': dec_label,
                                'DATE-OBS': data_fits_str,
                                'nome_file_fits': nome_fits,
                                'nome_cartella': str(nome_fits)[:8],
                                'segmentazione_trovata': False
                            })
        except Exception:
            continue

    # unisco le nuove rilevazioni extra a quelle elaborate prima
    if nuove_righe:
        df_nuovi = pd.DataFrame(nuove_righe)
        if not df_risultati.empty:
            df_risultati = pd.concat([df_risultati, df_nuovi], ignore_index=True)
        else:
            df_risultati = df_nuovi
else:
    print("ATTENZIONE: La cartella 'ASTRI1' non è stata trovata.")

# =============================================================================
# 4. SALVATAGGIO DEI RISULTATI FINALI
# =============================================================================

# assemblo e procedo al salvataggio solo se il dataframe finale contiene effettivamente dei dati
if not df_risultati.empty:
    # Cerco la cartella dove salvare i risultati
    cartella_output = cerca_cartella_nel_progetto(BASE_DIR, "presenza_consecutiva_multirun")

    # Definisco il percorso per il nuovo file
    percorso_finale = cartella_output / "candidati_frame.csv"

    # Salvo il dataframe finale in formato csv escludendo l'indice
    df_risultati.to_csv(percorso_finale, index=False)

    print(f"\nOperazione completata! Estratti in totale {len(df_risultati)} record.")
    print(f"File salvato correttamente in: {percorso_finale}")
else:
    print("\nNessun dato utile trovato da estrarre.")