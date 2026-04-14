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

cartella_oggetti = cerca_cartella_intero_pc('ASTRI1')

percorso_candidati_csv = cerca_file_nel_progetto(BASE_DIR, "candidati_frame.csv")

import re
import glob
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LogNorm

# Leggo il file CSV di partenza inserendolo in un dataframe
df_candidati = pd.read_csv(percorso_candidati_csv)

# Preparo una lista di tutti i file FITS presenti nella cartella sorgente e nelle sue sottocartelle
tutti_fits = glob.glob(os.path.join(cartella_oggetti, '**', '*.fits'), recursive=True)

# Creo una struttura per associare ogni file alla sua data di osservazione
catalogo_fits = []
for file_fits in tqdm(tutti_fits, desc="Scansione header FITS"):
    try:
        with fits.open(file_fits) as hdu_list:
            header = hdu_list[0].header
            if 'DATE-OBS' in header:
                # Aggiungo la data e il percorso del file corrispondente alla mia lista
                catalogo_fits.append((header['DATE-OBS'], file_fits))
    except Exception:
        continue

# Ordino il mio catalogo cronologicamente in base alla data
catalogo_fits.sort(key=lambda x: x[0])
date_ordinate = [elemento[0] for elemento in catalogo_fits]
file_ordinati = [elemento[1] for elemento in catalogo_fits]

# Creo la cartella principale dove andrò a salvare tutte le immagini dei riquadri
cartella_riquadri = BASE_DIR / "riquadri"
cartella_riquadri.mkdir(parents=True, exist_ok=True)

# Itero su ogni singola riga del mio dataframe dei candidati
for index, row in tqdm(df_candidati.iterrows(), total=len(df_candidati), desc="Elaborazione target"):
    label = str(row['label'])
    data_candidato = str(row['DATE-OBS'])

    # Estraggo i valori numerici di RA e DEC dalla stringa del label usando le espressioni regolari
    match = re.search(r'RA_([+-]?[\d\.]+)DEC([+-]?[\d\.]+)', label)
    if not match:
        continue

    ra = float(match.group(1))
    dec = float(match.group(2))

    # Cerco la riga corrente all'interno della mia lista ordinata per ricavare l'indice temporale esatto
    try:
        indice_centrale = date_ordinate.index(data_candidato)
    except ValueError:
        # Se non trovo una corrispondenza esatta di data, salto al candidato successivo
        continue

    # Definisco l'intervallo prendendo 5 elementi prima e 5 dopo (se non esco dai confini della lista)
    indice_inizio = max(0, indice_centrale - 5)
    indice_fine = min(len(date_ordinate), indice_centrale + 6)

    # Creo una sottocartella specifica nominata con il label su cui sto lavorando
    cartella_label = cartella_riquadri / label
    cartella_label.mkdir(parents=True, exist_ok=True)

    # Ciclo singolarmente sui file FITS che compongono l'intervallo temporale individuato
    for i in range(indice_inizio, indice_fine):
        data_file = date_ordinate[i]
        percorso_file = file_ordinati[i]

        try:
            with fits.open(percorso_file) as hdu_list:
                header = hdu_list[0].header
                dati_immagine = hdu_list[0].data

                # Converto le coordinate celesti di questo target nei pixel dell'immagine appena aperta
                w = WCS(header)
                px_float, py_float = w.all_world2pix(ra, dec, 0)
                px = int(round(float(px_float)))
                py = int(round(float(py_float)))

                # Calcolo le estensioni dei bordi per limitare e ottenere un riquadro di esattamente 40x40 pixel
                y_min = max(0, py - 20)
                y_max = min(dati_immagine.shape[0], py + 20)
                x_min = max(0, px - 20)
                x_max = min(dati_immagine.shape[1], px + 20)

                # Effettuo il taglio della matrice dati e sottraggo 1 a tutti i valori dei pixel
                riquadro = dati_immagine[y_min:y_max, x_min:x_max] - 1

                # Controllo di sicurezza: se il riquadro finisce fuori dai margini lo salto
                if riquadro.size == 0:
                    continue

                # Inizializzo e compongo graficamente la mia figura applicando le direttive visive fornite
                plt.figure()
                plt.imshow(riquadro, cmap="grey_r", norm=LogNorm(), interpolation='nearest')
                plt.title(data_file)
                plt.xlabel("X")
                plt.ylabel("Y")

                # Costruisco il nome del file di salvataggio convertendo i caratteri speciali per evitare errori
                nome_file_out = data_file.replace(":", "_").replace("-", "_") + ".png"
                percorso_out = cartella_label / nome_file_out

                # Salvo l'immagine nella relativa cartella e chiudo la finestra per rilasciare memoria preziosa
                plt.savefig(percorso_out, bbox_inches='tight')
                plt.close()
        except Exception:
            continue