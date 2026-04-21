import pandas as pd
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


def trova_cartella_base(nome_target="Lorenzo"):
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

cartella_tabelle = cerca_cartella_intero_pc('ASTRI1')

file_fits = list(cartella_tabelle.rglob(".fits"))

# inizializzo le mie liste per conservare tutte le coordinate lette
lista_ra = []
lista_dec = []

for file_p in tqdm(file_fits, desc="Analisi file"):
    # passo il singolo file alla funzione invece della lista intera

    hdu_list = fits.open(file_fits)
    header = hdu_list[0].header

    ra = header["RA"]
    dec = header["DEC"]

    # aggiungo i valori appena estratti alle mie liste
    lista_ra.append(ra)
    lista_dec.append(dec)

# creo la figura e salvo il mio plot con tutti i punti
plt.figure(figsize=(10, 8))
plt.scatter(lista_ra, lista_dec, s=1, color='blue', alpha=0.6)
plt.xlabel('RA (Gradi)')
plt.ylabel('DEC (Gradi)')
#plt.title('Mappa degli oggetti non catalogati in coordinate RA/DEC')
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig("mappa_puntamenti_ra_dec.png", dpi=300, bbox_inches='tight')
plt.close()