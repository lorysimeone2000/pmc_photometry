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
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


def trova_cartella_base(nome_target="Lorenzo"):
    # cerco la mia cartella base risalendo l'albero della directory
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

# converto la mia stringa in un oggetto Path per usare il metodo rglob
cartella_tabelle = cerca_cartella_nel_progetto(BASE_DIR, "tabelle_COLOSSALE_alleggerito")
file_fits = list(cartella_tabelle.rglob("*_oggetti_non_catalogati.parquet"))

# inizializzo la mia lista per conservare la coordinata letta
lista_ra = []
lista_dec = []

for file_p in tqdm(file_fits, desc="Analisi file"):
    try:
        # leggo il mio header dal file parquet
        header = leggi_header_da_parquet(file_p)

        # cerco la mia chiave in modo sicuro
        ra = header.get("RA")
        dec = header.get("DEC")

        # verifico il mio dato prima di aggiungerlo
        if ra is not None and dec is not None:
            lista_ra.append(ra)
            lista_dec.append(dec)
    except Exception:
        # ignoro il mio file in caso di errore
        continue

# converto la mia coordinata in sistema galattico
coords = SkyCoord(ra=lista_ra * u.degree, dec=lista_dec * u.degree, frame='icrs')
coords_gal = coords.galactic

# preparo la mia longitudine traslata e latitudine in radianti per la proiezione
l_rad = coords_gal.l.wrap_at(360 * u.deg).radian - np.pi
b_rad = coords_gal.b.radian


def genera_punti_circonferenza(centro, raggio_deg, num_punti=200):
    # genero i miei angoli per il cerchio
    angoli = np.linspace(0, 360, num_punti) * u.deg
    # calcolo il mio offset per creare la circonferenza sulla sfera
    cerchio_coords = centro.directional_offset_by(angoli, raggio_deg * u.deg)
    # trasformo il mio risultato in coordinate galattiche
    cerchio_gal = cerchio_coords.galactic

    # trovo i miei punti in radianti traslando la longitudine di 180 gradi
    l_rad_cerchio = cerchio_gal.l.wrap_at(360 * u.deg).radian - np.pi
    b_rad_cerchio = cerchio_gal.b.radian

    # calcolo la mia differenza tra i punti per trovare i salti da un bordo all'altro
    diffs = np.abs(np.diff(l_rad_cerchio))
    salti = np.where(diffs > np.pi)[0]

    # inserisco un valore nullo per spezzare la mia linea dove c'è il salto
    for salto in salti[::-1]:
        l_rad_cerchio = np.insert(l_rad_cerchio, salto + 1, np.nan)
        b_rad_cerchio = np.insert(b_rad_cerchio, salto + 1, np.nan)

    return l_rad_cerchio, b_rad_cerchio


# definisco il mio centro per i target richiesti
crab_centro = SkyCoord.from_name("Crab")
mrk421_centro = SkyCoord.from_name("Mrk 421")
polo_nord_centro = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame='icrs')

# genero il mio set di punti spezzando le linee ai bordi
l_crab, b_crab = genera_punti_circonferenza(crab_centro, 6.25)
l_mrk, b_mrk = genera_punti_circonferenza(mrk421_centro, 6.25)
l_polo, b_polo = genera_punti_circonferenza(polo_nord_centro, 6.25)

# creo la mia figura con proiezione aitoff
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='aitoff')

# mostro il mio punto originale
ax.scatter(l_rad, b_rad, s=1, color='blue', alpha=0.4, label="Runs")

# disegno la mia circonferenza per ogni target
ax.plot(l_crab, b_crab, color='red', linewidth=1, label='Crab')
ax.plot(l_mrk, b_mrk, color='orange', linewidth=1, label='Mrk 421')
ax.plot(l_polo, b_polo, color='green', linewidth=1, label='North Celestial Pole')

# attivo la mia griglia
ax.grid(True, linestyle='--', alpha=0.7)

# aggiorno le mie etichette per l'asse x in base al nuovo centro
ax.set_xticklabels(['30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°'])

# imposto la mia etichetta
ax.set_xlabel('Galactic Longitude')
ax.set_ylabel('Galactic Latitude')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

# salvo la mia immagine finale
plt.savefig("mappa_puntamenti.png", dpi=300, bbox_inches='tight')
plt.close()