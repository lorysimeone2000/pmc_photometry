import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings
from pathlib import Path
from tqdm import tqdm
from astropy.io.fits.verify import VerifyWarning
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
    return path_corrente.parent

BASE_DIR = trova_cartella_base("Lorenzo")
PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")
if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

from funzioni.utilita_parquet import *
from funzioni.astrometria_parquet import *

cartella_tabelle = cerca_cartella_nel_progetto(BASE_DIR, "tabelle_COLOSSALE_alleggerito")
file_fits = list(cartella_tabelle.rglob("*_oggetti_non_catalogati.parquet"))

lista_ra = []
lista_dec = []

for file_p in tqdm(file_fits, desc="Analisi file"):
    try:
        # leggo il mio header dal file parquet
        header = leggi_header_da_parquet(file_p)
        ra = header.get("RA")
        dec = header.get("DEC")
        if ra is not None and dec is not None:
            lista_ra.append(ra)
            lista_dec.append(dec)
    except Exception:
        continue

# definisco i miei centri per i target
crab_centro = SkyCoord.from_name("Crab")
mrk421_centro = SkyCoord.from_name("Mrk 421")
polo_nord_centro = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame='icrs')

raggio_fov = [5.75,1.5,1]
size = 40

# --- Riquadro 1: Crab (Cartesiano) ---
# creo la mia figura singola per la Crab
fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.scatter(lista_ra, lista_dec, s=2, color='blue', alpha=0.4, label="Runs")
# inserisco il mio puntino al centro al posto del cerchio
ax1.scatter(crab_centro.ra.deg, crab_centro.dec.deg, color='red', s=size, marker='+', zorder=5, label='Crab')

cos_dec_crab = np.cos(crab_centro.dec.radian)
delta_ra_crab = raggio_fov[0] / cos_dec_crab
ax1.set_xlim(crab_centro.ra.deg + delta_ra_crab, crab_centro.ra.deg - delta_ra_crab)
ax1.set_ylim(crab_centro.dec.deg - raggio_fov[0], crab_centro.dec.deg + raggio_fov[0])
# adatto la dimensione dei font per i titoli degli assi
ax1.set_xlabel("RA (deg)", fontsize=16)
ax1.set_ylabel("DEC (deg)", fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)
# adatto la dimensione dei font per la legenda
ax1.legend(loc='upper right', fontsize=14)

plt.tight_layout()
plt.savefig("mappa_puntamenti_crab.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Riquadro 2: Mrk 421 (Cartesiano) ---
# creo la mia figura singola per Mrk 421
fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.scatter(lista_ra, lista_dec, s=2, color='blue', alpha=0.4, label="Runs")
# inserisco il mio puntino al centro al posto del cerchio
ax2.scatter(mrk421_centro.ra.deg, mrk421_centro.dec.deg, color='orange', s=size, marker='+', zorder=5, label='Mrk 421')

cos_dec_mrk = np.cos(mrk421_centro.dec.radian)
delta_ra_mrk = raggio_fov[1] / cos_dec_mrk
ax2.set_xlim(mrk421_centro.ra.deg + delta_ra_mrk, mrk421_centro.ra.deg - delta_ra_mrk)
ax2.set_ylim(mrk421_centro.dec.deg - raggio_fov[1], mrk421_centro.dec.deg + raggio_fov[1])
# adatto la dimensione dei font per i titoli degli assi
ax2.set_xlabel("RA (deg)", fontsize=16)
ax2.set_ylabel("DEC (deg)", fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7)
# adatto la dimensione dei font per la legenda
ax2.legend(loc='upper right', fontsize=14)

plt.tight_layout()
plt.savefig("mappa_puntamenti_mrk421.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Riquadro 3: Polo Nord (Polare) ---
# creo la mia figura singola per il Polo Nord
fig3 = plt.figure(figsize=(6, 6))
ax3 = fig3.add_subplot(1, 1, 1, projection='polar')
theta_runs = np.radians(lista_ra)
r_runs = 90 - np.array(lista_dec)
ax3.scatter(theta_runs, r_runs, s=4, color='blue', alpha=0.4, label="Runs")

# inserisco il mio puntino nell'origine della proiezione polare (r=0)
ax3.scatter(0, 0, color='green', s=size, marker='+', zorder=5, label='North Celestial Pole')

ax3.set_ylim(0, raggio_fov[2] * 2)
ax3.set_yticklabels([])
ax3.grid(True, linestyle='--', alpha=0.7)
# adatto la dimensione dei font per la legenda
ax3.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=14)

plt.tight_layout()
plt.savefig("mappa_puntamenti_polo_nord.png", dpi=300, bbox_inches='tight')
plt.show()