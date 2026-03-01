import os
import sys
from pathlib import Path
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # risalgo l'albero delle directory per trovare la radice del progetto
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent

BASE_DIR = trova_cartella_base("pmc_photometry")

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from funzioni.utilita import *
from funzioni.astrometria import *

print(f"--- DEBUG ESTRAZIONE BLAZAR ---")
print(f"Cartella Base: {BASE_DIR}")

# Coordinate Mrk 421
coords_mrk421 = SkyCoord(ra=166.1138 * u.deg, dec=38.2088 * u.deg, frame='icrs')
raggio_fov_tolleranza = 6 * u.deg

PMC_DATA = cerca_cartella_nel_progetto(BASE_DIR, "PMC_DATA")
cartella_blazar = BASE_DIR / "blazar/PMC_DATA_BLAZAR"
cartella_blazar.mkdir(exist_ok=True, parents=True)

if PMC_DATA:
    # Cerco le cartelle risolvendo i symlink (fondamentale visto il tuo ls -ltr)
    sottocartelle = [d for d in PMC_DATA.iterdir() if d.is_dir() or d.is_symlink()]
    file_validi = []

    print(f"Trovate {len(sottocartelle)} cartelle/link in PMC_DATA.")

    for cartella_giorno in sottocartelle:
        # Risolvo il symlink per accedere ai file reali
        percorso_reale = cartella_giorno.resolve()
        
        file_fits_list = []
        for ext in ['*.fit', '*.fits', '*.FIT', '*.FITS']:
            file_fits_list.extend(percorso_reale.glob(ext))

        if not file_fits_list:
            continue

        for percorso_file in tqdm(file_fits_list, desc=f"Scansione {cartella_giorno.name}"):
            try:
                # Uso memmap=True per velocizzare la sola lettura dell'header
                with fits.open(percorso_file, memmap=True) as hdu:
                    header = hdu[0].header

                    # Provo diverse combinazioni di chiavi comuni
                    ra_val = header.get('RA') or header.get('RAJ2000') or header.get('OBJ-RA')
                    dec_val = header.get('DEC') or header.get('DEJ2000') or header.get('OBJ-DEC')
                    tempo_obs_str = header.get('DATE-OBS')

                    if ra_val is not None and dec_val is not None:
                        # Gestisco sia float (gradi) che stringhe (HH:MM:SS)
                        try:
                            if isinstance(ra_val, (int, float)):
                                coords_centro = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame='icrs')
                            else:
                                # Se sono stringhe, specifico le unità
                                coords_centro = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg), frame='icrs')
                            
                            separazione = coords_centro.separation(coords_mrk421)

                            if separazione <= raggio_fov_tolleranza:
                                file_validi.append({
                                    'percorso_originale': percorso_file,
                                    'nome_file': percorso_file.name,
                                    'nome_giorno': cartella_giorno.name,
                                    'tempo': Time(tempo_obs_str) if tempo_obs_str else Time(os.path.getmtime(percorso_file), format='unix'),
                                    'dej2000': coords_centro.dec.deg
                                })
                        except Exception as e:
                            # Salto il file se le coordinate non sono interpretabili
                            continue
            except Exception:
                continue

    if not file_validi:
        print("\nERRORE: Nessun file trovato. Possibili cause:")
        print("1. I symlink puntano a una cartella a cui l'utente lorysimeone non ha accesso (es. /home/astro/...).")
        print("2. Le chiavi RA/DEC nell'header hanno nomi differenti da quelli standard.")
        print("3. Il puntamento dei file è fuori dal raggio di 6 gradi da Mrk 421.")
        sys.exit()

    # --- Ordinamento e creazione RUN ---
    file_validi.sort(key=lambda x: x['tempo'])
    
    soglia_tempo = 600.0
    soglia_spazio = 0.2
    contatore_run = 1
    tempo_precedente = None
    dej2000_precedente = None
    file_trovati_blazar = []

    for dato in file_validi:
        if tempo_precedente is not None:
            delta_tempo = (dato['tempo'] - tempo_precedente).sec
            delta_spazio = abs(dato['dej2000'] - dej2000_precedente)
            if delta_tempo > soglia_tempo or delta_spazio > soglia_spazio:
                contatore_run += 1

        nome_run = f"run_{contatore_run:08d}"
        cartella_destinazione_run = cartella_blazar / dato['nome_giorno'] / nome_run
        cartella_destinazione_run.mkdir(parents=True, exist_ok=True)

        path_symlink = cartella_destinazione_run / dato['nome_file']
        
        # Creo il link usando il percorso assoluto reale
        if not path_symlink.exists():
            os.symlink(dato['percorso_originale'], path_symlink)

        file_trovati_blazar.append(f"{dato['nome_giorno']}/{nome_run}/{dato['nome_file']}")
        tempo_precedente = dato['tempo']
        dej2000_precedente = dato['dej2000']

    print(f"\n--- COMPLETATO ---")
    print(f"Creati {len(file_validi)} link in {contatore_run} run.")

else:
    print("Cartella PMC_DATA non trovata.")
