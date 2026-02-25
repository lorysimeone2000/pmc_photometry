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
    # risalgo l'albero delle directory
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

print(f"--- CONFIGURAZIONE ESTRAZIONE E SUDDIVISIONE RUN BLAZAR ---")
print(f"Cartella Base rilevata: {BASE_DIR}")

# definisco le coordinate esatte del blazar Markarian 421 (J2000)
coords_mrk421 = SkyCoord(ra=166.1138 * u.deg, dec=38.2088 * u.deg, frame='icrs')

# imposto il raggio di tolleranza
raggio_fov_tolleranza = 6 * u.deg

# preparo le cartelle
PMC_DATA = cerca_cartella_nel_progetto(BASE_DIR, "PMC_DATA")
cartella_blazar = BASE_DIR / "blazar/PMC_DATA_BLAZAR"
cartella_blazar.mkdir(exist_ok=True, parents=True)

print(f"Cartella di origine: {PMC_DATA}")
print(f"Cartella di destinazione: {cartella_blazar}\n")

if PMC_DATA:
    sottocartelle = [d for d in PMC_DATA.iterdir() if d.is_dir()]
    file_validi = []

    print("Fase 1: Scansione e filtro dei file sul Blazar in corso...")
    for cartella_giorno in sottocartelle:
        estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
        file_fits_list = []
        for ext in estensioni_valide:
            file_fits_list.extend(cartella_giorno.glob(ext))

        for percorso_file in tqdm(file_fits_list, desc=f"Scansione {cartella_giorno.name}"):
            try:
                with fits.open(percorso_file, memmap=False) as hdu:
                    header = hdu[0].header

                    # estraggo RA, DEC e TIME
                    ra = header.get('RA')
                    if ra is None: ra = header.get('RAJ2000')

                    dec = header.get('DEC')
                    if dec is None: dec = header.get('DEJ2000')

                    tempo_obs_str = header.get('DATE-OBS')

                    if ra is not None and dec is not None and tempo_obs_str is not None:
                        coords_centro = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame='icrs')
                        separazione = coords_centro.separation(coords_mrk421)

                        # se il blazar è nel campo visivo, lo salvo in memoria con i suoi metadati
                        if separazione <= raggio_fov_tolleranza:
                            file_validi.append({
                                'percorso_originale': percorso_file,
                                'nome_file': percorso_file.name,
                                'nome_giorno': cartella_giorno.name,
                                'tempo': Time(tempo_obs_str),
                                'dej2000': float(dec)
                            })
            except Exception:
                pass

    if not file_validi:
        print("Nessun file trovato per il Blazar Markarian 421.")
        sys.exit()

    print("\nFase 2: Ordinamento cronologico...")
    # ordino la lista in modo rigorosamente cronologico
    file_validi.sort(key=lambda x: x['tempo'])

    print("Fase 3: Valutazione soglie e creazione symlink nelle run...")
    # imposto le soglie ricavate dai grafici
    soglia_tempo = 600.0
    soglia_spazio = 0.2

    # inizializzo il mio contatore globale per le run e le variabili di confronto
    contatore_run = 1
    tempo_precedente = None
    dej2000_precedente = None

    file_trovati_blazar = []

    for dato in file_validi:
        if tempo_precedente is not None:
            # calcolo le variazioni rispetto allo scatto precedente
            delta_tempo = (dato['tempo'] - tempo_precedente).sec
            delta_spazio = abs(dato['dej2000'] - dej2000_precedente)

            # se supero le soglie, incremento il mio contatore
            if delta_tempo > soglia_tempo or delta_spazio > soglia_spazio:
                contatore_run += 1

        nome_run = f"run_{contatore_run:08d}"

        # costruisco il percorso: PMC_DATA_BLAZAR / giorno / run_0000000X
        cartella_destinazione_run = cartella_blazar / dato['nome_giorno'] / nome_run

        # creo la cartella se non esiste
        cartella_destinazione_run.mkdir(parents=True, exist_ok=True)

        # creo il symlink
        path_symlink = cartella_destinazione_run / dato['nome_file']
        path_relativo = os.path.relpath(dato['percorso_originale'], start=cartella_destinazione_run)

        if not path_symlink.exists():
            os.symlink(path_relativo, path_symlink)

        file_trovati_blazar.append(f"{dato['nome_giorno']}/{nome_run}/{dato['nome_file']}")

        # aggiorno le mie variabili per il ciclo successivo
        tempo_precedente = dato['tempo']
        dej2000_precedente = dato['dej2000']

    # salvo il file di testo riepilogativo
    file_txt_output = cartella_blazar / "lista_file_blazar.txt"
    with open(file_txt_output, "w") as f:
        f.write("# Elenco dei file FITS (organizzati per giorno e run) che osservano il blazar Markarian 421\n")
        f.write(f"# Totale file: {len(file_trovati_blazar)}\n")
        f.write(f"# Totale run create: {contatore_run}\n\n")
        for nome_file in file_trovati_blazar:
            f.write(f"{nome_file}\n")

    print(f"\n--- OPERAZIONE COMPLETATA ---")
    print(f"Trovati e collegati {len(file_validi)} file in {contatore_run} run totali.")
    print(f"La struttura delle cartelle è visibile in: {cartella_blazar}")

else:
    print("Elaborazione interrotta: cartella PMC_DATA non trovata.")