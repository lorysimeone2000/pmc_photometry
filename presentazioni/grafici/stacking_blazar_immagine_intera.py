import numpy as np
import os
import sys
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import warnings
from astropy.wcs import FITSFixedWarning
from tqdm import tqdm
import pandas as pd

# importo i moduli necessari per la statistica e la visualizzazione
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats

warnings.filterwarnings('ignore', category=FITSFixedWarning)
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


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    # cerco la mia cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


# trovo la mia cartella base del progetto
BASE_DIR = trova_cartella_base("Lorenzo")

PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")
if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

# =============================================================================
# 1. RICERCA E RACCOLTA DATI
# =============================================================================
print("--- INIZIO RICERCA IMMAGINI BLAZAR ---")

astri1 = cerca_cartella_intero_pc('ASTRI1')

# definisco l'elenco delle cartelle da cercare all'interno di astri1
cartelle_target = ["20251220", "20251221", "20251223", "20260114", "20260115", "20260116"]

tutti_file_fits = []
nomi_run_processate = []

if astri1 is None:
    print("ERRORE: Cartella ASTRI1 non trovata nel PC!")
    exit()

print(f"Cerco le cartelle specificate all'interno di {astri1}, potrebbe volerci del tempo...")

# esploro esclusivamente la mia directory astri1
for root, dirs, files in os.walk(astri1):
    # estraggo il nome della cartella corrente
    nome_cartella = os.path.basename(root)

    # verifico se la cartella corrente è una di quelle target
    if nome_cartella in cartelle_target:
        # cerco tutti i file FITS all'interno della cartella e li ordino alfabeticamente
        estensioni_valide = ['.fit', '.fits']
        file_run = sorted(
            [os.path.join(root, f) for f in files if os.path.splitext(f)[1].lower() in estensioni_valide])

        # salto la prima e le ultime due immagini della singola cartella per evitare scarti
        if len(file_run) > 3:
            file_run_validi = file_run[1:-2]
            tutti_file_fits.extend(file_run_validi)
            nomi_run_processate.append(nome_cartella)

if not tutti_file_fits:
    print("ERRORE: Nessun file FITS valido trovato nelle cartelle specificate!")
    exit()

print(f"Trovate {len(tutti_file_fits)} immagini valide in {len(nomi_run_processate)} cartelle totali.")

# =============================================================================
# 2. DEFINIZIONE DEL SISTEMA DI RIFERIMENTO GLOBALE (NO ZOOM)
# =============================================================================
# prendo un'immagine centrale (o la prima disponibile) come riferimento
indice_ref = min(12, len(tutti_file_fits) - 1)
print(f"Caricamento riferimento: {Path(tutti_file_fits[indice_ref]).name}")

hdu_ref = fits.open(tutti_file_fits[indice_ref])[0]
target_header_full = hdu_ref.header.copy()

# aggiungo relax=True per leggere i polinomi di distorsione senza warning
target_wcs = WCS(target_header_full, relax=True)

# utilizzo le dimensioni e il WCS dell'immagine originale intera senza ritagliare
target_shape = hdu_ref.data.shape
target_header = target_wcs.to_header()
target_header['DATE-OBS'] = target_header_full.get('DATE-OBS', 'UNKNOWN')

# creo la mia matrice finale (accumulatore) piena di zeri
final_image_sum = np.zeros(target_shape)
coverage_map = np.zeros(target_shape)

print(f"Inizio stacking globale sull'intero campo di vista...")
print(f"Dimensioni target: {target_shape} pixel")

# =============================================================================
# 3. LOOP DI RIPROIEZIONE E STACKING GLOBALE
# =============================================================================
for percorso_file_fits in tqdm(tutti_file_fits, desc="Stacking", unit="img"):
    try:
        with fits.open(percorso_file_fits) as hdu_list:
            data = hdu_list[0].data
            header = hdu_list[0].header

            # aggiungo relax=True anche qui per gestire le distorsioni dell'immagine corrente
            wcs_input = WCS(header, relax=True)

            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            print("Mediana: ", median)

            # Sottraggo il fondo
            data_sub = data - median

            # riproietto direttamente sul target_wcs dell'intera immagine
            array_reprojected, footprint = reproject_interp(
                (data_sub, wcs_input),
                target_wcs,
                shape_out=target_shape
            )

            # converto i NaN in zeri per permettere la somma matriciale
            array_reprojected = np.nan_to_num(array_reprojected, nan=0.0)
            final_image_sum += array_reprojected
            coverage_map += np.nan_to_num(footprint, nan=0.0)

    except Exception as e:
        tqdm.write(f"Errore nel file {Path(percorso_file_fits).name}: {e}")

# =============================================================================
# 4. SCALATURA E SALVATAGGIO
# =============================================================================
max_coverage = np.max(coverage_map)

# calcolo e applico il fattore di scala per uniformare l'immagine sui bordi di copertura
scale_factor_map = np.zeros_like(coverage_map, dtype=float)
np.divide(max_coverage, coverage_map,
          out=scale_factor_map,
          where=coverage_map > 0)

final_image_sum = final_image_sum * scale_factor_map

# imposto la mia cartella di output
output_dir = cerca_cartella_nel_progetto(BASE_DIR, "grafici")
output_dir.mkdir(parents=True, exist_ok=True)

output_filename = output_dir / 'stacked_sum_mrk421_COLOSSALE.fits'
header_finale = target_header.copy()
header_finale[
    'HISTORY'] = f'Stacking intero campo di vista ({len(tutti_file_fits)} immagini da {len(nomi_run_processate)} run)'

fits.writeto(str(output_filename), final_image_sum, header_finale, overwrite=True)
print(f"\nFatto! Immagine salvata come: {output_filename.name}")

coverage_filename = output_dir / 'coverage_map_mrk421_COLOSSALE.fits'
header_coverage = target_header.copy()
header_coverage['HISTORY'] = 'Mappa di copertura globale (numero di immagini per pixel)'

fits.writeto(str(coverage_filename), coverage_map, header_coverage, overwrite=True)
print(f"Fatto! Coverage map salvata come: {coverage_filename.name}")

# =============================================================================
# 5. VISUALIZZAZIONE VELOCE (CON REGOLAZIONE CONTEGGI)
# =============================================================================
# calcolo le statistiche sull'immagine totale
mean, median, std = sigma_clipped_stats(final_image_sum, sigma=3.0)
print(f"Mediana totale globale: {median}")

# sottraggo il fondo mediano
data_finale = final_image_sum - median

norm = simple_norm(data_finale, 'sqrt')
plt.figure(figsize=(10, 8))
ax = plt.subplot(projection=target_wcs)

# genero la mia immagine
im = ax.imshow(data_finale, origin='lower', norm=norm, cmap='viridis', interpolation='nearest')

# cerco la prima tabella catalogata
dir_tabelle_cat = BASE_DIR / "tabelle_blazar" / "tabelle_cataloghi"
if dir_tabelle_cat.exists():
    tabelle_csv = sorted(list(dir_tabelle_cat.rglob("*.csv")))
    if tabelle_csv:
        prima_tabella = tabelle_csv[0]
        print(f"Sovrappongo lo scatter della tabella: {prima_tabella.name}")

        # leggo i dati del catalogo
        df_cat = pd.read_csv(prima_tabella, comment='#')

        # identifico le colonne delle coordinate astronomiche
        col_ra = 'RAJ2000' if 'RAJ2000' in df_cat.columns else (
            'RA_centroid' if 'RA_centroid' in df_cat.columns else 'RA')
        col_dec = 'DEJ2000' if 'DEJ2000' in df_cat.columns else (
            'DEC_centroid' if 'DEC_centroid' in df_cat.columns else 'DEC')

        if col_ra in df_cat.columns and col_dec in df_cat.columns:
            # creo un array di coordinate celesti dal catalogo
            cat_coords = SkyCoord(ra=df_cat[col_ra].values * u.deg, dec=df_cat[col_dec].values * u.deg, frame='icrs')

            # converto le coordinate in pixel relativi alla mia immagine
            x_pix, y_pix = target_wcs.world_to_pixel(cat_coords)

            # estraggo le dimensioni della mia immagine
            ny, nx = data_finale.shape

            # creo la mia maschera logica per tenere solo le stelle che cadono dentro il riquadro dell'immagine
            mask_inside = (x_pix >= 0) & (x_pix <= nx) & (y_pix >= 0) & (y_pix <= ny)

            # filtro il dataframe usando la mia maschera
            df_cat_filtered = df_cat[mask_inside]

            print(f"Trovate {len(df_cat_filtered)} stelle di catalogo all'interno del campo visivo.")

            # applico il transform='world' per allineare direttamente RA/DEC all'immagine
            ax.scatter(df_cat_filtered[col_ra], df_cat_filtered[col_dec], transform=ax.get_transform('world'),
                       s=4, color='red', label='Catalogo (nel riquadro)', zorder=10)
            plt.legend()

plt.colorbar(im, label='Counts (Somma Totale)')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title(f'Stacking Globale Intero Campo (Copertura max={int(max_coverage)})')

output_png = output_dir / 'stacking_mrk421_COLOSSALE.png'
plt.savefig(str(output_png))