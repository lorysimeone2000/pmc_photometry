import numpy as np
import os
import sys
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from reproject import reproject_interp  # Assicurati di aver fatto: pip install reproject
import warnings
from astropy.wcs import FITSFixedWarning
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FITSFixedWarning)


# =============================================================================
# FUNZIONI DI GESTIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    # cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


# Trovo la cartella base del mio progetto
BASE_DIR = trova_cartella_base("Lorenzo")

PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")
if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

from funzioni.utilita import *
from funzioni.astrometria import *

# --- SETUP ---
run = 1
nome_cartella_run = f"20250120_run{run}"

# Costruisco il percorso della cartella che contiene le immagini FITS
cartella_run = BASE_DIR / "pmc_photometry/run_vecchie" / nome_cartella_run

# Verifico che la cartella esista
if not cartella_run.exists():
    raise FileNotFoundError(f"La cartella {cartella_run} non esiste!")

# Cerco tutti i file FITS all'interno della cartella della run e li ordino alfabeticamente
estensioni_valide = ['.fit', '.fits']
file_list = sorted([f for f in cartella_run.rglob('*') if f.suffix.lower() in estensioni_valide and f.is_file()])

# Trasformo in stringhe per compatibilità con il resto del codice
file_list = [str(f) for f in file_list]

if not file_list:
    raise ValueError(f"Nessun file FITS trovato nella cartella {cartella_run}!")

print(f"Trovati {len(file_list)} file FITS nella cartella della run {run}.")

# --- PASSO 1: DEFINIRE IL SISTEMA DI RIFERIMENTO (CANVAS) ---
# Usiamo la prima immagine come riferimento per il WCS e la dimensione finale.
immagine_di_riferimento = 12
if immagine_di_riferimento >= len(file_list):
    print("Indice immagine di riferimento fuori dai limiti, uso la prima immagine (0).")
    immagine_di_riferimento = 0

print(
    f"Caricamento riferimento: {file_list[immagine_di_riferimento]}")  # prendo come riferimento un'immagine più stabile
hdu_ref = fits.open(file_list[immagine_di_riferimento])[0]
target_header = hdu_ref.header.copy()
target_wcs = WCS(target_header)
target_shape = hdu_ref.data.shape

# Creiamo la matrice finale (accumulatore) piena di zeri
final_image_sum = np.zeros(target_shape)

# (Opzionale) Matrice di copertura se volessi fare la media invece della somma
coverage_map = np.zeros(target_shape)

print(f"Inizio stacking di {len(file_list)} immagini...")
print(f"Dimensioni target: {target_shape}")

i = 0

# --- PASSO 2: LOOP E RIPROIEZIONE ---
for percorso_file_fits in tqdm(file_list, desc="Stacking", unit="img"):
    i += 1
    # Salto il primo e gli ultimi due file come da tuo script originale
    if i == 1 or i == len(file_list) - 2 or i == len(file_list) - 1: continue

    try:
        with fits.open(percorso_file_fits) as hdu_list:
            # Caricamento dati
            data = hdu_list[0].data
            header = hdu_list[0].header
            wcs_input = WCS(header)

            # Non sottraggo il fondo
            data_sub = data

            # --- RIPROIEZIONE ---
            # Riproietta l'immagine corrente (data_sub) sul sistema di coordinate dell'immagine di riferimento (target_wcs)
            array_reprojected, footprint = reproject_interp(
                (data_sub, wcs_input),
                target_wcs,
                shape_out=target_shape
            )

            # reproject mette NaN dove l'immagine non si sovrappone.
            # Convertiamo i NaN in 0 per poter sommare.
            array_reprojected = np.nan_to_num(array_reprojected, nan=0.0)

            # Somma alla matrice finale
            final_image_sum += array_reprojected

            # (Opzionale) Aggiorna la mappa di copertura
            coverage_map += np.nan_to_num(footprint, nan=0.0)

    except Exception as e:
        tqdm.write(f"Errore nel file {percorso_file_fits}: {e}")

max_coverage = np.max(coverage_map)

# 2. Inizializza l'immagine scalata
final_image_scaled = final_image_sum.copy()

# 3. Identifica i pixel da scalare (copertura < massima ma > 0)
# Copertura Massima: coverage_map == max_coverage (il fattore sarà 1, quindi invariati)
# Copertura Parziale: 0 < coverage_map < max_coverage
# Copertura Zero: coverage_map == 0 (non li tocchiamo)

# Calcola l'array dei fattori di scala: max_coverage / coverage_map
# Usiamo np.divide con 'where' per evitare la divisione per zero e i calcoli non necessari
scale_factor_map = np.zeros_like(coverage_map, dtype=float)

np.divide(max_coverage, coverage_map,
          out=scale_factor_map,
          where=coverage_map > 0)

# 4. Applica il fattore di scala all'immagine somma
# Moltiplichiamo l'immagine somma per la mappa dei fattori di scala.
# Dove coverage_map == max_coverage, il fattore è 1, quindi final_image_scaled rimane invariato.
# Dove coverage_map < max_coverage (ma > 0), il fattore è > 1 e l'intensità viene aumentata.
# Dove coverage_map == 0, il fattore è 0, quindi il valore rimane 0 (come nell'originale).
final_image_sum = final_image_sum * scale_factor_map

# --- PASSO 3: SALVATAGGIO ---

# Imposto la cartella di output dinamicamente basandomi su BASE_DIR
output_dir = BASE_DIR / 'pmc_photometry' / 'prove_2' / 'stacking'
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Salvataggio Immagine Sommata
output_filename = output_dir / f'run_{run}_stacked_sum.fits'
header_finale = target_header.copy()  # Copiamo l'header per non modificare l'originale
header_finale[
    'HISTORY'] = f'Immagine ottenuta sommando N esposizioni riproiettate con reproject usando come riferimento l\' immagine {immagine_di_riferimento}'

fits.writeto(str(output_filename), final_image_sum, header_finale, overwrite=True)
print(f"Fatto! Immagine salvata come: {output_filename}")

# 2. Salvataggio Coverage Map
coverage_filename = output_dir / f'run_{run}_coverage_map.fits'
header_coverage = target_header.copy()
header_coverage['HISTORY'] = 'Mappa di copertura (numero di immagini per pixel)'

# La coverage map usa lo stesso WCS dell'immagine, così puoi sovrapporle in DS9
fits.writeto(str(coverage_filename), coverage_map, header_coverage, overwrite=True)
print(f"Fatto! Coverage map salvata come: {coverage_filename}")

# --- PASSO 4: VISUALIZZAZIONE VELOCE ---
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

# mi baso sul massimo e sul minimo dell'intero catalogo per uniformare la colorbar
vmin_cat = df_cat['Mag'].min()
vmax_cat = df_cat['Mag'].max()
scatter_cat = ax.scatter(x_cat_cutout, y_cat_cutout, c=mag_cat_cutout, cmap='viridis_r', s=15,
                         vmin=vmin_cat, vmax=vmax_cat, zorder=5)
# aggiungo la seconda colorbar dedicata alle magnitudini del catalogo
plt.colorbar(scatter_cat, ax=ax, label='Magnitudine catalogo', fraction=0.046, pad=0.04)

norm = simple_norm(final_image_sum, 'sqrt')
plt.figure(figsize=(10, 10))
plt.subplot(projection=target_wcs)
plt.imshow(final_image_sum, origin='lower', norm=norm, cmap='viridis')
plt.colorbar(label='Counts (Sum)')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title(f'Stacking Run {run}')
plt.savefig(str(f'run_{run}_stack_grab.png'))
plt.show()