import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from reproject import reproject_interp  # Assicurati di aver fatto: pip install reproject
import warnings
from astropy.wcs import FITSFixedWarning
from tqdm import tqdm
import os

warnings.filterwarnings('ignore', category=FITSFixedWarning)

# --- SETUP ---
run = 1
# Percorso del file lista
path_lista = f'/home/lorysimeone/tesi_magistrale/prove_2/liste_percorsi_run/lista_immagini_run_{run}.txt'

with open(path_lista, 'r') as file:
    file_list = file.read().splitlines()

if not file_list:
    raise ValueError("La lista dei file è vuota!")

# --- PASSO 1: DEFINIRE IL SISTEMA DI RIFERIMENTO (CANVAS) ---
# Usiamo la prima immagine come riferimento per il WCS e la dimensione finale.
immagine_di_riferimento=12
print(f"Caricamento riferimento: {file_list[immagine_di_riferimento]}") # prendo come riferimento un'immagine più stabile
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
    i=i+1
    if i==1 or i==len(file_list)-2 or i==len(file_list)-1: continue
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

output_dir = f'/home/lorysimeone/tesi_magistrale/prove_2/stacking/'

# 1. Salvataggio Immagine Sommata
output_filename = os.path.join(output_dir, f'run_{run}_stacked_sum.fits')
header_finale = target_header.copy() # Copiamo l'header per non modificare l'originale
header_finale['HISTORY'] = f'Immagine ottenuta sommando N esposizioni riproiettate con reproject usando come riferimento l\' immagine {immagine_di_riferimento}'

fits.writeto(output_filename, final_image_sum, header_finale, overwrite=True)
print(f"Fatto! Immagine salvata come: {output_filename}")

# 2. Salvataggio Coverage Map
coverage_filename = os.path.join(output_dir, f'run_{run}_coverage_map.fits')
header_coverage = target_header.copy()
header_coverage['HISTORY'] = 'Mappa di copertura (numero di immagini per pixel)'

# La coverage map usa lo stesso WCS dell'immagine, così puoi sovrapporle in DS9
fits.writeto(coverage_filename, coverage_map, header_coverage, overwrite=True)
print(f"Fatto! Coverage map salvata come: {coverage_filename}")

# --- PASSO 4: VISUALIZZAZIONE VELOCE ---
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

norm = simple_norm(final_image_sum, 'sqrt')
plt.figure(figsize=(10, 10))
plt.subplot(projection=target_wcs)
plt.imshow(final_image_sum, origin='lower', norm=norm, cmap='viridis')
plt.colorbar(label='Counts (Sum)')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title(f'Stacking Run {run}')
plt.show()