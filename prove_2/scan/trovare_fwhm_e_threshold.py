import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel, SourceFinder, SourceCatalog
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import warnings
from astropy.wcs import FITSFixedWarning

start_time = time.time()

# --- CONFIGURAZIONE ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- PARAMETRI ---
RUN = 1
IMG_RIFERIMENTO_IDX = 35
SOGLIA_CORRELAZIONE = 0.003349 * u.deg
FIXED_SIZE = 5  # Parametro fissato come richiesto

# Coordinate pixel target (ricalibrate o fisse)
TARGET_PIXEL_X = 1891.74
TARGET_PIXEL_Y = 1061.43

# --- GRIGLIA DI RICERCA ---
# 1. FWHM: Range ragionevole per il seeing
FWHM_RANGE = np.linspace(1.5, 5.0, 15)

# 2. THRESHOLD: Poiché i dati sono 0-254 (post background), esploriamo la parte bassa
# Partiamo da 2.0 (appena sopra il rumore tipico) fino a 15.0 (segnale chiaro)
THRESHOLD_RANGE = np.linspace(0.1, 10, 15)

# Parametri secondari (Filtri post-segmentazione)
# Nota: soglia_filtro_ass deve essere inferiore o uguale alla threshold minima testata per non interferire
PARAMETRI_FISSI = {
    'pixel': 3,
    'soglia_filtro_ass': 2.0,
    'soglia_filtro_rel': 0.4
}

# PERCORSI
BASE_PATH = "/home/lorysimeone/tesi_magistrale/prove_2"
FILE_LISTA_PATH = os.path.join(BASE_PATH, f"liste_percorsi_run/lista_immagini_run_{RUN}.txt")


# --- FUNZIONI ---
def get_file_list(list_path):
    with open(list_path, 'r') as file:
        return file.read().splitlines()


def elabora_file_fits(percorso_file):
    with fits.open(percorso_file) as hdu:
        header = hdu[0].header
        data = hdu[0].data
        wcs = WCS(header)
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        data_sub = data - median
        return data_sub, wcs, header


def segmentazione_veloce(convolved_data, data_originale, threshold, params):
    """
    Esegue solo la parte di thresholding e filtraggio.
    Accetta data_originale per il controllo dei filtri post-processing.
    """
    # 1. Source Finder
    finder = SourceFinder(npixels=params['pixel'], progress_bar=False)
    # Passiamo la soglia variabile qui
    segment_map = finder(convolved_data, threshold)

    if segment_map is None:
        return None

    # 2. Catalogo
    cat = SourceCatalog(data_originale, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    # 3. Filtraggio (Logica utente mantenuta)
    indici_validi = []
    for i, sorgente in enumerate(tbl):
        label = sorgente['label']
        mask_sorgente = (segment_map.data == label)
        valori_originali = data_originale[mask_sorgente]

        pixel_sopra_soglia_ass = np.sum(valori_originali > params['soglia_filtro_ass'])
        pixel_sopra_soglia_rel = np.sum(valori_originali > params['soglia_filtro_rel'] * sorgente['max_value'])

        if pixel_sopra_soglia_ass >= 2 and pixel_sopra_soglia_rel >= 2:
            indici_validi.append(i)

    return tbl[indici_validi]


# --- MAIN ---

print(f"=== ANALISI FWHM vs THRESHOLD (Run {RUN}) ===")
print(f"Fixed Kernel Size: {FIXED_SIZE}")
print(f"Thresholds: {THRESHOLD_RANGE}")

# Caricamento lista
file_paths = get_file_list(FILE_LISTA_PATH)
if not file_paths: exit()

# Identificazione Target (Fase Preliminare)
path_ref = file_paths[IMG_RIFERIMENTO_IDX]
data_ref, wcs_ref, _ = elabora_file_fits(path_ref)

# Kernel base per trovare il target
k_ref = make_2dgaussian_kernel(3.0, size=5)
data_conv_ref = convolve(data_ref, k_ref)
tbl_ref = segmentazione_veloce(data_conv_ref, data_ref, threshold=3.61, params=PARAMETRI_FISSI)

# Trova coordinate
dist_pix = np.sqrt((tbl_ref['xcentroid'] - TARGET_PIXEL_X) ** 2 + (tbl_ref['ycentroid'] - TARGET_PIXEL_Y) ** 2)
stella_target = tbl_ref[np.argmin(dist_pix)]
coord_target_world = wcs_ref.pixel_to_world(stella_target['xcentroid'], stella_target['ycentroid'])
print(f"Target locked at RA: {coord_target_world.ra.deg:.4f}")

# Inizializzazione Matrice: Righe=Threshold, Colonne=FWHM
matrice_conteggi = np.zeros((len(THRESHOLD_RANGE), len(FWHM_RANGE)), dtype=int)

print(f"\n--- INIZIO LOOP ---")

for i, path in enumerate(file_paths):
    try:
        data, wcs, _ = elabora_file_fits(path)

        # Ritaglio intelligente intorno al target
        cpixel = wcs.world_to_pixel(coord_target_world)
        xc, yc = cpixel[1], cpixel[0]  # xc=rows, yc=cols

        x_min, x_max = int(xc - 15), int(xc + 15)
        y_min, y_max = int(yc - 15), int(yc + 15)

        # Check boundaries
        if x_min < 0 or y_min < 0: continue
        riquadro = data[x_min:x_max, y_min:y_max]

        # --- OPTIMIZATION CORE ---
        # Loop esterno: FWHM (per minimizzare le convoluzioni)
        for f_idx, fwhm_val in enumerate(FWHM_RANGE):

            # 1. Convoluzione (Costosa: farla una sola volta per FWHM)
            kernel = make_2dgaussian_kernel(fwhm_val, size=FIXED_SIZE)
            riquadro_conv = convolve(riquadro, kernel)

            # Loop interno: Threshold (Veloce)
            for t_idx, thresh_val in enumerate(THRESHOLD_RANGE):

                tbl_sorgenti = segmentazione_veloce(riquadro_conv, riquadro, thresh_val, PARAMETRI_FISSI)

                if tbl_sorgenti is None or len(tbl_sorgenti) == 0:
                    continue

                # Verifica posizionale
                x_loc, y_loc = tbl_sorgenti['xcentroid'], tbl_sorgenti['ycentroid']
                coords_t = wcs.pixel_to_world(x_loc + y_min, y_loc + x_min)

                if np.min(coords_t.separation(coord_target_world)) <= SOGLIA_CORRELAZIONE:
                    matrice_conteggi[t_idx, f_idx] += 1

    except Exception as e:
        print(f"Err img {i}: {e}")
        continue

    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(file_paths)}", end='\r')

# --- SALVATAGGIO ---
print("\nSalvataggio CSV...")
records = []
for t_idx, t_val in enumerate(THRESHOLD_RANGE):
    for f_idx, f_val in enumerate(FWHM_RANGE):
        records.append({
            'Threshold': t_val,
            'FWHM': f_val,
            'Detected_Count': matrice_conteggi[t_idx, f_idx]
        })

pd.DataFrame(records).to_csv(os.path.join(BASE_PATH, f"grid_thresh_fwhm_run_{RUN}.csv"), index=False)

# --- PLOTTING ---
plt.figure(figsize=(12, 8))
img = plt.imshow(matrice_conteggi, interpolation='nearest', cmap='inferno', origin='lower', aspect='auto')

plt.xticks(np.arange(len(FWHM_RANGE)), [f"{x:.1f}" for x in FWHM_RANGE], rotation=45)
plt.yticks(np.arange(len(THRESHOLD_RANGE)), [f"{x:.1f}" for x in THRESHOLD_RANGE])

plt.xlabel('FWHM (pixel)', fontsize=12)
plt.ylabel('Threshold Assoluta (ADU)', fontsize=12)
plt.title(f'Robustezza Rilevamento: Threshold vs FWHM (Size={FIXED_SIZE})', fontsize=14)
plt.colorbar(img, label='Numero Immagini Rilevate')

# Annotazioni
for i in range(len(THRESHOLD_RANGE)):
    for j in range(len(FWHM_RANGE)):
        val = matrice_conteggi[i, j]
        col = "white" if val < np.max(matrice_conteggi) / 2 else "black"  # Invertito per cmap inferno (chiara in alto)
        plt.text(j, i, str(val), ha="center", va="center", color=col, fontsize=7)

plt.tight_layout()
plt.show()

print(f"Finito in {(time.time() - start_time):.2f}s")