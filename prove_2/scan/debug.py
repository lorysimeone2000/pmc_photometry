import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve
from matplotlib.colors import LogNorm
from photutils.segmentation import make_2dgaussian_kernel, SourceFinder, SourceCatalog
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from matplotlib.patches import Rectangle
import warnings
from astropy.wcs import FITSFixedWarning

# --- CONFIGURAZIONE E SOPPRESSIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- PARAMETRI ---
RUN = 1

# PERCORSI
BASE_PATH = "/home/lorysimeone/tesi_magistrale/prove_2"
FILE_LISTA_PATH = os.path.join(BASE_PATH, f"liste_percorsi_run/lista_immagini_run_{RUN}.txt")
FILE_PARAMETRI = '/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt'


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


def esegui_segmentazione_dinamica(data, fwhm, size, params):
    """
    Esegue la segmentazione.
    CORREZIONE: Estrae i valori specifici dal dizionario params.
    """
    # 1. Convoluzione
    try:
        kernel = make_2dgaussian_kernel(fwhm, size=size)
        convolved_data = convolve(data, kernel)
    except Exception as e:
        return None

    # 2. Source Finder
    # CORREZIONE: Qui passiamo solo il valore intero, non tutto il dizionario
    finder = SourceFinder(npixels=params['pixel'], progress_bar=False)

    # CORREZIONE: Qui passiamo solo il valore float della soglia
    segment_map = finder(convolved_data, params['threshold_assoluta'])

    if segment_map is None:
        return None

    # 3. Catalogo
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    # 4. Filtraggio
    indici_validi = []
    for i, sorgente in enumerate(tbl):
        label = sorgente['label']
        mask_sorgente = (segment_map.data == label)
        valori_originali = data[mask_sorgente]

        pixel_sopra_soglia_ass = np.sum(valori_originali > params['soglia_filtro_ass'])
        pixel_sopra_soglia_rel = np.sum(valori_originali > params['soglia_filtro_rel'] * sorgente['max_value'])

        if pixel_sopra_soglia_ass >= 2 and pixel_sopra_soglia_rel >= 2:
            indici_validi.append(i)

    return tbl[indici_validi]


# --- MAIN ---

print(f"=== DIAGNOSTICA IMMAGINI MANCANTI (Run {RUN}) ===")

# 1. Caricamento Parametri da file
try:
    parametri = {}
    with open(FILE_PARAMETRI, 'r') as file:
        next(file)  # Salta intestazione
        for riga in file:
            riga = riga.strip()
            if riga and not riga.startswith('#'):
                parts = riga.split()
                if len(parts) >= 2:
                    parametro, valore = parts[0], parts[1]
                    parametri[parametro] = float(valore) if '.' in valore else int(valore)

    DEBUG_FWHM = parametri.get('fwhm', 3.0)
    DEBUG_SIZE = parametri.get('size', 5)
    DEBUG_THRESH = parametri.get('threshold_assoluta', 3.61)

    print(f"Parametri caricati: FWHM={DEBUG_FWHM}, Size={DEBUG_SIZE}, Thresh={DEBUG_THRESH}")

except Exception as e:
    print(f"ATTENZIONE: Errore lettura file parametri ({e}). Uso valori default.")
    DEBUG_FWHM = 3.0
    DEBUG_SIZE = 5
    DEBUG_THRESH = 3.61

# Definiamo il dizionario completo per la funzione
PARAMETRI_COMPLETI = {
    'pixel': 3,
    'soglia_filtro_ass': 2.0,
    'soglia_filtro_rel': 0.4,
    'threshold_assoluta': DEBUG_THRESH  # Aggiunto qui per coerenza
}

# 2. Caricamento Lista File
try:
    file_paths = get_file_list(FILE_LISTA_PATH)
    print(f"Caricati {len(file_paths)} file.")
except FileNotFoundError:
    print(f"ERRORE: File lista non trovato.")
    exit()

# --- FASE 1: Identificazione Target (Reference) ---
print(f"\n--- FASE 1: Identificazione Target ---")

IMG_RIFERIMENTO_IDX = 35
TARGET_PIXEL_X = 1891.7445408352592
TARGET_PIXEL_Y = 1061.437287743088
SOGLIA_CORRELAZIONE = 0.003349 * u.deg

# Se il file hardcoded non esiste, usa quello della lista
path_ref_hardcoded = "/home/lorysimeone/tesi_magistrale/20250120_run1/20250120_213945.fits"
path_ref = path_ref_hardcoded if os.path.exists(path_ref_hardcoded) else file_paths[IMG_RIFERIMENTO_IDX]

data_ref, wcs_ref, header_ref = elabora_file_fits(path_ref)

# Esecuzione sulla reference image
# Ora la funzione riceverà il dizionario corretto
tbl_ref = esegui_segmentazione_dinamica(data_ref, fwhm=3.0, size=5, params=PARAMETRI_COMPLETI)

if tbl_ref is None or len(tbl_ref) == 0:
    print("ERRORE CRITICO: Non trovo nessuna stella nell'immagine di riferimento!")
    exit()

# Identificazione stella target
distanze_pixel = np.sqrt((tbl_ref['xcentroid'] - TARGET_PIXEL_X) ** 2 + (tbl_ref['ycentroid'] - TARGET_PIXEL_Y) ** 2)
idx_target = np.argmin(distanze_pixel)
stella_target = tbl_ref[idx_target]
coord_target_world = wcs_ref.pixel_to_world(stella_target['xcentroid'], stella_target['ycentroid'])

print(f"Target Agganciato: RA={coord_target_world.ra.deg:.5f}, DEC={coord_target_world.dec.deg:.5f}")

# --- FASE 2: Diagnostica Loop ---
print(f"\n--- FASE 2: Ricerca errori ---")
immagini_fallite = []

for i, path in enumerate(file_paths):
    # Caricamento
    data, wcs, header = elabora_file_fits(path)

    # 1. Coordinate Target
    coord_target_pixel = wcs.world_to_pixel(coord_target_world)
    xc, yc = coord_target_pixel[1], coord_target_pixel[0]  # xc=rows, yc=cols

    # 2. Creazione Riquadro
    x_min, x_max = int(xc - 15), int(xc + 15)
    y_min, y_max = int(yc - 15), int(yc + 15)

    # Controllo bordi
    if x_min < 0 or y_min < 0 or x_max > data.shape[0] or y_max > data.shape[1]:
        print(f"IMG {i}: FALLITO - Il target è troppo vicino al bordo!")
        immagini_fallite.append(i)
        continue

    riquadro = data[x_min:x_max, y_min:y_max]

    # 3. Segmentazione (Manuale per il debug, usando variabili esplicite)
    kernel = make_2dgaussian_kernel(DEBUG_FWHM, size=DEBUG_SIZE)
    convolved_data = convolve(riquadro, kernel)

    # Qui usiamo il valore diretto, non il dizionario
    finder = SourceFinder(npixels=PARAMETRI_COMPLETI['pixel'], progress_bar=False)
    segment_map = finder(convolved_data, DEBUG_THRESH)

    found = False
    motif_failure = "Nessuna sorgente nel riquadro (soglia non superata)"

    if segment_map is not None:
        cat = SourceCatalog(riquadro, segment_map, convolved_data=convolved_data)
        tbl = cat.to_table()

        # Filtro Custom
        indici_validi = []
        for idx, sorgente in enumerate(tbl):
            label = sorgente['label']
            mask = (segment_map.data == label)
            vals = riquadro[mask]

            # Applicazione filtro post-processing
            if (np.sum(vals > PARAMETRI_COMPLETI['soglia_filtro_ass']) >= 2):
                indici_validi.append(idx)

        tbl = tbl[indici_validi]

        if len(tbl) > 0:
            # Controllo Distanza
            x_loc = tbl['xcentroid']
            y_loc = tbl['ycentroid']
            coords_trovate = wcs.pixel_to_world(x_loc + y_min, y_loc + x_min)
            distanze = coords_trovate.separation(coord_target_world)

            if np.min(distanze) <= SOGLIA_CORRELAZIONE:
                found = True
            else:
                motif_failure = f"Sorgenti trovate ma lontane (Min Dist: {np.min(distanze).deg:.5f} deg)"
        else:
            motif_failure = "Sorgenti trovate ma eliminate dal filtro di qualità"

    if not found:
        immagini_fallite.append(i)
        print(f"\n>>> FALLIMENTO IMG {i} ({os.path.basename(path)})")
        print(f"    Motivo: {motif_failure}")

        # --- PLOT DIAGNOSTICO ---
        plt.figure(figsize=(10, 5))

        # Subplot 1: Il Riquadro "Visto" dall'algoritmo
        plt.subplot(1, 2, 1)
        plt.title(f"Riquadro $30\\times30$ pixel\nImg Index {i}")
        plt.imshow(riquadro, origin='lower', cmap='gray_r', norm=LogNorm())
        # Centro teorico (dove il WCS dice che dovrebbe essere la stella)
        plt.plot(15, 15, 'rx', markersize=10, label='Centro Atteso')
        plt.legend()

        # Subplot 2: L'immagine INTERA (Zoomata) per il contesto
        plt.subplot(1, 2, 2)
        plt.title("Contesto Globale (Zoom)")

        # Mostra un'area più grande (es. 100x100) attorno al target
        big_xmin, big_xmax = int(xc - 50), int(xc + 50)
        big_ymin, big_ymax = int(yc - 50), int(yc + 50)

        # Gestione bordi per il plot
        big_xmin = max(0, big_xmin)
        big_ymin = max(0, big_ymin)

        # Ritaglio grande
        big_riquadro = data[big_xmin:big_xmax, big_ymin:big_ymax]

        plt.imshow(big_riquadro, origin='lower', cmap='viridis', norm=LogNorm())

        # Disegna il box rosso che stiamo analizzando
        rect_x = (y_min - big_ymin)
        rect_y = (x_min - big_xmin)
        rect = Rectangle((rect_x, rect_y), 30, 30, linewidth=2, edgecolor='red', facecolor='none',
                         label='Riquadro Analisi')
        plt.gca().add_patch(rect)
        plt.legend()

        plt.tight_layout()
        plt.show()

print(f"\nTotale immagini fallite: {len(immagini_fallite)}")