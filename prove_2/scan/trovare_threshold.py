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
import astropy.units as u
import warnings
from astropy.wcs import FITSFixedWarning

# Registra il tempo di inizio
start_time = time.time()

# --- CONFIGURAZIONE E SOPPRESSIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- PARAMETRI ---
RUN = 1
IMG_RIFERIMENTO_IDX = 35
SOGLIA_CORRELAZIONE = 0.003349 * u.deg

# Coordinate pixel della stella target
TARGET_PIXEL_X = 1891.7445408352592
TARGET_PIXEL_Y = 1061.437287743088

# --- MODIFICA: PARAMETRI DI TEST ---
# Fissiamo la FWHM
FIXED_FWHM = 3.0

# Definiamo il range di Threshold da testare
# Il valore di partenza era 3.61. Creiamo un range intorno a questo valore (es. da 1.5 a 6.5)
THRESHOLD_RANGE = np.linspace(1.5, 6.5, 15)

# Parametri fissi di segmentazione (il valore 'threshold_assoluta' verrà sovrascritto nel loop)
PARAMETRI_BASE = {
    'size': 5,
    'pixel': 3,
    'soglia_filtro_ass': 2.5,
    'soglia_filtro_rel': 0.4,
    'threshold_assoluta': 3.61  # Valore di default per la fase di identificazione
}

# PERCORSI
BASE_PATH = "/home/lorysimeone/tesi_magistrale/prove_2"
FILE_LISTA_PATH = os.path.join(BASE_PATH, f"liste_percorsi_run/lista_immagini_run_{RUN}.txt")


# --- FUNZIONI ---

def get_file_list(list_path):
    """Legge la lista dei file FITS"""
    with open(list_path, 'r') as file:
        return file.read().splitlines()


def elabora_file_fits(percorso_file):
    """Apre il file FITS e sottrae il background"""
    with fits.open(percorso_file) as hdu:
        header = hdu[0].header
        data = hdu[0].data
        wcs = WCS(header)

        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        data_sub = data - median
        return data_sub, wcs


def esegui_segmentazione(data, fwhm, params):
    """
    Esegue la segmentazione (SourceFinder) usando i parametri forniti.
    """
    # 1. Convoluzione
    kernel = make_2dgaussian_kernel(fwhm, size=params['size'])
    convolved_data = convolve(data, kernel)

    # 2. Source Finder (usa la threshold passata nel dizionario params)
    finder = SourceFinder(npixels=params['pixel'], progress_bar=False)
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

print(f"=== INIZIO ANALISI RUN {RUN} (Variazione Threshold) ===")

# 1. Caricamento lista file
try:
    file_paths = get_file_list(FILE_LISTA_PATH)
    print(f"Caricati {len(file_paths)} percorsi file.")
except FileNotFoundError:
    print(f"ERRORE: Non trovo il file lista in {FILE_LISTA_PATH}")
    exit()

# 2. Identificazione Stella Target (Fase Preliminare STANDARD)
# Usiamo i parametri base (FWHM=3.0, Thresh=3.61) per trovare dov'è la stella
print(f"\n--- FASE 1: Identificazione Target (Setup Standard) ---")
path_ref = "/home/lorysimeone/tesi_magistrale/20250120_run1/20250120_213945.fits"

if not os.path.exists(path_ref):
    print(f"ATTENZIONE: File hardcoded non trovato: {path_ref}")
    print(f"Uso il file all'indice {IMG_RIFERIMENTO_IDX} della lista.")
    path_ref = file_paths[IMG_RIFERIMENTO_IDX]

data_ref, wcs_ref = elabora_file_fits(path_ref)

# Segmentazione di riferimento
tbl_ref = esegui_segmentazione(data_ref, fwhm=FIXED_FWHM, params=PARAMETRI_BASE)

# Ricerca Target
distanze_pixel = np.sqrt((tbl_ref['xcentroid'] - TARGET_PIXEL_X) ** 2 + (tbl_ref['ycentroid'] - TARGET_PIXEL_Y) ** 2)
idx_target = np.argmin(distanze_pixel)
stella_target = tbl_ref[idx_target]

# Calcola le coordinate celesti del target (RA, DEC)
coord_target_world = wcs_ref.pixel_to_world(stella_target['xcentroid'], stella_target['ycentroid'])

print(f"Stella Target identificata:")
print(f"  - Coordinate Celesti (RA, DEC): {coord_target_world.ra.deg:.5f}, {coord_target_world.dec.deg:.5f}")

# 3. Loop su THRESHOLD
print(f"\n--- FASE 2: Analisi variabilità THRESHOLD (FWHM fissa a {FIXED_FWHM}) ---")
print(f"Valori Threshold da testare: {THRESHOLD_RANGE}")

risultati_conteggi = []

for thresh_test in THRESHOLD_RANGE:
    print(f"\nTestando Threshold = {thresh_test:.3f} ...")
    immagini_trovate_count = 0

    # Creiamo una copia dei parametri e aggiorniamo la soglia
    current_params = PARAMETRI_BASE.copy()
    current_params['threshold_assoluta'] = thresh_test

    # Loop su tutte le immagini della run
    for i, path in enumerate(file_paths):
        # Carica e Segmenta
        data, wcs = elabora_file_fits(path)

        # Passiamo i parametri correnti (con la nuova threshold)
        tbl_sorgenti = esegui_segmentazione(data, fwhm=FIXED_FWHM, params=current_params)

        if tbl_sorgenti is None or len(tbl_sorgenti) == 0:
            continue

        # Converti i centroidi trovati in coordinate SkyCoord
        coords_trovate = wcs.pixel_to_world(tbl_sorgenti['xcentroid'], tbl_sorgenti['ycentroid'])

        # Calcola distanza dal Target
        distanze = coords_trovate.separation(coord_target_world)

        # Controlla se almeno una sorgente è entro la soglia
        if np.min(distanze) <= SOGLIA_CORRELAZIONE:
            immagini_trovate_count += 1

        # Barra di avanzamento
        if i % 10 == 0:
            print(f"  Elaborate {i}/{len(file_paths)} immagini...", end='\r')

    print(f"  -> Target trovato in {immagini_trovate_count} su {len(file_paths)} immagini.")
    risultati_conteggi.append(immagini_trovate_count)

# --- CREAZIONE DATAFRAME E SALVATAGGIO ---
print("\n--- SALVATAGGIO RISULTATI ---")
output_filename = os.path.join(BASE_PATH, f"risultati_threshold_run_{RUN}.csv")

df_risultati = pd.DataFrame({
    'Threshold': THRESHOLD_RANGE,
    'Immagini_Trovate': risultati_conteggi,
    'Totale_Immagini': len(file_paths),
    'Percentuale_Rilevamento': (np.array(risultati_conteggi) / len(file_paths)) * 100
})

# Salvataggio in CSV
df_risultati.to_csv(output_filename, index=False)
print(f"File salvato con successo in:\n{output_filename}")
print(df_risultati)

# 4. Plotting
print("\n--- GENERAZIONE GRAFICO ---")

plt.figure(figsize=(10, 6))
plt.plot(THRESHOLD_RANGE, risultati_conteggi, marker='o', linestyle='-', linewidth=2, color='green')

# Estetica
plt.title(f'Rilevamento Stella Target (Run {RUN})\nFWHM fissa: {FIXED_FWHM} - Variando Threshold', fontsize=14)
plt.xlabel('Soglia di Rilevamento (sigma)', fontsize=12)
plt.ylabel('Numero di immagini in cui è stata trovata', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, len(file_paths) * 1.05)

# Aggiungi etichette sui punti
for x, y in zip(THRESHOLD_RANGE, risultati_conteggi):
    plt.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

plt.tight_layout()
plt.show()

# --- CALCOLO TEMPO TRASCORSO ---
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n==========================================")
print(f"Tempo totale di esecuzione: {elapsed_time:.2f} secondi")
print(f"({elapsed_time / 60:.2f} minuti)")
print(f"==========================================")
print("Finito.")