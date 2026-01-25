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
# Fissiamo FWHM e Threshold ai valori standard
FIXED_FWHM = 3.0
FIXED_THRESHOLD = 3.61

# Definiamo il range di SIZE da testare
# Il valore di base era 5. Testiamo valori interi dispari attorno a 5.
# Genera: [3, 5, 7, 9, 11, 13, 15]
SIZE_RANGE = np.arange(3, 16, 2)

# Parametri fissi di segmentazione (il valore 'size' verrà sovrascritto nel loop)
PARAMETRI_BASE = {
    'size': 5,            # Valore di default
    'pixel': 3,
    'soglia_filtro_ass': 2.5,
    'soglia_filtro_rel': 0.4,
    'threshold_assoluta': FIXED_THRESHOLD
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
    # 1. Convoluzione (Usa params['size'])
    # Assicuriamo che size sia un intero
    kernel_size = int(params['size'])
    kernel = make_2dgaussian_kernel(fwhm, size=kernel_size)
    convolved_data = convolve(data, kernel)

    # 2. Source Finder
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

print(f"=== INIZIO ANALISI RUN {RUN} (Variazione Kernel Size) ===")

# 1. Caricamento lista file
try:
    file_paths = get_file_list(FILE_LISTA_PATH)
    print(f"Caricati {len(file_paths)} percorsi file.")
except FileNotFoundError:
    print(f"ERRORE: Non trovo il file lista in {FILE_LISTA_PATH}")
    exit()

# 2. Identificazione Stella Target (Fase Preliminare STANDARD)
print(f"\n--- FASE 1: Identificazione Target (Setup Standard: Size=5) ---")
path_ref = "/home/lorysimeone/tesi_magistrale/20250120_run1/20250120_213945.fits"

if not os.path.exists(path_ref):
    print(f"ATTENZIONE: File hardcoded non trovato: {path_ref}")
    print(f"Uso il file all'indice {IMG_RIFERIMENTO_IDX} della lista.")
    path_ref = file_paths[IMG_RIFERIMENTO_IDX]

data_ref, wcs_ref = elabora_file_fits(path_ref)

# Segmentazione di riferimento (con size=5 definito in PARAMETRI_BASE)
tbl_ref = esegui_segmentazione(data_ref, fwhm=FIXED_FWHM, params=PARAMETRI_BASE)

# Ricerca Target
distanze_pixel = np.sqrt((tbl_ref['xcentroid'] - TARGET_PIXEL_X) ** 2 + (tbl_ref['ycentroid'] - TARGET_PIXEL_Y) ** 2)
idx_target = np.argmin(distanze_pixel)
stella_target = tbl_ref[idx_target]

# Calcola le coordinate celesti del target (RA, DEC)
coord_target_world = wcs_ref.pixel_to_world(stella_target['xcentroid'], stella_target['ycentroid'])

print(f"Stella Target identificata:")
print(f"  - Coordinate Celesti (RA, DEC): {coord_target_world.ra.deg:.5f}, {coord_target_world.dec.deg:.5f}")

# 3. Loop su SIZE
print(f"\n--- FASE 2: Analisi variabilità SIZE (FWHM={FIXED_FWHM}, Thresh={FIXED_THRESHOLD}) ---")
print(f"Valori Size da testare: {SIZE_RANGE}")

risultati_conteggi = []

for size_test in SIZE_RANGE:
    print(f"\nTestando Kernel Size = {size_test} ...")
    immagini_trovate_count = 0

    # Creiamo una copia dei parametri e aggiorniamo il size
    current_params = PARAMETRI_BASE.copy()
    current_params['size'] = size_test

    # Loop su tutte le immagini della run
    for i, path in enumerate(file_paths):
        # Carica e Segmenta
        data, wcs = elabora_file_fits(path)

        # Passiamo i parametri correnti (con il nuovo size)
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
output_filename = os.path.join(BASE_PATH, f"risultati_size_run_{RUN}.csv")

df_risultati = pd.DataFrame({
    'Kernel_Size': SIZE_RANGE,
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
plt.plot(SIZE_RANGE, risultati_conteggi, marker='s', linestyle='-', linewidth=2, color='red')

# Estetica
plt.title(f'Rilevamento Stella Target (Run {RUN})\nVariando Kernel Size (FWHM={FIXED_FWHM})', fontsize=14)
plt.xlabel('Dimensione Kernel (pixel)', fontsize=12)
plt.ylabel('Numero di immagini in cui è stata trovata', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, len(file_paths) * 1.05)
plt.xticks(SIZE_RANGE) # Assicura che l'asse X mostri solo i valori interi testati

# Aggiungi etichette sui punti
for x, y in zip(SIZE_RANGE, risultati_conteggi):
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