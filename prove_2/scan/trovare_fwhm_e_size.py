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
import warnings
from astropy.wcs import FITSFixedWarning


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
    Versione modificata per accettare sia fwhm che size come argomenti variabili
    """
    # 1. Convoluzione
    try:
        kernel = make_2dgaussian_kernel(fwhm, size=size)
        convolved_data = convolve(data, kernel)
    except Exception as e:
        # Se il kernel è troppo piccolo per il FWHM o altri errori matematici
        return None

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



# Registra il tempo di inizio
start_time = time.time()

# --- CONFIGURAZIONE E SOPPRESSIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- PARAMETRI ---
RUN = 1
IMG_RIFERIMENTO_IDX = 35
SOGLIA_CORRELAZIONE = 0.003349 * u.deg

# Coordinate pixel della stella target (da calibrazione precedente)
TARGET_PIXEL_X = 1891.7445408352592
TARGET_PIXEL_Y = 1061.437287743088

# Parametri fissi di segmentazione (rimossi size e fwhm dal dizionario fisso)
PARAMETRI_FISSI = {
    'threshold_assoluta': 3.61,
    'pixel': 3,
    'soglia_filtro_ass': 2.5,
    'soglia_filtro_rel': 0.05
}

PARAMETRI_FISSI = {}
with open('/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt', 'r') as file:
    next(file)  # Salta intestazione
    for riga in file:
        riga = riga.strip()
        if riga and not riga.startswith('#'):
            parametro, valore = riga.split()
            PARAMETRI_FISSI[parametro] = float(valore) if '.' in valore else int(valore)

# --- GRIGLIA DI RICERCA PARAMETRI ---
# Range di FWHM da testare
FWHM_RANGE = np.linspace(0.1, 1.7, 45)
# Range di SIZE del kernel (devono essere numeri dispari: 3, 5, 7, 9...)
SIZE_RANGE = np.array([3])  # Genera [3, 5, 7, 9, 11]

# PERCORSI
BASE_PATH = "/home/lorysimeone/tesi_magistrale/prove_2"
FILE_LISTA_PATH = os.path.join(BASE_PATH, f"liste_percorsi_run/lista_immagini_run_{RUN}.txt")


# --- MAIN ---

print(f"=== INIZIO ANALISI PARAMETRICA AVANZATA RUN {RUN} ===")
print(f"Parametri variabili:")
print(f" - FWHM: {FWHM_RANGE}")
print(f" - Kernel Size: {SIZE_RANGE}")

# 1. Caricamento lista file
try:
    file_paths = get_file_list(FILE_LISTA_PATH)
    print(f"Caricati {len(file_paths)} file.")
except FileNotFoundError:
    print(f"ERRORE: File lista non trovato.")
    exit()

# 2. Identificazione Stella Target (Fase Preliminare)
# Necessaria per avere le coordinate celesti di riferimento
path_ref = "/home/lorysimeone/tesi_magistrale/20250120_run1/20250120_213945.fits"
if not os.path.exists(path_ref):
    path_ref = file_paths[IMG_RIFERIMENTO_IDX]

data_ref, wcs_ref, header_ref = elabora_file_fits(path_ref)

# Eseguiamo una segmentazione "standard" solo per trovare RA/DEC del target
tbl_ref = esegui_segmentazione_dinamica(data_ref, fwhm=3.0, size=5, params=PARAMETRI_FISSI)

# Ricerca Target
distanze_pixel = np.sqrt((tbl_ref['xcentroid'] - TARGET_PIXEL_X) ** 2 + (tbl_ref['ycentroid'] - TARGET_PIXEL_Y) ** 2)
idx_target = np.argmin(distanze_pixel)
stella_target = tbl_ref[idx_target]
coord_target_world = wcs_ref.pixel_to_world(stella_target['xcentroid'], stella_target['ycentroid'])

print(f"Target Agganciato: RA={coord_target_world.ra.deg:.5f}, DEC={coord_target_world.dec.deg:.5f}")

# 3. Inizializzazione Matrice Risultati
# Righe = SIZE, Colonne = FWHM
matrice_conteggi = np.zeros((len(SIZE_RANGE), len(FWHM_RANGE)), dtype=int)

print(f"\n--- INIZIO LOOP SULLE IMMAGINI ---")
print("Strategia: Carico immagine -> Ritaglio -> Testo TUTTE le combinazioni Size/FWHM")

for i, path in enumerate(file_paths):
    # A. Caricamento e Ritaglio (Fatto una sola volta per immagine)
    try:
        data, wcs, header = elabora_file_fits(path)

        # Calcolo coordinate pixel target nell'immagine corrente
        coord_target_pixel = wcs.world_to_pixel(coord_target_world)

        # Definizione Riquadro (preservando la logica originale delle variabili x_min/y_min)
        # Nota: nel codice originale x_min indicava l'asse delle righe (Y) e y_min le colonne (X)
        x_rows_center = coord_target_pixel[1]
        y_cols_center = coord_target_pixel[0]

        x_min = int(x_rows_center - 15)
        x_max = int(x_rows_center + 15)
        y_min = int(y_cols_center - 15)
        y_max = int(y_cols_center + 15)

        # Controllo bordi
        if x_min < 0 or y_min < 0 or x_max > data.shape[0] or y_max > data.shape[1]:
            continue

        riquadro = data[x_min:x_max, y_min:y_max]

        # B. Loop sui Parametri (In memoria, molto veloce)
        for s_idx, size_val in enumerate(SIZE_RANGE):
            for f_idx, fwhm_val in enumerate(FWHM_RANGE):

                # Esegui segmentazione sul ritaglio
                '''tbl_sorgenti = esegui_segmentazione_dinamica(riquadro, fwhm=fwhm_val, size=size_val,
                                                             params=PARAMETRI_FISSI)'''

                kernel = make_2dgaussian_kernel(fwhm_val, size=size_val)
                convolved_data = convolve(riquadro, kernel)

                finder = SourceFinder(npixels=PARAMETRI_FISSI['pixel'], progress_bar=False)
                segment_map = finder(convolved_data, PARAMETRI_FISSI['threshold_assoluta'])
                cat = SourceCatalog(riquadro, segment_map, convolved_data=convolved_data)
                tbl_sorgenti = cat.to_table()

                if tbl_sorgenti is None or len(tbl_sorgenti) == 0:
                    continue

                # Ricostruzione coordinate globali
                # x_locali sono colonne (X), y_locali sono righe (Y)
                x_locali = tbl_sorgenti['xcentroid']
                y_locali = tbl_sorgenti['ycentroid']

                # Applico la logica inversa del ritaglio originale
                x_globali = x_locali + y_min
                y_globali = y_locali + x_min

                coords_trovate = wcs.pixel_to_world(x_globali, y_globali)
                distanze = coords_trovate.separation(coord_target_world)

                if np.min(distanze) <= SOGLIA_CORRELAZIONE:
                    matrice_conteggi[s_idx, f_idx] += 1

    except Exception as e:
        print(f"Errore su file {i}: {e}")
        continue

    # Barra di avanzamento
    if (i + 1) % 10 == 0 or (i + 1) == len(file_paths):
        print(f"  Elaborate {i + 1}/{len(file_paths)} immagini...", end='\r')

# --- CREAZIONE DATAFRAME E SALVATAGGIO ---
print("\n\n--- SALVATAGGIO RISULTATI ---")

# Creiamo una lista piatta per il CSV
records = []
for s_idx, size_val in enumerate(SIZE_RANGE):
    for f_idx, fwhm_val in enumerate(FWHM_RANGE):
        count = matrice_conteggi[s_idx, f_idx]
        perc = (count / len(file_paths)) * 100
        records.append({
            'Kernel_Size': size_val,
            'FWHM': fwhm_val,
            'Immagini_Trovate': count,
            'Percentuale': perc
        })

df_risultati = pd.DataFrame(records)
output_filename = os.path.join(BASE_PATH, f"grid_search_params_run_{RUN}.csv")
df_risultati.to_csv(output_filename, index=False)
print(f"Risultati salvati in: {output_filename}")

# Trova la combinazione migliore
best_row = df_risultati.loc[df_risultati['Immagini_Trovate'].idxmax()]
print("\n*** MIGLIORE COMBINAZIONE ***")
print(f"Size: {int(best_row['Kernel_Size'])} pixel")
print(f"FWHM: {best_row['FWHM']:.2f}")
print(f"Trovata in {int(best_row['Immagini_Trovate'])} su {len(file_paths)} immagini ({best_row['Percentuale']:.2f}%)")

# --- PLOTTING (HEATMAP) ---
print("\nGenerazione grafico Heatmap...")

plt.figure(figsize=(12, 8))

# Usiamo imshow per creare la heatmap dalla matrice
# aspect='auto' adatta i pixel alla finestra, origin='lower' mette (0,0) in basso a sx
img = plt.imshow(matrice_conteggi, interpolation='nearest', cmap='viridis', origin='lower', aspect='auto')

# Impostazione assi
plt.xticks(np.arange(len(FWHM_RANGE)), [f"{x:.1f}" for x in FWHM_RANGE], rotation=45)
plt.yticks(np.arange(len(SIZE_RANGE)), SIZE_RANGE)

plt.xlabel('FWHM (pixel)', fontsize=12)
plt.ylabel('Kernel Size (pixel)', fontsize=12)
plt.title(f'Performance Rilevamento Target - Run {RUN}\n(Conteggio immagini in cui la stella è stata trovata)',
          fontsize=14)

# Colorbar
cbar = plt.colorbar(img)
cbar.set_label('Numero di Rilevamenti', rotation=270, labelpad=15)

# Annotazioni sui blocchi della heatmap
for i in range(len(SIZE_RANGE)):
    for j in range(len(FWHM_RANGE)):
        valore = matrice_conteggi[i, j]
        # Scrivi il testo bianco se lo sfondo è scuro, nero se chiaro
        colore_testo = "white" if valore < np.max(matrice_conteggi) / 2 else "black"
        plt.text(j, i, str(valore), ha="center", va="center", color=colore_testo, fontsize=8)

plt.tight_layout()
plt.show()

# --- TEMPO ---
end_time = time.time()
print(f"Tempo totale: {(end_time - start_time):.2f} sec")