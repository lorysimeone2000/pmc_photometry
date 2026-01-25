import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time  # Importato per il calcolo del tempo
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica
from photutils.segmentation import make_2dgaussian_kernel, SourceFinder, SourceCatalog
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.patches import Circle
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

# Range di FWHM da testare
FWHM_RANGE = np.linspace(1.5, 5.0, 15)

# Parametri fissi di segmentazione
PARAMETRI_FISSI = {
    'size': 5,
    'threshold_assoluta': 3.61,
    'pixel': 3,
    'soglia_filtro_ass': 2.5,
    'soglia_filtro_rel': 0.4
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
        return data_sub, wcs, header


def esegui_segmentazione(data, fwhm, params):
    """
    Esegue la segmentazione (SourceFinder) e restituisce la tabella delle sorgenti.
    """
    # 1. Convoluzione
    kernel = make_2dgaussian_kernel(fwhm, size=params['size'])
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

print(f"=== INIZIO ANALISI RUN {RUN} ===")

# 1. Caricamento lista file
try:
    file_paths = get_file_list(FILE_LISTA_PATH)
    print(f"Caricati {len(file_paths)} percorsi file.")
except FileNotFoundError:
    print(f"ERRORE: Non trovo il file lista in {FILE_LISTA_PATH}")
    exit()

# 2. Identificazione Stella Target (Fase Preliminare)
print(f"\n--- FASE 1: Identificazione Target (FWHM=3.0) ---")
path_ref = "/home/lorysimeone/tesi_magistrale/20250120_run1/20250120_213945.fits"

if not os.path.exists(path_ref):
    print(f"ATTENZIONE: File hardcoded non trovato: {path_ref}")
    print(f"Uso il file all'indice {IMG_RIFERIMENTO_IDX} della lista.")
    path_ref = file_paths[IMG_RIFERIMENTO_IDX]

data_ref, wcs_ref, header_ref = elabora_file_fits(path_ref)

# catalogo
file_csv_catalogate = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{RUN}/run_1_stelle_catalogate_immagine_{IMG_RIFERIMENTO_IDX:03d}.csv"

dataframe2 = pd.read_csv(file_csv_catalogate, comment='#')
tbl_catalogate_ref = Table.from_pandas(dataframe2)
coo_cel_cat_ref = SkyCoord(ra=tbl_catalogate_ref['RAJ2000'],
                   dec=tbl_catalogate_ref['DEJ2000'],
                                 unit = 'deg',
                                 frame= 'icrs')
magnitudini = tbl_catalogate_ref['Mag']
cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())

# Eseguiamo la segmentazione sull'immagine di riferimento
tbl_ref = esegui_segmentazione(data_ref, fwhm=3.0, params=PARAMETRI_FISSI)

# Ricerca basata sulla distanza euclidea dai pixel target
distanze_pixel = np.sqrt((tbl_ref['xcentroid'] - TARGET_PIXEL_X)**2 + (tbl_ref['ycentroid'] - TARGET_PIXEL_Y)**2)
idx_target = np.argmin(distanze_pixel)
stella_target = tbl_ref[idx_target]
distanza_minima = distanze_pixel[idx_target]

# Calcola le coordinate celesti del target (RA, DEC)
coord_target_world = wcs_ref.pixel_to_world(stella_target['xcentroid'], stella_target['ycentroid'])

print(f"Stella Target identificata:")
print("Immagine di riferimento: " , path_ref)
print(f"  - Coordinate cercate (pixel): ({TARGET_PIXEL_X}, {TARGET_PIXEL_Y})")
print(f"  - Coordinate trovate (pixel): ({stella_target['xcentroid']:.2f}, {stella_target['ycentroid']:.2f})")
print(f"  - ID locale (label): {stella_target['label']}")
print(f"  - Kron Flux: {stella_target['kron_flux']:.2f}")
print(f"  - Coordinate Celesti (RA, DEC): {coord_target_world.ra.deg:.5f}, {coord_target_world.dec.deg:.5f}")

# 3. Loop su FWHM
print(f"\n--- FASE 2: Analisi variabilità FWHM ---")
print(f"Valori FWHM da testare: {FWHM_RANGE}")

risultati_conteggi = []

for fwhm_test in FWHM_RANGE:
    print(f"\nTestando FWHM = {fwhm_test:.2f} ...")
    immagini_trovate_count = 0

    # Loop su tutte le immagini della run
    for i, path in enumerate(file_paths):

        i = i+1
        # Carica
        data, wcs, header = elabora_file_fits(path)
        coo_pixel = wcs.world_to_pixel(coo_cel_cat_ref)

        # Creo il riquadro
        coord_target_pixel = wcs.world_to_pixel(coord_target_world)
        x_min = int(coord_target_pixel[1] - 15)
        x_max = int(coord_target_pixel[1] + 15)
        y_min = int(coord_target_pixel[0] - 15)
        y_max = int(coord_target_pixel[0] + 15)

        riquadro = data[x_min:x_max , y_min:y_max] # Ritaglia un'area tot x tot pixel

        if i == 1 and fwhm_test == 1.50:
            # Calcolo le coordinate per il plot
            # 1. wcs.world_to_pixel restituisce una TUPLA (x_array, y_array), la spacchettiamo:
            x_globali, y_globali = coo_pixel

            # 2. Trasliamo le coordinate globali nel sistema di riferimento del Riquadro
            # Nota: Hai definito y_min come start delle colonne (X) e x_min come start delle righe (Y)
            x_locali = x_globali - y_min
            y_locali = y_globali - x_min

            plt.figure(figsize=(8, 8))  # Opzionale: rende la figura più grande

            # Visualizza il ritaglio
            plt.imshow(riquadro, cmap="gray_r", norm=LogNorm(), origin='lower', interpolation='nearest')

            # Aggiungi la colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), label='Magnitudine V')

            # 3. Plot delle stelle catalogate (usando le coordinate LOCALI traslate)
            # Nota: ho sostituito 'c=colors' con 'c=magnitudini' perché 'colors' non era definito in questo script
            plt.scatter(x_locali, y_locali, c=magnitudini, s = 36, alpha=0.7, cmap='viridis_r')

            # --- AGGIUNTA CERCHIO SOGLIA CORRELAZIONE ---

            # A. Calcolo il raggio in pixel
            # proj_plane_pixel_scales restituisce la scala [deg/pix_x, deg/pix_y]
            scales = proj_plane_pixel_scales(wcs)
            scale_avg = np.mean(scales) * u.deg  # Scala media in gradi/pixel

            # Converto la soglia (che è in gradi) in pixel puri
            raggio_pixel = (SOGLIA_CORRELAZIONE / scale_avg).decompose().value

            # B. Calcolo il centro del target nel sistema locale del riquadro
            # Ottengo i pixel globali del target (che è il centro della ricerca)
            target_glob_x, target_glob_y = wcs.world_to_pixel(coord_target_world)

            # Traslo nel sistema locale (sottraendo l'offset del ritaglio)
            target_loc_x = target_glob_x - y_min
            target_loc_y = target_glob_y - x_min

            # C. Creo e aggiungo il cerchio
            cerchio = Circle((target_loc_x, target_loc_y), raggio_pixel,
                             edgecolor='red', facecolor='none',
                             lw=2, linestyle='--', label='Soglia Corr.')

            plt.gca().add_patch(cerchio)

            # Imposto i limiti per essere sicuro di vedere solo il riquadro
            plt.xlim(0, riquadro.shape[1])
            plt.ylim(0, riquadro.shape[0])

            plt.legend()
            plt.title(f"Check Riquadro - FWHM {fwhm_test}\nRaggio Soglia: {raggio_pixel:.1f} pixel")
            plt.show()
            #quit()

        # segmentazione
        tbl_sorgenti = esegui_segmentazione(riquadro, fwhm=fwhm_test, params=PARAMETRI_FISSI)

        if tbl_sorgenti is None or len(tbl_sorgenti) == 0:
            continue

        # Converti i centroidi trovati in coordinate SkyCoord
        x_locali = tbl_sorgenti['xcentroid']
        y_locali = tbl_sorgenti['ycentroid']

        x_globali = x_locali + y_min
        y_globali = y_locali + x_min

        # Ora chiedo al WCS le coordinate celesti usando i pixel corretti
        coords_trovate = wcs.pixel_to_world(x_globali, y_globali)

        # Calcola distanza dal Target
        distanze = coords_trovate.separation(coord_target_world)

        # Controlla se almeno una sorgente è entro la soglia
        if np.min(distanze) <= SOGLIA_CORRELAZIONE:
            immagini_trovate_count += 1

        # Barra di avanzamento
        if i-1 % 10 == 0:
            print(f"  Elaborate {i}/{len(file_paths)} immagini...", end='\r')

    print(f"  -> Target trovato in {immagini_trovate_count} su {len(file_paths)} immagini.")
    risultati_conteggi.append(immagini_trovate_count)

# --- CREAZIONE DATAFRAME E SALVATAGGIO ---
print("\n--- SALVATAGGIO RISULTATI ---")
output_filename = os.path.join(BASE_PATH, f"risultati_fwhm_run_{RUN}.csv")

df_risultati = pd.DataFrame({
    'FWHM': FWHM_RANGE,
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
plt.plot(FWHM_RANGE, risultati_conteggi, marker='o', linestyle='-', linewidth=2, color='blue')

# Estetica
plt.title(f'Rilevamento Stella Target (Run {RUN})\nTarget Pixel iniziali: ({TARGET_PIXEL_X:.1f}, {TARGET_PIXEL_Y:.1f})', fontsize=14)
plt.xlabel('FWHM della PSF (pixel)', fontsize=12)
plt.ylabel('Numero di immagini in cui è stata trovata', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, len(file_paths) * 1.05)

# Aggiungi etichette sui punti
for x, y in zip(FWHM_RANGE, risultati_conteggi):
    plt.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

plt.tight_layout()
plt.show()

# --- CALCOLO TEMPO TRASCORSO ---
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n==========================================")
print(f"Tempo totale di esecuzione: {elapsed_time:.2f} secondi")
print(f"({elapsed_time/60:.2f} minuti)")
print(f"==========================================")
print("Finito.")