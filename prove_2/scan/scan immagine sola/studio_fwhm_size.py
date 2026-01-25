import pandas as pd
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
import numpy as np
import os
import time
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture

# Set up wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from astroquery.vizier import Vizier
from astropy.coordinates import Angle

from shapely.geometry import Point, Polygon
# warning
import warnings
from astropy.io.fits.verify import VerifyWarning
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

from pathlib import Path

# Registra il tempo di inizio
start_time = time.time()

def converti_valore(valore):
    valore = valore.strip()
    if not valore: return valore
    try: return int(valore)
    except ValueError: pass
    try: return float(valore)
    except ValueError: pass
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']: return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']: return False
    return valore

def leggi_header_da_csv(filename):
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':
                break
    return header_dict

def freedman_diaconis_bins(data):
    data_clean = data.compressed() if hasattr(data, 'mask') else data
    if len(data_clean) < 2: return 1
    iqr = np.percentile(data_clean, 75) - np.percentile(data_clean, 25)
    if iqr == 0: return 1
    bin_width = 2 * iqr / (len(data_clean) ** (1/3))
    data_range = np.max(data_clean) - np.min(data_clean)
    bins = int(np.ceil(data_range / bin_width))
    return max(bins, 1)

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

def elabora_file_fits(percorso_file):
    with fits.open(percorso_file) as hdu:
        header = hdu[0].header
        data = hdu[0].data
        wcs = WCS(header)
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        data_sub = data - median
        return data_sub, wcs, header

def unione_tabelle(tbl_seg, tbl_cat, percorso_file_fits):
    # Rimuovi colonne non necessarie e aggiungi coordinate celesti
    tbl_seg.keep_columns(['label', 'xcentroid', 'ycentroid', 'area', 'max_value', 'kron_flux'])

    with fits.open(percorso_file_fits) as hdu_list:
        header = hdu_list[0].header
        w = WCS(header)

    coords_trovate = w.pixel_to_world(tbl_seg['xcentroid'], tbl_seg['ycentroid'])
    tbl_seg['RA_centroid'] = coords_trovate.ra
    tbl_seg['DEC_centroid'] = coords_trovate.dec

    # 2. Preparazione Tabella Finale Vuota
    tbl_finale = Table()

    # Aggiungi colonne dalla prima tabella preservando i tipi
    for colname in tbl_seg.colnames:
        tbl_finale[colname] = tbl_seg[colname][:0]

    # Colonna chiave per il matching (tipo stringa per Rank)
    tbl_finale['Corrispondenza'] = 'SI (Rank 1)'

    # Aggiungi colonne dalla seconda tabella preservando i tipi
    for colname in tbl_cat.colnames:
        tbl_finale[colname] = tbl_cat[colname][:0]

    # Metto la colonna "Catalogo" prima di "ID"
    colonne = tbl_finale.colnames
    pos_id = colonne.index('ID')
    nuovo_ordine = colonne[:pos_id] + ['Catalogo'] + [col for col in colonne if
                                                      col != 'Catalogo' and col not in colonne[:pos_id]]
    tbl_finale = tbl_finale[nuovo_ordine]

    # 3. Conversione Coordinate Catalogate
    try:
        if hasattr(tbl_cat['RAJ2000'], 'value'):
            ra_values = tbl_cat['RAJ2000'].value
            dec_values = tbl_cat['DEJ2000'].value
        else:
            ra_values = np.array(tbl_cat['RAJ2000'])
            dec_values = np.array(tbl_cat['DEJ2000'])
        coords_catalogate = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)
    except Exception:
        coords_catalogate = SkyCoord(ra=tbl_cat['RAJ2000'],
                                     dec=tbl_cat['DEJ2000'],
                                     unit=u.deg)

    print(f"Cercata correlazione in {len(coords_trovate)} stelle per size = {size_val} e FWHM = {fwhm_val}")

    # 4. Logica di Correlazione (Rank 1, 2, 3 per luminosità)
    colonna_magnitudine = 'Mag'
    righe_da_aggiungere = []

    for idx_trovato in range(len(coords_trovate)):
        coord_trovata = coords_trovate[idx_trovato]
        distanza_da_trovata = coords_catalogate.separation(coord_trovata)

        # Criterio 1: Stelle entro la soglia di distanza
        mask_distanza = distanza_da_trovata <= soglia_correlazione

        '''# Criterio 2: Stelle più luminose di Mag < MAG_CUTOFF (10.4)
        magnitudini_candidate = tbl_cat[colonna_magnitudine]
        mask_luminosita = magnitudini_candidate < MAG_CUTOFF'''

        # Maschera finale: deve soddisfare entrambi i criteri
        # maschera_finale = mask_distanza & mask_luminosita
        maschera_finale = mask_distanza
        indici_catalogate_vicine = np.where(maschera_finale)[0]

        ha_corrispondenza = len(indici_catalogate_vicine) > 0

        # Prepara la riga base (dati della sorgente trovata)
        riga_base = {}
        for colname in tbl_seg.colnames:
            riga_base[colname] = tbl_seg[colname][idx_trovato]

        if ha_corrispondenza:
            # 1. Ordina le stelle corrispondenti per luminosità
            magnitudini_vicine = tbl_cat[colonna_magnitudine][indici_catalogate_vicine]

            # Ottieni gli indici SORTATI per luminosità
            indici_luminosita_sort = np.argsort(magnitudini_vicine)

            # Ottieni gli indici originali (nella tbl_cat) delle stelle più luminose, ordinate
            indici_originali_ordinati = indici_catalogate_vicine[indici_luminosita_sort]

            indici_da_considerare = indici_originali_ordinati

            # 2. Crea una riga per OGNI corrispondenza valida trovata (Rank 1, 2, o 3)
            for rank, idx_catalogata in enumerate(indici_da_considerare, 1):
                nuova_riga = riga_base.copy()

                # Dati dalla stella catalogata corrispondente
                for colname in tbl_cat.colnames:
                    nuova_riga[colname] = tbl_cat[colname][idx_catalogata]

                # Aggiungi informazioni sul "rank"
                nuova_riga['Corrispondenza'] = f'SI (Rank {rank})'

                righe_da_aggiungere.append(nuova_riga)

        else:
            # Nessuna corrispondenza valida trovata -> UNA SOLA riga NO
            nuova_riga_no_match = riga_base.copy()
            nuova_riga_no_match['Corrispondenza'] = 'NO'

            # Imposta i valori sentinella per i dati del catalogo
            for colname in tbl_cat.colnames:
                if colname == 'Catalogo':
                    nuova_riga_no_match[colname] = 'N/A'
                elif tbl_cat[colname].dtype.kind in ['i', 'u']:
                    nuova_riga_no_match[colname] = -999
                elif tbl_cat[colname].dtype.kind == 'f':
                    nuova_riga_no_match[colname] = np.nan
                elif tbl_cat[colname].dtype.kind == 'O':
                    nuova_riga_no_match[colname] = 'N/A'
                else:
                    nuova_riga_no_match[colname] = None

            righe_da_aggiungere.append(nuova_riga_no_match)

    # 5. Aggiungi tutte le righe raccolte alla tabella finale
    for riga in righe_da_aggiungere:
        tbl_finale.add_row(riga)
        
    return tbl_finale

PARAMETRI_FISSI = {
    'threshold_assoluta': 3.61,
    'pixel': 3,
    'soglia_filtro_ass': 2.5,
    'soglia_filtro_rel': 0.05
}

run = 1

cartella_csv_cat = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}"
file_csv_cat = sorted([f for f in os.listdir(cartella_csv_cat) if f.endswith('.csv')])
lista_percorsi_csv_cat = [os.path.join(cartella_csv_cat, file) for file in file_csv_cat]

n_immagine = 35

percorso_file_csv_cat = lista_percorsi_csv_cat[n_immagine]
dataframe = pd.read_csv(percorso_file_csv_cat, comment='#')
tbl_catalogate = Table.from_pandas(dataframe) # tabella di tutte le stelle catalogate

header_info = leggi_header_da_csv(percorso_file_csv_cat)
percorso_fits = header_info.get('PERCORSO_FILE', '')
print(percorso_fits)

# --- GRIGLIA DI RICERCA PARAMETRI ---
# Range di FWHM da testare
FWHM_RANGE = np.linspace(0.1, 1.7, 45)
# Range di SIZE del kernel (devono essere numeri dispari: 3, 5, 7, 9...)
SIZE_RANGE = np.array([3])  # Genera [3, 5, 7, 9, 11]
soglia_correlazione = 0.003349 * u.deg  # soglia fissa (in gradi)

data, wcs, header = elabora_file_fits(percorso_fits)

# Righe = SIZE, Colonne = FWHM
matrice_conteggi = np.zeros((len(SIZE_RANGE), len(FWHM_RANGE)), dtype=int)

# Loop sui parametri

i = 0
for s_idx, size_val in enumerate(SIZE_RANGE):
    for f_idx, fwhm_val in enumerate(FWHM_RANGE):

        i += 1
        # image segmentation
        tbl_trovate = esegui_segmentazione_dinamica(data, fwhm=fwhm_val, size=size_val, params=PARAMETRI_FISSI)
        
        # unione tabelle
        tbl = unione_tabelle(tbl_trovate, tbl_catalogate, percorso_fits)
        mask_si = np.char.startswith(tbl['Corrispondenza'], 'SI')
        tbl_corr = tbl[mask_si] # tabella delle stelle trovate con correlazione

        matrice_conteggi[s_idx, f_idx] += len(tbl_corr)

        if (i + 1) % 10 == 0 or (i + 1) == matrice_conteggi.size:
            print(f"  Elaborate {i + 1}/{matrice_conteggi.size} immagini...", end='\r')

# --- CREAZIONE DATAFRAME E SALVATAGGIO ---
print("\n\n--- SALVATAGGIO RISULTATI ---")

# Creiamo una lista piatta per il CSV
records = []
for s_idx, size_val in enumerate(SIZE_RANGE):
    for f_idx, fwhm_val in enumerate(FWHM_RANGE):
        count = matrice_conteggi[s_idx, f_idx]
        perc = (count / len(tbl_catalogate)) * 100
        records.append({
            'Kernel_Size': size_val,
            'FWHM': fwhm_val,
            'Immagini_Trovate': count,
            'Percentuale': perc
        })

df_risultati = pd.DataFrame(records)
output_filename = os.path.join("/home/lorysimeone/tesi_magistrale/prove_2", f"_search_params_run_{run}_image_{n_immagine}.csv")
df_risultati.to_csv(output_filename, index=False)
print(f"Risultati salvati in: {output_filename}")

# Trova la combinazione migliore
best_row = df_risultati.loc[df_risultati['Immagini_Trovate'].idxmax()]
print("\n*** MIGLIORE COMBINAZIONE ***")
print(f"Size: {int(best_row['Kernel_Size'])} pixel")
print(f"FWHM: {best_row['FWHM']:.2f}")
print(f"Trovata in {int(best_row['Immagini_Trovate'])} su {len(tbl_catalogate)} immagini ({best_row['Percentuale']:.2f}%)")

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
plt.title(f'Performance Rilevamento Target - Run {run}\n(Conteggio immagini in cui la stella è stata trovata)',
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

# --- TEMPO ---
end_time = time.time()
print(f"Tempo totale: {(end_time - start_time):.2f} sec")

plt.tight_layout()
plt.show()

