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


def converti_valore(valore):
    valore = valore.strip()
    if not valore: return valore
    try:
        return int(valore)
    except ValueError:
        pass
    try:
        return float(valore)
    except ValueError:
        pass
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False
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
    bin_width = 2 * iqr / (len(data_clean) ** (1 / 3))
    data_range = np.max(data_clean) - np.min(data_clean)
    bins = int(np.ceil(data_range / bin_width))
    return max(bins, 1)


# --- INIZIO CODICE ---
run = 1
cartella_csv = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

cartella_csv_cat = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}"
file_csv_cat = sorted([f for f in os.listdir(cartella_csv_cat) if f.endswith('.csv')])
lista_percorsi_csv_cat = [os.path.join(cartella_csv_cat, file) for file in file_csv_cat]

n_immagine = 35

percorso_file_csv = lista_percorsi_csv[n_immagine]
dataframe = pd.read_csv(percorso_file_csv, comment="#")
tbl = Table.from_pandas(dataframe)

mask_si = np.char.startswith(tbl['Corrispondenza'], 'SI')
mask_no = np.char.startswith(tbl['Corrispondenza'], 'NO')
tbl_catalogate_corr = tbl[mask_si]
tbl_trovate_non_corr = tbl[mask_no]

percorso_file_csv_cat = lista_percorsi_csv_cat[n_immagine]
dataframe_cat = pd.read_csv(percorso_file_csv_cat, comment="#")
tbl_cat = Table.from_pandas(dataframe_cat)

print("Tabella completa:\n", tbl)

magnitudini = tbl_catalogate_corr['Mag']
magnitudini_cat = tbl_cat['Mag']

# Gestione dei dati (conversione in array numpy e rimozione masked/NaN)
if hasattr(magnitudini, 'compressed'):
    mag_data = magnitudini.compressed()
else:
    mag_data = np.array(magnitudini)
    mag_data = mag_data[~np.isnan(mag_data)]

if hasattr(magnitudini_cat, 'compressed'):
    mag_cat_data = magnitudini_cat.compressed()
else:
    mag_cat_data = np.array(magnitudini_cat)
    mag_cat_data = mag_cat_data[~np.isnan(mag_cat_data)]

# Calcolo bin comuni
dati_totali = np.concatenate((mag_data, mag_cat_data))
n_bin = freedman_diaconis_bins(dati_totali)
hist_range = (np.min(dati_totali), np.max(dati_totali))
bins = np.histogram_bin_edges(dati_totali, bins=n_bin, range=hist_range)

# plt.figure(figsize=(20, 10))

# Istogrammi
counts_cat, bins_used_cat, patches_cat = plt.hist(
    mag_cat_data,
    bins=bins,
    alpha=1.0,
    color='purple',
    edgecolor='black',
    label='Sorgenti Catalogate (Totali)'
)

counts_corr, bins_used_corr, patches_corr = plt.hist(
    mag_data,
    bins=bins,
    alpha=0.4,
    color='red',
    edgecolor='black',
    label='Sorgenti Correlate (Filtrate)'
)

# --- SISTEMAZIONE ASSE X (MODIFICATA) ---
# Calcola tutti i centri dei bin
bin_centers = (bins[:-1] + bins[1:]) / 2

# Crea tutte le etichette formattate
tick_labels = [f'{c:.2f}' for c in bin_centers]

# Seleziona solo 1 tick ogni 4 (start:stop:step)
step = 4
subset_ticks = bin_centers[::step]
subset_labels = tick_labels[::step]

# --- CODICE AGGIUNTIVO PER VISUALIZZARE LE SORGENTI MANCANTI ---

# plt.figure(figsize=(20, 10))

# 2. Calcoliamo la differenza: (Stelle nel Catalogo) - (Stelle Trovate)
# Queste sono le stelle perse (False Negative)
counts_missing = counts_cat - counts_corr

# Evitiamo valori negativi (non dovrebbero esserci se i dati sono coerenti, ma per sicurezza)
counts_missing = np.maximum(counts_missing, 0)

# Calcolo dei centri dei bin per il plot a barre
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

# 3. Plot delle stelle MANCANTI
plt.bar(bin_centers, counts_missing, width=bin_width,
        color='tab:blue', edgecolor='black', alpha=1, label='Sorgenti NON Rilevate (Missing)')

# 4. Evidenziamo l'area critica intorno a 8.80
plt.axvspan(8.5, 10.01, color='orange', alpha=0.2, label='Zona Anomalie')

plt.yscale('log')
plt.xlabel('Magnitudine')
plt.ylabel('Numero di Stelle Mancanti')
plt.title('Distribuzione delle stelle presenti nel catalogo ma NON rilevate dal software')
plt.gca().invert_xaxis()  # Magnitudini luminose a destra, deboli a sinistra (o viceversa in base al tuo standard)

# Applico la stessa formattazione dell'asse X del tuo grafico originale
plt.gca().set_xticks(subset_ticks)
plt.gca().set_xticklabels(subset_labels, rotation=45, ha='right')

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# --- STAMPA DI DEBUG PER L'INTERVALLO 8.80 ---
# Identifichiamo i bin vicini a 8.80 per vedere quante ne mancano esattamente
idx_interest = np.where((bin_centers >= 8.5) & (bin_centers <= 10.01))[0]
print("\n--- Analisi Stelle Mancanti intorno a Mag 8.80 ---")
for i in idx_interest:
    print(f"Mag {bin_centers[i]:.2f}: Perse {int(counts_missing[i])} su {int(counts_cat[i])} catalogate "
          f"({(counts_missing[i] / counts_cat[i]) * 100:.1f}% perse)")

# --- 1. DEFINIZIONE DELLE COORDINATE ---

ra_col_cat = 'RAJ2000' if 'RAJ2000' in tbl_cat.colnames else 'ra'
dec_col_cat = 'DEJ2000' if 'DEJ2000' in tbl_cat.colnames else 'dec'

ra_col_det = 'RAJ2000' if 'RAJ2000' in tbl_catalogate_corr.colnames else 'ra'
dec_col_det = 'DEJ2000' if 'DEJ2000' in tbl_catalogate_corr.colnames else 'dec'

ra_non_corr = 'RAJ2000' if 'RAJ2000' in tbl_trovate_non_corr.colnames else 'ra'
dec_non_corr = 'DEJ2000' if 'DEJ2000' in tbl_trovate_non_corr.colnames else 'dec'

# Creiamo gli oggetti coordinate per il Catalogo (tutte le stelle attese)
coords_cat = SkyCoord(ra=tbl_cat[ra_col_cat], dec=tbl_cat[dec_col_cat], unit=u.deg)

# Creiamo gli oggetti coordinate per le Sorgenti Rilevate (quelle trovate)
coords_det = SkyCoord(ra=tbl_catalogate_corr[ra_col_det], dec=tbl_catalogate_corr[dec_col_det], unit=u.deg)

# Creiamo gli oggetti coordinate per le Sorgenti Rilevate (quelle trovate) senza corrispondenza
coords_non_corr = SkyCoord(ra=tbl_trovate_non_corr[ra_non_corr], dec=tbl_trovate_non_corr[dec_non_corr], unit=u.deg)

# --- 2. MATCHING INVERSO (TROVARE CHI MANCA) ---
# Per ogni stella del catalogo, cerchiamo la sorgente rilevata più vicina
idx, d2d, _ = coords_cat.match_to_catalog_sky(coords_det)

# Definiamo una soglia di tolleranza (es. 2 arcosecondi)
# Se la distanza è maggiore, consideriamo la stella come "NON TROVATA"
max_sep = 0.003349 * u.deg
mask_not_found = d2d > max_sep

# Tabella delle stelle perse
tbl_missing = tbl_cat[mask_not_found]

# --- 3. FILTRAGGIO PER L'INTERVALLO DI INTERESSE ---
# Definiamo l'intervallo "sospetto" di saturazione
mag_min_sat = 8.0
mag_max_sat = 10.01
mask_sat_range = (tbl_missing['Mag'] >= mag_min_sat) & (tbl_missing['Mag'] <= mag_max_sat)

stars_missing_sat = tbl_missing[mask_sat_range]

print(f"\n--- REPORT STELLE MANCANTI ---")
print(f"Totale stelle nel catalogo: {len(tbl_cat)}")
print(f"Totale stelle NON rilevate: {len(tbl_missing)}")
print(f"Stelle mancarti nell'intervallo critico ({mag_min_sat} < Mag < {mag_max_sat}): {len(stars_missing_sat)}")

print(f"\n--- COORDINATE STELLE MANCANTI tra {mag_min_sat} e {mag_max_sat} ---")
# Mostriamo le colonne principali: ID (se c'è), RA, DEC, Mag
cols_to_show = [c for c in ['id', 'ID', 'Source', ra_col_cat, dec_col_cat, 'Mag'] if c in tbl_cat.colnames]
print(stars_missing_sat[cols_to_show])

if len(stars_missing_sat) > 0:
    # A. Recupero il percorso del file FITS dall'header del CSV catalogo
    header_info = leggi_header_da_csv(percorso_file_csv_cat)
    percorso_fits = header_info.get('PERCORSO_FILE', '')

    # Se il percorso non è nell'header, prova a ricostruirlo o avvisa
    if not percorso_fits or not os.path.exists(percorso_fits):
        print(f"ATTENZIONE: Impossibile trovare il file FITS originale in: {percorso_fits}")
        print("Impossibile calcolare i pixel esatti senza il WCS del FITS.")
    else:
        print(f"Caricamento WCS da: {percorso_fits}")

        # B. Carico il WCS
        with fits.open(percorso_fits) as hdu:
            wcs = WCS(hdu[0].header)
            image_data = hdu[0].data
            mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
            image_data = image_data - median
            data = image_data

            # C. Conversione RA/Dec -> Pixel
            # Creiamo coordinate solo per le stelle mancanti selezionate
            coords_target = SkyCoord(ra=stars_missing_sat[ra_col_cat],
                                     dec=stars_missing_sat[dec_col_cat],
                                     unit=u.deg)

            # world_to_pixel restituisce due array: x e y
            x_pix, y_pix = wcs.world_to_pixel(coords_target)

            # D. Aggiungo le colonne alla tabella risultati
            stars_missing_sat['X_Pixel'] = x_pix
            stars_missing_sat['Y_Pixel'] = y_pix

    # --- 5. STAMPA FINALE ---
    print(f"\n--- LISTA STELLE CATALOGATE MANCANTI CON COORDINATE PIXEL tra {mag_min_sat} e {mag_max_sat}---")

    # Seleziono le colonne da stampare (incluso ID se esiste)
    cols_base = ['ID', 'Mag', 'X_Pixel', 'Y_Pixel', ra_col_cat, dec_col_cat]
    cols_final = [c for c in cols_base if c in stars_missing_sat.colnames]

    # Converto in Pandas per una stampa pulita
    df_result = stars_missing_sat[cols_final].to_pandas()

    # Formattazione per leggere meglio i numeri
    pd.options.display.float_format = '{:.2f}'.format
    print(df_result.to_string(index=False))

    # Salvataggio su CSV opzionale
    csv_out = f"stelle_mancanti_pixel_img_{n_immagine}.csv"
    df_result.to_csv(csv_out, index=False)
    print(f"\nTabella salvata in: {csv_out}")

    print(f"\n--- LISTA STELLE TROVATE SENZA CORRISPONDENZA CON COORDINATE PIXEL")
    print(tbl_trovate_non_corr)

    # =============================================================================
    # INIZIO GENERAZIONE CUTOUT GRAFICI COMPLETI (ZOOM)
    # =============================================================================
    print(f"\n--- INIZIO CREAZIONE FIGURE ZOOOMATE ---")

    # 1. Configurazione cartella output (TROVATE MA NON CATALOGATE)
    output_dir_found = f"cutouts_trovate_non_cat_img_{n_immagine}"
    if not os.path.exists(output_dir_found):
        os.makedirs(output_dir_found)

    # 2. Configurazione cartella output (CATALOGATE MA NON TROVATE - MISSING)
    output_dir_missing = f"cutouts_cat_non_trovate_img_{n_immagine}_tra_{mag_min_sat}_e_{mag_max_sat}"
    if not os.path.exists(output_dir_missing):
        os.makedirs(output_dir_missing)

    print(f"Immagini 'Trovate Non Catalogate' in: {output_dir_found}")
    print(f"Immagini 'Mancanti (Sat)' in: {output_dir_missing}")

    # 3. Preparazione della figura "Master"
    fig_master, ax_master = plt.subplots()

    # A. Immagine di sfondo
    ax_master.imshow(data, cmap="grey_r", norm=LogNorm(), interpolation='nearest', origin='lower')

    # B. Tutti gli scatter plot
    # Coordinate per il plot (le abbiamo già calcolate nel codice originale)
    all_x, all_y = wcs.world_to_pixel(coords_cat)
    x_non_corr_pix = tbl_trovate_non_corr['xcentroid']
    y_non_corr_pix = tbl_trovate_non_corr['ycentroid']

    # Plot Catalogate (Tutte - Verde/Giallo)
    ax_master.scatter(all_x, all_y, s=20, c=magnitudini_cat, alpha=0.6, cmap='viridis_r', label='Catalogo (Tutte)')
    # Plot Mancanti (Rosse) - usiamo x_pix calcolato sopra per stars_missing_sat
    ax_master.scatter(x_pix, y_pix, s=40, edgecolor='red', linewidth=2,
                      label=f'Catalogate non trovate (mag tra {mag_min_sat} e {mag_max_sat})')
    # Plot Trovate NON catalogate (Ciano - i nostri target)
    ax_master.scatter(x_non_corr_pix, y_non_corr_pix, s=50, color='cyan', linewidth=2, marker='x',
                      label='Trovate, ma non catalogate')

    # C. Aperture (Cerchi gialli su quelle trovate non catalogate)
    scales = proj_plane_pixel_scales(wcs)
    pixel_scale_deg = np.mean(scales)
    r_in_pixels = max_sep.to(u.deg).value / pixel_scale_deg
    positions = np.transpose((x_non_corr_pix, y_non_corr_pix))
    aperture = CircularAperture(positions, r=r_in_pixels)
    aperture.plot(ax=ax_master, color='yellow', lw=1.5, alpha=0.8, label='Regione di correlazione')

    # D. Configurazione Legenda e Assi
    ax_master.set_xlabel("Pixel X")
    ax_master.set_ylabel("Pixel Y")
    ax_master.grid(color='white', alpha=0.2)

    # Gestione legenda
    handles, labels = ax_master.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_master.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small', framealpha=0.8)

    # Dimensione zoom (30x30 pixel totali -> +/- 15 dal centro)
    half_size = 15

    # =============================================================================
    # CICLO 1: TROVATE MA NON CATALOGATE
    # =============================================================================
    print(f"\nSalvataggio {len(tbl_trovate_non_corr)} immagini (Trovate ma non Catalogate)...")

    for i, row in enumerate(tbl_trovate_non_corr):
        x_cen = row['xcentroid']
        y_cen = row['ycentroid']

        # Zoom
        ax_master.set_xlim(x_cen - half_size, x_cen + half_size)
        ax_master.set_ylim(y_cen - half_size, y_cen + half_size)

        # Titolo
        ax_master.set_title(f"FOUND_NOT_CAT {i + 1}/{len(tbl_trovate_non_corr)}\nCentro: X={x_cen:.1f}, Y={y_cen:.1f}")

        # Salvataggio
        filename = f"target_{i:03d}_x{x_cen:.1f}_y{y_cen:.1f}_zoom.png"
        filepath = os.path.join(output_dir_found, filename)
        fig_master.savefig(filepath, dpi=150, bbox_inches='tight')

    # =============================================================================
    # CICLO 2: CATALOGATE MA NON TROVATE (MISSING)
    # =============================================================================
    # Usiamo stars_missing_sat che ha già X_Pixel e Y_Pixel calcolati

    print(f"\nSalvataggio {len(stars_missing_sat)} immagini (Catalogate ma non Trovate, Mag 8-10)...")

    for i, row in enumerate(stars_missing_sat):
        x_cen = row['X_Pixel']
        y_cen = row['Y_Pixel']
        mag_val = row['Mag']
        id_obj = row['ID'] if 'ID' in row.colnames else 'N/A'

        # Zoom
        ax_master.set_xlim(x_cen - half_size, x_cen + half_size)
        ax_master.set_ylim(y_cen - half_size, y_cen + half_size)

        # Titolo
        ax_master.set_title(
            f"MISSING {i + 1}/{len(stars_missing_sat)}\nID: {id_obj} | Mag: {mag_val:.2f}\nX={x_cen:.1f}, Y={y_cen:.1f}")

        # Salvataggio
        filename = f"missing_{i:03d}_mag{mag_val:.2f}_x{x_cen:.1f}_y{y_cen:.1f}.png"
        filepath = os.path.join(output_dir_missing, filename)
        fig_master.savefig(filepath, dpi=150, bbox_inches='tight')

    # Chiudiamo la figura master solo alla fine di TUTTI i cicli
    plt.close(fig_master)
    print(f"\nOperazioni completate.")
    print(f"Controlla le cartelle:\n1. {output_dir_found}\n2. {output_dir_missing}")

    # =============================================================================
    # FINE GENERAZIONE CUTOUT
    # =============================================================================

    print("Mostro l'immagine completa a schermo per riferimento...")
    plt.figure()
    plt.imshow(data, cmap="grey_r", norm=LogNorm(), interpolation='nearest', origin='lower')
    plt.scatter(all_x, all_y, s=20, c=magnitudini_cat, alpha=0.6, cmap='viridis_r', label='Catalogo (Tutte)')
    plt.scatter(x_pix, y_pix, s=40, edgecolor='red', linewidth=2,
                label='Catalogate non trovate (mag tra 8 e 10)')
    plt.scatter(x_non_corr_pix, y_non_corr_pix, s=50, color='cyan', linewidth=2, marker='x',
                label='Trovate, ma non catalogate')
    aperture.plot(color='yellow', lw=1)
    plt.legend(loc='upper right', fontsize='small', framealpha=0.8)
    plt.show()

else:
    print("Nessuna stella critica mancante trovata in questo intervallo.")