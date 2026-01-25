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
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from shapely.geometry import Point, Polygon
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning
from pathlib import Path
import matplotlib.ticker as ticker

# --- GESTIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='Units from inserted quantities will be ignored.')

# Registra il tempo di inizio
start_time = time.time()


def converti_valore(valore):
    valore = str(valore).strip()
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


def esegui_segmentazione_dinamica(data, fwhm, size, params):
    try:
        kernel = make_2dgaussian_kernel(fwhm, size=size)
        convolved_data = convolve(data, kernel)
    except Exception as e:
        return None

    finder = SourceFinder(npixels=params['pixel'], progress_bar=False)
    segment_map = finder(convolved_data, params['threshold_assoluta'])

    if segment_map is None:
        return None

    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

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
        # Restituisco anche la mediana per calcolare la soglia di saturazione corretta
        return data_sub, wcs, header, median


def filtra_vicini_saturi(tbl, median_bg, dist_limit=30, val_sat=254):
    """
    Rimuove dalla tabella le sorgenti che distano meno di dist_limit
    da una sorgente satura (valore >= val_sat), eccetto la sorgente satura stessa.
    """
    if tbl is None or len(tbl) == 0:
        return tbl

    # I dati in 'tbl' sono background-subtracted.
    soglia_effettiva = val_sat

    # Identifica le sorgenti sature
    mask_sature = tbl['max_value'] >= soglia_effettiva
    idx_sature = np.where(mask_sature)[0]

    if len(idx_sature) == 0:
        return tbl

    x = tbl['xcentroid']
    y = tbl['ycentroid']
    to_remove = set()

    for i_sat in idx_sature:
        # Calcola distanza da QUESTA sorgente satura a TUTTE le altre
        # (Vettorializzato per velocità)
        dx = x - x[i_sat]
        dy = y - y[i_sat]
        dists = np.hypot(dx, dy)

        # Trova indici entro il limite
        vicini = np.where(dists <= dist_limit)[0]

        for v in vicini:
            # Rimuovi se:
            # 1. Non è la stella satura stessa (v != i_sat)
            # 2. La stella vicina NON è essa stessa satura (per evitare di cancellare stelle doppie reali molto luminose)
            if v != i_sat and not mask_sature[v]:
                to_remove.add(v)

    if len(to_remove) > 0:
        # Crea una maschera per mantenere quelle NON nella lista di rimozione
        mask_keep = np.ones(len(tbl), dtype=bool)
        mask_keep[list(to_remove)] = False
        return tbl[mask_keep]

    return tbl


def unione_tabelle_ottimizzata(tbl_seg, tbl_cat, wcs, coords_catalogate, soglia_correlazione):
    tbl_seg.keep_columns(['label', 'xcentroid', 'ycentroid', 'area', 'max_value', 'kron_flux'])
    coords_trovate = wcs.pixel_to_world(tbl_seg['xcentroid'], tbl_seg['ycentroid'])
    tbl_seg['RA_centroid'] = coords_trovate.ra
    tbl_seg['DEC_centroid'] = coords_trovate.dec

    tbl_finale = Table()
    for colname in tbl_seg.colnames:
        tbl_finale[colname] = tbl_seg[colname][:0]
    tbl_finale['Corrispondenza'] = 'SI (Rank 1)'

    for colname in tbl_cat.colnames:
        tbl_finale[colname] = tbl_cat[colname][:0]

    colonne = tbl_finale.colnames
    if 'ID' in colonne:
        pos_id = colonne.index('ID')
        nuovo_ordine = colonne[:pos_id] + ['Catalogo'] + [col for col in colonne if
                                                          col != 'Catalogo' and col not in colonne[:pos_id]]
        tbl_finale = tbl_finale[nuovo_ordine]

    colonna_magnitudine = 'Mag'
    righe_da_aggiungere = []

    for idx_trovato in range(len(coords_trovate)):
        coord_trovata = coords_trovate[idx_trovato]
        distanza_da_trovata = coords_catalogate.separation(coord_trovata)
        mask_distanza = distanza_da_trovata <= soglia_correlazione
        indici_catalogate_vicine = np.where(mask_distanza)[0]
        ha_corrispondenza = len(indici_catalogate_vicine) > 0
        riga_base = {col: tbl_seg[col][idx_trovato] for col in tbl_seg.colnames}

        if ha_corrispondenza:
            magnitudini_vicine = tbl_cat[colonna_magnitudine][indici_catalogate_vicine]
            indici_luminosita_sort = np.argsort(magnitudini_vicine)
            indici_da_considerare = indici_catalogate_vicine[indici_luminosita_sort]

            for rank, idx_catalogata in enumerate(indici_da_considerare, 1):
                nuova_riga = riga_base.copy()
                for colname in tbl_cat.colnames:
                    nuova_riga[colname] = tbl_cat[colname][idx_catalogata]
                nuova_riga['Corrispondenza'] = f'SI (Rank {rank})'
                righe_da_aggiungere.append(nuova_riga)
        else:
            nuova_riga = riga_base.copy()
            nuova_riga['Corrispondenza'] = 'NO'
            for colname in tbl_cat.colnames:
                if colname == 'Catalogo':
                    nuova_riga[colname] = 'N/A'
                elif tbl_cat[colname].dtype.kind in ['i', 'u']:
                    nuova_riga[colname] = -999
                elif tbl_cat[colname].dtype.kind == 'f':
                    nuova_riga[colname] = np.nan
                else:
                    nuova_riga[colname] = None
            righe_da_aggiungere.append(nuova_riga)

    if righe_da_aggiungere:
        tbl_finale = Table(rows=righe_da_aggiungere, names=tbl_finale.colnames)

    return tbl_finale


# --- CONFIGURAZIONE ---
PARAMETRI_FISSI = {
    'threshold_assoluta': 3.61,
    'pixel': 3,
    'soglia_filtro_ass': 2.5,
    'soglia_filtro_rel': 0.05
}

run = 1
n_immagine = 35
cartella_csv_cat = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}"
file_csv_cat = sorted([f for f in os.listdir(cartella_csv_cat) if f.endswith('.csv')])
percorso_file_csv_cat = os.path.join(cartella_csv_cat, file_csv_cat[n_immagine])

header_info = leggi_header_da_csv(percorso_file_csv_cat)
percorso_fits = header_info.get('PERCORSO_FILE', '')
print(f"File FITS: {percorso_fits}")

dataframe = pd.read_csv(percorso_file_csv_cat, comment='#')
tbl_catalogate = Table.from_pandas(dataframe)

try:
    if hasattr(tbl_catalogate['RAJ2000'], 'value'):
        ra_values = tbl_catalogate['RAJ2000'].value
        dec_values = tbl_catalogate['DEJ2000'].value
    else:
        ra_values = np.array(tbl_catalogate['RAJ2000'])
        dec_values = np.array(tbl_catalogate['DEJ2000'])
    coords_catalogate = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)
except Exception:
    coords_catalogate = SkyCoord(ra=tbl_catalogate['RAJ2000'], dec=tbl_catalogate['DEJ2000'], unit=u.deg)

# Recupero data_sub e MEDIAN
data, wcs, header, median_val = elabora_file_fits(percorso_fits)

FWHM_RANGE = np.linspace(0.5, 4.5, 100)
SIZES_TO_TEST = [3, 5]
soglia_correlazione = 0.003349 * u.deg

plt.figure(figsize=(12, 7))
colors_corr = {3: 'red', 5: 'blue'}
colors_fp = {3: 'darkred', 5: 'darkblue'}

# --- LOOP ---
for size_val in SIZES_TO_TEST:
    print(f"\nInizio elaborazione per Kernel Size = {size_val}")
    n_corr = []
    n_non_corr = []

    last_halo_fwhm = 0.0

    for i, fwhm_val in enumerate(FWHM_RANGE):
        tbl_trovate = esegui_segmentazione_dinamica(data, fwhm=fwhm_val, size=size_val, params=PARAMETRI_FISSI)

        if tbl_trovate is None or len(tbl_trovate) == 0:
            n_corr.append(0)
            n_non_corr.append(0)
            continue

        # --- FILTRO AGGIUNTO QUI ---
        # Rimuove artefatti attorno alle stelle sature PRIMA del matching
        tbl_trovate = filtra_vicini_saturi(tbl_trovate, median_bg=median_val, dist_limit=30, val_sat=254)
        # ---------------------------

        tbl = unione_tabelle_ottimizzata(tbl_trovate, tbl_catalogate, wcs, coords_catalogate, soglia_correlazione)

        if len(tbl) > 0:
            mask_si = np.char.startswith(tbl['Corrispondenza'], 'SI')
            tbl_corr = tbl[mask_si]
            tbl_corr = tbl_corr[tbl_corr['Mag'] < 10.01]

            valore_corr = max(0, len(tbl_corr) - 270)
            n_corr.append(valore_corr)

            mask_no = np.char.startswith(tbl['Corrispondenza'], 'NO')
            n_non_corr.append(np.sum(mask_no))
            tbl_non_corr = tbl[mask_no]

            # --- CONTROLLO ALONE ---
            if len(tbl_non_corr) > 0:
                x_gigante = 1902
                y_gigante = 1351
                x_non_corr = tbl_non_corr['xcentroid']
                y_non_corr = tbl_non_corr['ycentroid']

                dx = x_non_corr - x_gigante
                dy = y_non_corr - y_gigante
                distanze_da_gigante = np.hypot(dx, dy)

                distanza_minima_gigante = np.min(distanze_da_gigante)

                if distanza_minima_gigante < 30:
                    last_halo_fwhm = fwhm_val
            # -----------------------

        else:
            n_corr.append(0)
            n_non_corr.append(0)

        if (i + 1) % 1 == 0:
            print(f"  FWHM step {i + 1}/{len(FWHM_RANGE)} completati...", end='\r')

    plt.plot(FWHM_RANGE, n_corr, color=colors_corr[size_val], linestyle='-', linewidth=2,
             label=f'Corrispondenze (Size {size_val})')
    plt.plot(FWHM_RANGE, n_non_corr, color=colors_fp[size_val], linestyle='--', linewidth=2,
             label=f'Falsi Positivi (Size {size_val})')

    if last_halo_fwhm > 0:
        plt.axvline(x=last_halo_fwhm, color=colors_fp[size_val], linestyle=':', alpha=0.8,
                    label=f'Fine Halo FPs (Size {size_val}) @ {last_halo_fwhm:.2f}')

# --- PLOTTING FINALE ---
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.xlabel('FWHM')
plt.ylabel('Numero di stelle (Log)')
plt.yscale('log')
plt.title(f'Analisi Sensibilità: Size 3 vs 5')

ax = plt.gca()
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y')

plt.legend()

end_time = time.time()
tempo_totale = end_time - start_time
print(f"\n\nProcesso completato in: {tempo_totale:.2f} secondi")

plt.show()