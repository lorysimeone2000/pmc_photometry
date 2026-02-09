import pandas as pd
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import os
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
import warnings
from astropy.wcs import FITSFixedWarning
from photutils.datasets import make_100gaussians_image
from scipy.optimize import curve_fit
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.utils.data import download_file
from astropy.table import Table, vstack
from photutils.detection import find_peaks
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from shapely.geometry import Point, Polygon
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning

# --- IMPORT FONDAMENTALE PER LA PORTABILITÀ ---
from pathlib import Path

# --- GESTIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI DINAMICA (PORTABILITÀ TOTALE)
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
        print(
            f"INFO: Trovati {len(files_trovati)} file '{nome_file_esatto}'. Uso il primo: {files_trovati[0].relative_to(base_dir)}")
    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    if len(cartelle_trovate) > 1:
        print(
            f"INFO: Trovate {len(cartelle_trovate)} cartelle '{nome_cartella_esatto}'. Uso la prima: {cartelle_trovate[0].relative_to(base_dir)}")
    return cartelle_trovate[0]


BASE_DIR = trova_cartella_base("pmc_photometry")
print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")

RUN = [1, 2, 3]

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)


# =============================================================================
# 1. FUNZIONI HELPER
# =============================================================================

def leggi_file_parametri(percorso):
    parametri = {}
    if not os.path.exists(percorso): return {}
    with open(percorso, 'r') as file:
        next(file, None)
        for riga in file:
            riga = riga.split('#')[0].strip()
            if riga:
                parts = riga.split()
                if len(parts) >= 2:
                    try:
                        valore = float(parts[1]) if '.' in parts[1] else int(parts[1])
                        parametri[parts[0]] = valore
                    except ValueError:
                        pass
    return parametri


def elabora_file_fits(percorso_file_):
    with fits.open(percorso_file_, memmap=False) as hdu_list_:
        image_data_ = hdu_list_[0].data
        w_ = WCS(hdu_list_[0].header)
        mean_, median_, std_ = sigma_clipped_stats(image_data_, sigma=3.0)
        image_data_ = image_data_ - median_
        return image_data_, median_, w_


def calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pixel, k=2.5, r_min=3.5):
    somma_intensita = np.sum(valori_pixel)
    if somma_intensita <= 0: return np.nan, np.nan
    somma_momenti = np.sum(valori_pixel * distanze_pixel)
    r_1 = somma_momenti / somma_intensita
    r_kron_finale = max(k * r_1, r_min)
    aper = CircularAperture((xc, yc), r=r_kron_finale)
    phot = aperture_photometry(data, aper)
    return phot['aperture_sum'][0], r_kron_finale


def tabella_catalogo(image_file_, magnitudine_massima):
    hdu_list_ = fits.open(image_file_)
    wcs = WCS(hdu_list_[0].header)
    data_ = hdu_list_[0].data
    bordo = 7
    h, w = data_.shape[0], data_.shape[1]

    mag_limite_tra_hipparco_e_vizier = 7.
    tbl_catalogo_vizier = tbl_riquadro_esterno_vizier[
        (tbl_riquadro_esterno_vizier['gmag'] >= mag_limite_tra_hipparco_e_vizier)
    ]

    colonne_vizier = {
        'ID': tbl_catalogo_vizier['objID'], 'RAJ2000': tbl_catalogo_vizier['RAJ2000'],
        'DEJ2000': tbl_catalogo_vizier['DEJ2000'], 'Mag': tbl_catalogo_vizier['gmag'],
        'Catalogo': ["II/389/ps1_dr2"] * len(tbl_catalogo_vizier)
    }
    colonne_hipparco = {
        'Catalogo': ["I/239/hip_main"] * len(tbl_catalogo_hipparco),
        'ID': tbl_catalogo_hipparco['HIP'], 'RAJ2000': tbl_catalogo_hipparco['_RAJ2000'],
        'DEJ2000': tbl_catalogo_hipparco['_DEJ2000'], 'Mag': tbl_catalogo_hipparco['Vmag'],
    }

    t1 = Table(colonne_vizier)
    t2 = Table(colonne_hipparco)
    tbl_unita_estesa = vstack([t1, t2])

    coords_catalogo = SkyCoord(ra=tbl_unita_estesa['RAJ2000'], dec=tbl_unita_estesa['DEJ2000'], unit=u.deg)
    x_pix, y_pix = wcs.world_to_pixel(coords_catalogo)
    mask_bordo = ((x_pix >= bordo) & (x_pix < (w - bordo)) & (y_pix >= bordo) & (y_pix < (h - bordo)))
    tbl_cataloghi_ = tbl_unita_estesa[mask_bordo]
    hdu_list_.close()
    return tbl_cataloghi_


def esegui_fotometria_variabile(data, positions, raggi):
    flussi = []
    for (xc, yc), r in zip(positions, raggi):
        if r > 0 and not np.isnan(r):
            aper = CircularAperture((xc, yc), r=r)
            phot = aperture_photometry(data, aper)
            flussi.append(phot['aperture_sum'][0])
        else:
            flussi.append(np.nan)
    return flussi


def salva_csv_con_header_fits(dataframe, header_fits, filename, nome_file_fits, parametri_seg=None):
    nome_solo = os.path.basename(str(nome_file_fits))
    with open(filename, 'w') as f:
        f.write("# Header FITS:\n")
        for key, value in header_fits.items():
            clean_val = str(value).replace('\n', ' ')
            f.write(f"# {key}: {clean_val}\n")
        f.write(f"# NOME_FILE: {nome_solo}\n")
        f.write("#\n# PARAMETRI SEGMENTAZIONE:\n")
        if parametri_seg:
            for key, value in parametri_seg.items():
                f.write(f"# {key}: {value}\n")
        f.write("#\n")
        dataframe.to_csv(f, index=False)


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
    return valore


def leggi_header_da_csv(filename):
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            else:
                break
    return header_dict


# =============================================================================
# 2. FUNZIONE DI ANALISI PRINCIPALE
# =============================================================================

def analisi_image_segmentation(percorso_file_, parametri_globali):
    data, fondo_iniziale, w = elabora_file_fits(percorso_file_)
    fwhm = parametri_globali.get('fwhm', 3.0)
    size = parametri_globali.get('size', 5)
    threshold = parametri_globali.get('threshold_assoluta', 3.0)
    pixel_n = parametri_globali.get('pixel', 5)

    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    finder = SourceFinder(npixels=pixel_n, progress_bar=False)
    segment_map = finder(convolved_data, threshold)
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    if len(tbl) == 0: return tbl, parametri_globali

    for col in ['xcentroid', 'ycentroid', 'kron_flux']:
        tbl[col].info.format = '.2f'

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    livello_saturazione = 255 - fondo_iniziale - median
    tbl['saturazione'] = np.where(tbl['max_value'] >= livello_saturazione, 'SI', 'NO')

    K_KRON, R_MIN_KRON = 2.5, 3.5
    soglia_assoluta, soglia_relativa = 2.5, 0.05
    bordo, ny, nx = 7, data.shape[0], data.shape[1]

    lista_raggi_max, kron_manuale_seg, kron_manuale_aper, raggi_kron_aper, mask_keep = [], [], [], [], []

    for prop in cat:
        xc, yc = prop.xcentroid, prop.ycentroid
        if not ((xc >= bordo) and (xc < nx - bordo) and (yc >= bordo) and (yc < ny - bordo)):
            lista_raggi_max.append(0.5);
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan);
            raggi_kron_aper.append(np.nan);
            mask_keep.append(False)
            continue

        slices = prop.slices
        valori_pixel = data[prop.slices][segment_map.data[prop.slices] == prop.label]

        if len(valori_pixel) == 0:
            lista_raggi_max.append(0.5);
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan);
            raggi_kron_aper.append(np.nan);
            mask_keep.append(False)
            continue

        y_idx, x_idx = np.indices(segment_map.data[prop.slices].shape)
        ypix = y_idx[segment_map.data[prop.slices] == prop.label] + slices[0].start
        xpix = x_idx[segment_map.data[prop.slices] == prop.label] + slices[1].start
        distanze_pix = np.hypot(xpix - xc, ypix - yc)

        r_max_pix = max(np.max(distanze_pix) if len(distanze_pix) > 0 else 0.5, 0.5)
        lista_raggi_max.append(r_max_pix)

        r_int = int(np.ceil(r_max_pix))
        y_min, y_max = max(0, int(yc - r_int)), min(data.shape[0], int(yc + r_int + 1))
        x_min, x_max = max(0, int(xc - r_int)), min(data.shape[1], int(xc + r_int + 1))
        cutout = data[y_min:y_max, x_min:x_max]
        y_g, x_g = np.ogrid[y_min:y_max, x_min:x_max]
        dist_box = np.hypot(x_g - xc, y_g - yc)
        mask_circle = dist_box <= r_max_pix

        fl_aper, r_used = calcola_flusso_kron_completo(data, xc, yc, cutout[mask_circle], dist_box[mask_circle], K_KRON,
                                                       R_MIN_KRON)
        kron_manuale_aper.append(fl_aper)
        raggi_kron_aper.append(r_used)

        fl_seg, _ = calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pix, K_KRON, R_MIN_KRON)
        kron_manuale_seg.append(fl_seg)

        is_good = (np.sum(valori_pixel > soglia_assoluta) >= 3) and (
                np.sum(valori_pixel > soglia_relativa * prop.max_value) >= 2)
        mask_keep.append(is_good)

    tbl['kron_manuale_seg'] = kron_manuale_seg
    tbl['kron_manuale_aper'] = kron_manuale_aper
    tbl['raggio_kron_aper'] = raggi_kron_aper
    tbl['somma_apertura_ultimo_pixel'] = esegui_fotometria_variabile(data,
                                                                     np.transpose((tbl['xcentroid'], tbl['ycentroid'])),
                                                                     lista_raggi_max)

    for col in ['somma_apertura_ultimo_pixel', 'kron_manuale_seg', 'kron_manuale_aper', 'raggio_kron_aper']:
        tbl[col].info.format = '%.2f'

    tbl_filtrato = tbl[mask_keep]
    if len(tbl_filtrato) > 0: tbl_filtrato['label'] = np.arange(1, len(tbl_filtrato) + 1)

    return tbl_filtrato, parametri_globali


# =============================================================================
# 3. BLOCCO DI ESECUZIONE (MAIN)
# =============================================================================

if __name__ == "__main__":

    soglia_correlazione = 0.003349 * u.deg
    dist_ripetizione = 0.0011 * u.deg

    nome_params = 'parametri_image_segmentation.txt'
    file_parametri = cerca_file_nel_progetto(BASE_DIR, nome_params)
    if file_parametri is None: exit()
    parametri_caricati = leggi_file_parametri(file_parametri)

    # Accumulatore per tutti i file CSV generati in tutte le run
    tutti_i_file_csv_generati = []

    # --- STRUTTURE PER IL TRACKING GLOBALE DEGLI OGGETTI (LABEL PERSISTENTI) ---
    # MODIFICA: Ottimizzazione Tracking
    # global_tracker_coords: contiene SOLO le coordinate degli oggetti NON CATALOGATI (NO)
    global_tracker_coords = None
    global_tracker_labels = []  # Label corrispondenti per gli oggetti NO
    global_max_label = 0  # Contatore globale
    global_catalog_label_map = {}  # Dizionario {ID_CATALOGO: LABEL_ASSEGNATO} per gli oggetti SI

    # --- CICLO PER OGNI RUN ---
    for run in RUN:
        print(f"\n==================== ELABORAZIONE RUN {run} ====================")
        nome_cartella_run = f"20250120_run{run}"
        found_folders = list(BASE_DIR.rglob(nome_cartella_run))
        if not found_folders:
            print(f"Run {run} non trovata, salto.")
            continue
        run_folder = found_folders[0]

        estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
        file_list = []
        for ext in estensioni_valide: file_list.extend(run_folder.glob(ext))
        file_list = sorted([str(f) for f in file_list])
        if not file_list:
            print(f"Nessun FITS in Run {run}, salto.")
            continue

        cartella_tabelle = cerca_cartella_nel_progetto(BASE_DIR, "tabelle")
        if cartella_tabelle is None:
            cartella_tabelle = BASE_DIR / "tabelle"
            cartella_tabelle.mkdir(exist_ok=True)

        output_dir = cartella_tabelle / f"tabelle_unite/tabelle_unite_run_{run}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_str = str(output_dir)

        # --- FASE 1: CREAZIONE TABELLE UNITE ---
        print(f"--- FASE 1: Segmentazione & Unione ({len(file_list)} files) ---")

        for n, percorso_file in enumerate(tqdm(file_list, desc=f"Fase 1 Run {run}"), 1):
            if n == 1:
                hdu_list = fits.open(percorso_file)
                w = WCS(hdu_list[0].header)
                ra_c, dec_c = hdu_list[0].header["RA"], hdu_list[0].header["DEC"]

                alto_destra = w.pixel_to_world(3071, 2047)
                centro = SkyCoord(ra_c, dec_c, unit=u.deg)

                riquadro_esterno_vizier = vizier.query_region(
                    coord.SkyCoord(ra=ra_c, dec=dec_c, unit=(u.deg, u.deg), frame='icrs'),
                    radius=Angle(centro.separation(alto_destra) * 1.5, "deg"),
                    column_filters={'gmag': f'<{15}'}
                )
                tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]

                file_hipparco = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
                hdu_list_hipparco = fits.open(file_hipparco)
                table_data = Table(hdu_list_hipparco[1].data)
                tbl_catalogo_hipparco = table_data[(table_data['Vmag']) < 7.]
                hdu_list.close()

            tbl_catalogate = tabella_catalogo(percorso_file, 15)
            try:
                tbl, _ = analisi_image_segmentation(percorso_file, parametri_caricati)
            except Exception:
                continue

            df_trovate = tbl.to_pandas()
            df_catalogate = tbl_catalogate.to_pandas()

            all_cols = df_trovate.columns.tolist()
            cols_keep = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']
            for c in ['saturazione', 'kron_flux']:
                if c in all_cols: cols_keep.append(c)
            extra_flux = ['kron_manuale_seg', 'kron_manuale_aper', 'somma_apertura_ultimo_pixel', 'raggio_kron_aper']
            for c in extra_flux:
                if c in all_cols: cols_keep.append(c)

            df_trovate = df_trovate[[c for c in cols_keep if c in df_trovate.columns]].copy()

            # Calcolo coordinate astronomiche per le sorgenti trovate
            with fits.open(percorso_file, memmap=False) as hdu:
                w = WCS(hdu[0].header)
            coords = w.pixel_to_world(df_trovate['xcentroid'], df_trovate['ycentroid'])
            df_trovate['RA_centroid'] = coords.ra.deg
            df_trovate['DEC_centroid'] = coords.dec.deg

            # Spostamento colonne RA/DEC per ordine visuale
            cols_order = df_trovate.columns.tolist()
            if 'ycentroid' in cols_order:
                for c in ['RA_centroid', 'DEC_centroid']:
                    if c in cols_order: cols_order.remove(c)
                idx_y = cols_order.index('ycentroid')
                cols_order.insert(idx_y + 1, 'RA_centroid')
                cols_order.insert(idx_y + 2, 'DEC_centroid')
                df_trovate = df_trovate[cols_order]

            # =================================================================
            # MATCHING COL CATALOGO (Spostato PRIMA del Tracking)
            # =================================================================
            if 'RAJ2000' in df_catalogate.columns:
                c_cat = SkyCoord(ra=df_catalogate['RAJ2000'].values * u.deg,
                                 dec=df_catalogate['DEJ2000'].values * u.deg)
                idx_t, idx_c, d2d, _ = c_cat.search_around_sky(coords, soglia_correlazione)

                matches = pd.DataFrame(
                    {'idx_t': idx_t, 'idx_c': idx_c, 'dist': d2d.deg, 'mag': df_catalogate.iloc[idx_c]['Mag'].values})
                matches.sort_values(by=['idx_t', 'mag'], inplace=True)
                matches['rank'] = matches.groupby('idx_t').cumcount() + 1
                matches['Corrispondenza'] = 'SI (Rank ' + matches['rank'].astype(str) + ')'

                df_si = pd.concat([
                    df_trovate.iloc[matches['idx_t']].reset_index(drop=True),
                    matches[['Corrispondenza']].reset_index(drop=True),
                    df_catalogate.iloc[matches['idx_c']].reset_index(drop=True)
                ], axis=1)

                unmatched = list(set(range(len(df_trovate))) - set(matches['idx_t']))
                df_no = df_trovate.iloc[unmatched].copy()
                df_no['Corrispondenza'] = 'NO'
                for c in df_catalogate.columns: df_no[c] = np.nan

                df_final = pd.concat([df_si, df_no], ignore_index=True)
            else:
                df_final = df_trovate.copy()
                df_final['Corrispondenza'] = 'NO'

            # =================================================================
            # INIZIO BLOCCO: TRACKING GLOBALE OTTIMIZZATO
            # =================================================================
            # Inizializza contatore label se prima volta
            if global_max_label == 0 and len(df_final) > 0:
                # Assicuriamoci di partire da 0 o da max(df_trovate) ma qui assegniamo noi
                pass

            # Prepara colonna label finale
            final_labels = np.zeros(len(df_final), dtype=int)

            # 1. GESTIONE OGGETTI CATALOGATI (SI) - Usa ID per matching veloce
            mask_cat = df_final['Corrispondenza'] != 'NO'
            if mask_cat.any():
                indices_cat = np.where(mask_cat)[0]
                ids_cat = df_final.loc[mask_cat, 'ID'].values

                for idx_row, cat_id in zip(indices_cat, ids_cat):
                    if cat_id in global_catalog_label_map:
                        final_labels[idx_row] = global_catalog_label_map[cat_id]
                    else:
                        global_max_label += 1
                        global_catalog_label_map[cat_id] = global_max_label
                        final_labels[idx_row] = global_max_label

            # 2. GESTIONE OGGETTI NON CATALOGATI (NO) - Usa Matching Spaziale
            mask_no = ~mask_cat
            if mask_no.any():
                indices_no = np.where(mask_no)[0]
                coords_no = SkyCoord(ra=df_final.loc[mask_no, 'RA_centroid'].values * u.deg,
                                     dec=df_final.loc[mask_no, 'DEC_centroid'].values * u.deg)

                assigned_no_labels = np.zeros(len(indices_no), dtype=int)

                if global_tracker_coords is None:
                    # Primo batch assoluto di NO
                    start_id = global_max_label + 1
                    end_id = start_id + len(indices_no)
                    new_ids = np.arange(start_id, end_id)
                    assigned_no_labels = new_ids

                    global_tracker_coords = coords_no
                    global_tracker_labels = list(new_ids)
                    global_max_label = end_id - 1
                else:
                    # Match spaziale contro i NO già noti
                    idx, d2d, _ = coords_no.match_to_catalog_sky(global_tracker_coords)
                    matched_mask_no = d2d < soglia_correlazione

                    # Case A: Trovati nel tracker
                    if matched_mask_no.any():
                        idx_matched = idx[matched_mask_no]
                        assigned_no_labels[matched_mask_no] = np.array(global_tracker_labels)[idx_matched]

                    # Case B: Nuovi NO
                    unmatched_mask_no = ~matched_mask_no
                    num_new = np.sum(unmatched_mask_no)
                    if num_new > 0:
                        start_id = global_max_label + 1
                        end_id = start_id + num_new
                        new_ids = np.arange(start_id, end_id)
                        assigned_no_labels[unmatched_mask_no] = new_ids

                        # Aggiorna tracker
                        new_coords_obj = coords_no[unmatched_mask_no]
                        combined_ra = np.concatenate([global_tracker_coords.ra.deg, new_coords_obj.ra.deg])
                        combined_dec = np.concatenate([global_tracker_coords.dec.deg, new_coords_obj.dec.deg])
                        global_tracker_coords = SkyCoord(ra=combined_ra * u.deg, dec=combined_dec * u.deg)
                        global_tracker_labels.extend(new_ids)
                        global_max_label = end_id - 1

                final_labels[indices_no] = assigned_no_labels

            # Applica i label calcolati
            df_final['label'] = final_labels

            # Aggiunta colonne identificative Run e Immagine
            df_final['run_id'] = run
            df_final['img_index'] = n

            # =================================================================
            # FINE BLOCCO TRACKING
            # =================================================================

            if 'label' in df_final.columns: df_final.sort_values('label', inplace=True)

            cols = df_final.columns.tolist()
            if 'ID' in cols and 'Catalogo' in cols:
                cols.remove('Catalogo')
                cols.insert(cols.index('ID'), 'Catalogo')
                # Riordino finale per mettere run_id e img_index prima di Corrispondenza
                # Cerchiamo dove sono finiti run_id e img_index (sono in df_trovate, quindi in df_si e df_no)

            # Logica riordino colonne richiesta
            final_cols = df_final.columns.tolist()
            # Rimuoviamo temporaneamente
            for c in ['run_id', 'img_index']:
                if c in final_cols: final_cols.remove(c)

            # Cerchiamo l'indice di Corrispondenza
            if 'Corrispondenza' in final_cols:
                idx_corr = final_cols.index('Corrispondenza')
                final_cols.insert(idx_corr, 'img_index')
                final_cols.insert(idx_corr, 'run_id')
            else:
                # Fallback se Corrispondenza non c'è
                final_cols.insert(0, 'run_id')
                final_cols.insert(1, 'img_index')

            df_final = df_final[final_cols]

            file_out = output_dir / f'run_{run}_stelle_trovate_e_catalogate_immagine_{n:03d}.csv'
            salva_csv_con_header_fits(df_final, dict(fits.getheader(percorso_file)),
                                      file_out, str(percorso_file), parametri_caricati)

        # =============================================================================
        # FASE 2 & 3: RAGGI MAX E FLUSSO FISSO (PER RUN)
        # =============================================================================
        print(f"--- FASE 2 & 3: Analisi Fotometria Fissa per Run {run} ---")

        file_csv_list = sorted([f for f in output_dir.glob('*.csv')])

        # Salviamo la tupla (percorso_file, numero_run) per uso futuro
        for f in file_csv_list:
            tutti_i_file_csv_generati.append((f, run))

        all_ids = []
        all_radii = []

        for file_csv in tqdm(file_csv_list, desc="Scansione Raggi"):
            try:
                df_temp = pd.read_csv(file_csv, comment='#', usecols=['ID', 'raggio_kron_aper'])
                df_temp = df_temp.dropna(subset=['ID', 'raggio_kron_aper'])
                all_ids.append(df_temp['ID'].values)
                all_radii.append(df_temp['raggio_kron_aper'].values)
            except Exception:
                pass

        if len(all_ids) > 0:
            big_ids = np.concatenate(all_ids)
            big_radii = np.concatenate(all_radii)
            df_global = pd.DataFrame({'ID': big_ids, 'R': big_radii})
            map_raggi_max = df_global.groupby('ID')['R'].quantile(0.95).to_dict()
        else:
            map_raggi_max = {}


        def salva_csv_con_header_aggiornato(df, header_dict, output_file):
            with open(output_file, 'w') as f:
                f.write("# Header FITS:\n")
                for k, v in header_dict.items(): f.write(f"# {k}: {v}\n")
                f.write("#\n")
                df.to_csv(f, index=False)


        for file_csv in tqdm(file_csv_list, desc="Ricalcolo Flussi"):
            df_frame = pd.read_csv(file_csv, comment='#')
            header_info = leggi_header_da_csv(file_csv)

            # Recuperiamo percorso e nome file dall'header del CSV
            path_fits = header_info.get('PERCORSO_FILE', '')
            nome_fits = header_info.get('NOME_FILE', '')

            # Se il percorso scritto nel CSV non esiste o è vuoto, proviamo a trovarlo
            if not path_fits or not os.path.exists(path_fits):

                # Tentativo 1: Ricostruzione path (metodo vecchio)
                if path_fits:
                    p_obj = Path(path_fits)
                    try:
                        if "pmc_photometry" in p_obj.parts:
                            idx = p_obj.parts.index("pmc_photometry")
                            new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                            if new_path.exists():
                                path_fits = str(new_path)
                    except:
                        pass

                # Tentativo 2: Ricerca brutale per NOME in tutta la cartella BASE_DIR (pmc_photometry)
                # Questo scatta se il percorso è ancora invalido ma abbiamo il nome del file
                if (not path_fits or not os.path.exists(path_fits)) and nome_fits:
                    # rglob('*' + nome_fits) cerca il file ovunque sotto BASE_DIR
                    # Usiamo il nome esatto per evitare ambiguità
                    files_trovati = list(BASE_DIR.rglob(str(nome_fits).strip()))

                    if files_trovati:
                        # Prendiamo il primo trovato (di solito è unico)
                        path_fits = str(files_trovati[0])

            # Se dopo tutto questo il file non c'è, salto
            if not path_fits or not os.path.exists(path_fits):
                # print(f"ATTENZIONE: File FITS originale non trovato per {nome_fits}, salto.")
                continue

            with fits.open(path_fits, memmap=False) as hdu:
                data = hdu[0].data
                _, median_bg, _ = sigma_clipped_stats(data[::10, ::10], sigma=3.0)
                data_sub = data - median_bg

            raggi_fissi = []
            ids_presenti = df_frame['ID'].values
            flussi_calcolati = []

            for i, star_id in enumerate(ids_presenti):
                r_globale = map_raggi_max.get(star_id, np.nan)
                if np.isnan(r_globale) or r_globale <= 0:
                    if 'raggio_kron_aper' in df_frame.columns:
                        r_globale = df_frame.at[i, 'raggio_kron_aper']
                    else:
                        r_globale = np.nan
                raggi_fissi.append(r_globale)

                if r_globale > 0 and not np.isnan(r_globale):
                    pos = (df_frame.at[i, 'xcentroid'], df_frame.at[i, 'ycentroid'])
                    aper = CircularAperture(pos, r=r_globale)
                    phot = aperture_photometry(data_sub, aper)
                    flussi_calcolati.append(phot['aperture_sum'][0])
                else:
                    flussi_calcolati.append(np.nan)

            df_frame['flusso_fisso_max_run'] = flussi_calcolati
            df_frame['raggio_fisso_max_run'] = raggi_fissi

            df_frame['flusso_fisso_max_run'] = df_frame['flusso_fisso_max_run'].map(
                lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')
            df_frame['raggio_fisso_max_run'] = df_frame['raggio_fisso_max_run'].map(
                lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')

            if 'label' in df_frame.columns: df_frame.sort_values(by=['label', 'Corrispondenza'], inplace=True)
            salva_csv_con_header_aggiornato(df_frame, header_info, file_csv)

    # =============================================================================
    # FASE FINALE GLOBALE: STATISTICHE E ID SU TUTTE LE RUN
    # =============================================================================
    print("\n==================== FASE FINALE GLOBALE (TUTTE LE RUN) ====================")
    print(f"Elaborazione di {len(tutti_i_file_csv_generati)} file totali...")

    if not tutti_i_file_csv_generati:
        print("Nessun file generato. Esco.")
        exit()

    # 1. Caricamento Dati Globale
    lista_df = []
    # Ordiniamo per percorso per avere ordine cronologico
    tutti_i_file_csv_generati = sorted(tutti_i_file_csv_generati, key=lambda x: str(x[0]))

    for idx_file, (file_csv, run_number) in enumerate(tqdm(tutti_i_file_csv_generati, desc="Lettura Dati Globali")):
        df_temp = pd.read_csv(file_csv, comment='#')
        df_temp['file_index'] = idx_file
        df_temp['run_number'] = run_number  # Salviamo a che run appartiene
        df_temp['original_file_path'] = str(file_csv)
        df_temp['original_idx'] = df_temp.index
        lista_df.append(df_temp)

    big_df = pd.concat(lista_df, ignore_index=True)

    # 2. Assegnazione ID Univoco (run_unique_id) SU TUTTO IL DATASET
    # CORREZIONE TIPO: Inizializza come object per evitare errori di tipo
    big_df['run_unique_id'] = np.nan
    big_df['run_unique_id'] = big_df['run_unique_id'].astype(object)

    mask_si = big_df['Corrispondenza'].str.startswith('SI', na=False)
    big_df.loc[mask_si, 'run_unique_id'] = "CAT_" + big_df.loc[mask_si, 'ID'].astype(str)

    mask_no = big_df['Corrispondenza'] == 'NO'
    df_no = big_df[mask_no].copy()

    # Ordiniamo per file_index per rispettare la cronologia assoluta
    df_no.sort_values('file_index', inplace=True)

    known_clusters_coords = []
    known_clusters_ids = []
    threshold_deg = 0.0011
    unique_files = df_no['file_index'].unique()
    next_internal_id = 1
    no_mapping = {}  # Mappa globale (file_index, original_idx) -> ID

    for f_idx in tqdm(unique_files, desc="Matching oggetti NO (Multi-Run)"):
        subset = df_no[df_no['file_index'] == f_idx]
        if subset.empty: continue
        coords_subset = SkyCoord(ra=subset['RA_centroid'].values * u.deg,
                                 dec=subset['DEC_centroid'].values * u.deg)
        indices_subset = subset.index.tolist()

        if not known_clusters_coords:
            for i, (ra, dec) in enumerate(zip(subset['RA_centroid'], subset['DEC_centroid'])):
                cid = f"INT_{next_internal_id}"
                known_clusters_coords.append((ra, dec))
                known_clusters_ids.append(cid)
                no_mapping[indices_subset[i]] = cid
                next_internal_id += 1
        else:
            cluster_sc = SkyCoord(known_clusters_coords, unit=u.deg)
            idx_cluster, d2d, _ = coords_subset.match_to_catalog_sky(cluster_sc)
            for i, (match_idx, dist, ra_curr, dec_curr) in enumerate(
                    zip(idx_cluster, d2d, subset['RA_centroid'], subset['DEC_centroid'])):
                global_idx = indices_subset[i]
                if dist.deg <= threshold_deg:
                    no_mapping[global_idx] = known_clusters_ids[match_idx]
                else:
                    cid = f"INT_{next_internal_id}"
                    known_clusters_ids.append(cid)
                    known_clusters_coords.append((ra_curr, dec_curr))
                    known_clusters_ids.append(cid)
                    no_mapping[global_idx] = cid
                    next_internal_id += 1

    for idx, uid in no_mapping.items(): big_df.at[idx, 'run_unique_id'] = uid

    # 3. Calcolo Statistiche Globali (Su tutte le run) e Repetition per Run
    print("Calcolo statistiche globali e riorganizzazione colonne...")
    cols_flux = ['somma_apertura_ultimo_pixel', 'kron_manuale_seg', 'kron_manuale_aper', 'flusso_fisso_max_run']
    cols_flux_presenti = [c for c in cols_flux if c in big_df.columns]
    for c in cols_flux_presenti: big_df[c] = pd.to_numeric(big_df[c], errors='coerce')

    # Groupby ID per stats globali
    grouped = big_df.groupby('run_unique_id')
    means = grouped[cols_flux_presenti].mean()

    # CORREZIONE DEVIAZIONE STANDARD DELLA MEDIA (SEM)
    counts_grouped = grouped[cols_flux_presenti].count()
    stds_sample = grouped[cols_flux_presenti].std()
    stds = stds_sample / np.sqrt(counts_grouped)

    # Per repetitioni, dobbiamo contare per (ID, Run)
    # Crea una tabella pivot: Index=ID, Columns=Run, Values=Count
    repetition_pivot = pd.pivot_table(big_df, index='run_unique_id', columns='run_number', aggfunc='size', fill_value=0)

    # Mapping back to big_df
    stat_columns = []
    for c in cols_flux_presenti:
        col_mean = f'media_{c}'
        col_std = f'std_{c}'
        big_df[col_mean] = big_df['run_unique_id'].map(means[c])
        big_df[col_std] = big_df['run_unique_id'].map(stds[c])
        stat_columns.extend([col_mean, col_std])

    for c in stat_columns: big_df[c] = big_df[c].map(lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')

    # CORREZIONE: Convertiamo ID in object per poter ospitare stringhe "INT_X"
    big_df['ID'] = big_df['ID'].astype(object)

    mask_no_match = big_df['Corrispondenza'] == 'NO'
    big_df.loc[mask_no_match, 'ID'] = big_df.loc[mask_no_match, 'run_unique_id']

    # 4. Salvataggio su File
    files_groups = big_df.groupby('original_file_path')


    def salva_finale_global(df, header_dict, output_file, fp_count):
        # Estrae solo il nome del file dal percorso
        nome_solo = os.path.basename(str(output_file))
        with open(output_file, 'w') as f:
            f.write("# Header FITS:\n")
            f.write(f"# Numero di falsi positivi esclusi sicuramente: {fp_count}\n")
            for k, v in header_dict.items():
                if k != 'PERCORSO_FILE':  # Scriveremo noi il nome pulito
                    f.write(f"# {k}: {v}\n")
            f.write(f"# NOME_FILE: {nome_solo}\n")
            f.write("#\n")
            df.to_csv(f, index=False)


    # Calcolo i conteggi totali (su tutte le run) per ogni ID
    global_repetition_counts = big_df['run_unique_id'].value_counts()

    for file_path, df_file in tqdm(files_groups, desc="Salvataggio file globali aggiornati"):

        # Recuperiamo il numero della run da una qualsiasi riga del gruppo (sono tutte dello stesso file)
        current_run_num = df_file['run_number'].iloc[0]

        # MODIFICA: Colonna 'ripetizioni' con conteggio totale su tutte le run
        col_rip_name = 'ripetizioni'

        # Mappo il conteggio globale sulla colonna
        df_file[col_rip_name] = df_file['run_unique_id'].map(global_repetition_counts)

        # --- FILTRO TRANSIENTI DISATTIVATO (COMMENTATO) ---
        '''
        # Nota: qui "ripetizioni" deve riferirsi a cosa? Alla run corrente o globale?
        # Solitamente si filtra se appare 1 volta sola in TOTALE o nella RUN.

        mask_trash = (df_file['Corrispondenza'] == 'NO') & (df_file[col_rip_name] <= 1)
        num_falsi_positivi = mask_trash.sum()
        df_final_save = df_file[~mask_trash].copy()
        '''
        df_final_save = df_file.copy()
        num_falsi_positivi = 0
        # --------------------------------------------------

        header_orig = leggi_header_da_csv(file_path)

        cols = df_final_save.columns.tolist()
        for temp_c in ['file_index', 'original_file_path', 'original_idx', 'run_unique_id', 'run_number']:
            if temp_c in cols: cols.remove(temp_c)

        if col_rip_name in cols: cols.remove(col_rip_name)
        if 'saturazione' in cols:
            cols.insert(cols.index('saturazione') + 1, col_rip_name)
        else:
            cols.append(col_rip_name)

        for c_flux in cols_flux_presenti:
            c_mean, c_std = f'media_{c_flux}', f'std_{c_flux}'
            if c_flux in cols and c_mean in cols:
                cols.remove(c_mean);
                cols.remove(c_std)
                idx_flux = cols.index(c_flux)
                cols.insert(idx_flux + 1, c_mean);
                cols.insert(idx_flux + 2, c_std)

        df_final_save = df_final_save[cols]
        salva_finale_global(df_final_save, header_orig, file_path, num_falsi_positivi)

    print("\n--- ELABORAZIONE GLOBALE MULTI-RUN COMPLETATA CON SUCCESSO ---")