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
    """
    Risale la directory partendo dalla posizione dello script fino a trovare
    la cartella target (es. 'pmc_photometry').
    """
    path_corrente = Path(__file__).resolve()

    # Risaliamo fino a trovare la cartella target
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent

    # Fallback: se non la trova, usa la cartella dello script
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    """
    Cerca un file ricorsivamente in tutte le sottocartelle di base_dir.
    """
    files_trovati = list(base_dir.rglob(nome_file_esatto))

    if not files_trovati:
        return None

    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
        print(
            f"INFO: Trovati {len(files_trovati)} file '{nome_file_esatto}'. Uso il primo: {files_trovati[0].relative_to(base_dir)}")

    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    """
    Cerca una CARTELLA ricorsivamente in tutte le sottocartelle di base_dir.
    """
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]

    if not cartelle_trovate:
        return None

    cartelle_trovate.sort(key=lambda p: len(str(p)))

    if len(cartelle_trovate) > 1:
        print(
            f"INFO: Trovate {len(cartelle_trovate)} cartelle '{nome_cartella_esatto}'. Uso la prima: {cartelle_trovate[0].relative_to(base_dir)}")

    return cartelle_trovate[0]


# Definisco la BASE_DIR dinamicamente
BASE_DIR = trova_cartella_base("pmc_photometry")

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")

# Inizializzo Vizier
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)


# =============================================================================
# 1. FUNZIONI HELPER
# =============================================================================

def leggi_file_parametri(percorso):
    """Legge il file dei parametri in un dizionario."""
    parametri = {}
    if not os.path.exists(percorso):
        print(f"Warning: File parametri non trovato in {percorso}")
        return {}
    with open(percorso, 'r') as file:
        next(file, None)
        for riga in file:
            riga = riga.split('#')[0].strip()
            if riga:
                parts = riga.split()
                if len(parts) >= 2:
                    parametro = parts[0]
                    valore_str = parts[1]
                    try:
                        valore = float(valore_str) if '.' in valore_str else int(valore_str)
                        parametri[parametro] = valore
                    except ValueError:
                        pass
    return parametri


def elabora_file_fits(percorso_file_):
    """Carica il FITS e sottrae il fondo."""
    # memmap=False previene errori con BZERO/BSCALE
    with fits.open(percorso_file_, memmap=False) as hdu_list_:
        image_data_ = hdu_list_[0].data
        w_ = WCS(hdu_list_[0].header)
        mean_, median_, std_ = sigma_clipped_stats(image_data_, sigma=3.0)
        image_data_ = image_data_ - median_
        return image_data_, median_, w_


def calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pixel, k=2.5, r_min=3.5):
    """Esegue l'INTERA pipeline Kron per una singola stella."""
    somma_intensita = np.sum(valori_pixel)

    if somma_intensita <= 0:
        return np.nan, np.nan

    somma_momenti = np.sum(valori_pixel * distanze_pixel)
    r_1 = somma_momenti / somma_intensita

    # Raggio finale con soglia minima
    r_kron_finale = max(k * r_1, r_min)

    # Misura Fotometrica (Integrazione)
    aper = CircularAperture((xc, yc), r=r_kron_finale)
    phot = aperture_photometry(data, aper)

    return phot['aperture_sum'][0], r_kron_finale


def tabella_catalogo(image_file_, magnitudine_massima):
    """Seleziona le stelle del catalogo che rientrano nel riquadro e le filtra ai bordi."""
    hdu_list_ = fits.open(image_file_)
    wcs = WCS(hdu_list_[0].header)
    data_ = hdu_list_[0].data

    bordo = 7
    h, w = data_.shape[0], data_.shape[1]

    mag_limite_tra_hipparco_e_vizier = 7.
    tbl_catalogo_vizier = tbl_riquadro_esterno_vizier[
        (tbl_riquadro_esterno_vizier['gmag'] >= mag_limite_tra_hipparco_e_vizier)
    ]

    nome_catalogo_vizier = ["II/389/ps1_dr2"] * len(tbl_catalogo_vizier)
    colonne_vizier = {
        'ID': tbl_catalogo_vizier['objID'],
        'RAJ2000': tbl_catalogo_vizier['RAJ2000'],
        'DEJ2000': tbl_catalogo_vizier['DEJ2000'],
        'Mag': tbl_catalogo_vizier['gmag'],
        'Catalogo': nome_catalogo_vizier
    }

    nome_catalogo_hipparco = ["I/239/hip_main"] * len(tbl_catalogo_hipparco)
    colonne_hipparco = {
        'Catalogo': nome_catalogo_hipparco,
        'ID': tbl_catalogo_hipparco['HIP'],
        'RAJ2000': tbl_catalogo_hipparco['_RAJ2000'],
        'DEJ2000': tbl_catalogo_hipparco['_DEJ2000'],
        'Mag': tbl_catalogo_hipparco['Vmag'],
    }

    t1 = Table(colonne_vizier)
    t2 = Table(colonne_hipparco)

    tbl_unita_estesa = vstack([t1, t2])
    tbl_unita_estesa['Mag'].description = 'Magnitudine AB nel filtro g di Pan-STARRS'

    # --- FILTRAGGIO BORDI ---
    coords_catalogo = SkyCoord(
        ra=tbl_unita_estesa['RAJ2000'],
        dec=tbl_unita_estesa['DEJ2000'],
        unit=u.deg
    )

    x_pix, y_pix = wcs.world_to_pixel(coords_catalogo)

    mask_bordo = (
            (x_pix >= bordo) &
            (x_pix < (w - bordo)) &
            (y_pix >= bordo) &
            (y_pix < (h - bordo))
    )

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
    with open(filename, 'w') as f:
        f.write("# Header FITS:\n")
        for key, value in header_fits.items():
            clean_val = str(value).replace('\n', ' ')
            f.write(f"# {key}: {clean_val}\n")
        f.write(f"# PERCORSO_FILE: {nome_file_fits}\n")
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
# 2. FUNZIONE DI ANALISI PRINCIPALE (IMAGE SEGMENTATION)
# =============================================================================

def analisi_image_segmentation(percorso_file_, parametri_globali):
    # A. Setup
    data, fondo_iniziale, w = elabora_file_fits(percorso_file_)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # B. Parametri
    fwhm = parametri_globali.get('fwhm', 3.0)
    size = parametri_globali.get('size', 5)
    threshold = parametri_globali.get('threshold_assoluta', 3.0)
    pixel_n = parametri_globali.get('pixel', 5)

    # C. Segmentazione
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    finder = SourceFinder(npixels=pixel_n, progress_bar=False)
    segment_map = finder(convolved_data, threshold)

    # D. Catalogo Base
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    if len(tbl) == 0:
        return tbl, parametri_globali

    for col in ['xcentroid', 'ycentroid', 'kron_flux']:
        tbl[col].info.format = '.2f'

    livello_saturazione = 255 - fondo_iniziale - median
    tbl['saturazione'] = np.where(tbl['max_value'] >= livello_saturazione, 'SI', 'NO')

    # E. CALCOLO E FILTRAGGIO
    K_KRON = 2.5
    R_MIN_KRON = 3.5
    soglia_assoluta = 2.5
    soglia_relativa = 0.05
    bordo = 7
    ny, nx = data.shape

    lista_raggi_max = []
    kron_manuale_seg = []
    kron_manuale_aper = []
    raggi_kron_aper = []
    mask_keep = []

    for prop in cat:
        xc, yc = prop.xcentroid, prop.ycentroid

        # 1. Check Bordo
        dentro_riquadro = (xc >= bordo) and (xc < nx - bordo) and \
                          (yc >= bordo) and (yc < ny - bordo)

        if not dentro_riquadro:
            lista_raggi_max.append(0.5)
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan)
            raggi_kron_aper.append(np.nan)
            mask_keep.append(False)
            continue

        slices = prop.slices
        cutout_seg = segment_map.data[slices]
        y_loc, x_loc = np.where(cutout_seg == prop.label)

        ypix = y_loc + slices[0].start
        xpix = x_loc + slices[1].start

        valori_pixel = data[ypix, xpix]

        if len(valori_pixel) == 0:
            lista_raggi_max.append(0.5)
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan)
            raggi_kron_aper.append(np.nan)
            mask_keep.append(False)
            continue

        distanze_pix = np.hypot(xpix - xc, ypix - yc)

        r_max_pix = np.max(distanze_pix) if len(distanze_pix) > 0 else 0.5
        r_max_pix = max(r_max_pix, 0.5)
        lista_raggi_max.append(r_max_pix)

        # Ritaglio Box
        r_int = int(np.ceil(r_max_pix))
        y_min_box = int(max(0, yc - r_int))
        y_max_box = int(min(data.shape[0], yc + r_int + 1))
        x_min_box = int(max(0, xc - r_int))
        x_max_box = int(min(data.shape[1], xc + r_int + 1))
        cutout_box = data[y_min_box:y_max_box, x_min_box:x_max_box]

        y_grid, x_grid = np.ogrid[y_min_box:y_max_box, x_min_box:x_max_box]
        distanze_box = np.hypot(x_grid - xc, y_grid - yc)
        mask_circle = distanze_box <= r_max_pix

        pixels_apertura_completa = cutout_box[mask_circle]
        distanze_apertura_completa = distanze_box[mask_circle]

        flusso_kron_apertura, raggio_usato = calcola_flusso_kron_completo(
            data=data, xc=xc, yc=yc,
            valori_pixel=pixels_apertura_completa,
            distanze_pixel=distanze_apertura_completa,
            k=K_KRON, r_min=R_MIN_KRON
        )
        kron_manuale_aper.append(flusso_kron_apertura)
        raggi_kron_aper.append(raggio_usato)

        flusso_kron_seg, raggio_valore = calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pix,
                                                                      k=K_KRON, r_min=R_MIN_KRON)
        kron_manuale_seg.append(flusso_kron_seg)

        # 4. Check Soglie
        pixel_sopra_soglia_assoluta = np.sum(valori_pixel > soglia_assoluta)
        pixel_sopra_soglia_relativa = np.sum(valori_pixel > soglia_relativa * prop.max_value)

        is_good = (pixel_sopra_soglia_assoluta >= 3) and (pixel_sopra_soglia_relativa >= 2)
        mask_keep.append(is_good)

    tbl['kron_manuale_seg'] = kron_manuale_seg
    tbl['kron_manuale_aper'] = kron_manuale_aper
    tbl['raggio_kron_aper'] = raggi_kron_aper

    positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))
    tbl['somma_apertura_ultimo_pixel'] = esegui_fotometria_variabile(data, positions, lista_raggi_max)

    for col in ['somma_apertura_ultimo_pixel', 'kron_manuale_seg', 'kron_manuale_aper', 'raggio_kron_aper']:
        tbl[col].info.format = '%.2f'

    tbl_filtrato = tbl[mask_keep]

    if len(tbl_filtrato) > 0:
        tbl_filtrato['label'] = np.arange(1, len(tbl_filtrato) + 1)

    return tbl_filtrato, parametri_globali


# =============================================================================
# 3. BLOCCO DI ESECUZIONE (MAIN)
# =============================================================================

if __name__ == "__main__":

    soglia_correlazione = 0.003349 * u.deg

    try:
        run = int(input("Quale run vuoi elaborare: "))
    except ValueError:
        print("Input non valido.")
        exit()

    # --- RICERCA FILE PARAMETRI ---
    nome_params = 'parametri_image_segmentation.txt'
    file_parametri = cerca_file_nel_progetto(BASE_DIR, nome_params)

    if file_parametri is None:
        print(f"ERRORE CRITICO: File '{nome_params}' non trovato in nessuna sottocartella di {BASE_DIR}")
        exit()

    print(f"Caricamento parametri da: {file_parametri.relative_to(BASE_DIR)}")
    parametri_caricati = leggi_file_parametri(file_parametri)

    # --- RICERCA CARTELLE RUN ---
    nome_cartella_run = f"20250120_run{run}"
    found_folders = list(BASE_DIR.rglob(nome_cartella_run))

    if not found_folders:
        print(f"ERRORE: Cartella '{nome_cartella_run}' non trovata in nessuna sottocartella di {BASE_DIR}")
        exit()

    run_folder = found_folders[0]
    if len(found_folders) > 1:
        print(f"AVVISO: Trovate {len(found_folders)} cartelle. Uso la prima: {run_folder}")
    else:
        print(f"Cartella dati trovata: {run_folder.relative_to(BASE_DIR)}")

    # Cerca i file FITS
    estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
    file_list = []
    for ext in estensioni_valide:
        file_list.extend(run_folder.glob(ext))

    file_list = sorted(file_list, key=lambda x: x.name)
    file_list = [str(f) for f in file_list]

    if not file_list:
        print(f"ERRORE: Nessun file FITS trovato in {run_folder}")
        exit()

    print(f"Trovati {len(file_list)} file da elaborare.")

    # --- CARTELLA OUTPUT ---
    cartella_tabelle = cerca_cartella_nel_progetto(BASE_DIR, "tabelle")

    if cartella_tabelle is None:
        print(f"AVVISO: Cartella 'tabelle' non trovata. La creo nella root del progetto.")
        cartella_tabelle = BASE_DIR / "tabelle"
        cartella_tabelle.mkdir(exist_ok=True)
    else:
        print(f"Cartella output 'tabelle' rilevata in: {cartella_tabelle.relative_to(BASE_DIR)}")

    output_dir = cartella_tabelle / f"tabelle_unite/tabelle_unite_run_{run}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir)

    numero_stelle = []
    n = 0

    i = 0
    print(f"\n--- FASE 1: Segmentazione & Unione ({len(file_list)} files) ---")

    for percorso_file in tqdm(file_list, desc="Fase 1"):

        # Creazione tabella catalogo
        i += 1

        # Gestione Vizier e Hipparco (Solo al primo giro)
        if i == 1:
            hdu_list = fits.open(percorso_file)
            image_header = hdu_list[0].header
            ra_centro = image_header["RA"]
            dec_centro = image_header["DEC"]

            w = WCS(hdu_list[0].header)
            alto_destra = w.pixel_to_world(3071, 2047)
            centro = SkyCoord(ra_centro, dec_centro, unit=u.deg)

            riquadro_esterno_vizier = vizier.query_region(
                coord.SkyCoord(ra=ra_centro, dec=dec_centro, unit=(u.deg, u.deg), frame='icrs'),
                radius=Angle(centro.separation(alto_destra) * 1.5, "deg"),
                column_filters={'gmag': f'<{15}'},
            )
            tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]

            file_hipparco = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
            if file_hipparco is None:
                print(f"ERRORE CRITICO: Catalogo 'hipparco.fit' non trovato.")
                exit()

            hdu_list_hipparco = fits.open(file_hipparco)
            table_data = Table(hdu_list_hipparco[1].data)
            mag_limite_tra_hipparco_e_vizier = 7.
            tbl_catalogo_hipparco = table_data[(table_data['Vmag']) < mag_limite_tra_hipparco_e_vizier]
            hdu_list.close()

        mag_max = 15
        tbl_catalogate = tabella_catalogo(percorso_file, mag_max)

        # Creazione tabella segmentazione
        n += 1

        try:
            tbl, _ = analisi_image_segmentation(percorso_file, parametri_caricati)
            numero_stelle.append(len(tbl))
        except Exception as e:
            print(f"\n[ERRORE FATALE] Impossibile elaborare {Path(percorso_file).name}: {e}")
            continue

        header_fits = fits.getheader(percorso_file)

        # Selezione colonne dinamica
        df_trovate = tbl.to_pandas()
        df_catalogate = tbl_catalogate.to_pandas()

        all_cols = df_trovate.columns.tolist()
        cols_base = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']

        cols_extra = []
        if 'saturazione' in all_cols: cols_extra.append('saturazione')
        if 'kron_flux' in all_cols: cols_extra.append('kron_flux')

        start_search = all_cols.index('kron_flux') + 1 if 'kron_flux' in all_cols else 0
        for c in all_cols[start_search:]:
            if c not in cols_base and c not in cols_extra:
                cols_extra.append(c)

        cols_finali = cols_base + cols_extra
        cols_presenti = [c for c in cols_finali if c in df_trovate.columns]
        df_trovate = df_trovate[cols_presenti].copy()

        # WCS
        with fits.open(percorso_file, memmap=False) as hdu_list:
            w = WCS(hdu_list[0].header)
        coords_trovate = w.pixel_to_world(df_trovate['xcentroid'], df_trovate['ycentroid'])
        df_trovate['RA_centroid'] = coords_trovate.ra.deg
        df_trovate['DEC_centroid'] = coords_trovate.dec.deg

        # Matching
        if 'RAJ2000' in df_catalogate.columns:
            coords_catalogate = SkyCoord(ra=df_catalogate['RAJ2000'].values * u.deg,
                                         dec=df_catalogate['DEJ2000'].values * u.deg)

            idx_trovate, idx_catalogate, d2d, _ = coords_catalogate.search_around_sky(coords_trovate,
                                                                                      soglia_correlazione)

            matches = pd.DataFrame({
                'idx_t': idx_trovate,
                'idx_c': idx_catalogate,
                'dist': d2d.deg,
                'mag': df_catalogate.iloc[idx_catalogate]['Mag'].values
            })

            matches.sort_values(by=['idx_t', 'mag'], inplace=True)
            matches['rank'] = matches.groupby('idx_t').cumcount() + 1
            matches['Corrispondenza'] = 'SI (Rank ' + matches['rank'].astype(str) + ')'

            part_trovate = df_trovate.iloc[matches['idx_t']].reset_index(drop=True)
            part_catalogate = df_catalogate.iloc[matches['idx_c']].reset_index(drop=True)
            part_rank = matches[['Corrispondenza']].reset_index(drop=True)
            df_si = pd.concat([part_trovate, part_rank, part_catalogate], axis=1)

            matched_indices = set(matches['idx_t'].unique())
            unmatched_indices = list(set(range(len(df_trovate))) - matched_indices)

            if unmatched_indices:
                df_no = df_trovate.iloc[unmatched_indices].copy()
                df_no['Corrispondenza'] = 'NO'
                for col in df_catalogate.columns:
                    df_no[col] = np.nan
                df_no = df_no.reindex(columns=df_si.columns, fill_value=np.nan)
            else:
                df_no = pd.DataFrame(columns=df_si.columns)

            df_finale = pd.concat([df_si, df_no], ignore_index=True)
            if 'label' in df_finale.columns:
                df_finale.sort_values('label', inplace=True)

            cols = df_finale.columns.tolist()
            if 'ID' in cols and 'Catalogo' in cols:
                cols.remove('Catalogo')
                cols.insert(cols.index('ID'), 'Catalogo')
                df_finale = df_finale[cols]

            filename_out = output_dir / f'run_{run}_stelle_trovate_e_catalogate_immagine_{n:03d}.csv'

            header_dict_save = dict(header_fits)
            header_dict_save['PERCORSO_FILE'] = str(percorso_file)

            salva_csv_con_header_fits(df_finale, header_dict_save, filename_out, str(percorso_file),
                                      parametri_seg=parametri_caricati)

    print("\n--- FASE 2: Analisi Globale (Calcolo Raggi Max & Filtro Transienti) ---")

    file_csv_list = sorted([f for f in output_dir.glob('*.csv')])

    # Strutture dati
    all_ids = []
    all_radii = []
    map_index_to_meta = {}
    coords_ra = []
    coords_dec = []

    # UNICO CICLO DI LETTURA
    for file_csv in tqdm(file_csv_list, desc="Scansione Dati Globale"):
        try:
            # FIX: Carichiamo anche 'label' per identificare le sorgenti NO
            cols_to_load = ['ID', 'raggio_kron_aper', 'Corrispondenza', 'RA_centroid', 'DEC_centroid', 'label']
            df_temp = pd.read_csv(file_csv, comment='#', usecols=cols_to_load)

            # A. Dati per Raggi Max
            df_raggi = df_temp.dropna(subset=['ID', 'raggio_kron_aper'])
            all_ids.append(df_raggi['ID'].values)
            all_radii.append(df_raggi['raggio_kron_aper'].values)

            # B. Dati per Transienti
            mask_no = df_temp['Corrispondenza'] == 'NO'
            df_no = df_temp[mask_no]
            fname_str = str(file_csv)

            current_ra = df_no['RA_centroid'].values
            current_dec = df_no['DEC_centroid'].values
            current_labels = df_no['label'].values  # Usiamo label, non ID!

            start_idx = len(coords_ra)
            coords_ra.extend(current_ra)
            coords_dec.extend(current_dec)

            for k in range(len(current_labels)):
                map_index_to_meta[start_idx + k] = (fname_str, current_labels[k])

        except Exception as e:
            pass

    # Calcolo Raggi
    if len(all_ids) > 0:
        big_ids = np.concatenate(all_ids)
        big_radii = np.concatenate(all_radii)
        df_global = pd.DataFrame({'ID': big_ids, 'R': big_radii})
        map_raggi_max = df_global.groupby('ID')['R'].quantile(0.95).to_dict()
        print(f"\nMappate {len(map_raggi_max)} stelle uniche per apertura fissa.")
    else:
        print("ATTENZIONE: Nessun raggio trovato.")
        map_raggi_max = {}

    # Filtro Transienti
    set_da_cancellare = set()
    if coords_ra:
        print("Esecuzione cross-match spaziale sui candidati 'NO'...")
        c_tot = SkyCoord(ra=coords_ra * u.deg, dec=coords_dec * u.deg)
        dist_limit = 0.001 * u.deg

        idx1, idx2, _, _ = c_tot.search_around_sky(c_tot, dist_limit)

        indices_with_valid_neighbor = set()

        for k in range(len(idx1)):
            i1 = idx1[k]
            i2 = idx2[k]
            if i1 == i2: continue

            file1 = map_index_to_meta[i1][0]
            file2 = map_index_to_meta[i2][0]

            if file1 != file2:
                indices_with_valid_neighbor.add(i1)

        for i in range(len(coords_ra)):
            if i not in indices_with_valid_neighbor:
                set_da_cancellare.add(map_index_to_meta[i])

        print(f"Individuate {len(set_da_cancellare)} sorgenti 'NO' isolate da rimuovere.")
    else:
        print("Nessuna sorgente 'NO' trovata.")

    print("\n--- FASE 3: Fotometria Finale & Pulizia ---")


    def salva_finale(df, header_dict, output_file, fp_count):
        """Scrive il CSV inserendo il conteggio FP prima del percorso file."""
        with open(output_file, 'w') as f:
            f.write("# Header FITS:\n")
            # 1. Inserimento Conteggio FP
            f.write(f"# Numero di falsi positivi sicuri: {fp_count}\n")

            # 2. Inserimento PERCORSO_FILE
            if 'PERCORSO_FILE' in header_dict:
                f.write(f"# PERCORSO_FILE: {header_dict['PERCORSO_FILE']}\n")

            # 3. Resto dell'header
            for k, v in header_dict.items():
                if k != 'PERCORSO_FILE':
                    f.write(f"# {k}: {v}\n")
            f.write("#\n")
            df.to_csv(f, index=False)


    for file_csv in tqdm(file_csv_list, desc="Fase 3"):
        fname_str = str(file_csv)
        df_frame = pd.read_csv(file_csv, comment='#')
        header_info = leggi_header_da_csv(file_csv)

        # --- FILTRO TRANSIENTI (CORRETTO CON LABEL) ---
        fp_count = 0
        if set_da_cancellare:
            # Identifichiamo i label da rimuovere per questo specifico file
            labels_to_drop = [lbl for lbl in df_frame['label'] if (fname_str, lbl) in set_da_cancellare]
            fp_count = len(labels_to_drop)

            if labels_to_drop:
                # Rimozione basata su LABEL
                df_frame = df_frame[~df_frame['label'].isin(labels_to_drop)]
        # -------------------------

        path_fits = header_info.get('PERCORSO_FILE', '')
        if not os.path.exists(path_fits):
            p_obj = Path(path_fits)
            try:
                if "pmc_photometry" in p_obj.parts:
                    idx = p_obj.parts.index("pmc_photometry")
                    new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                    if new_path.exists(): path_fits = str(new_path)
                elif "prove_2" in p_obj.parts:
                    idx = p_obj.parts.index("prove_2")
                    new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                    if new_path.exists(): path_fits = str(new_path)
            except:
                pass

        if not os.path.exists(path_fits): continue

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
                # Fallback sicuro: cerca nel dataframe usando l'ID
                try:
                    val = df_frame.loc[df_frame['ID'] == star_id, 'raggio_kron_aper'].values[0]
                    r_globale = val
                except:
                    r_globale = np.nan
            raggi_fissi.append(r_globale)

            if r_globale > 0 and not np.isnan(r_globale):
                row = df_frame.loc[df_frame['ID'] == star_id].iloc[0]
                pos = (row['xcentroid'], row['ycentroid'])
                aper = CircularAperture(pos, r=r_globale)
                phot = aperture_photometry(data_sub, aper)
                flussi_calcolati.append(phot['aperture_sum'][0])
            else:
                flussi_calcolati.append(np.nan)

        df_frame['flusso_fisso_max_run'] = flussi_calcolati
        df_frame['raggio_fisso_max_run'] = raggi_fissi

        # Format
        df_frame['flusso_fisso_max_run'] = df_frame['flusso_fisso_max_run'].map(
            lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')
        df_frame['raggio_fisso_max_run'] = df_frame['raggio_fisso_max_run'].map(
            lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')

        if 'label' in df_frame.columns and 'Corrispondenza' in df_frame.columns:
            df_frame.sort_values(by=['label', 'Corrispondenza'], ascending=[True, True], inplace=True)

        cols = df_frame.columns.tolist()
        cols_move = ['flusso_fisso_max_run', 'raggio_fisso_max_run']
        target = 'RA_centroid'
        for c in cols_move:
            if c in cols: cols.remove(c)
        if target in cols:
            idx = cols.index(target)
            for c in reversed(cols_move): cols.insert(idx, c)
        else:
            cols.extend(cols_move)

        df_frame = df_frame[cols]
        # Salvataggio con la nuova funzione header
        salva_finale(df_frame, header_info, file_csv, fp_count)

    print(f"\nElaborazione Finale completata. I file in {output_dir_str} sono stati aggiornati.")