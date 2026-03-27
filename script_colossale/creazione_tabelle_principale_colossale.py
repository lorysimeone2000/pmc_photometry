import pandas as pd
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import time
import os
import sys
import gc
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
from astropy.coordinates import search_around_sky
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from shapely.geometry import Point, Polygon
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import label
import re

# --- IMPORT FONDAMENTALE PER LA MIA PORTABILITÀ ---
from pathlib import Path

# importo il modulo per la mia ricerca spaziale ultra veloce
from scipy.spatial import cKDTree

# --- GESTIONE WARNING ---
# ignoro i warning per mantenere pulito il mio output
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory del mio script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")
PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

from funzioni.utilita import *
from funzioni.astrometria import *

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"Moduli esterni caricati con successo.")
print(f"------------------------------")

# uso il mirror di Harvard scaricando le 5 bande fondamentali per simulare il mio sensore FLIR
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1,
)

def stampa_descrizioni_colonne_ps1():
    """
    Stampo tutte le informazioni disponibili per le colonne di magnitudine
    del catalogo Pan-STARRS DR2.
    """
    from astroquery.vizier import Vizier
    import pandas as pd

    print("\n" + "=" * 80)
    print("ANALISI COMPLETA COLONNE MAGNITUDINE - PAN-STARRS DR2 (II/389/ps1_dr2)")
    print("=" * 80)

    # recupero il catalogo
    print("\n1. Scaricamento metadati del catalogo...")
    catalogo = Vizier.get_catalogs("II/389/ps1_dr2")[0]

    # definisco i colori di magnitudine che voglio analizzare
    bande = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']
    bande_err = ['e_gmag', 'e_rmag', 'e_imag', 'e_zmag', 'e_ymag']
    bande_std = ['gmagStd', 'rmagStd', 'imagStd', 'zmagStd', 'ymagStd']

    tutte_colonne = bande + bande_err + bande_std

    print("\n2. Analisi dettagliata per colonna:")
    print("-" * 80)

    risultati = {}

    for col_name in tutte_colonne:
        if col_name in catalogo.columns:
            print(f"\n>>> COLONNA: {col_name}")
            print("-" * 50)

            # ottengo la mia colonna
            col = catalogo[col_name]

            # stampo tutte le informazioni disponibili
            print(f"  Descrizione: {col.description if hasattr(col, 'description') else 'N/A'}")
            print(f"  Unità: {col.unit if hasattr(col, 'unit') else 'N/A'}")
            print(f"  Formato: {col.format if hasattr(col, 'format') else 'N/A'}")

            # gestisco i meta dati
            if hasattr(col, 'meta'):
                print(f"  Meta dati:")
                for key, value in col.meta.items():
                    print(f"    {key}: {value}")

            # estraggo l'UCD
            if hasattr(col, 'meta') and 'ucd' in col.meta:
                print(f"  UCD: {col.meta['ucd']}")

            # cerco informazioni sulla lunghezza d'onda
            desc = col.description if hasattr(col, 'description') else ""
            if desc:
                # cerco pattern di lunghezza d'onda come "4866Å", "4866 A", "4866A", "4866 nm"
                import re
                patterns = [
                    r'(\d+)\s*[ÅA]',
                    r'(\d+)\s*nm',
                    r'(\d+)\s*microns',
                    r'(\d+\.?\d*)\s*μm',
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, desc)
                    if matches:
                        print(f"  Lunghezza d'onda trovata: {matches[0]} Å")
                        break

            risultati[col_name] = {
                'description': desc,
                'ucd': col.meta.get('ucd', 'N/A') if hasattr(col, 'meta') else 'N/A'
            }
        else:
            print(f"\n>>> COLONNA: {col_name} - NON TROVATA nel mio catalogo")

    print("\n" + "=" * 80)
    print("RIASSUNTO UCD TROVATI:")
    print("=" * 80)
    for col_name, info in risultati.items():
        print(f"{col_name:10s} -> UCD: {info['ucd']}")

    print("\n" + "=" * 80)

    return risultati


def scarica_intervalli_bande_ps1_da_descrizioni():
    """
    Scarico gli intervalli delle bande Pan-STARRS DR2 estraendo
    le lunghezze d'onda centrali dalle descrizioni delle colonne.
    """
    from astroquery.vizier import Vizier
    import re

    print("\n" + "=" * 70)
    print("SCARICAMENTO INTERVALLI BANDE PAN-STARRS DR2")
    print("=" * 70)

    # recupero il mio catalogo
    catalogo = Vizier.get_catalogs("II/389/ps1_dr2")[0]

    # annoto le FWHM delle bande Pan-STARRS che ho ricavato da Tonry+ 2012
    fwhm_nm = {
        'gmag': 137,
        'rmag': 140,
        'imag': 130,
        'zmag': 104,
        'ymag': 83
    }

    # imposto i miei valori di fallback validi (in nm)
    fallback_validi = {
        'gmag': 486.6,
        'rmag': 621.5,
        'imag': 754.5,
        'zmag': 867.9,
        'ymag': 963.3
    }

    # definisco il pattern per estrarre la lunghezza d'onda centrale prestando attenzione all'ordine
    # prima cerco il pattern con {AA} che mi serve in modo specifico per gmag
    patterns = [
        r'\((\d+)\s*\{AA\}\)',
        r'\((\d+)\s*A\)',
        r'(\d+)\s*[ÅA]',
        r'(\d+)\s*nm',
    ]

    bande = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']
    limiti_bande = {}

    print("\nEstrazione lunghezze d'onda dalle descrizioni:")
    print("-" * 70)

    for banda in bande:
        if banda in catalogo.columns:
            col = catalogo[banda]
            descrizione = col.description if hasattr(col, 'description') else ""

            print(f"\n{banda}:")
            print(f"  Descrizione: {descrizione}")

            # estraggo la mia lunghezza d'onda centrale
            lambda_centro = None

            # provo prima a trovare il pattern specifico per questa banda
            if banda == 'gmag':
                # cerco specificamente il mio pattern con {AA}
                match = re.search(r'\((\d+)\s*\{AA\}\)', descrizione)
                if match:
                    valore = float(match.group(1))
                    lambda_centro = valore / 10.0
                    # raddoppio le mie parentesi graffe per non farle interpretare come variabile
                    print(f"  -> Lunghezza d'onda estratta (pattern {{AA}}): {valore:.0f} Å = {lambda_centro:.1f} nm")

            # se non lo trovo, provo tutti i miei pattern
            if lambda_centro is None:
                for pattern in patterns:
                    match = re.search(pattern, descrizione)
                    if match:
                        valore = float(match.group(1))
                        # verifico che il mio valore sia in un range plausibile (300-2000 nm o 3000-20000 Å)
                        # aggiungo la 'r' per rendere la stringa raw ed evitare il SyntaxWarning
                        if '{AA}' in pattern or 'Å' in pattern or pattern.endswith(r'A\)') or pattern.endswith('[ÅA]'):
                            # è in Å, quindi lo converto in nm
                            lambda_centro = valore / 10.0
                            print(f"  -> Lunghezza d'onda estratta: {valore:.0f} Å = {lambda_centro:.1f} nm")
                        else:
                            lambda_centro = valore
                            print(f"  -> Lunghezza d'onda estratta: {lambda_centro:.1f} nm")
                        break

            # verifico che il valore che ho ottenuto sia plausibile (tra 300 e 2000 nm)
            if lambda_centro is not None:
                if lambda_centro < 300 or lambda_centro > 2000:
                    print(f"  -> ATTENZIONE: Valore {lambda_centro:.1f} nm non plausibile! Uso il mio fallback.")
                    lambda_centro = fallback_validi.get(banda, 500.0)
            else:
                print(f"  -> ATTENZIONE: Nessuna lunghezza d'onda trovata! Uso il mio fallback.")
                lambda_centro = fallback_validi.get(banda, 500.0)
                print(f"  -> Valore di fallback: {lambda_centro:.1f} nm")

            # calcolo l'intervallo usando 1.5×FWHM per coprire circa il 93% della mia risposta
            fwhm = fwhm_nm.get(banda, 100)
            fattore = 0.75
            w_min = int(round(lambda_centro - fwhm * fattore))
            w_max = int(round(lambda_centro + fwhm * fattore))

            # limito al range del mio sensore (300-1100 nm)
            w_min = max(w_min, 300)
            w_max = min(w_max, 1100)

            # verifico alla fine che w_min < w_max
            if w_min >= w_max:
                print(f"  -> ERRORE: Intervallo non valido ({w_min}-{w_max})! Uso il mio fallback.")
                # imposto un intervallo di fallback sicuro
                fallback_intervalli = {
                    'gmag': (418, 555),
                    'rmag': (552, 692),
                    'imag': (690, 820),
                    'zmag': (816, 920),
                    'ymag': (922, 1005)
                }
                w_min, w_max = fallback_intervalli.get(banda, (400, 550))

            limiti_bande[banda] = (w_min, w_max)
            print(f"  -> FWHM: {fwhm} nm")
            print(f"  -> Intervallo finale: {w_min} - {w_max} nm")

    print("\n" + "=" * 70)
    print("DIZIONARIO FINALE:")
    print("=" * 70)
    for banda, (w_min, w_max) in limiti_bande.items():
        print(f"    '{banda}': ({w_min}, {w_max}),")
    print("=" * 70 + "\n")

    return limiti_bande


# =============================================================================
# BLOCCO DI ESECUZIONE (MAIN)
# =============================================================================

if __name__ == "__main__":

    soglia_correlazione = 35 / 3600 * u.deg
    dist_ripetizione = soglia_correlazione
    magnitudine_massima = 15

    nome_params = 'parametri_image_segmentation.txt'
    file_parametri = cerca_file_nel_progetto(BASE_DIR, nome_params)
    if file_parametri is None:
        print("File dei parametri non trovato")
        exit()
    parametri_caricati = leggi_file_parametri(file_parametri)

    # scarico il mio catalogo globale Hipparcos da VizieR
    print("Scaricamento catalogo globale Hipparcos da VizieR in corso...")
    vizier_hip = Vizier(
        catalog="I/239/hip_main",
        columns=['HIP', '_RA.icrs', '_DE.icrs', 'Vmag', 'B-V'],
        row_limit=-1
    )
    risultato_hip = vizier_hip.query_constraints(Vmag="<16")
    tbl_catalogo_hipparco = risultato_hip[0]

    # rinomino le colonne per allinearmi al mio formato standard
    if '_RA.icrs' in tbl_catalogo_hipparco.colnames:
        tbl_catalogo_hipparco.rename_column('_RA.icrs', '_RAJ2000')
        tbl_catalogo_hipparco.rename_column('_DE.icrs', '_DEJ2000')
    print(f"Scaricati {len(tbl_catalogo_hipparco)} oggetti da Hipparcos.")

    exclusion_radii_deg = np.full(len(tbl_catalogo_hipparco), 2.5 / 3600.0)

    # creo il mio SkyCoord Hipparcos globale
    coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco['_RAJ2000'],
                                      dec=tbl_catalogo_hipparco['_DEJ2000'],
                                      unit=u.deg)

    dati_tutte_le_run = []
    mappa_headers_globali = {}

    # cerco il file contenente la mia curva di efficienza quantica per calcolare i pesi esatti
    file_curva_pmc = cerca_file_nel_progetto(BASE_DIR, "curva_PMC.csv")
    if file_curva_pmc is not None:
        # leggo il dataframe della mia curva
        df_curva = pd.read_csv(file_curva_pmc)

        # stabilisco i limiti di lunghezza d'onda delle singole bande estraendoli da VizieR
        limiti_bande = scarica_intervalli_bande_ps1_da_descrizioni()
        print(f"Limiti bande: \n{limiti_bande}")

        pesi_estratti = []

        # itero sulle bande per calcolare l'area sottesa alla mia curva per ciascun range
        for nome_banda, (w_min, w_max) in limiti_bande.items():
            # applico la mia maschera di taglio per l'intervallo corrente
            maschera_w = (df_curva['Wavelength'] >= w_min) & (df_curva['Wavelength'] <= w_max)
            # calcolo l'integrale tramite il metodo dei trapezi per estrarre la mia porzione di efficienza
            area = np.trapezoid(df_curva['QE'][maschera_w], x=df_curva['Wavelength'][maschera_w])
            pesi_estratti.append(area)

        # converto in array e normalizzo in modo che la mia somma finale sia pari a 1
        pesi_estratti = np.array(pesi_estratti)
        pesi_ideali_globali = pesi_estratti / np.sum(pesi_estratti)
    else:
        # imposto i pesi standard in caso di mancato ritrovamento del mio csv
        pesi_ideali_globali = np.array([0.458, 0.326, 0.133, 0.055, 0.028])

    file_somma_pixel = cerca_file_nel_progetto(BASE_DIR, "risultati_somma_pixel.csv")
    if file_somma_pixel:
        df_somma_pixel = pd.read_csv(file_somma_pixel)
        s_ref_df = df_somma_pixel[(df_somma_pixel['Run'] == 1) & (df_somma_pixel['im'] == 35)]
        if not s_ref_df.empty:
            s_ref = s_ref_df['Somma_Pixel_Esterni'].values[0]
        else:
            s_ref = 1.0
    else:
        df_somma_pixel = None
        s_ref = 1.0

    global_tracker_coords = None
    global_tracker_labels = []

    cartella_dati = BASE_DIR / "PMC_DATA_COLOSSALE"
    cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE" / "tabelle_unite"

    if not cartella_dati.exists():
        print(f"ERRORE: Cartella dati {cartella_dati} non trovata.")
        exit()

    # --- CICLO PER OGNI RUN ---
    for cartella_giorno in sorted([d for d in cartella_dati.iterdir() if d.is_dir()]):
        for run_folder in sorted([d for d in cartella_giorno.iterdir() if d.is_dir()]):

            run_name = run_folder.name
            print(f"\n==================== ELABORAZIONE {cartella_giorno.name} - {run_name} ====================")

            estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
            file_list = []
            for ext in estensioni_valide:
                file_list.extend(run_folder.glob(ext))

            file_list = sorted([str(f) for f in file_list])
            if not file_list:
                print(f"Nessun FITS in {run_name}, salto.")
                continue

            output_dir = cartella_tabelle / cartella_giorno.name / run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Cartella di output: {output_dir}")
            file_csv_generati_nella_run = []

            # --- FASE 1: CREAZIONE TABELLE UNITE ---
            print(f"--- FASE 1: Segmentazione & Unione ({len(file_list)} files) ---")

            for n, percorso_file in enumerate(tqdm(file_list, desc=f"Fase 1 {run_name}"), 1):
                if n == 1:
                    hdu_list = fits.open(percorso_file)
                    w = WCS(hdu_list[0].header)
                    ra_c, dec_c = hdu_list[0].header["RA"], hdu_list[0].header["DEC"]

                    alto_destra = w.pixel_to_world(3071, 2047)
                    centro = SkyCoord(ra_c, dec_c, unit=u.deg)

                    raggio_ricerca = Angle(centro.separation(alto_destra) * 1.5, "deg")

                    hdu_list.close()

                    # implemento il ciclo di tentativi per il mio scaricamento da vizier
                    tentativi_massimi = 5
                    attesa = 10
                    for tentativo in range(tentativi_massimi):
                        try:
                            riquadro_esterno_vizier = vizier.query_region(
                                coord.SkyCoord(ra=ra_c, dec=dec_c, unit=(u.deg, u.deg), frame='icrs'),
                                radius=raggio_ricerca
                            )
                            tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]
                            break
                        except Exception as e:
                            if tentativo < tentativi_massimi - 1:
                                time.sleep(attesa)
                            else:
                                raise

                    # CALCOLO MAGNITUDINE SINTETICA FLIR
                    bande = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']

                    # recupero i miei pesi precedentemente calcolati in modo efficiente
                    pesi_ideali = pesi_ideali_globali

                    flussi = []

                    for banda in bande:
                        colonna = tbl_riquadro_esterno_vizier[banda]
                        array_dati = colonna.filled(np.nan) if hasattr(colonna, 'filled') else np.array(colonna)
                        flusso = 10 ** (-0.4 * array_dati)

                        # sostituisco i dati mancanti con un flusso pari a zero
                        # assumo che se il catalogo non ha visto la mia stella in questa banda, il contributo di luce è nullo
                        flusso_pulito = np.nan_to_num(flusso, nan=0.0)
                        flussi.append(flusso_pulito)

                    flussi = np.array(flussi)
                    array_pesi = pesi_ideali[:, None]

                    # calcolo il mio flusso pesato totale senza normalizzare per le bande mancanti
                    flusso_finale = np.sum(flussi * array_pesi, axis=0)

                    with np.errstate(divide='ignore', invalid='ignore'):
                        # assegno una magnitudine fittizia di 99.0 ai miei oggetti che risultano avere flusso totalmente zero
                        mag_sintetica_globale = np.where(flusso_finale > 0, -2.5 * np.log10(flusso_finale),
                                                         99.0)

                    tbl_riquadro_esterno_vizier['Mag_sintetica'] = mag_sintetica_globale

                    distanze_hip = centro.separation(coords_hipparco_global)

                    mask_hip_fov = distanze_hip < raggio_ricerca

                    tbl_hipparco_run_subset = tbl_catalogo_hipparco[mask_hip_fov]
                    coords_hipparco_run_subset = coords_hipparco_global[mask_hip_fov]
                    exclusion_radii_run_subset = exclusion_radii_deg[mask_hip_fov]

                    # =================================================================
                    # --- FILTRAGGIO COMPETITIVO A SINGOLA FASE ---
                    # =================================================================
                    print("Avvio il mio filtraggio competitivo a singola fase Vizier vs Hipparcos...")

                    coords_vizier = SkyCoord(ra=tbl_riquadro_esterno_vizier['RAJ2000'],
                                             dec=tbl_riquadro_esterno_vizier['DEJ2000'],
                                             unit=u.deg)

                    max_threshold_deg = np.max(exclusion_radii_run_subset)
                    seplimit = max_threshold_deg * u.deg

                    idx_A, idx_B, d2d_1, _ = coords_hipparco_run_subset.search_around_sky(coords_vizier, seplimit)

                    if len(idx_A) > 0 and np.max(idx_A) >= len(coords_hipparco_run_subset):
                        idx_viz_1, idx_hip_1 = idx_A, idx_B
                    else:
                        idx_hip_1, idx_viz_1 = idx_A, idx_B

                    mask_threshold = d2d_1.deg <= exclusion_radii_run_subset[idx_hip_1]

                    idx_hip_valid = idx_hip_1[mask_threshold]
                    idx_viz_valid = idx_viz_1[mask_threshold]

                    mask_keep_hipparco = np.ones(len(tbl_hipparco_run_subset), dtype=bool)
                    mask_keep_vizier = np.ones(len(tbl_riquadro_esterno_vizier), dtype=bool)

                    unique_hip_idx = np.unique(idx_hip_valid)

                    array_mag_vizier = np.nan_to_num(tbl_riquadro_esterno_vizier['Mag_sintetica'].data, nan=99.0)
                    array_mag_hipparco = np.nan_to_num(tbl_hipparco_run_subset['Vmag'].data, nan=99.0)

                    for i_hip in unique_hip_idx:
                        viz_matches = idx_viz_valid[idx_hip_valid == i_hip]

                        if len(viz_matches) > 0:
                            mag_viz_matches = array_mag_vizier[viz_matches]

                            idx_min_mag = np.argmin(mag_viz_matches)
                            best_viz_idx = viz_matches[idx_min_mag]
                            best_viz_mag = mag_viz_matches[idx_min_mag]

                            hip_mag = array_mag_hipparco[i_hip]

                            # implemento la logica di filtraggio per eliminare i conflitti
                            if hip_mag < 9.0:
                                mask_keep_vizier[viz_matches] = False
                            elif best_viz_mag <= hip_mag:
                                mask_keep_hipparco[i_hip] = False
                            else:
                                mask_keep_vizier[best_viz_idx] = False

                    hipparco_escluse = np.sum(~mask_keep_hipparco)
                    vizier_escluse = np.sum(~mask_keep_vizier)
                    print(f"Risolti {len(unique_hip_idx)} conflitti spaziali:")
                    print(f" -> Escluse {hipparco_escluse} stelle Hipparco (tenute Vizier perché più brillanti)")
                    print(f" -> Escluse {vizier_escluse} stelle Vizier (tenute Hipparco perché più brillanti)")

                    mask_keep_hipparco[tbl_hipparco_run_subset['Vmag'] >= 15] = False

                    tbl_hipparco_run_clean = tbl_hipparco_run_subset[mask_keep_hipparco].copy()
                    tbl_riquadro_esterno_vizier_CLEAN = tbl_riquadro_esterno_vizier[mask_keep_vizier]

                    with np.errstate(invalid='ignore'):
                        mask_taglio = tbl_riquadro_esterno_vizier_CLEAN['Mag_sintetica'] < magnitudine_massima
                    tbl_vizier_cut = tbl_riquadro_esterno_vizier_CLEAN[mask_taglio].copy()

                tbl_vizier_cut['Mag'] = tbl_vizier_cut['Mag_sintetica']
                tbl_hipparco_run_clean['Mag'] = tbl_hipparco_run_clean['Vmag']

                tbl_catalogate = tabella_catalogo(percorso_file, tbl_vizier_cut, tbl_hipparco_run_clean)
                tbl_trovate, _ = analisi_image_segmentation(percorso_file, parametri_caricati)

                df_trovate = tbl_trovate.to_pandas()
                df_catalogate = tbl_catalogate.to_pandas()

                all_cols = df_trovate.columns.tolist()
                cols_keep = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']
                for c in ['saturazione']:
                    if c in all_cols: cols_keep.append(c)

                # tengo solamente il raggio calcolato in fase 1 ed elimino i flussi superflui
                extra_flux = ['raggio_kron_aper']
                for c in extra_flux:
                    if c in all_cols: cols_keep.append(c)

                df_trovate = df_trovate[[c for c in cols_keep if c in df_trovate.columns]].copy()

                with fits.open(percorso_file, memmap=False) as hdu:
                    w = WCS(hdu[0].header)
                coords = w.pixel_to_world(df_trovate['xcentroid'], df_trovate['ycentroid'])
                df_trovate['RA_centroid'] = coords.ra.deg
                df_trovate['DEC_centroid'] = coords.dec.deg

                cols_order = df_trovate.columns.tolist()
                if 'ycentroid' in cols_order:
                    for c in ['RA_centroid', 'DEC_centroid']:
                        if c in cols_order: cols_order.remove(c)
                    idx_y = cols_order.index('ycentroid')
                    cols_order.insert(idx_y + 1, 'RA_centroid')
                    cols_order.insert(idx_y + 2, 'DEC_centroid')
                    df_trovate = df_trovate[cols_order]

                    if 'RAJ2000' in df_catalogate.columns:
                        c_cat = SkyCoord(ra=df_catalogate['RAJ2000'].values * u.deg,
                                         dec=df_catalogate['DEJ2000'].values * u.deg)
                        idx_t, idx_c, d2d, _ = c_cat.search_around_sky(coords, soglia_correlazione)

                        matches = pd.DataFrame(
                            {'idx_t': idx_t, 'idx_c': idx_c, 'dist': d2d.deg,
                             'mag': df_catalogate.iloc[idx_c]['Mag'].values})
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
                # INIZIO BLOCCO: TRACKING GLOBALE OTTIMIZZATO (cKDTree - SCIPY)
                # =================================================================
                ra_rad = np.radians(df_final['RA_centroid'].values)
                dec_rad = np.radians(df_final['DEC_centroid'].values)
                x = np.cos(dec_rad) * np.cos(ra_rad)
                y = np.cos(dec_rad) * np.sin(ra_rad)
                z = np.sin(dec_rad)
                coords_cart = np.column_stack((x, y, z))

                soglia_rad = np.radians(dist_ripetizione.value)
                soglia_3d = 2.0 * np.sin(soglia_rad / 2.0)

                final_labels = np.empty(len(df_final), dtype=object)

                if global_tracker_coords is None:
                    # al primo giro, inizializzo la mia memoria storica
                    global_tracker_coords = coords_cart
                    global_tracker_labels = [f"RA_{ra:.3f}DEC{dec:.3f}" for ra, dec in
                                             zip(df_final['RA_centroid'].values, df_final['DEC_centroid'].values)]
                    final_labels[:] = global_tracker_labels
                else:
                    # creo il mio albero KD per una ricerca spaziale ultra veloce
                    albero = cKDTree(global_tracker_coords)
                    distanze, indici = albero.query(coords_cart, distance_upper_bound=soglia_3d)

                    # trovo quali oggetti hanno un match sotto la mia soglia
                    mask_match = distanze <= soglia_3d

                    # assegno le etichette già note ai miei oggetti
                    for i in np.where(mask_match)[0]:
                        final_labels[i] = global_tracker_labels[indici[i]]

                    # isolo i miei nuovi oggetti non trovati
                    nuovi_idx = np.where(~mask_match)[0]
                    if len(nuovi_idx) > 0:
                        nuove_coords_cart = coords_cart[nuovi_idx]
                        nuove_labels = [
                            f"RA_{df_final['RA_centroid'].values[i]:.3f}__DEC_{df_final['DEC_centroid'].values[i]:.3f}"
                            for i in nuovi_idx
                        ]

                        for i, l_idx in enumerate(nuovi_idx):
                            final_labels[l_idx] = nuove_labels[i]

                        # aggiorno la mia memoria accodando i nuovi array in puro numpy
                        global_tracker_coords = np.vstack([global_tracker_coords, nuove_coords_cart])
                        global_tracker_labels.extend(nuove_labels)

                df_final['label'] = final_labels

                # aggiungo le mie colonne identificative
                df_final['run_id'] = run_name
                df_final['img_index'] = n

                # =================================================================
                # FINE BLOCCO TRACKING
                # =================================================================

                if 'label' in df_final.columns: df_final.sort_values('label', inplace=True)

                cols = df_final.columns.tolist()
                if 'ID' in cols and 'Catalogo' in cols:
                    cols.remove('Catalogo')
                    cols.insert(cols.index('ID'), 'Catalogo')

                final_cols = df_final.columns.tolist()
                for c in ['run_id', 'img_index']:
                    if c in final_cols: final_cols.remove(c)

                if 'Corrispondenza' in final_cols:
                    idx_corr = final_cols.index('Corrispondenza')
                    final_cols.insert(idx_corr, 'img_index')
                    final_cols.insert(idx_corr, 'run_id')
                else:
                    final_cols.insert(0, 'run_id')
                    final_cols.insert(1, 'img_index')

                df_final = df_final[final_cols]

                file_out = output_dir / f'{run_name}_stelle_trovate_e_catalogate_immagine_{n:03d}.csv'
                salva_csv_con_header_fits(df_final, dict(fits.getheader(percorso_file)),
                                          file_out, str(percorso_file), parametri_caricati)

            # =============================================================================
            # FASE 2 & 3: RAGGI MAX E FLUSSO FISSO (PER RUN)
            # =============================================================================
            print(f"--- FASE 2 & 3: Analisi Fotometria Fissa per {run_name} ---")

            file_csv_list = sorted([f for f in output_dir.glob('*.csv')])

            for f in file_csv_list:
                file_csv_generati_nella_run.append((f, run_name))

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

                path_fits = header_info.get('PERCORSO_FILE', '')
                nome_fits = header_info.get('NOME_FILE_FITS', '')

                if not path_fits or not os.path.exists(path_fits):
                    if path_fits:
                        p_obj = Path(path_fits)
                        try:
                            if "pmc_photometry" in p_obj.parts:
                                idx_part = p_obj.parts.index("pmc_photometry")
                                new_path = BASE_DIR.joinpath(*p_obj.parts[idx_part + 1:])
                                if new_path.exists():
                                    path_fits = str(new_path)
                        except:
                            pass

                    if (not path_fits or not os.path.exists(path_fits)) and nome_fits:
                        found = cerca_file_nel_progetto(BASE_DIR, str(nome_fits).strip())
                        if found:
                            path_fits = str(found)

                if not path_fits or not os.path.exists(path_fits):
                    print(f"ATTENZIONE: File FITS {path_fits} originale non trovato per {nome_fits}, salto.")
                    continue

                with fits.open(path_fits, memmap=False) as hdu:
                    data_fits = hdu[0].data
                    _, median_bg, _ = sigma_clipped_stats(data_fits[::10, ::10], sigma=3.0)
                    data_sub = data_fits - median_bg

                img_idx = df_frame['img_index'].iloc[0] if 'img_index' in df_frame.columns else int(
                    nome_fits.split('.')[0][-3:])

                # estraggo il numero della mia run dalla stringa per i calcoli dei pixel
                match_run = re.search(r'run_?(\d+)', run_name, re.IGNORECASE)
                run_idx = int(match_run.group(1)) if match_run else 1

                fondo_pp = 0.0
                if df_somma_pixel is not None:
                    s_t_df = df_somma_pixel[(df_somma_pixel['Run'] == run_idx) & (df_somma_pixel['im'] == img_idx)]
                    if not s_t_df.empty:
                        s_t = s_t_df['Somma_Pixel_Esterni'].values[0]
                        if 'fondo_per_pixel' in s_t_df.columns:
                            fondo_pp = s_t_df['fondo_per_pixel'].values[0]
                    else:
                        s_t = s_ref
                else:
                    s_t = s_ref

                n_tot = data_sub.size
                diff_s = s_ref - s_t

                raggi_fissi = []
                ids_presenti = df_frame['ID'].values

                # mantengo solo il flusso che mi serve per arrivare alla correzione globale
                flussi_calcolati_add = []

                for idx_star, star_id in enumerate(ids_presenti):
                    r_globale = map_raggi_max.get(star_id, np.nan)

                    if np.isnan(r_globale) or r_globale <= 0:
                        if 'raggio_kron_aper' in df_frame.columns:
                            r_globale = df_frame.at[idx_star, 'raggio_kron_aper']
                        else:
                            r_globale = np.nan

                    raggi_fissi.append(r_globale)

                    pos = (df_frame.at[idx_star, 'xcentroid'], df_frame.at[idx_star, 'ycentroid'])

                    if r_globale > 0 and not np.isnan(r_globale):
                        aper = CircularAperture(pos, r=r_globale)
                        phot = aperture_photometry(data_sub, aper)
                        fl_calcolato = phot['aperture_sum'][0]

                        # calcolo direttamente la sola correzione additiva
                        flussi_calcolati_add.append(fl_calcolato + (np.pi * r_globale ** 2 / n_tot) * diff_s)
                    else:
                        flussi_calcolati_add.append(np.nan)

                df_frame['raggio_fisso_max_run'] = raggi_fissi

                # assegno unicamente il mio flusso
                df_frame['flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura'] = flussi_calcolati_add

                df_frame['fondo_per_pixel'] = fondo_pp

                if 'label' in df_frame.columns:
                    df_frame.sort_values(by=['label', 'Corrispondenza'], inplace=True)

                salva_csv_con_header_aggiornato(df_frame, header_info, file_csv)

            # =============================================================================
            # Avvio la mia fase 4 per calcolare statistiche e ID per la singola run corrente
            # =============================================================================
            print(f"\n--- FASE 4: Statistiche Locali e Ripetizioni per {run_name} ---")

            if not file_csv_generati_nella_run:
                continue

            lista_df_run = []
            file_csv_generati_nella_run = sorted(file_csv_generati_nella_run, key=lambda x: str(x[0]))

            for idx_file, (file_csv, run_number_val) in enumerate(
                    tqdm(file_csv_generati_nella_run, desc="Lettura Dati Run")):
                df_temp = pd.read_csv(file_csv, comment='#')
                df_temp['file_index'] = idx_file
                df_temp['run_number'] = run_number_val
                df_temp['original_file_path'] = str(file_csv)
                df_temp['original_idx'] = df_temp.index
                lista_df_run.append(df_temp)

            run_df = pd.concat(lista_df_run, ignore_index=True)
            run_df['run_unique_id'] = np.nan
            run_df['run_unique_id'] = run_df['run_unique_id'].astype(object)

            mask_si = run_df['Corrispondenza'].str.startswith('SI', na=False)
            run_df.loc[mask_si, 'run_unique_id'] = "CAT_" + run_df.loc[mask_si, 'ID'].astype(str)

            mask_no = run_df['Corrispondenza'] == 'NO'
            df_no_run = run_df[mask_no].copy()
            df_no_run.sort_values('file_index', inplace=True)

            known_clusters_coords_run = []
            known_clusters_ids_run = []
            threshold_deg_run = 35 / 3600
            unique_files_run = df_no_run['file_index'].unique()

            # imposto localmente per iterare sui NO
            next_internal_id_run = 1
            no_mapping_run = {}

            for f_idx in tqdm(unique_files_run, desc="Matching oggetti NO (Intra-Run)"):
                subset = df_no_run[df_no_run['file_index'] == f_idx]
                if subset.empty: continue
                coords_subset = SkyCoord(ra=subset['RA_centroid'].values * u.deg,
                                         dec=subset['DEC_centroid'].values * u.deg)
                indices_subset = subset.index.tolist()

                if not known_clusters_coords_run:
                    for i, (ra, dec) in enumerate(zip(subset['RA_centroid'], subset['DEC_centroid'])):
                        cid = f"INT_{next_internal_id_run}"
                        known_clusters_coords_run.append((ra, dec))
                        known_clusters_ids_run.append(cid)
                        no_mapping_run[indices_subset[i]] = cid
                        next_internal_id_run += 1
                else:
                    cluster_sc = SkyCoord(known_clusters_coords_run, unit=u.deg)
                    idx_cluster, d2d, _ = coords_subset.match_to_catalog_sky(cluster_sc)
                    for i, (match_idx, dist, ra_curr, dec_curr) in enumerate(
                            zip(idx_cluster, d2d, subset['RA_centroid'], subset['DEC_centroid'])):
                        global_idx = indices_subset[i]
                        if dist.deg <= threshold_deg_run:
                            no_mapping_run[global_idx] = known_clusters_ids_run[match_idx]
                            known_clusters_coords_run[match_idx] = (ra_curr, dec_curr)
                        else:
                            cid = f"INT_{next_internal_id_run}"
                            known_clusters_ids_run.append(cid)
                            known_clusters_coords_run.append((ra_curr, dec_curr))
                            no_mapping_run[global_idx] = cid
                            next_internal_id_run += 1

            for idx, uid in no_mapping_run.items(): run_df.at[idx, 'run_unique_id'] = uid

            print("Eseguo il mio filtraggio temporale e delle ripetizioni minime...")

            run_df['da_eliminare_temporale'] = False
            mask_no_temp = run_df['Corrispondenza'] == 'NO'

            for uid, group in run_df[mask_no_temp].groupby('run_unique_id'):
                indici_file = group['file_index'].values
                for idx_row, f_idx in zip(group.index, indici_file):
                    vicini = [x for x in indici_file if x != f_idx and abs(x - f_idx) <= 2]
                    if len(vicini) == 0:
                        run_df.at[idx_row, 'da_eliminare_temporale'] = True

            run_df = run_df[~run_df['da_eliminare_temporale']].drop(columns=['da_eliminare_temporale'])

            conteggi_aggiornati = run_df['run_unique_id'].value_counts()
            id_da_scartare = conteggi_aggiornati[conteggi_aggiornati < 2].index

            mask_da_scartare_rip = (run_df['Corrispondenza'] == 'NO') & (run_df['run_unique_id'].isin(id_da_scartare))
            run_df = run_df[~mask_da_scartare_rip]

            print("Riorganizzazione colonne e salvataggio per fase 5...")
            # tengo traccia solo del flusso base richiesto e rimuovo tutti gli step di decorrelazione locale
            cols_flux = ['flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura']

            cols_flux_presenti_run = [c for c in cols_flux if c in run_df.columns]
            for c in cols_flux_presenti_run: run_df[c] = pd.to_numeric(run_df[c], errors='coerce')

            run_df['fondo_per_pixel'] = pd.to_numeric(run_df.get('fondo_per_pixel', np.nan), errors='coerce')

            run_df['ID'] = run_df['ID'].astype(object)
            mask_no_match = run_df['Corrispondenza'] == 'NO'
            run_df.loc[mask_no_match, 'ID'] = run_df.loc[mask_no_match, 'run_unique_id']
            files_groups_run = run_df.groupby('original_file_path')

            # salvo le mie modifiche intra-run e accumulo in memoria per la Fase 5 globale
            for file_path, df_file in tqdm(files_groups_run, desc="Salvataggio Fase 4"):
                df_final_save = df_file.copy()
                header_orig = leggi_header_da_csv(file_path)

                cols = df_final_save.columns.tolist()
                for temp_c in ['file_index', 'original_file_path', 'original_idx', 'run_unique_id', 'run_number']:
                    if temp_c in cols: cols.remove(temp_c)

                df_final_save = df_final_save[cols]
                salva_csv_con_header_aggiornato(df_final_save, header_orig, file_path)

                dati_tutte_le_run.append(df_file)
                mappa_headers_globali[str(file_path)] = header_orig

    # =============================================================================
    # FASE 5 FINALE GLOBALE: STATISTICHE E DECORRELAZIONE GLOBALE SU TUTTE LE RUN
    # =============================================================================
    if dati_tutte_le_run:
        print("\n==================== FASE FINALE GLOBALE (TUTTE LE RUN) ====================")
        print(f"--- FASE 5: Calcolo Decorrelazione Globale delle Stelle ---")
        df_totale = pd.concat(dati_tutte_le_run, ignore_index=True)

        # deframmento il mio dataframe iniziale per ottimizzare e liberare la mia RAM
        df_totale = df_totale.copy()
        del dati_tutte_le_run
        gc.collect()

        # estraggo l'unica colonna di flusso che ho mantenuto
        cols_base_flux = [c for c in df_totale.columns if ('flusso_' in c or 'somma_' in c or 'kron_' in c)]
        cols_tutti_flussi_glob = [c for c in cols_base_flux if not any(
            x in c for x in ['media_', 'std_', 'DECORRELAZIONE_STELLE', 'ripetizioni', 'raggio_'])]

        nuovi_flussi_globali = []
        dizionario_nuove_colonne = {}

        for c in tqdm(cols_tutti_flussi_glob, desc="Decorrelazione Ensemble Globale"):
            new_col = f"{c}_DECORRELAZIONE_STELLE_GLOBALE"
            nuovi_flussi_globali.append(new_col)

            # converto la mia colonna in formato numerico
            flusso_numerico = pd.to_numeric(df_totale[c], errors='coerce')

            # trovo il primo valore registrato per ogni oggetto e lo imposto come mio riferimento
            primo_flusso_stella_globale = flusso_numerico.groupby(df_totale['run_unique_id']).transform('first')

            # calcolo il mio rapporto tra il flusso e il primo valore di riferimento
            with np.errstate(divide='ignore', invalid='ignore'):
                rapporto_relativo = np.where(primo_flusso_stella_globale > 0,
                                             flusso_numerico / primo_flusso_stella_globale,
                                             np.nan)

            # calcolo il fattore di correzione per l'immagine senza aggiungere colonne temporanee al mio dataframe
            temp_rapporto_series = pd.Series(rapporto_relativo)
            fattore_immagine = temp_rapporto_series.groupby(df_totale['original_file_path']).transform('median')

            # salvo il mio risultato nel dizionario temporaneo
            dizionario_nuove_colonne[new_col] = flusso_numerico / fattore_immagine

        # unisco tutte le mie nuove colonne calcolate al dataframe principale in un colpo solo per ridurre la frammentazione
        df_totale = pd.concat([df_totale, pd.DataFrame(dizionario_nuove_colonne)], axis=1)

        # calcolo la mia media e deviazione standard per le mie nuove colonne globali
        stat_columns_globali = []
        dizionario_statistiche = {}

        for c in nuovi_flussi_globali:
            col_mean = f'media_{c}'
            col_std = f'std_{c}'

            # converto temporaneamente per calcolare le mie statistiche
            flusso_num_globale = pd.to_numeric(df_totale[c], errors='coerce')

            dizionario_statistiche[col_mean] = flusso_num_globale.groupby(df_totale['run_unique_id']).transform('mean')
            stds_sample = flusso_num_globale.groupby(df_totale['run_unique_id']).transform('std')
            counts_grouped = flusso_num_globale.groupby(df_totale['run_unique_id']).transform('count')
            dizionario_statistiche[col_std] = stds_sample / np.sqrt(counts_grouped)

            stat_columns_globali.extend([col_mean, col_std])

        # unisco le mie statistiche al dataframe in un colpo solo
        df_totale = pd.concat([df_totale, pd.DataFrame(dizionario_statistiche)], axis=1)

        # riordino e salvo tutti i miei file
        files_groups_globale = df_totale.groupby('original_file_path')
        run_repetition_counts_global = df_totale['run_unique_id'].value_counts()

        for file_path, df_file in tqdm(files_groups_globale, desc="Salvataggio finale FASE 5"):
            header_orig = mappa_headers_globali[file_path]
            cols = df_file.columns.tolist()

            for temp_c in ['file_index', 'original_file_path', 'original_idx', 'run_unique_id', 'run_number']:
                if temp_c in cols: cols.remove(temp_c)

            # elimino la colonna base per lasciare solamente quella richiesta
            colonna_base = 'flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura'
            if colonna_base in cols:
                cols.remove(colonna_base)

            df_file = df_file.copy()
            df_file['ripetizioni'] = df_file['ID'].map(run_repetition_counts_global)
            if 'ripetizioni' not in cols:
                if 'saturazione' in cols:
                    cols.insert(cols.index('saturazione') + 1, 'ripetizioni')
                else:
                    cols.append('ripetizioni')

            for c_flux in nuovi_flussi_globali:
                c_mean, c_std = f'media_{c_flux}', f'std_{c_flux}'
                if c_flux in cols and c_mean in cols:
                    cols.remove(c_mean)
                    cols.remove(c_std)
                    idx_flux = cols.index(c_flux)
                    cols.insert(idx_flux + 1, c_mean)
                    cols.insert(idx_flux + 2, c_std)

            df_final_save = df_file[cols]
            nome_solo = os.path.basename(str(file_path))

            with open(file_path, 'w') as f:
                f.write("# Header FITS:\n")
                f.write("# Numero di falsi positivi esclusi sicuramente: 0\n")
                for k, v in header_orig.items():
                    if k not in ['PERCORSO_FILE', 'NOME_FILE'] and not k.startswith("Numero di falsi"):
                        f.write(f"# {k}: {v}\n")
                f.write(f"# NOME_FILE: {nome_solo}\n")
                f.write("#\n")
                df_final_save.to_csv(f, index=False)

    print("\n--- ELABORAZIONE COMPLETATA CON SUCCESSO ---")