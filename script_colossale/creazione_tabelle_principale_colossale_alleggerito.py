import pandas as pd
import matplotlib
import argparse
import json
import pyarrow.parquet as pq
import shutil
from astropy.config import paths

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import time
import os
import sys
import gc
from scipy.optimize import curve_fit
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
import warnings
from astropy.wcs import FITSFixedWarning
from photutils.datasets import make_100gaussians_image
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
from pathlib import Path
from astropy.time import Time

# gestisco i warning ignorandoli per mantenere pulito il mio output
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

# importo i moduli per il salvataggio in parquet e la relativa utilità
from funzioni.utilita_parquet import *
from funzioni.astrometria_parquet import *

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


# =============================================================================
# BLOCCO DI ESECUZIONE (MAIN)
# =============================================================================

if __name__ == "__main__":

    # aggiungo il mio analizzatore di argomenti da terminale
    parser = argparse.ArgumentParser(description="Elaborazione dati ASTRI1 con filtro temporale")
    parser.add_argument("reprocess", type=str, choices=['s', 'n'], help="Effettuare il riprocessamento totale? ('s' per sì, 'n' per no)")
    parser.add_argument("start_date", type=str, help="La mia data di inizio nel formato YYYYMMDD")
    parser.add_argument("end_date", type=str, help="La mia data di fine nel formato YYYYMMDD")
    args = parser.parse_args()

    soglia_correlazione = 35 / 3600 * u.deg
    dist_ripetizione = soglia_correlazione
    magnitudine_massima = 15

    nome_params = 'parametri_image_segmentation.txt'
    file_parametri = cerca_file_nel_progetto(BASE_DIR, nome_params)
    if file_parametri is None:
        print("File dei parametri non trovato")
        exit()
    parametri_caricati = leggi_file_parametri(file_parametri)

    print("Scaricamento catalogo globale Hipparcos da VizieR in corso...")
    
    # individuo il percorso esatto della cache di astroquery nel mio sistema
    astroquery_cache_dir = os.path.join(paths.get_cache_dir(), 'astroquery')
    
    # verifico se la cartella esiste prima di procedere
    if os.path.exists(astroquery_cache_dir):
        # elimino l'intera cartella e tutto il suo contenuto per liberare spazio
        shutil.rmtree(astroquery_cache_dir)
        # ricreo la cartella vuota per evitare errori nelle esecuzioni future
        os.makedirs(astroquery_cache_dir)

    vizier_hip = Vizier(
        catalog="I/239/hip_main",
        columns=['HIP', '_RA.icrs', '_DE.icrs', 'Vmag', 'B-V'],
        row_limit=-1
    )
    risultato_hip = vizier_hip.query_constraints(Vmag="<16")
    tbl_catalogo_hipparco = risultato_hip[0]

    if '_RA.icrs' in tbl_catalogo_hipparco.colnames:
        tbl_catalogo_hipparco.rename_column('_RA.icrs', '_RAJ2000')
        tbl_catalogo_hipparco.rename_column('_DE.icrs', '_DEJ2000')
    print(f"Scaricati {len(tbl_catalogo_hipparco)} oggetti da Hipparcos.")

    exclusion_radii_deg = np.full(len(tbl_catalogo_hipparco), 2.5 / 3600.0)

    coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco['_RAJ2000'],
                                      dec=tbl_catalogo_hipparco['_DEJ2000'],
                                      unit=u.deg)

    file_curva_pmc = cerca_file_nel_progetto(BASE_DIR, "curva_PMC.csv")
    if file_curva_pmc is not None:
        df_curva = pd.read_csv(file_curva_pmc)

        limiti_bande = scarica_intervalli_bande_ps1_da_descrizioni()
        print(f"Limiti bande: \n{limiti_bande}")

        pesi_estratti = []

        for nome_banda, (w_min, w_max) in limiti_bande.items():
            maschera_w = (df_curva['Wavelength'] >= w_min) & (df_curva['Wavelength'] <= w_max)
            area = np.trapezoid(df_curva['QE'][maschera_w], x=df_curva['Wavelength'][maschera_w])
            pesi_estratti.append(area)

        pesi_estratti = np.array(pesi_estratti)
        somma_pesi = np.sum(pesi_estratti)
        pesi_ideali_globali = pesi_estratti / somma_pesi

        # calcolo il peso per la banda Vmag di Hipparco nell'intervallo 500-600 nm
        maschera_vmag = (df_curva['Wavelength'] >= 500) & (df_curva['Wavelength'] <= 600)
        area_vmag = np.trapezoid(df_curva['QE'][maschera_vmag], x=df_curva['Wavelength'][maschera_vmag])
        peso_hipparco = area_vmag / somma_pesi
    else:
        pesi_ideali_globali = np.array([0.458, 0.326, 0.133, 0.055, 0.028])
        # imposto un peso di fallback per hipparco
        peso_hipparco = 0.35

    s_ref = 1.0
    dizionario_sfondi = {}

    global_tracker_coords = None
    global_tracker_labels = []

    next_internal_id = 1

    cartella_dati = cerca_cartella_intero_pc("ASTRI1")
    cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito" / "tabelle_unite"

    if cartella_dati is None or not Path(cartella_dati).exists():
        print(f"ERRORE: Cartella dati ASTRI1 non trovata.")
        sys.exit()

    # inizializzo il mio set per i file già processati e controllo i metadati se scelgo 'n'
    file_gia_processati = set()
    if args.reprocess == 'n':
        print(f"\nModalità 'no riprocessamento' attivata. Estraggo i nomi dai metadati dei file Parquet...")
        if cartella_tabelle.exists():
            for pq_file in cartella_tabelle.rglob('*.parquet'):
                try:
                    # estraggo rapidamente lo schema senza caricare l'intero file in memoria
                    schema = pq.read_schema(pq_file)
                    meta_raw = schema.metadata.get(b"astro_metadata")
                    if meta_raw:
                        metadata = json.loads(meta_raw.decode("utf-8"))
                        nome_fits = metadata.get("NOME_FILE_FITS")
                        if nome_fits:
                            file_gia_processati.add(nome_fits)
                except Exception:
                    continue
        print(f"Ho trovato {len(file_gia_processati)} file FITS già processati.")

    cartella_dati = Path(cartella_dati)
    
    # filtro le mie sottocartelle verificando che il loro nome rientri nell'intervallo temporale che ho specificato
    sottocartelle = [d for d in cartella_dati.iterdir() if (d.is_dir() or d.is_symlink()) and args.start_date <= d.name <= args.end_date]

    # inizializzo un dizionario per raggruppare preventivamente i file per cartella/giorno
    files_per_giorno = {}

    print(f"Trovate {len(sottocartelle)} cartelle/link in ASTRI1 valide per l'intervallo. Inizio scansione header...")

    for cartella_giorno in sottocartelle:
        percorso_reale = cartella_giorno.resolve()

        file_fits_list = []
        for ext in ['*.fit', '*.fits', '*.FIT', '*.FITS']:
            # cerco i file fits includendo eventuali sottocartelle
            file_fits_list.extend(percorso_reale.rglob(ext))

        # salto la cartella se contiene 10 o meno file
        if len(file_fits_list) <= 10:
            print(
                f"Salto la cartella {cartella_giorno.name}: contiene solo {len(file_fits_list)} file FITS (richiesti > 10).")
            continue

        nome_giorno = cartella_giorno.name
        files_per_giorno[nome_giorno] = []

        for percorso_file in tqdm(file_fits_list, desc=f"Scansione {nome_giorno}"):
            # salto questo file se ho scelto 'n' ed è già presente nell'elenco dei processati
            if args.reprocess == 'n' and percorso_file.name in file_gia_processati:
                continue

            try:
                # uso memmap=True per velocizzare la sola lettura dell'header
                with fits.open(percorso_file, memmap=True) as hdu:
                    header = hdu[0].header

                    # provo diverse combinazioni di chiavi comuni
                    ra_val = header.get('RA') or header.get('RAJ2000') or header.get('OBJ-RA')
                    dec_val = header.get('DEC') or header.get('DEJ2000') or header.get('OBJ-DEC')
                    tempo_obs_str = header.get('DATE-OBS')

                    if ra_val is not None and dec_val is not None:
                        try:
                            if isinstance(ra_val, (int, float)):
                                coords_centro = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame='icrs')
                            else:
                                coords_centro = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg),
                                                         frame='icrs')

                            files_per_giorno[nome_giorno].append({
                                'percorso_originale': str(percorso_file),
                                'nome_file': percorso_file.name,
                                'nome_giorno': nome_giorno,
                                'tempo': Time(tempo_obs_str) if tempo_obs_str else Time(os.path.getmtime(percorso_file),
                                                                                        format='unix'),
                                'dej2000': coords_centro.dec.deg
                            })
                        except Exception:
                            continue
            except Exception:
                continue

    if not files_per_giorno:
        print("\nERRORE: Nessun file FITS valido trovato (oppure tutti i file sono già stati processati).")
        sys.exit()

    soglia_tempo = 300.0
    soglia_spazio = 0.1
    run_groups = {}
    totale_file_validi = 0
    totale_run_create = 0

    # suddivido i file in run logiche, processando UN GIORNO ALLA VOLTA
    for giorno in sorted(files_per_giorno.keys()):
        file_del_giorno = files_per_giorno[giorno]
        if not file_del_giorno:
            continue

        # ordino cronologicamente i file solo all'interno della notte corrente
        file_del_giorno.sort(key=lambda x: x['tempo'])

        # inizializzo i contatori azzerandoli per ogni nuovo giorno
        contatore_run = 1
        tempo_precedente = None
        dej2000_precedente = None

        for dato in file_del_giorno:
            if tempo_precedente is not None:
                delta_tempo = abs((dato['tempo'] - tempo_precedente).sec)
                delta_spazio = abs(dato['dej2000'] - dej2000_precedente)
                # se supero le soglie, faccio scattare la mia nuova run all'interno della stessa notte
                if delta_tempo > soglia_tempo or delta_spazio > soglia_spazio:
                    contatore_run += 1

            nome_run = f"{giorno}_run_{contatore_run:03d}"  # questo nome dovrebbe essere univoco per la run
            chiave_gruppo = (giorno, nome_run)

            if chiave_gruppo not in run_groups:
                run_groups[chiave_gruppo] = []

            run_groups[chiave_gruppo].append(dato['percorso_originale'])

            tempo_precedente = dato['tempo']
            dej2000_precedente = dato['dej2000']
            totale_file_validi += 1

        totale_run_create += contatore_run

    print(f"\nRaggruppamento completato: {totale_file_validi} file divisi in {totale_run_create} run logiche.")

    # inizializzo le mie variabili per la verifica della distanza tra le run
    centro_run_precedente = None
    tbl_vizier_cut_precedente = None
    tbl_hipparco_run_clean_precedente = None

    # --- CICLO PER OGNI RUN LOGICA ---
    for (cartella_giorno_name, run_name) in sorted(run_groups.keys()):
        file_list = sorted(run_groups[(cartella_giorno_name, run_name)])

        # salto la mia run se contiene 10 o meno file
        if len(file_list) <= 10:
            print(f"Salto la run {run_name} di {cartella_giorno_name}: contiene solo {len(file_list)} file FITS (richiesti > 10).")
            continue

        print(f"\n==================== ELABORAZIONE {cartella_giorno_name} - {run_name} ====================")

        # estraggo il numero della mia run dalla stringa per i calcoli dei pixel
        match_run = re.search(r'run_?(\d+)', run_name, re.IGNORECASE)
        run_idx = int(match_run.group(1)) if match_run else 1

        output_dir = cartella_tabelle / cartella_giorno_name / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Cartella di output: {output_dir}")
        file_parquet_generati_nella_run = []

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

                # verifico la distanza dal centro della run precedente per evitare il download superfluo
                scarica_nuovo_catalogo = True
                if centro_run_precedente is not None:
                    distanza = centro.separation(centro_run_precedente)
                    if distanza.deg <= 1.1:
                        scarica_nuovo_catalogo = False

                if scarica_nuovo_catalogo:
                    # individuo il percorso esatto della cache di astroquery nel mio sistema
                    astroquery_cache_dir = os.path.join(paths.get_cache_dir(), 'astroquery')
                    
                    # verifico se la cartella esiste prima di procedere
                    if os.path.exists(astroquery_cache_dir):
                        # elimino l'intera cartella e tutto il suo contenuto per liberare spazio
                        shutil.rmtree(astroquery_cache_dir)
                        # ricreo la cartella vuota per evitare errori nelle esecuzioni future
                        os.makedirs(astroquery_cache_dir)
                        
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

                    bande = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']
                    pesi_ideali = pesi_ideali_globali
                    flussi = []

                    for banda in bande:
                        colonna = tbl_riquadro_esterno_vizier[banda]
                        array_dati = colonna.filled(np.nan) if hasattr(colonna, 'filled') else np.array(colonna)
                        flusso = 10 ** (-0.4 * array_dati)
                        flusso_pulito = np.nan_to_num(flusso, nan=0.0)
                        flussi.append(flusso_pulito)

                    flussi = np.array(flussi)
                    array_pesi = pesi_ideali[:, None]

                    flusso_finale = np.sum(flussi * array_pesi, axis=0)

                    with np.errstate(divide='ignore', invalid='ignore'):
                        mag_sintetica_globale = np.where(flusso_finale > 0, -2.5 * np.log10(flusso_finale), 99.0)

                    tbl_riquadro_esterno_vizier['Mag_sintetica'] = mag_sintetica_globale

                    distanze_hip = centro.separation(coords_hipparco_global)
                    mask_hip_fov = distanze_hip < raggio_ricerca

                    tbl_hipparco_run_subset = tbl_catalogo_hipparco[mask_hip_fov]

                    # calcolo il flusso di Hipparco, applico il peso e riconverto in magnitudine
                    colonna_vmag = tbl_hipparco_run_subset['Vmag']
                    array_dati_hip = colonna_vmag.filled(np.nan) if hasattr(colonna_vmag, 'filled') else np.array(colonna_vmag)
                    flusso_hip = 10 ** (-0.4 * array_dati_hip)
                    flusso_hip_pulito = np.nan_to_num(flusso_hip, nan=0.0)
                    flusso_hip_pesato = flusso_hip_pulito * peso_hipparco

                    with np.errstate(divide='ignore', invalid='ignore'):
                        mag_hip_pesata = np.where(flusso_hip_pesato > 0, -2.5 * np.log10(flusso_hip_pesato), 99.0)

                    tbl_hipparco_run_subset['Vmag'] = mag_hip_pesata

                    coords_hipparco_run_subset = coords_hipparco_global[mask_hip_fov]
                    exclusion_radii_run_subset = exclusion_radii_deg[mask_hip_fov]

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
                    
                    # salvo il mio stato attuale per il controllo nella run successiva
                    centro_run_precedente = centro
                    tbl_vizier_cut_precedente = tbl_vizier_cut.copy()
                    tbl_hipparco_run_clean_precedente = tbl_hipparco_run_clean.copy()
                else:
                    # recupero la mia tabella precedentemente calcolata se la distanza è inferiore a 1.1 gradi
                    print("Distanza dal centro della run precedente <= 1.1 gradi: riutilizzo il catalogo Vizier e Hipparcos.")
                    tbl_vizier_cut = tbl_vizier_cut_precedente.copy()
                    tbl_hipparco_run_clean = tbl_hipparco_run_clean_precedente.copy()

            tbl_vizier_cut['Mag'] = tbl_vizier_cut['Mag_sintetica']
            tbl_hipparco_run_clean['Mag'] = tbl_hipparco_run_clean['Vmag']

            tbl_catalogate = tabella_catalogo(percorso_file, tbl_vizier_cut, tbl_hipparco_run_clean)
            tbl_trovate, _, somma_totale, fondo_medio = analisi_image_segmentation(percorso_file,
                                                                                   parametri_caricati)

            dizionario_sfondi[(run_idx, n)] = {'somma': somma_totale, 'fondo_pp': fondo_medio}

            if run_idx == 1 and n == 35:
                s_ref = somma_totale

            df_trovate = tbl_trovate.to_pandas()
            df_catalogate = tbl_catalogate.to_pandas()

            all_cols = df_trovate.columns.tolist()
            cols_keep = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']
            for c in ['saturazione']:
                if c in all_cols: cols_keep.append(c)

            extra_flux = ['raggio_kron_aper']
            for c in extra_flux:
                if c in all_cols: cols_keep.append(c)

            df_trovate = df_trovate[[c for c in cols_keep if c in df_trovate.columns]].copy()

            with fits.open(percorso_file, memmap=False) as hdu:
                w = WCS(hdu[0].header)
            coords = w.pixel_to_world(df_trovate['xcentroid'], df_trovate['ycentroid'])
            df_trovate['RA_centroid'] = coords.ra.deg
            df_trovate['DE_centroid'] = coords.dec.deg

            cols_order = df_trovate.columns.tolist()
            if 'ycentroid' in cols_order:
                for c in ['RA_centroid', 'DE_centroid']:
                    if c in cols_order: cols_order.remove(c)
                idx_y = cols_order.index('ycentroid')
                cols_order.insert(idx_y + 1, 'RA_centroid')
                cols_order.insert(idx_y + 2, 'DE_centroid')
                df_trovate = df_trovate[cols_order]

                if 'RAJ2000' in df_catalogate.columns:
                    c_cat = SkyCoord(ra=df_catalogate['RAJ2000'].values * u.deg,
                                     dec=df_catalogate['DEJ2000'].values * u.deg)

                    idx_cat, idx_trov, d2d, _ = search_around_sky(c_cat, coords, soglia_correlazione)

                    matches = pd.DataFrame(
                        {'idx_t': idx_trov, 'idx_c': idx_cat, 'dist': d2d.deg,
                         'mag': df_catalogate.iloc[idx_cat]['Mag'].values})

                    matches.sort_values(by=['idx_t', 'mag'], inplace=True)
                    matches['Rank'] = matches.groupby('idx_t').cumcount() + 1
                    matches['Corrispondenza'] = True

                    df_si = pd.concat([
                        df_trovate.iloc[matches['idx_t']].reset_index(drop=True),
                        matches[['Corrispondenza', 'Rank']].reset_index(drop=True),
                        df_catalogate.iloc[matches['idx_c']].reset_index(drop=True)
                    ], axis=1)

                    unmatched = list(set(range(len(df_trovate))) - set(matches['idx_t']))
                    df_no = df_trovate.iloc[unmatched].copy()

                    df_no['Corrispondenza'] = False
                    df_no['Rank'] = 0

                    for c in df_catalogate.columns:
                        if c == 'ID':
                            df_no[c] = 0
                        else:
                            df_no[c] = np.nan

                    df_final = pd.concat([df_si, df_no], ignore_index=True)

                    if 'ID' in df_final.columns:
                        df_final['ID'] = df_final['ID'].fillna(0).astype(np.int64)

                else:
                    df_final = df_trovate.copy()
                    df_final['Corrispondenza'] = False
                    df_final['Rank'] = 0

            # =================================================================
            # INIZIO BLOCCO: TRACKING GLOBALE OTTIMIZZATO (Astropy SkyCoord)
            # =================================================================
            coords_obj_all = SkyCoord(ra=np.atleast_1d(df_final['RA_centroid'].values) * u.deg,
                                      dec=np.atleast_1d(df_final['DE_centroid'].values) * u.deg)
            final_labels = np.empty(len(df_final), dtype=object)

            if global_tracker_coords is None:
                global_tracker_coords = coords_obj_all
                global_tracker_labels = [f"RA_{ra:.3f}DEC{dec:.3f}" for ra, dec in
                                         zip(coords_obj_all.ra.deg, coords_obj_all.dec.deg)]
                final_labels[:] = global_tracker_labels
            else:
                idx_match, d2d, _ = coords_obj_all.match_to_catalog_sky(global_tracker_coords)
                mask_match = d2d < dist_ripetizione
                for i in np.where(mask_match)[0]:
                    final_labels[i] = global_tracker_labels[idx_match[i]]

                nuovi_idx = np.where(~mask_match)[0]
                if len(nuovi_idx) > 0:
                    nuove_coords = coords_obj_all[nuovi_idx]
                    nuove_labels = [f"RA_{ra:.3f}__DEC_{dec:.3f}" for ra, dec in
                                    zip(nuove_coords.ra.deg, nuove_coords.dec.deg)]
                    for i, l_idx in enumerate(nuovi_idx):
                        final_labels[l_idx] = nuove_labels[i]

                    nuovi_ra = np.concatenate([global_tracker_coords.ra.deg, nuove_coords.ra.deg])
                    nuovi_dec = np.concatenate([global_tracker_coords.dec.deg, nuove_coords.dec.deg])
                    global_tracker_coords = SkyCoord(ra=nuovi_ra * u.deg, dec=nuovi_dec * u.deg)
                    global_tracker_labels.extend(nuove_labels)

            df_final['label'] = final_labels

            # =================================================================
            # FINE BLOCCO TRACKING
            # =================================================================

            if 'label' in df_final.columns: df_final.sort_values('label', inplace=True)

            cols = df_final.columns.tolist()
            if 'ID' in cols and 'Catalogo' in cols:
                cols.remove('Catalogo')
                cols.insert(cols.index('ID'), 'Catalogo')

            df_final = df_final[cols]

            file_out = output_dir / f'{run_name}_immagine_{n:04d}.parquet'

            salva_tabella_parquet(
                df_final,
                dict(fits.getheader(percorso_file)),
                file_out,
                str(percorso_file),
                parametri_caricati,
                num_run=run_idx,
                num_immagine=n
            )

        # =============================================================================
        # FASE 2 & 3: RAGGI MAX E FLUSSO FISSO (PER RUN)
        # =============================================================================
        print(f"--- FASE 2 & 3: Analisi Fotometria Fissa per {run_name} ---")

        file_parquet_list = sorted([f for f in output_dir.glob('*.parquet')])

        for f in file_parquet_list:
            file_parquet_generati_nella_run.append((f, run_name))

        all_ids = []
        all_radii = []

        for file_pq in tqdm(file_parquet_list, desc="Scansione Raggi"):
            try:
                df_temp, _ = leggi_tabella_parquet(file_pq)
                if 'ID' in df_temp.columns and 'raggio_kron_aper' in df_temp.columns:
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

        for file_pq in tqdm(file_parquet_list, desc="Ricalcolo Flussi"):
            df_frame, meta_info = leggi_tabella_parquet(file_pq)
            header_info = meta_info.get("FITS_HEADER", {})

            nome_fits = meta_info.get('NOME_FILE_FITS', '')
            if not nome_fits:
                nome_fits = os.path.basename(str(file_pq)).replace('.parquet', '.fit')

            nome_fits = str(nome_fits).strip()
            file_trovato = cerca_file_nel_progetto(BASE_DIR, nome_fits)

            if file_trovato is None:
                continue

            path_fits = str(file_trovato)
            data_sub, median_bg, _ = elabora_file_fits(path_fits)

            img_idx = meta_info.get('NUMERO_IMMAGINE', int(nome_fits.split('.')[0][-3:]))
            run_idx_meta = meta_info.get('NUMERO_RUN', run_idx)

            fondo_pp = 0.0

            sfondi_correnti = dizionario_sfondi.get((run_idx_meta, img_idx))
            if sfondi_correnti is not None:
                s_t = sfondi_correnti['somma']
                fondo_pp = sfondi_correnti['fondo_pp']
            else:
                s_t = s_ref

            n_tot = data_sub.size
            diff_s = s_ref - s_t

            raggi_fissi = []
            ids_presenti = df_frame['ID'].values

            flussi_calcolati_add = []
            flusso_fisso_max_run_senza_correzioni = []

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
                    flusso_fisso_max_run_senza_correzioni.append(fl_calcolato)

                    flussi_calcolati_add.append(fl_calcolato + (np.pi * r_globale ** 2 / n_tot) * diff_s)
                else:
                    flussi_calcolati_add.append(np.nan)
                    flusso_fisso_max_run_senza_correzioni.append(np.nan)

            df_frame['raggio_fisso_max_run'] = raggi_fissi
            df_frame['flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura'] = flussi_calcolati_add
            df_frame['flusso_fisso_max_run_senza_correzioni'] = flusso_fisso_max_run_senza_correzioni
            df_frame['fondo_per_pixel'] = fondo_pp

            if 'label' in df_frame.columns:
                df_frame.sort_values(by=['label', 'Corrispondenza'], inplace=True)

            salva_tabella_parquet(
                df_frame,
                header_info,
                file_pq,
                nome_fits,
                parametri_caricati,
                num_run=run_idx_meta,
                num_immagine=img_idx
            )

        # =============================================================================
        # Avvio la mia fase 4 per calcolare statistiche e ID per la singola run corrente
        # =============================================================================
        print(f"\n--- FASE 4: Statistiche Locali e Ripetizioni per {run_name} ---")

        if not file_parquet_generati_nella_run:
            continue

        lista_df_run = []
        file_parquet_generati_nella_run = sorted(file_parquet_generati_nella_run, key=lambda x: str(x[0]))

        for idx_file, (file_pq, run_number_val) in enumerate(
                tqdm(file_parquet_generati_nella_run, desc="Lettura Dati Run")):
            df_temp, meta = leggi_tabella_parquet(file_pq)
            df_temp['file_index'] = idx_file
            df_temp['original_file_path'] = str(file_pq)
            df_temp['original_idx'] = df_temp.index
            lista_df_run.append(df_temp)

        run_df = pd.concat(lista_df_run, ignore_index=True)

        run_df['ID'] = pd.to_numeric(run_df['ID'], errors='coerce').fillna(0).astype('int64')

        mask_no = run_df['Corrispondenza'] == False
        df_no_run = run_df[mask_no].copy()
        df_no_run.sort_values('file_index', inplace=True)

        known_clusters_coords_run = []
        known_clusters_ids_run = []
        threshold_deg_run = 35 / 3600
        unique_files_run = df_no_run['file_index'].unique()

        no_mapping_run = {}

        for f_idx in tqdm(unique_files_run, desc="Matching oggetti NO (Intra-Run)"):
            subset = df_no_run[df_no_run['file_index'] == f_idx]
            if subset.empty: continue
            coords_subset = SkyCoord(ra=subset['RA_centroid'].values * u.deg,
                                     dec=subset['DE_centroid'].values * u.deg)
            indices_subset = subset.index.tolist()

            if not known_clusters_coords_run:
                for i, (ra, dec) in enumerate(zip(subset['RA_centroid'], subset['DE_centroid'])):
                    cid = -next_internal_id
                    known_clusters_coords_run.append((ra, dec))
                    known_clusters_ids_run.append(cid)
                    no_mapping_run[indices_subset[i]] = cid
                    next_internal_id += 1
            else:
                cluster_sc = SkyCoord(known_clusters_coords_run, unit=u.deg)
                idx_cluster, d2d, _ = coords_subset.match_to_catalog_sky(cluster_sc)
                for i, (match_idx, dist, ra_curr, dec_curr) in enumerate(
                        zip(idx_cluster, d2d, subset['RA_centroid'], subset['DE_centroid'])):
                    global_idx = indices_subset[i]
                    if dist.deg <= threshold_deg_run:
                        no_mapping_run[global_idx] = known_clusters_ids_run[match_idx]
                        known_clusters_coords_run[match_idx] = (ra_curr, dec_curr)
                    else:
                        cid = -next_internal_id
                        known_clusters_ids_run.append(cid)
                        known_clusters_coords_run.append((ra_curr, dec_curr))
                        no_mapping_run[global_idx] = cid
                        next_internal_id += 1

        for idx, uid in no_mapping_run.items(): run_df.at[idx, 'ID'] = uid

        print("Eseguo il mio filtraggio temporale e delle ripetizioni minime...")

        run_df['da_eliminare_temporale'] = False
        mask_no_temp = run_df['Corrispondenza'] == False

        for uid, group in run_df[mask_no_temp].groupby('ID'):
            indici_file = group['file_index'].values
            for idx_row, f_idx in zip(group.index, indici_file):
                vicini = [x for x in indici_file if x != f_idx and abs(x - f_idx) <= 2]
                if len(vicini) == 0:
                    run_df.at[idx_row, 'da_eliminare_temporale'] = True

        run_df = run_df[~run_df['da_eliminare_temporale']].drop(columns=['da_eliminare_temporale'])

        conteggi_aggiornati = run_df['ID'].value_counts()
        id_da_scartare = conteggi_aggiornati[conteggi_aggiornati < 2].index

        mask_da_scartare_rip = (run_df['Corrispondenza'] == False) & (run_df['ID'].isin(id_da_scartare))
        run_df = run_df[~mask_da_scartare_rip]

        print("Calcolo statistiche di run e riorganizzazione colonne...")

        cols_flux = ['flusso_fisso_max_run_senza_correzioni',
                     'flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura']

        cols_flux_presenti_run = [c for c in cols_flux if c in run_df.columns]
        for c in cols_flux_presenti_run: run_df[c] = pd.to_numeric(run_df[c], errors='coerce')

        run_df['fondo_per_pixel'] = pd.to_numeric(run_df.get('fondo_per_pixel', np.nan), errors='coerce')

        run_df['ID'] = run_df['ID'].astype('int64')

        # =============================================================================
        # FASE 5: Decorrelazione della singola Run e Formattazione Dinamica
        # =============================================================================
        print(f"\n--- FASE 5: Calcolo Decorrelazione e Formattazione Dinamica per {run_name} ---")

        cols_tutti_flussi_glob = ['flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura']

        run_df['stat_group_id'] = np.where(
            run_df['ID'] < 0,
            run_df['label'],
            run_df['ID']
        )

        dizionario_nuove_colonne = {}
        for c in tqdm(cols_tutti_flussi_glob, desc=f"Decorrelazione Ensemble {run_name}"):
            new_col = f"{c}_DECORRELAZIONE_STELLE_GLOBALE"
            flusso_numerico = pd.to_numeric(run_df[c], errors='coerce')
            primo_flusso_stella_globale = flusso_numerico.groupby(run_df['stat_group_id']).transform('first')

            with np.errstate(divide='ignore', invalid='ignore'):
                rapporto_relativo = np.where(primo_flusso_stella_globale > 0,
                                             flusso_numerico / primo_flusso_stella_globale,
                                             np.nan)

            fattore_immagine = pd.Series(rapporto_relativo).groupby(run_df['original_file_path']).transform('median')
            dizionario_nuove_colonne[new_col] = flusso_numerico / fattore_immagine

        run_df = pd.concat([run_df, pd.DataFrame(dizionario_nuove_colonne)], axis=1)

        colonne_per_statistiche = [f"{c}_DECORRELAZIONE_STELLE_GLOBALE" for c in cols_tutti_flussi_glob] + [
            'flusso_fisso_max_run_senza_correzioni']
        dizionario_statistiche = {}

        for c in colonne_per_statistiche:
            col_mean = f'media_{c}'
            col_std = f'std_{c}'
            flusso_num_globale = pd.to_numeric(run_df[c], errors='coerce')

            dizionario_statistiche[col_mean] = flusso_num_globale.groupby(run_df['stat_group_id']).transform('mean')
            stds_sample = flusso_num_globale.groupby(run_df['stat_group_id']).transform('std')
            counts_grouped = flusso_num_globale.groupby(run_df['stat_group_id']).transform('count')

            dizionario_statistiche[col_std] = stds_sample / np.sqrt(counts_grouped)

        run_df = pd.concat([run_df, pd.DataFrame(dizionario_statistiche)], axis=1)

        print("Applico il mio arrotondamento dinamico basato sull'incertezza...")

        for c in colonne_per_statistiche:
            col_mean = f'media_{c}'
            col_std = f'std_{c}'

            valori = run_df[c].values
            medie = run_df[col_mean].values
            incertezze = run_df[col_std].values

            nuovi_v = []
            nuove_m = []
            nuove_s = []

            for v, m, s in zip(valori, medie, incertezze):
                if pd.notna(s) and s > 0:
                    ordine = int(np.floor(np.log10(s)))
                    dec = max(0, -(ordine - 2))
                    nuovi_v.append(round(v, dec) if pd.notna(v) else v)
                    nuove_m.append(round(m, dec) if pd.notna(m) else m)
                    nuove_s.append(round(s, dec))
                else:
                    nuovi_v.append(v)
                    nuove_m.append(m)
                    nuove_s.append(s)

            run_df[c] = nuovi_v
            run_df[col_mean] = nuove_m
            run_df[col_std] = nuove_s

        # =================================================================
        # ESECUZIONE FIT DELLA RUN PRIMA DEL SALVATAGGIO
        # =================================================================
        print("--- Esecuzione FIT Fotometrico della Run ---")
        flusso_target = 'flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE'
        col_media = f'media_{flusso_target}'
        col_std = f'std_{flusso_target}'

        fit_results = {}

        # isolo unicamente le righe necessarie per non analizzare duplicati (una riga per oggetto)
        df_fit_base = run_df.drop_duplicates(subset=['stat_group_id']).copy()

        # filtro i miei oggetti mantendendo solo quelli catalogati validi sotto la magnitudine limite e non saturi
        mask_catalogati = df_fit_base['Corrispondenza'] == True
        mask_mag = df_fit_base['Mag'] <= 10.0
        mask_sature = df_fit_base['saturazione'] == True
        mask_valida = (df_fit_base[col_media].notna()) & (df_fit_base[col_media] > 0) & (
            df_fit_base[col_std].notna()) & (df_fit_base[col_std] > 0)

        df_fit_clean = df_fit_base[mask_catalogati & mask_mag & ~mask_sature & mask_valida].copy()

        if len(df_fit_clean) > 2:
            X = df_fit_clean['Mag'].values
            Y_flux = df_fit_clean[col_media].values
            sigma_flux = df_fit_clean[col_std].values

            # calcolo il numero di bin
            n_bins = max(5, int(np.sqrt(len(X))))

            # ordino i miei array basandomi sulla magnitudine
            sort_idx = np.argsort(X)
            X_sorted = X[sort_idx]
            Y_sorted = Y_flux[sort_idx]
            Err_sorted = sigma_flux[sort_idx]

            # divido i miei array in chunk di uguale dimensione
            X_chunks = np.array_split(X_sorted, n_bins)
            Y_chunks = np.array_split(Y_sorted, n_bins)
            Err_chunks = np.array_split(Err_sorted, n_bins)

            X_binned, Y_binned, Err_binned = [], [], []

            # assemblo i miei bin
            for x_bin, y_bin, err_bin in zip(X_chunks, Y_chunks, Err_chunks):
                if len(x_bin) > 0:
                    y_media_semplice = np.mean(y_bin)
                    if len(y_bin) > 1:
                        y_errore_semplice = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin))
                        if y_errore_semplice == 0:
                            y_errore_semplice = np.mean(err_bin)
                    else:
                        y_errore_semplice = err_bin[0]
                    X_binned.append(np.mean(x_bin))
                    Y_binned.append(y_media_semplice)
                    Err_binned.append(y_errore_semplice)

            X_binned = np.array(X_binned)
            Y_binned = np.array(Y_binned)
            Err_binned = np.array(Err_binned)

            # converto i miei flussi usando la scala logaritmica e propago l'errore
            Y_log_binned = np.log10(Y_binned)
            sigma_log_binned = (1 / np.log(10)) * (Err_binned / Y_binned)

            try:
                # eseguo il mio fit lineare sui dati binnati
                popt, pcov = curve_fit(modello_lineare, X_binned, Y_log_binned, sigma=sigma_log_binned,
                                       absolute_sigma=True)
                m_fit, q_fit = popt
                err_m, err_q = np.sqrt(np.diag(pcov))

                y_model_binned = modello_lineare(X_binned, m_fit, q_fit)
                dof = len(X_binned) - 2
                chi2 = np.sum(((Y_log_binned - y_model_binned) / sigma_log_binned) ** 2)
                chi2_red = chi2 / dof if dof > 0 else 0

                # raccolgo i risultati del mio fit in un dizionario
                fit_results = {
                    'FIT_M': round(m_fit, 4),
                    'FIT_EM': round(err_m, 4),
                    'FIT_Q': round(q_fit, 4),
                    'FIT_EQ': round(err_q, 4),
                    'FIT_CHI2': round(chi2_red, 4)
                }
                print(f"Fit completato: m={m_fit:.4f}±{err_m:.4f}, q={q_fit:.4f}±{err_q:.4f}, chi2_red={chi2_red:.2f}")

                # disegno e salvo il mio grafico del fit
                plt.figure(figsize=(10, 7))
                plt.errorbar(X, Y_flux, yerr=sigma_flux, fmt='o', markersize=2, color='blue', ecolor='lightblue',
                             alpha=0.5, label='Catalogati Validi')
                plt.errorbar(X_binned, Y_binned, yerr=Err_binned, fmt='o', markersize=6, color='green', ecolor='green',
                             capsize=3, label='Bin Fit')
                x_plot = np.linspace(min(X) - 0.5, max(X) + 0.5, 100)
                y_plot_log = modello_lineare(x_plot, m_fit, q_fit)
                plt.plot(x_plot, 10 ** y_plot_log, 'g--', linewidth=2,
                         label=f'Fit: log(F)=({m_fit:.2f})M + ({q_fit:.2f})')
                plt.yscale('log')
                plt.gca().invert_xaxis()
                plt.xlabel("Mag")
                plt.ylabel("Flusso Base")
                plt.legend()
                plt.grid(True, alpha=0.3)
                # plt.savefig(output_dir / f"{run_name}_fit_bin.png")
                plt.close()

            except Exception as e:
                print(f"Errore nel curve_fit: {e}")
        else:
            print("Punti insufficienti per il fit in questa run.")

        # =================================================================
        # FASE DI SALVATAGGIO DEFINITIVA
        # =================================================================

        files_groups_globale = run_df.groupby('original_file_path')
        run_repetition_counts_global = run_df['stat_group_id'].value_counts()

        for file_path, df_file in tqdm(files_groups_globale, desc=f"Salvataggio finale PARQUET {run_name}"):
            _, meta_orig = leggi_tabella_parquet(file_path)
            header_orig = meta_orig.get("FITS_HEADER", {})

            # aggiorno il mio header aggiungendo i valori del fit se sono stati calcolati con successo
            if fit_results:
                header_orig.update(fit_results)

            nome_fits_orig = meta_orig.get("NOME_FILE_FITS", "")
            run_idx_orig = meta_orig.get("NUMERO_RUN", run_idx)
            img_idx_orig = meta_orig.get("NUMERO_IMMAGINE", 1)

            cols = df_file.columns.tolist()

            for temp_c in ['file_index', 'original_file_path', 'original_idx', 'stat_group_id']:
                if temp_c in cols: cols.remove(temp_c)

            if 'flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura' in cols:
                cols.remove('flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura')

            df_file = df_file.copy()
            df_file['ripetizioni'] = df_file['stat_group_id'].map(run_repetition_counts_global)

            if 'ripetizioni' not in cols:
                idx_ins = cols.index('saturazione') + 1 if 'saturazione' in cols else len(cols)
                cols.insert(idx_ins, 'ripetizioni')

            for c_flux in colonne_per_statistiche:
                c_mean, c_std = f'media_{c_flux}', f'std_{c_flux}'
                if c_flux in cols and c_mean in cols:
                    cols.remove(c_mean)
                    cols.remove(c_std)
                    idx_flux = cols.index(c_flux)
                    cols.insert(idx_flux + 1, c_mean)
                    cols.insert(idx_flux + 2, c_std)

            df_final_save = df_file[cols]
            percorso_parquet = Path(file_path)

            salva_tabella_parquet(
                df_final_save,
                header_orig,
                percorso_parquet,
                nome_fits_orig,
                parametri_caricati,
                num_run=run_idx_orig,
                num_immagine=img_idx_orig
            )

        # inizializzo il mio nuovo dataframe richiesto includendo le coordinate
        mydf = pd.DataFrame(columns=[
            'label',
            'RA_centroid',
            'DE_centroid',
            'ripetizioni',
            'media_flusso_fisso_max_run_senza_correzioni',
            'std_flusso_fisso_max_run_senza_correzioni',
            'media_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE',
            'std_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE'
        ])

        # filtro i miei oggetti isolando solo quelli senza corrispondenza (valore booleano False)
        maschera_no_cat = run_df['Corrispondenza'] == False
        dati_senza_corrispondenza = run_df[maschera_no_cat].copy()

        # calcolo la mia colonna delle ripetizioni
        dati_senza_corrispondenza['ripetizioni'] = dati_senza_corrispondenza['stat_group_id'].map(
            run_repetition_counts_global)

        # elimino i miei duplicati per mantenere un'unica riga riassuntiva per oggetto
        dati_senza_corrispondenza = dati_senza_corrispondenza.drop_duplicates(subset=['label'])

        # popolo il mio dataframe con i dati estratti per le colonne indicate
        for col in mydf.columns:
            if col in dati_senza_corrispondenza.columns:
                mydf[col] = dati_senza_corrispondenza[col].values

        # decido il nome del mio nuovo file parquet per gli oggetti estranei
        file_out_mydf = output_dir / f"{run_name}_oggetti_non_catalogati.parquet"

        # aggiorno l'header anche per il file degli estranei
        header_per_non_cat = header_orig.copy() if 'header_orig' in locals() else {}
        if fit_results:
            header_per_non_cat.update(fit_results)

        # salvo la mia nuova tabella
        salva_tabella_parquet(
            mydf,
            header_per_non_cat,
            file_out_mydf,
            str(file_out_mydf.name),
            parametri_caricati
        )

        del run_df
        gc.collect()

print("\n--- ELABORAZIONE COMPLETATA CON SUCCESSO ---")
