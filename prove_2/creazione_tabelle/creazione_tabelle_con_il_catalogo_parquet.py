import pandas as pd
# pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # mi permette di avere la scala logaritmica
from scipy.optimize import curve_fit
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
import numpy as np
import os
import sys
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture

# imposto il mio wcs
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
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)  # sopprimo il mio warning FITSFixedWarning

# --- IMPORT FONDAMENTALE PER LA PORTABILITÀ ---
from pathlib import Path

# sopprimo i miei warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI DINAMICA (PORTABILITÀ TOTALE)
# =============================================================================

import pandas as pd
import matplotlib
import argparse
import json
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import concurrent.futures
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
warnings.filterwarnings('ignore', message='.*deblending mode.*')


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
print(f"------------------------------")

# inizializzo il mio Vizier estraendo tutte e 5 le bande necessarie
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1
)

# definisco le mie run da analizzare
RUN = [1, 2, 3]

# cerco il mio file contenente la curva di efficienza quantica per calcolare i pesi esatti
file_curva_pmc = cerca_file_nel_progetto(BASE_DIR, "curva_PMC.csv")
if file_curva_pmc is not None:
    # leggo il mio dataframe della curva
    df_curva = pd.read_csv(file_curva_pmc)

    # scarico i miei intervalli delle bande direttamente dalle descrizioni
    limiti_bande = scarica_intervalli_bande_ps1_da_descrizioni()
    print(f"Limiti bande: \n{limiti_bande}")

    pesi_estratti = []

    # itero sulle bande per calcolare l'area sottesa alla curva per ciascun range
    for nome_banda, (w_min, w_max) in limiti_bande.items():
        # applico la mia maschera di taglio per l'intervallo corrente
        maschera_w = (df_curva['Wavelength'] >= w_min) & (df_curva['Wavelength'] <= w_max)
        # calcolo il mio integrale tramite il metodo dei trapezi per estrarre la porzione di efficienza
        area = np.trapezoid(df_curva['QE'][maschera_w], x=df_curva['Wavelength'][maschera_w])
        pesi_estratti.append(area)

    # converto in array e normalizzo in modo che la somma finale sia pari a 1
    pesi_estratti = np.array(pesi_estratti)
    somma_pesi = np.sum(pesi_estratti)
    pesi_ideali_globali = pesi_estratti / somma_pesi

    # calcolo il mio peso per la banda Vmag di Hipparco nell'intervallo 500-600 nm
    maschera_vmag = (df_curva['Wavelength'] >= 500) & (df_curva['Wavelength'] <= 600)
    area_vmag = np.trapezoid(df_curva['QE'][maschera_vmag], x=df_curva['Wavelength'][maschera_vmag])
    peso_hipparco = area_vmag / somma_pesi
else:
    # imposto i miei pesi standard in caso di mancato ritrovamento del csv
    pesi_ideali_globali = np.array([0.458, 0.326, 0.133, 0.055, 0.028])
    # imposto il mio peso di fallback per hipparco
    peso_hipparco = 0.35

# inizio il mio ciclo per ogni run
for run in RUN:
    print(f"\n==================== ELABORAZIONE RUN {run} ====================")

    # --- RICERCA CARTELLE RUN ---
    nome_cartella_run = f"20250120_run{run}"
    found_folders = list(BASE_DIR.rglob(nome_cartella_run))

    if not found_folders:
        print(
            f"AVVISO: Cartella '{nome_cartella_run}' non trovata in nessuna sottocartella di {BASE_DIR}. Salto la run.")
        continue

    run_folder = found_folders[0]
    if len(found_folders) > 1:
        print(f"AVVISO: Trovate {len(found_folders)} cartelle. Uso la prima: {run_folder}")
    else:
        print(f"Cartella dati trovata: {run_folder.relative_to(BASE_DIR)}")

    # cerco i miei file FITS
    estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
    file_list = []
    for ext in estensioni_valide:
        file_list.extend(run_folder.glob(ext))

    file_list = sorted(file_list, key=lambda x: x.name)
    file_list = [str(f) for f in file_list]

    if not file_list:
        print(f"AVVISO: Nessun file FITS trovato in {run_folder}. Salto la run.")
        continue

    print(f"Trovati {len(file_list)} file da elaborare.")

    # --- GESTIONE DINAMICA CARTELLA OUTPUT ---
    cartella_tabelle = cerca_cartella_nel_progetto(BASE_DIR, "tabelle_alleggerite")
    if cartella_tabelle is None:
        # creo se non esiste in base_dir
        cartella_tabelle = BASE_DIR / "tabelle"
        cartella_tabelle.mkdir(exist_ok=True)

    # creo la mia sottocartella specifica per la run corrente
    output_dir = cartella_tabelle / "sorgenti_catalogate_run" / f"sorgenti_catalogate_run_{run}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir)

    print(f"Cartella Output: {output_dir.relative_to(BASE_DIR)}")

    i = 0
    j = 0
    posizioni_lista = []  # lista che dovrà essere riempita con tutte le mie poszioni di tutte le tabelle
    distanze = []
    numero_stelle_catalogate = []
    tempo = []

    # inizializzo le mie variabili dei cataloghi puliti fuori dal ciclo per la run
    tbl_vizier_cut = None
    tbl_hipparco_run_clean = None

    # itero su tutti i file fits
    for percorso_file_fits in file_list:
        i += 1
        # controllo la mia esistenza del file
        if not os.path.exists(percorso_file_fits):
            print(f"AVVISO: File non trovato, salto: {percorso_file_fits}")
            continue

        hdu_list = fits.open(percorso_file_fits)
        image_header = hdu_list[0].header

        if i == 1:  # chiamo il sito una volta sola su un'immagine più grande per non mandarlo in down

            # coordinate centro
            ra_centro = image_header["RA"]
            dec_centro = image_header["DEC"]
            data = hdu_list[0].data

            w = WCS(hdu_list[0].header)  # creo un mio oggetto WCS usando l'header del file FITS
            alto_destra = w.pixel_to_world(3071, 2047)
            alto_sinistra = w.pixel_to_world(3071, 0)
            basso_sinistra = w.pixel_to_world(0, 0)
            basso_destra = w.pixel_to_world(0, 2047)

            centro = SkyCoord(ra_centro, dec_centro, unit=u.deg)

            # creo un mio riquadro esterno leggermente più grande
            raggio_ricerca = Angle(centro.separation(alto_destra) * 1.5, "deg")
            riquadro_esterno_vizier = vizier.query_region(coord.SkyCoord(ra=ra_centro, dec=dec_centro,
                                                                         unit=(u.deg, u.deg),
                                                                         frame='icrs'),
                                                          radius=raggio_ricerca,
                                                          )  # ho messo un limite di magnitudine per non scaricare milioni di stelle
            tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]

            # calcolo la mia magnitudine sintetica combinata per tutto il riquadro
            bande = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']

            flussi = []

            for banda in bande:
                colonna = tbl_riquadro_esterno_vizier[banda]
                array_dati = colonna.filled(np.nan) if hasattr(colonna, 'filled') else np.array(colonna)
                flusso = 10 ** (-0.4 * array_dati)

                # sostituisco i dati mancanti con un flusso pari a zero
                # assumo che se il catalogo non ha visto la mia stella in questa banda,
                # il suo contributo di luce qui è nullo
                flusso_pulito = np.nan_to_num(flusso, nan=0.0)
                flussi.append(flusso_pulito)

            flussi = np.array(flussi)
            array_pesi = pesi_ideali_globali[:, None]

            # calcolo il mio flusso pesato totale senza normalizzare per le bande mancanti
            # in questo modo le stelle fredde mantengono i loro pesi ottici alti moltiplicati per zero
            flusso_finale = np.sum(flussi * array_pesi, axis=0)

            with np.errstate(divide='ignore', invalid='ignore'):
                # assegno una magnitudine fittizia di 99.0 agli oggetti che risultano avere flusso totalmente zero
                mag_sintetica_globale = np.where(flusso_finale > 0, -2.5 * np.log10(flusso_finale),
                                                 99.0)

            tbl_riquadro_esterno_vizier['Mag_sintetica'] = mag_sintetica_globale

            # --- RICERCA DINAMICA HIPPARCO TRAMITE VIZIER ---
            print("Scaricamento catalogo globale Hipparcos da VizieR in corso...")
            vizier_hip = Vizier(
                catalog="I/239/hip_main",
                # estraggo le coordinate ICRS all'epoca J2000 esatte pre-calcolate da VizieR
                columns=['HIP', '_RA.icrs', '_DE.icrs', 'Vmag', 'B-V'],
                row_limit=-1
            )
            risultato_hip = vizier_hip.query_constraints(Vmag="<16")
            tbl_catalogo_hipparco = risultato_hip[0]

            # rinomino le mie colonne ICRS J2000 per mantenere la compatibilità col resto dello script
            if '_RA.icrs' in tbl_catalogo_hipparco.colnames:
                tbl_catalogo_hipparco.rename_column('_RA.icrs', '_RAJ2000')
                tbl_catalogo_hipparco.rename_column('_DE.icrs', '_DEJ2000')

            print(f"Scaricati {len(tbl_catalogo_hipparco)} oggetti da Hipparcos.")

            # imposto la mia soglia fissa a 2.5 arcosecondi
            exclusion_radii_deg = np.full(len(tbl_catalogo_hipparco), 2.5 / 3600.0)

            # creo il mio oggetto SkyCoord globale per Hipparcos
            coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco['_RAJ2000'],
                                              dec=tbl_catalogo_hipparco['_DEJ2000'],
                                              unit=u.deg)

            # filtro spazialmente Hipparcos per la run corrente
            distanze_hip = centro.separation(coords_hipparco_global)
            mask_hip_fov = distanze_hip < raggio_ricerca

            tbl_hipparco_run_subset = tbl_catalogo_hipparco[mask_hip_fov]

            # calcolo il mio flusso di Hipparco, applico il mio peso e riconverto in magnitudine
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

            # =================================================================
            # --- FILTRAGGIO COMPETITIVO A SINGOLA FASE (AGGIUNTO) ---
            # =================================================================
            print("Avvio filtraggio competitivo a singola fase Vizier vs Hipparcos...")

            # preparo le mie coordinate Vizier
            coords_vizier = SkyCoord(ra=tbl_riquadro_esterno_vizier['RAJ2000'],
                                     dec=tbl_riquadro_esterno_vizier['DEJ2000'],
                                     unit=u.deg)

            # ricavo il mio limite massimo di ricerca per coprire tutte le tolleranze
            max_threshold_deg = np.max(exclusion_radii_run_subset)
            seplimit = max_threshold_deg * u.deg

            # cerco tutte le mie stelle Vizier attorno a ogni stella Hipparcos
            idx_A, idx_B, d2d_1, _ = coords_hipparco_run_subset.search_around_sky(coords_vizier, seplimit)

            # implemento il mio controllo di sicurezza per aggirare l'inversione degli indici di astropy
            if len(idx_A) > 0 and np.max(idx_A) >= len(coords_hipparco_run_subset):
                idx_viz_1, idx_hip_1 = idx_A, idx_B
            else:
                idx_hip_1, idx_viz_1 = idx_A, idx_B

            # applico la mia tolleranza dinamica esatta
            mask_threshold = d2d_1.deg <= exclusion_radii_run_subset[idx_hip_1]

            # filtro i miei indici per tenere solo quelli entro la tolleranza
            idx_hip_valid = idx_hip_1[mask_threshold]
            idx_viz_valid = idx_viz_1[mask_threshold]

            # genero le mie maschere di mantenimento inizializzate a True
            mask_keep_hipparco = np.ones(len(tbl_hipparco_run_subset), dtype=bool)
            mask_keep_vizier = np.ones(len(tbl_riquadro_esterno_vizier), dtype=bool)

            # estraggo i miei indici univoci di Hipparcos che hanno almeno un match
            unique_hip_idx = np.unique(idx_hip_valid)

            # --- TRUCCO DI VELOCIZZAZIONE ---
            # estraggo le mie magnitudini in array numpy puri prima del ciclo per evitare l'overhead di Astropy
            array_mag_vizier = np.nan_to_num(tbl_riquadro_esterno_vizier['Mag_sintetica'].data, nan=99.0)
            array_mag_hipparco = np.nan_to_num(tbl_hipparco_run_subset['Vmag'].data, nan=99.0)

            # itero su ogni stella Hipparcos coinvolta
            for i_hip in unique_hip_idx:
                # trovo i miei indici delle stelle Vizier associate a questa specifica stella Hipparcos
                viz_matches = idx_viz_valid[idx_hip_valid == i_hip]

                if len(viz_matches) > 0:
                    # pesco le mie magnitudini in modo ultra-rapido dall'array numpy
                    mag_viz_matches = array_mag_vizier[viz_matches]

                    # individuo la mia stella Vizier più luminosa (valore di magnitudine minore)
                    idx_min_mag = np.argmin(mag_viz_matches)
                    best_viz_idx = viz_matches[idx_min_mag]
                    best_viz_mag = mag_viz_matches[idx_min_mag]

                    # pesco la magnitudine della stella Hipparcos in esame dall'array
                    hip_mag = array_mag_hipparco[i_hip]

                    # confronto e scarto applicando la mia soglia prioritaria per Hipparcos
                    if hip_mag < 9.0:
                        mask_keep_vizier[viz_matches] = False
                    elif best_viz_mag <= hip_mag:
                        # Vizier è più luminosa (o uguale), scarto la mia stella Hipparcos
                        mask_keep_hipparco[i_hip] = False
                    else:
                        # Hipparcos è più luminosa, scarto la mia Vizier più luminosa
                        mask_keep_vizier[best_viz_idx] = False

            hipparco_escluse = np.sum(~mask_keep_hipparco)
            vizier_escluse = np.sum(~mask_keep_vizier)
            print(f"Risolti {len(unique_hip_idx)} conflitti spaziali:")
            print(f" -> Escluse {hipparco_escluse} stelle Hipparco (tenute Vizier perché più brillanti)")
            print(f" -> Escluse {vizier_escluse} stelle Vizier (tenute Hipparco perché più brillanti)")

            mask_keep_hipparco[tbl_hipparco_run_subset['Vmag'] >= 15] = False

            # uso il mio copy() per svincolare i dati
            tbl_hipparco_run_clean = tbl_hipparco_run_subset[mask_keep_hipparco].copy()
            tbl_riquadro_esterno_vizier_CLEAN = tbl_riquadro_esterno_vizier[mask_keep_vizier]

            # calcolo la mia magnitudine sintetica finale
            mag_max = 15
            with np.errstate(invalid='ignore'):
                mask_taglio = tbl_riquadro_esterno_vizier_CLEAN['Mag_sintetica'] < mag_max
            tbl_vizier_cut = tbl_riquadro_esterno_vizier_CLEAN[mask_taglio].copy()

            # --- UNIFICAZIONE MAGNITUDINI ---
            # ridefinisco la mia colonna Mag per Pan-STARRS usando la magnitudine sintetica
            tbl_vizier_cut['Mag'] = tbl_vizier_cut['Mag_sintetica']

            # ridefinisco la mia colonna Mag per Hipparcos usando direttamente la Vmag
            tbl_hipparco_run_clean['Mag'] = tbl_hipparco_run_clean['Vmag']

            print("-----------------------------")

        print(f"Elaborando {percorso_file_fits}")
        print("\n")

        tbl_catalogate = tabella_catalogo(percorso_file_fits, tbl_vizier_cut, tbl_hipparco_run_clean)

        numero_stelle_catalogate.append(len(tbl_catalogate))
        print(f"Trovate {len(tbl_catalogate)} stelle dei cataloghi nel riquadro {i}")

        # creo i miei file csv
        dataframe = tbl_catalogate.to_pandas()
        # uso output_dir (Path object)
        filename = output_dir / f'run_{run}_stelle_catalogate_immagine_{i:03d}.parquet'
        salva_tabella_parquet(dataframe, image_header, str(filename), percorso_file_fits)

        # if i == 10: break