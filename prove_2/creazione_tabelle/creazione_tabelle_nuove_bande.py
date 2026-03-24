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
from pathlib import Path

# gestisco i warning ignorandoli
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
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

# definisco le mie run da analizzare
RUN = [1, 2, 3]

# uso il mirror di Harvard scaricando le 5 bande fondamentali per simulare il sensore FLIR
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1,
)

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

    print("Scaricamento catalogo globale Hipparcos da VizieR in corso...")
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

    next_internal_id = 1
    dati_tutte_le_run = []
    mappa_headers_globali = {}

    # cerco il file contenente la curva di efficienza quantica per calcolare i pesi esatti
    file_curva_pmc = cerca_file_nel_progetto(BASE_DIR, "curva_PMC.csv")
    if file_curva_pmc is not None:
        # leggo il dataframe della curva
        df_curva = pd.read_csv(file_curva_pmc)

        # stabilisco i limiti di lunghezza d'onda delle singole bande
        limiti_bande = {
            'gmag': (400, 550),
            'rmag': (550, 700),
            'imag': (680, 840),
            'zmag': (820, 920),
            'ymag': (920, 1050)
        }

        pesi_estratti = []

        # itero sulle bande per calcolare l'area sottesa alla curva per ciascun range
        for nome_banda, (w_min, w_max) in limiti_bande.items():
            # applico la maschera di taglio per l'intervallo corrente
            maschera_w = (df_curva['Wavelength'] >= w_min) & (df_curva['Wavelength'] <= w_max)
            # calcolo l'integrale tramite il metodo dei trapezi per estrarre la porzione di efficienza
            area = np.trapz(df_curva['QE'][maschera_w], x=df_curva['Wavelength'][maschera_w])
            pesi_estratti.append(area)

        # converto in array e normalizzo in modo che la somma finale sia pari a 1
        pesi_estratti = np.array(pesi_estratti)
        pesi_ideali_globali = pesi_estratti / np.sum(pesi_estratti)
    else:
        # imposto i pesi standard in caso di mancato ritrovamento del csv
        pesi_ideali_globali = np.array([0.458, 0.326, 0.133, 0.055, 0.028])

    for run in RUN:
        print(f"\n==================== ELABORAZIONE RUN {run} ====================")
        nome_cartella_run = f"20250120_run{run}"
        found_folders = list(BASE_DIR.rglob(nome_cartella_run))
        if not found_folders:
            continue
        run_folder = found_folders[0]

        estensioni_valide = ['*.fit', '*.fits', '*.FIT', '*.FITS']
        file_list = []
        for ext in estensioni_valide:
            file_list.extend(run_folder.glob(ext))

        file_list = sorted([str(f) for f in file_list])
        if not file_list:
            continue

        cartella_prove = BASE_DIR
        cartella_tabelle = cartella_prove / "tabelle"
        output_dir = cartella_tabelle / "tabelle_unite" / f"tabelle_unite_run_{run}"
        output_dir.mkdir(parents=True, exist_ok=True)

        global_tracker_coords = None
        global_tracker_labels = []
        file_csv_generati_nella_run = []

        print(f"--- FASE 1: Segmentazione & Unione ({len(file_list)} files) ---")

        for n, percorso_file in enumerate(tqdm(file_list, desc=f"Fase 1 Run {run}"), 1):
            if n == 1:
                hdu_list = fits.open(percorso_file)
                w = WCS(hdu_list[0].header)
                ra_c, dec_c = hdu_list[0].header["RA"], hdu_list[0].header["DEC"]
                alto_destra = w.pixel_to_world(3071, 2047)
                centro = SkyCoord(ra_c, dec_c, unit=u.deg)
                raggio_ricerca = Angle(centro.separation(alto_destra) * 1.5, "deg")
                hdu_list.close()

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

                # calcolo la magnitudine sintetica FLIR
                bande = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']

                # recupero i pesi precedentemente calcolati in modo efficiente
                pesi_ideali = pesi_ideali_globali

                flussi = []
                maschere_valide = []

                for banda in bande:
                    colonna = tbl_riquadro_esterno_vizier[banda]
                    array_dati = colonna.filled(np.nan) if hasattr(colonna, 'filled') else np.array(colonna)
                    flusso = 10 ** (-0.4 * array_dati)
                    flussi.append(flusso)
                    maschere_valide.append(~np.isnan(flusso))

                flussi = np.array(flussi)
                maschere_valide = np.array(maschere_valide)
                array_pesi = pesi_ideali[:, None]

                pesi_attivi = array_pesi * maschere_valide
                somma_pesi = np.sum(pesi_attivi, axis=0)
                flussi_sicuri = np.nan_to_num(flussi)
                flusso_pesato_totale = np.sum(flussi_sicuri * pesi_attivi, axis=0)

                with np.errstate(divide='ignore', invalid='ignore'):
                    flusso_finale = flusso_pesato_totale / somma_pesi
                    mag_sintetica_globale = -2.5 * np.log10(flusso_finale)

                tbl_riquadro_esterno_vizier['Mag_sintetica'] = mag_sintetica_globale

                distanze_hip = centro.separation(coords_hipparco_global)
                mask_hip_fov = distanze_hip < raggio_ricerca
                tbl_hipparco_run_subset = tbl_catalogo_hipparco[mask_hip_fov]
                coords_hipparco_run_subset = coords_hipparco_global[mask_hip_fov]
                exclusion_radii_run_subset = exclusion_radii_deg[mask_hip_fov]

                # avvio il filtraggio competitivo a singola fase Vizier vs Hipparcos
                print("Avvio filtraggio competitivo a singola fase Vizier vs Hipparcos...")
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

            tbl_vizier_cut['Mag'] = tbl_vizier_cut['Mag_sintetica']
            tbl_hipparco_run_clean['Mag'] = tbl_hipparco_run_clean['Vmag']

            tbl_catalogate = tabella_catalogo(percorso_file, tbl_vizier_cut, tbl_hipparco_run_clean)
            tbl_trovate, _ = analisi_image_segmentation(percorso_file, parametri_caricati)

            df_trovate = tbl_trovate.to_pandas()
            df_catalogate = tbl_catalogate.to_pandas()

            all_cols = df_trovate.columns.tolist()
            cols_keep = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']
            for c in ['saturazione', 'kron_flux']:
                if c in all_cols: cols_keep.append(c)
            extra_flux = ['kron_manuale_seg', 'kron_manuale_aper', 'somma_apertura_ultimo_pixel',
                          'raggio_kron_aper']
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

                    # assegno correttamente gli indici
                    idx_cat, idx_trov, d2d, _ = search_around_sky(c_cat, coords, soglia_correlazione)

                    matches = pd.DataFrame(
                        {'idx_t': idx_trov, 'idx_c': idx_cat, 'dist': d2d.deg,
                         'mag': df_catalogate.iloc[idx_cat]['Mag'].values})
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

            coords_obj_all = SkyCoord(ra=np.atleast_1d(df_final['RA_centroid'].values) * u.deg,
                                      dec=np.atleast_1d(df_final['DEC_centroid'].values) * u.deg)
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
            df_final['run_id'] = run
            df_final['img_index'] = n

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
            file_out = output_dir / f'run_{run}_stelle_trovate_e_catalogate_immagine_{n:03d}.csv'
            salva_csv_con_header_fits(df_final, dict(fits.getheader(percorso_file)),
                                      file_out, str(percorso_file), parametri_caricati)

        # =============================================================================
        # Avvio la fase 2 e 3 per estrarre i raggi massimi e il flusso fisso per la mia run
        # =============================================================================
        print(f"--- FASE 2 & 3: Analisi Fotometria Fissa per Run {run} ---")

        file_csv_list = sorted([f for f in output_dir.glob('*.csv')])

        for f in file_csv_list:
            file_csv_generati_nella_run.append((f, run))

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

            nome_fits = header_info.get('NOME_FILE_FITS', '')
            if not nome_fits:
                percorso_raw = header_info.get('PERCORSO_FILE', '')
                nome_fits = os.path.basename(str(percorso_raw))

            nome_fits = str(nome_fits).strip()
            file_trovato = cerca_file_nel_progetto(BASE_DIR, nome_fits)

            if file_trovato is None:
                continue

            path_fits = str(file_trovato)
            data_sub, median_bg, _ = elabora_file_fits(path_fits)

            img_idx = df_frame['img_index'].iloc[0] if 'img_index' in df_frame.columns else int(
                nome_fits.split('.')[0][-3:])
            run_idx = df_frame['run_id'].iloc[0] if 'run_id' in df_frame.columns else run

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
            molt_corr = s_ref / s_t if s_t != 0 else 1.0
            diff_s = s_ref - s_t

            if 'kron_manuale_aper' in df_frame.columns and 'raggio_kron_aper' in df_frame.columns:
                df_frame['kron_manuale_aper_CORRETTO_Normalizzazione_Moltiplicativa'] = pd.to_numeric(
                    df_frame['kron_manuale_aper'], errors='coerce') * molt_corr
                df_frame['kron_manuale_aper_CORRETTO_Correzione_Additiva_dell_Apertura'] = pd.to_numeric(
                    df_frame['kron_manuale_aper'], errors='coerce') + (np.pi * pd.to_numeric(
                    df_frame['raggio_kron_aper'], errors='coerce') ** 2 / n_tot) * diff_s

            if 'kron_manuale_seg' in df_frame.columns:
                df_frame['kron_manuale_seg_CORRETTO_Normalizzazione_Moltiplicativa'] = pd.to_numeric(
                    df_frame['kron_manuale_seg'], errors='coerce') * molt_corr

            if 'somma_apertura_ultimo_pixel' in df_frame.columns:
                df_frame['somma_apertura_ultimo_pixel_CORRETTO_Normalizzazione_Moltiplicativa'] = pd.to_numeric(
                    df_frame['somma_apertura_ultimo_pixel'], errors='coerce') * molt_corr

            raggi_fissi = []
            ids_presenti = df_frame['ID'].values

            flussi_calcolati = []
            flussi_calcolati_molt = []
            flussi_calcolati_add = []
            flussi_calcolati_fondo_sottratto = []

            flussi_doppi = []
            flussi_doppi_molt = []
            flussi_doppi_add = []
            flussi_doppi_fondo_sottratto = []

            flussi_kron_manuale_aper_fondo_sottratto = []

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
                    flussi_calcolati.append(fl_calcolato)

                    flussi_calcolati_molt.append(fl_calcolato * molt_corr)
                    flussi_calcolati_add.append(fl_calcolato + (np.pi * r_globale ** 2 / n_tot) * diff_s)
                    fl_sottratto = fl_calcolato - (fondo_pp * aper.area)
                    flussi_calcolati_fondo_sottratto.append(fl_sottratto)

                    r_doppio = r_globale * 2
                    aper_doppia = CircularAperture(pos, r=r_doppio)
                    phot_doppia = aperture_photometry(data_sub, aper_doppia)
                    fl_doppio = phot_doppia['aperture_sum'][0]
                    flussi_doppi.append(fl_doppio)

                    flussi_doppi_molt.append(fl_doppio * molt_corr)
                    flussi_doppi_add.append(fl_doppio + (np.pi * r_doppio ** 2 / n_tot) * diff_s)
                    fl_doppio_sottratto = fl_doppio - (fondo_pp * aper_doppia.area)
                    flussi_doppi_fondo_sottratto.append(fl_doppio_sottratto)
                else:
                    flussi_calcolati.append(np.nan)
                    flussi_calcolati_molt.append(np.nan)
                    flussi_calcolati_add.append(np.nan)
                    flussi_calcolati_fondo_sottratto.append(np.nan)

                    flussi_doppi.append(np.nan)
                    flussi_doppi_molt.append(np.nan)
                    flussi_doppi_add.append(np.nan)
                    flussi_doppi_fondo_sottratto.append(np.nan)

                if 'kron_manuale_aper' in df_frame.columns and 'raggio_kron_aper' in df_frame.columns:
                    r_kron_man = pd.to_numeric(df_frame.at[idx_star, 'raggio_kron_aper'], errors='coerce')
                    fl_kron_man = pd.to_numeric(df_frame.at[idx_star, 'kron_manuale_aper'], errors='coerce')
                    if pd.notnull(r_kron_man) and r_kron_man > 0 and pd.notnull(fl_kron_man):
                        fl_kron_man_sottr = fl_kron_man - (fondo_pp * (np.pi * r_kron_man ** 2))
                        flussi_kron_manuale_aper_fondo_sottratto.append(fl_kron_man_sottr)
                    else:
                        flussi_kron_manuale_aper_fondo_sottratto.append(np.nan)

            df_frame['raggio_fisso_max_run'] = raggi_fissi

            df_frame['flusso_fisso_max_run'] = flussi_calcolati
            df_frame['flusso_fisso_max_run_CORRETTO_Normalizzazione_Moltiplicativa'] = flussi_calcolati_molt
            df_frame['flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura'] = flussi_calcolati_add
            df_frame['flusso_fisso_max_run_FONDO_SOTTRATTO'] = flussi_calcolati_fondo_sottratto

            df_frame['flusso_raggio_fisso_doppio'] = flussi_doppi
            df_frame['flusso_raggio_fisso_doppio_CORRETTO_Normalizzazione_Moltiplicativa'] = flussi_doppi_molt
            df_frame['flusso_raggio_fisso_doppio_CORRETTO_Correzione_Additiva_dell_Apertura'] = flussi_doppi_add
            df_frame['flusso_raggio_fisso_doppio_FONDO_SOTTRATTO'] = flussi_doppi_fondo_sottratto

            if 'kron_manuale_aper' in df_frame.columns and 'raggio_kron_aper' in df_frame.columns:
                df_frame['kron_manuale_aper_FONDO_SOTTRATTO'] = flussi_kron_manuale_aper_fondo_sottratto

            df_frame['fondo_per_pixel'] = fondo_pp

            if 'label' in df_frame.columns:
                df_frame.sort_values(by=['label', 'Corrispondenza'], inplace=True)

            salva_csv_con_header_aggiornato(df_frame, header_info, file_csv)

        # =============================================================================
        # Avvio la fase 4 per calcolare statistiche e ID per la mia singola run corrente
        # =============================================================================
        print(f"\n--- FASE 4: Statistiche Locali e Ripetizioni per Run {run} ---")

        if not file_csv_generati_nella_run:
            continue

        lista_df_run = []
        file_csv_generati_nella_run = sorted(file_csv_generati_nella_run, key=lambda x: str(x[0]))

        for idx_file, (file_csv, run_number) in enumerate(tqdm(file_csv_generati_nella_run, desc="Lettura Dati Run")):
            df_temp = pd.read_csv(file_csv, comment='#')
            df_temp['file_index'] = idx_file
            df_temp['run_number'] = run_number
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
        threshold_deg = 35 / 3600
        unique_files_run = df_no_run['file_index'].unique()

        no_mapping = {}

        for f_idx in tqdm(unique_files_run, desc="Matching oggetti NO (Intra-Run)"):
            subset = df_no_run[df_no_run['file_index'] == f_idx]
            if subset.empty: continue
            coords_subset = SkyCoord(ra=subset['RA_centroid'].values * u.deg,
                                     dec=subset['DEC_centroid'].values * u.deg)
            indices_subset = subset.index.tolist()

            if not known_clusters_coords_run:
                for i, (ra, dec) in enumerate(zip(subset['RA_centroid'], subset['DEC_centroid'])):
                    cid = f"INT_{next_internal_id}"
                    known_clusters_coords_run.append((ra, dec))
                    known_clusters_ids_run.append(cid)
                    no_mapping[indices_subset[i]] = cid
                    next_internal_id += 1
            else:
                cluster_sc = SkyCoord(known_clusters_coords_run, unit=u.deg)
                idx_cluster, d2d, _ = coords_subset.match_to_catalog_sky(cluster_sc)
                for i, (match_idx, dist, ra_curr, dec_curr) in enumerate(
                        zip(idx_cluster, d2d, subset['RA_centroid'], subset['DEC_centroid'])):
                    global_idx = indices_subset[i]
                    if dist.deg <= threshold_deg:
                        no_mapping[global_idx] = known_clusters_ids_run[match_idx]
                        known_clusters_coords_run[match_idx] = (ra_curr, dec_curr)
                    else:
                        cid = f"INT_{next_internal_id}"
                        known_clusters_ids_run.append(cid)
                        known_clusters_coords_run.append((ra_curr, dec_curr))
                        no_mapping[global_idx] = cid
                        next_internal_id += 1

        for idx, uid in no_mapping.items(): run_df.at[idx, 'run_unique_id'] = uid

        print("Eseguo il filtraggio temporale e delle ripetizioni minime...")

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

        print("Calcolo statistiche di run e riorganizzazione colonne...")
        cols_flux = ['somma_apertura_ultimo_pixel',
                     'somma_apertura_ultimo_pixel_CORRETTO_Normalizzazione_Moltiplicativa',
                     'kron_manuale_seg', 'kron_manuale_seg_CORRETTO_Normalizzazione_Moltiplicativa',
                     'kron_manuale_aper', 'kron_manuale_aper_CORRETTO_Normalizzazione_Moltiplicativa',
                     'kron_manuale_aper_CORRETTO_Correzione_Additiva_dell_Apertura',
                     'kron_manuale_aper_FONDO_SOTTRATTO',
                     'flusso_fisso_max_run', 'flusso_fisso_max_run_CORRETTO_Normalizzazione_Moltiplicativa',
                     'flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura',
                     'flusso_fisso_max_run_FONDO_SOTTRATTO',
                     'flusso_raggio_fisso_doppio', 'flusso_raggio_fisso_doppio_CORRETTO_Normalizzazione_Moltiplicativa',
                     'flusso_raggio_fisso_doppio_CORRETTO_Correzione_Additiva_dell_Apertura',
                     'flusso_raggio_fisso_doppio_FONDO_SOTTRATTO']

        cols_flux_presenti = [c for c in cols_flux if c in run_df.columns]
        for c in cols_flux_presenti: run_df[c] = pd.to_numeric(run_df[c], errors='coerce')

        run_df['fondo_per_pixel'] = pd.to_numeric(run_df.get('fondo_per_pixel', np.nan), errors='coerce')
        nuovi_flussi_decorrelati = []

        for c in cols_flux_presenti:
            new_col = f"{c}_DECORRELAZIONE_LINEARE"
            run_df[new_col] = np.nan
            nuovi_flussi_decorrelati.append(new_col)

            for uid, group in run_df.groupby('run_unique_id'):
                idx = group.index
                F = group[c].values
                B = group['fondo_per_pixel'].values
                mask_valid = ~np.isnan(F) & ~np.isnan(B)

                if np.sum(mask_valid) > 1 and np.std(B[mask_valid]) > 1e-12:
                    F_v = F[mask_valid]
                    B_v = B[mask_valid]
                    m, q = np.polyfit(B_v, F_v, 1)
                    B_mean = np.mean(B_v)
                    F_corr = F_v - m * (B_v - B_mean)
                    run_df.loc[idx[mask_valid], new_col] = F_corr
                else:
                    run_df.loc[idx[mask_valid], new_col] = F[mask_valid]

        cols_tutti_flussi = cols_flux_presenti + nuovi_flussi_decorrelati
        nuovi_flussi_ensemble = []

        for c in cols_tutti_flussi:
            new_col = f"{c}_DECORRELAZIONE_STELLE"
            run_df[new_col] = np.nan
            nuovi_flussi_ensemble.append(new_col)

            mediane_stella = run_df.groupby('run_unique_id')[c].transform('median')

            with np.errstate(divide='ignore', invalid='ignore'):
                rapporto_relativo = np.where(mediane_stella > 0, run_df[c] / mediane_stella, np.nan)

            run_df['temp_rapporto'] = rapporto_relativo
            fattore_immagine = run_df.groupby('file_index')['temp_rapporto'].transform('median')
            run_df[new_col] = run_df[c] / fattore_immagine

        if 'temp_rapporto' in run_df.columns:
            run_df.drop(columns=['temp_rapporto'], inplace=True)

        cols_tutti_flussi.extend(nuovi_flussi_ensemble)
        grouped_per_run_id = run_df.groupby(['run_unique_id'])
        stat_columns = []

        for c in cols_tutti_flussi:
            col_mean = f'media_{c}'
            col_std = f'std_{c}'
            run_df[col_mean] = grouped_per_run_id[c].transform('mean')
            stds_sample = grouped_per_run_id[c].transform('std')
            counts_grouped = grouped_per_run_id[c].transform('count')
            run_df[col_std] = stds_sample / np.sqrt(counts_grouped)
            stat_columns.extend([col_mean, col_std])

        run_df['ID'] = run_df['ID'].astype(object)
        mask_no_match = run_df['Corrispondenza'] == 'NO'
        run_df.loc[mask_no_match, 'ID'] = run_df.loc[mask_no_match, 'run_unique_id']
        files_groups_run = run_df.groupby('original_file_path')
        run_repetition_counts = run_df['run_unique_id'].value_counts()

        for file_path, df_file in tqdm(files_groups_run):
            df_final_save = df_file.copy()
            header_orig = leggi_header_da_csv(file_path)
            dati_tutte_le_run.append(df_final_save)
            mappa_headers_globali[str(file_path)] = header_orig

    # =============================================================================
    # FASE 5: Decorrelazione Globale (Senza Formattazione Decimali)
    # =============================================================================
    if dati_tutte_le_run:
        print("\n--- FASE 5: Calcolo Decorrelazione Globale delle Stelle ---")
        df_totale = pd.concat(dati_tutte_le_run, ignore_index=True)

        # deframmento il dataframe iniziale per ottimizzare e liberare la RAM
        df_totale = df_totale.copy()
        del dati_tutte_le_run
        gc.collect()

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

            # calcolo la vera mediana su tutte le run combinate
            mediane_stella_globale = flusso_numerico.groupby(df_totale['run_unique_id']).transform('median')

            # calcolo il mio rapporto tra il flusso e la vera mediana globale
            with np.errstate(divide='ignore', invalid='ignore'):
                rapporto_relativo = np.where(mediane_stella_globale > 0,
                                             flusso_numerico / mediane_stella_globale,
                                             np.nan)

            # calcolo il fattore di correzione per l'immagine senza aggiungere colonne temporanee al dataframe
            temp_rapporto_series = pd.Series(rapporto_relativo)
            fattore_immagine = temp_rapporto_series.groupby(df_totale['original_file_path']).transform('median')

            # salvo il risultato nel dizionario temporaneo
            dizionario_nuove_colonne[new_col] = flusso_numerico / fattore_immagine

        # unisco tutte le nuove colonne calcolate al dataframe principale in un colpo solo (Zero frammentazione)
        df_totale = pd.concat([df_totale, pd.DataFrame(dizionario_nuove_colonne)], axis=1)

        # calcolo la media e deviazione standard per le mie nuove colonne globali
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

        # unisco le statistiche al dataframe in un colpo solo
        df_totale = pd.concat([df_totale, pd.DataFrame(dizionario_statistiche)], axis=1)

        # riordino e salvo tutti i miei file
        files_groups_globale = df_totale.groupby('original_file_path')
        run_repetition_counts_global = df_totale['run_unique_id'].value_counts()

        for file_path, df_file in tqdm(files_groups_globale, desc="Salvataggio finale FASE 5"):
            header_orig = mappa_headers_globali[file_path]
            cols = df_file.columns.tolist()

            # elimino le mie colonne temporanee necessarie solo allo script
            for temp_c in ['file_index', 'original_file_path', 'original_idx', 'run_unique_id', 'run_number']:
                if temp_c in cols: cols.remove(temp_c)

            df_file = df_file.copy()

            # sistemo la colonna delle ripetizioni calcolandola sull'insieme globale
            df_file['ripetizioni'] = df_file['ID'].map(run_repetition_counts_global)
            if 'ripetizioni' not in cols:
                if 'saturazione' in cols:
                    cols.insert(cols.index('saturazione') + 1, 'ripetizioni')
                else:
                    cols.append('ripetizioni')

            # riordino le mie colonne statistiche globali accanto ai rispettivi flussi
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