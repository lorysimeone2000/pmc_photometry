import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import sys
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import simple_norm
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from astroquery.vizier import Vizier
from astropy.coordinates import Angle, SkyCoord
from astropy.io.fits.verify import VerifyWarning
from pathlib import Path
from astropy.time import Time

# --- GESTIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    # risalgo l'albero delle directory per trovare la radice del progetto
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")
PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")
if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

# importo le mie funzioni di utilità e astrometria
from funzioni.utilita import *
from funzioni.astrometria import *

print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)

# =============================================================================
# BLOCCO DI ESECUZIONE (MAIN)
# =============================================================================

if __name__ == "__main__":

    soglia_correlazione = 35 / 3600 * u.deg
    dist_ripetizione = soglia_correlazione
    magnitudine_massima = 15

    # --- PRE-CALCOLO GLOBALE HIPPARCOS ---
    file_hipparco = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
    if file_hipparco:
        hdu_list_hipparco = fits.open(file_hipparco)
        tbl_catalogo_hipparco = Table(hdu_list_hipparco[1].data)

        # rinomino la colonna della magnitudine in Mag se presente come Vmag
        if 'Vmag' in tbl_catalogo_hipparco.colnames:
            tbl_catalogo_hipparco.rename_column('Vmag', 'Mag')

        hdu_list_hipparco.close()

        coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco['_RAJ2000'],
                                          dec=tbl_catalogo_hipparco['_DEJ2000'],
                                          unit=u.deg)
    else:
        print("Errore: hipparco.fit non trovato.")
        exit()

    tutti_i_file_csv_generati = []
    cartella_dati = BASE_DIR / "PMC_DATA_BLAZAR"
    cartella_tabelle = BASE_DIR / "blazar" / "tabelle" / "tabelle_unite"
    cartella_tabelle.mkdir(parents=True, exist_ok=True)

    if not cartella_dati.exists():
        print(f"ERRORE: Cartella dati {cartella_dati} non trovata.")
        exit()

    contatore_globale = 1

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
                continue

            output_dir = cartella_tabelle / cartella_giorno.name / run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # --- FASE 1: ANALISI E VISUALIZZAZIONE ---
            for n, percorso_file in enumerate(tqdm(file_list, desc=f"Elaborazione {run_name}"), 1):
                if n == 1:
                    # estraggo i dati del FOV dal primo file della run per interrogare Vizier
                    hdu_list = fits.open(percorso_file)
                    w_ref = WCS(hdu_list[0].header)
                    ra_c, dec_c = hdu_list[0].header["RA"], hdu_list[0].header["DEC"]

                    alto_destra = w_ref.pixel_to_world(3071, 2047)
                    centro = SkyCoord(ra_c, dec_c, unit=u.deg)
                    raggio_ricerca = Angle(centro.separation(alto_destra) * 1.5, "deg")
                    hdu_list.close()

                    # interrogo Vizier e filtro Hipparcos per l'area di interesse
                    riquadro_esterno_vizier = \
                        vizier.query_region(centro, radius=raggio_ricerca, column_filters={'gmag': f'<{15}'})[0]
                    distanze_hip = centro.separation(coords_hipparco_global)
                    mask_hip_fov = distanze_hip < raggio_ricerca
                    tbl_hipparco_run_clean = tbl_catalogo_hipparco[mask_hip_fov]
                    tbl_vizier_cut = riquadro_esterno_vizier[riquadro_esterno_vizier['gmag'] < magnitudine_massima]

                    # rinomino la colonna da gmag a Mag per allinearla a quanto richiesto dalla funzione tabella_catalogo
                    tbl_vizier_cut.rename_column('gmag', 'Mag')

                # creo la tabella catalogata per il file corrente
                tbl_catalogate = tabella_catalogo(percorso_file, tbl_vizier_cut, tbl_hipparco_run_clean)
                df_catalogate = tbl_catalogate.to_pandas()

                with fits.open(percorso_file, memmap=False) as hdu:
                    w = WCS(hdu[0].header)
                    header_date_obs = hdu[0].header['DATE-OBS']
                    data_fits = hdu[0].data.astype(float)

                    # --- ESTRAZIONE RIQUADRO BLAZAR ---
                    # calcolo il fondo per pulire la visualizzazione
                    _, median_bg, _ = sigma_clipped_stats(data_fits[::10, ::10], sigma=3.0)
                    data_sub = data_fits - median_bg

                    # coordinate Markarian 421
                    coords_mrk421 = SkyCoord(ra=166.1138 * u.deg, dec=38.2088 * u.deg, frame='icrs')
                    x_mrk, y_mrk = w.world_to_pixel(coords_mrk421)

                    # calcolo la scala dei pixel in arcsec/pixel per il mio WCS
                    pixel_scale = proj_plane_pixel_scales(w)[0] * 3600.0

                    # calcolo quanti pixel corrispondono a metà del mio lato da 1.6 arcominuti (96 arcsec in totale, quindi 48 arcsec dal centro)
                    half_size_px = int(np.round(48.0 / pixel_scale))

                    # definisco il mio cutout usando la mezza ampiezza calcolata
                    x_min, x_max = int(np.round(x_mrk)) - half_size_px, int(np.round(x_mrk)) + half_size_px
                    y_min, y_max = int(np.round(y_mrk)) - half_size_px, int(np.round(y_mrk)) + half_size_px

                    x_min_s, x_max_s = max(0, x_min), min(data_fits.shape[1], x_max)
                    y_min_s, y_max_s = max(0, y_min), min(data_fits.shape[0], y_max)

                    cutout_data = data_sub[y_min_s:y_max_s, x_min_s:x_max_s]

                    # --- GENERAZIONE PLOT ---
                    img_dir = BASE_DIR / "blazar" / "analisi" / "studio_blazar" / "immagini_campo_blazar"
                    img_dir.mkdir(parents=True, exist_ok=True)

                    # imposto una dimensione della figura ridotta per mantenere leggibili i font grandi
                    fig, ax = plt.subplots(figsize=(4, 4))
                    cutout_plot = np.clip(cutout_data, a_min=1e-3, a_max=None)

                    # imposto l'extent per rappresentare gli assi con le coordinate assolute del riquadro
                    img_plot = ax.imshow(cutout_plot, cmap="grey_r", norm=LogNorm(), interpolation='nearest',
                                         extent=[x_min_s, x_max_s, y_max_s, y_min_s])
                    ax.invert_yaxis()

                    # aggiungo i titoli agli assi X e Y con dimensioni font maggiorate
                    ax.set_xlabel("X", fontsize=10)
                    ax.set_ylabel("Y", fontsize=10)

                    # ingrandisco i numeri sui tick degli assi
                    ax.tick_params(axis='both', which='major', labelsize=14)

                    # configuro la colorbar con testi e tick adatti al restringimento
                    cbar = plt.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Pixel value (ADU)', size=16)
                    cbar.ax.tick_params(labelsize=14)

                    try:
                        titolo_data_ora = Time(header_date_obs).to_datetime().strftime('%Y/%m/%d %H:%M:%S')
                    except:
                        titolo_data_ora = str(header_date_obs)

                    # imposto il titolo ingrandito
                    ax.set_title(f"{titolo_data_ora}", fontsize=16)

                    # calcolo raggio in pixel per il cerchio di riferimento (35 arcsec)
                    pixel_scale = proj_plane_pixel_scales(w)[0] * 3600.0
                    raggio_pixel = 35.0 / pixel_scale

                    # segno la posizione teorica del Blazar utilizzando le coordinate assolute
                    x_c_cut, y_c_cut = x_mrk, y_mrk
                    circle = plt.Circle((x_c_cut, y_c_cut), raggio_pixel, color='green', fill=False, linewidth=1.5)
                    ax.add_patch(circle)

                    # sovrappongo le stelle di catalogo identificate nel riquadro usando le coordinate assolute
                    if not df_catalogate.empty:
                        c_cat = SkyCoord(ra=df_catalogate['RAJ2000'].values * u.deg,
                                         dec=df_catalogate['DEJ2000'].values * u.deg)
                        x_cat, y_cat = w.world_to_pixel(c_cat)

                        mask = (x_cat >= x_min_s) & (x_cat < x_max_s) & (y_cat >= y_min_s) & (y_cat < y_max_s)
                        if np.any(mask):
                            scatter_cat = ax.scatter(x_cat[mask], y_cat[mask],
                                                     c=df_catalogate['Mag'].values[mask], cmap='viridis_r',
                                                     s=25, zorder=5, vmin=df_catalogate['Mag'].min(),
                                                     vmax=df_catalogate['Mag'].max())

                    plt.savefig(img_dir / f"markarian_{cartella_giorno.name}_{run_name}_{contatore_globale:05d}.png",
                                bbox_inches='tight')
                    plt.close()
                    contatore_globale += 1

                # salvo il file CSV base senza la parte di segmentation
                file_out = output_dir / f'{run_name}_stelle_catalogo_immagine_{n:03d}.csv'
                df_catalogate.to_csv(file_out, index=False)

    print("\n--- OPERAZIONE COMPLETATA ---")