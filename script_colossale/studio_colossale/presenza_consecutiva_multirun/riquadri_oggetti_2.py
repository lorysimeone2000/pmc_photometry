import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.optimize import curve_fit
import warnings
from pathlib import Path
from tqdm import tqdm
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import FITSFixedWarning
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
import time
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import astropy.coordinates as coord
from datetime import datetime, timedelta
import pandas as pd
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib

matplotlib.use('Agg')
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
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


def trova_cartella_base(nome_target="pmc_photometry"):
    # cerco la mia cartella base risalendo l'albero delle directory
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

from funzioni.utilita_parquet import *
from funzioni.astrometria_parquet import *

cartella_oggetti = cerca_cartella_intero_pc('ASTRI1')

percorso_candidati_csv = cerca_file_nel_progetto(BASE_DIR, "candidati_frame.csv")

# converto il mio file csv in un dataframe pandas
df_candidati = pd.read_csv(percorso_candidati_csv)

# creo la mia cartella principale dove inserire i riquadri
cartella_riquadri = "riquadri"
os.makedirs(cartella_riquadri, exist_ok=True)

# inizializzo il mio dizionario di cache per velocizzare le ricerche delle cartelle nel pc
cache_cartelle = {}
centro_run_precedente = None
tbl_vizier_cut_precedente = None
tbl_hipparco_run_clean_precedente = None

# raggruppo per label in modo da eseguire la query Vizier una sola volta per etichetta e avvolgo il ciclo con tqdm
for label, gruppo_label in tqdm(df_candidati.groupby('label'), desc="Elaborazione labels"):
    # creo la cartella del label specifico
    cartella_salvataggio_label = os.path.join(cartella_riquadri, str(label))
    os.makedirs(cartella_salvataggio_label, exist_ok=True)

    # estraggo la mia prima riga utile del gruppo per impostare il centro della query
    prima_riga = gruppo_label.iloc[0]
    nome_cartella_query = prima_riga['nome_cartella']
    nome_file_query = prima_riga['nome_file_fits']

    # converto il nome della cartella in un formato data per poter fare calcoli temporali
    data_centrale = datetime.strptime(str(nome_cartella_query), "%Y%m%d")

    # calcolo il nome delle cartelle adiacenti sottraendo e aggiungendo un giorno
    cartella_precedente = (data_centrale - timedelta(days=1)).strftime("%Y%m%d")
    cartella_successiva = (data_centrale + timedelta(days=1)).strftime("%Y%m%d")

    # preparo la mia lista di cartelle in cui effettuare la ricerca
    cartelle_da_esplorare = [str(nome_cartella_query), cartella_precedente, cartella_successiva]

    percorso_file_query = None

    # cerco il mio file scorrendo le tre cartelle possibili e salvando le scoperte in cache
    for cartella_target in cartelle_da_esplorare:
        if cartella_target not in cache_cartelle:
            # cerco la mia cartella specificatamente all'interno di ASTRI1
            risultato = cerca_cartella_nel_progetto(cartella_oggetti, cartella_target)
            cache_cartelle[cartella_target] = str(risultato) if risultato else None

        percorso_cartella_cache = cache_cartelle[cartella_target]
        # mi assicuro che la cartella esista fisicamente prima di unire il percorso
        if percorso_cartella_cache is not None:
            percorso_temporaneo = os.path.join(percorso_cartella_cache, nome_file_query)
            if os.path.exists(percorso_temporaneo):
                percorso_file_query = percorso_temporaneo
                break  # interrompo la ricerca non appena trovo il mio file

    if percorso_file_query is None:
        print(f"ATTENZIONE: Impossibile trovare il file query {nome_file_query}. Salto il label {label}.")
        continue

    hdu_list = fits.open(percorso_file_query)
    w = WCS(hdu_list[0].header)
    ra_c, dec_c = hdu_list[0].header["RA"], hdu_list[0].header["DEC"]
    alto_destra = w.pixel_to_world(3071, 2047)
    centro = SkyCoord(ra_c, dec_c, unit=u.deg)
    raggio_ricerca = Angle(centro.separation(alto_destra) * 1.5, "deg")
    hdu_list.close()

    scarica_nuovo_catalogo = True
    if centro_run_precedente is not None:
        distanza = centro.separation(centro_run_precedente)
        if distanza.deg <= 1.1:
            scarica_nuovo_catalogo = False

    if scarica_nuovo_catalogo:
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

        # calcolo la mia magnitudine sintetica FLIR
        bande = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']
        flussi = []

        for banda in bande:
            colonna = tbl_riquadro_esterno_vizier[banda]
            array_dati = colonna.filled(np.nan) if hasattr(colonna, 'filled') else np.array(colonna)
            flusso = 10 ** (-0.4 * array_dati)
            flusso_pulito = np.nan_to_num(flusso, nan=0.0)
            flussi.append(flusso_pulito)

        flussi = np.array(flussi)
        array_pesi = pesi_ideali_globali[:, None]
        flusso_finale = np.sum(flussi * array_pesi, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            mag_sintetica_globale = np.where(flusso_finale > 0, -2.5 * np.log10(flusso_finale), 99.0)

        tbl_riquadro_esterno_vizier['Mag_sintetica'] = mag_sintetica_globale

        distanze_hip = centro.separation(coords_hipparco_global)
        mask_hip_fov = distanze_hip < raggio_ricerca
        tbl_hipparco_run_subset = tbl_catalogo_hipparco[mask_hip_fov]

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

        mask_keep_hipparco[tbl_hipparco_run_subset['Vmag'] >= 15] = False
        tbl_hipparco_run_clean = tbl_hipparco_run_subset[mask_keep_hipparco].copy()
        tbl_riquadro_esterno_vizier_CLEAN = tbl_riquadro_esterno_vizier[mask_keep_vizier]

        with np.errstate(invalid='ignore'):
            mask_taglio = tbl_riquadro_esterno_vizier_CLEAN['Mag_sintetica'] < magnitudine_massima
        tbl_vizier_cut = tbl_riquadro_esterno_vizier_CLEAN[mask_taglio].copy()

        centro_run_precedente = centro
        tbl_vizier_cut_precedente = tbl_vizier_cut.copy()
        tbl_hipparco_run_clean_precedente = tbl_hipparco_run_clean.copy()
    else:
        tbl_vizier_cut = tbl_vizier_cut_precedente.copy()
        tbl_hipparco_run_clean = tbl_hipparco_run_clean_precedente.copy()

    tbl_vizier_cut['Mag'] = tbl_vizier_cut['Mag_sintetica']
    tbl_hipparco_run_clean['Mag'] = tbl_hipparco_run_clean['Vmag']

    tbl_catalogate = tabella_catalogo(percorso_file_query, tbl_vizier_cut, tbl_hipparco_run_clean)

    # estraggo le mie coordinate in modo robusto in base alle colonne presenti
    ha_raj2000 = 'RAJ2000' in tbl_catalogate.colnames
    col_ra = 'RAJ2000' if ha_raj2000 else 'RA'
    col_dec = 'DEJ2000' if ha_raj2000 else 'DEC'
    coord_catalogo_cielo = SkyCoord(ra=tbl_catalogate[col_ra], dec=tbl_catalogate[col_dec], unit=u.deg)

    # itero sulle singole occorrenze per creare i riquadri monitorando il progresso con tqdm
    for indice, riga in tqdm(gruppo_label.iterrows(), total=len(gruppo_label), desc="Elaborazione frames", leave=False):
        nome_cartella = riga['nome_cartella']
        nome_file_fits = riga['nome_file_fits']

        # applico la mia logica del giorno prima/dopo anche per tutti i file di ritaglio
        data_corrente = datetime.strptime(str(nome_cartella), "%Y%m%d")
        cart_prec = (data_corrente - timedelta(days=1)).strftime("%Y%m%d")
        cart_succ = (data_corrente + timedelta(days=1)).strftime("%Y%m%d")

        cartelle_possibili = [str(nome_cartella), cart_prec, cart_succ]
        percorso_file_fits = None

        for cartella_target in cartelle_possibili:
            if cartella_target not in cache_cartelle:
                # cerco la mia cartella specificatamente all'interno di ASTRI1
                risultato = cerca_cartella_nel_progetto(cartella_oggetti, cartella_target)
                cache_cartelle[cartella_target] = str(risultato) if risultato else None

            percorso_cartella_cache = cache_cartelle[cartella_target]
            if percorso_cartella_cache is not None:
                percorso_temporaneo = os.path.join(percorso_cartella_cache, nome_file_fits)
                if os.path.exists(percorso_temporaneo):
                    percorso_file_fits = percorso_temporaneo
                    break

        # salto l'elaborazione del singolo frame se il file FITS risulta introvabile ovunque
        if percorso_file_fits is None:
            print(f"ATTENZIONE: File FITS {nome_file_fits} non trovato nelle cartelle adiacenti. Salto il riquadro.")
            continue

        # apro il mio file corrente
        with fits.open(percorso_file_fits) as hdul:
            dati_fits = hdul[0].data - 1
            wcs_img = WCS(hdul[0].header)

        x_centroid = int(round(riga['xcentroid']))
        y_centroid = int(round(riga['ycentroid']))

        # definisco i miei limiti del riquadro 60x60 gestendo i bordi
        x_min = max(0, x_centroid - 30)
        x_max = min(dati_fits.shape[1], x_centroid + 30)
        y_min = max(0, y_centroid - 30)
        y_max = min(dati_fits.shape[0], y_centroid + 30)

        riquadro = dati_fits[y_min:y_max, x_min:x_max]

        # converto le coordinate in pixel per l'immagine specifica
        x_cat, y_cat = wcs_img.world_to_pixel(coord_catalogo_cielo)

        fig, ax = plt.subplots(figsize=(8, 6))

        # uso l'attributo extent per proiettare in modo automatico i pixel originari sugli assi
        im = ax.imshow(riquadro, cmap="grey_r", norm=LogNorm(), interpolation='nearest',
                       origin='lower', extent=[x_min, x_max, y_min, y_max])

        # sovrappongo i punti convertiti colorandoli secondo la magnitudine
        sc = ax.scatter(x_cat, y_cat, c=tbl_catalogate['Mag'], cmap='viridis', s=4, zorder=5)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Magnitudine')

        # converto i 35 arcosecondi della circonferenza in scala di pixel
        scala_pixel_arcsec = proj_plane_pixel_scales(wcs_img)[0] * 3600
        raggio_pixel = 35.0 / scala_pixel_arcsec

        # disegno la mia zona di correlazione
        cerchio = Circle((riga['xcentroid'], riga['ycentroid']), raggio_pixel, edgecolor='red',
                         facecolor='none', lw=1.5, label='correlation region', zorder=10)
        ax.add_patch(cerchio)

        # predispongo il titolo e la configurazione degli assi
        transient_flag = "YES" if riga['segmentazione_trovata'] else "NO"
        titolo_grafico = f"{riga['DATE-OBS']}\n{label}\ntransient found: {transient_flag}"

        ax.set_title(titolo_grafico)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()

        # salvo ed evito accumulo di memoria
        nome_img_salvata = f"{nome_file_fits.replace('.fits', '')}.png"
        percorso_img_salvata = os.path.join(cartella_salvataggio_label, nome_img_salvata)

        plt.savefig(percorso_img_salvata, bbox_inches='tight')
        plt.close(fig)