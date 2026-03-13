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
from scipy.ndimage import label
from astropy.time import Time

# eseguo l'import fondamentale per la mia portabilità
from pathlib import Path

# gestisco i warning ignorandoli
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


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

# imposto le coordinate del mio telescopio usando la mia funzione importata
lat_oss, lon_oss, alt_oss = ottieni_coordinate_telescopio('ASTRI 1', BASE_DIR)

# definisco le mie run da analizzare
RUN = [1, 2, 3]

# uso il mirror di Harvard per aggirare i blocchi IP del server principale francese
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1,
)

next_internal_id = 1

if __name__ == "__main__":

    # inizializzo i dizionari per salvare i dati temporali e i flussi
    tempi_assoluti = {}
    somme_pixel = {}
    tempo_zero_globale = None

    nome_params = 'parametri_image_segmentation.txt'
    file_parametri = cerca_file_nel_progetto(BASE_DIR, nome_params)
    if file_parametri is None:
        print("File dei parametri non trovato")
        exit()
    parametri_caricati = leggi_file_parametri(file_parametri)

    # definisco i parametri soliti per la segmentazione
    fwhm_seg = parametri_caricati['fwhm']
    size_seg = parametri_caricati['size']
    threshold_seg = parametri_caricati['threshold_assoluta']
    pixel_n_seg = parametri_caricati['pixel']

    # creo il kernel una sola volta fuori dal ciclo per ottimizzare
    kernel_seg = make_2dgaussian_kernel(fwhm_seg, size=size_seg)
    finder_seg = SourceFinder(npixels=pixel_n_seg, progress_bar=False)

    # inizio il ciclo per ogni mia run
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
        for ext in estensioni_valide:
            file_list.extend(run_folder.glob(ext))

        file_list = sorted([str(f) for f in file_list])
        if not file_list:
            print(f"Nessun FITS in Run {run}, salto.")
            continue

        # creo le liste temporanee per la run corrente
        tempi_run = []
        somme_run = []

        for n, percorso_file in enumerate(tqdm(file_list, desc=f"Analisi pixel Run {run}"), 1):

            # apro il mio file fits
            with fits.open(percorso_file) as hdu_list:
                header = hdu_list[0].header
                # converto i dati a float64 per evitare errori di overflow durante la somma totale
                data_sub, median_bg, _ = elabora_file_fits(percorso_file)
                data = data_sub

                # convolvo i dati per la segmentazione
                convolved_data = convolve(data, kernel_seg)

                # trovo le sorgenti con i parametri soliti
                segment_map = finder_seg(convolved_data, threshold_seg)

                # creo la mia maschera impostando tutto a True inizialmente
                mask_sfondo = np.ones(data.shape, dtype=bool)

                if segment_map is not None:
                    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)

                    # analizzo ogni sorgente trovata
                    for prop in cat:
                        xc, yc = prop.xcentroid, prop.ycentroid
                        slices = prop.slices

                        # estraggo i pixel associati alla sorgente per trovarne la larghezza massima
                        y_idx, x_idx = np.indices(segment_map.data[slices].shape)
                        mask_label = segment_map.data[slices] == prop.label
                        ypix = y_idx[mask_label] + slices[0].start
                        xpix = x_idx[mask_label] + slices[1].start

                        # calcolo le distanze dal centroide
                        distanze_pix = np.hypot(xpix - xc, ypix - yc)
                        r_max_pix = max(np.max(distanze_pix) if len(distanze_pix) > 0 else 0.5, 0.5)

                        # raddoppio l'apertura rispetto alla larghezza massima
                        r_apertura = 2.0 * r_max_pix
                        r_int = int(np.ceil(r_apertura))

                        # definisco i limiti per mascherare solo l'area di interesse
                        y_min = max(0, int(yc - r_int))
                        y_max = min(data.shape[0], int(yc + r_int + 1))
                        x_min = max(0, int(xc - r_int))
                        x_max = min(data.shape[1], int(xc + r_int + 1))

                        # creo la griglia locale e calcolo le distanze
                        y_g_box, x_g_box = np.ogrid[y_min:y_max, x_min:x_max]
                        dist_box = np.hypot(x_g_box - xc, y_g_box - yc)

                        # escludo i pixel che cadono dentro l'apertura calcolata
                        mask_sfondo[y_min:y_max, x_min:x_max][dist_box <= r_apertura] = False

                # sommo tutti i pixel rimasti fuori dalle aperture
                somma_parziale = np.sum(data[mask_sfondo])

                # conto il numero di pixel usati per la somma e calcolo il rapporto sul totale
                pixel_contati = np.sum(mask_sfondo)
                pixel_totali = data.size
                rapporto_pixel = pixel_contati / pixel_totali

                # divido la somma per il rapporto calcolato
                somma_totale = somma_parziale / rapporto_pixel

                # recupero l'orario di scatto e lo converto in un formato analizzabile
                tempo_scatto = Time(header['DATE-OBS'], format='isot', scale='utc')

                # imposto il mio tempo zero al primo scatto assoluto trovato
                if tempo_zero_globale is None:
                    tempo_zero_globale = tempo_scatto

                tempi_run.append(tempo_scatto)
                somme_run.append(somma_totale)

        # salvo i risultati finali della mia run
        tempi_assoluti[run] = tempi_run
        somme_pixel[run] = somme_run

    # preparo e salvo i dati estratti nel file csv richiesto
    dati_per_csv = []
    for run in RUN:
        if run in tempi_assoluti and tempi_assoluti[run]:
            # ricavo il numero dell'immagine partendo da 1 per ogni run usando enumerate
            for im_num, (t, s) in enumerate(zip(tempi_assoluti[run], somme_pixel[run]), start=1):
                dati_per_csv.append({'Run': run, 'im': im_num, 'Tempo_UTC': t.isot, 'Somma_Pixel_Esterni': s})

    df_csv = pd.DataFrame(dati_per_csv)
    nome_file_csv = "risultati_somma_pixel.csv"
    df_csv.to_csv(nome_file_csv, index=False)
    print(f"\nDati salvati correttamente su: {nome_file_csv}")

    # =============================================================================
    # CREAZIONE E SALVATAGGIO DEL GRAFICO
    # =============================================================================
    print("Generazione del grafico...")
    plt.figure(figsize=(12, 6))

    colori = ['red', 'green', 'blue', 'orange', 'purple']

    for idx, run in enumerate(RUN):
        if run in tempi_assoluti and tempi_assoluti[run]:
            # calcolo i minuti trascorsi dal mio primissimo scatto assoluto
            minuti_trascorsi = [(t - tempo_zero_globale).sec / 60.0 for t in tempi_assoluti[run]]

            colore = colori[idx % len(colori)]
            plt.plot(minuti_trascorsi, somme_pixel[run], marker='o', markersize=3,
                     linestyle='-', linewidth=1.5, color=colore, label=f'Run {run}')

    # applico la mia formattazione grafica personalizzata
    if tempo_zero_globale:
        plt.xlabel(f"Tempo in minuti (T0 = {tempo_zero_globale.isot})", fontsize=12)
    else:
        plt.xlabel("Tempo in minuti", fontsize=12)

    plt.ylabel("Somma Totale dei Pixel non stellari", fontsize=12)
    plt.title("Variazione del conteggio totale dei pixel non stellari nel tempo sulle run di prova, normalizzati sul totale", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()

    file_grafico = "andamento_pixel_totalinon_stellari_run_di_prova.png"

    plt.savefig(file_grafico, dpi=300)
    print(f"Grafico salvato con successo: {file_grafico}")
    plt.show()