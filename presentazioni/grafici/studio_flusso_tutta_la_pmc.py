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

                # sommo tutti i pixel presenti nella mia immagine
                somma_totale = np.sum(data)

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

    # =============================================================================
    # CREAZIONE E SALVATAGGIO DEL GRAFICO
    # =============================================================================
    print("\nGenerazione del grafico...")

    # imposto dimensioni adatte a un testo A4 con width=0.85\textwidth
    plt.figure(figsize=(8.5, 5))

    # imposto il colore blu come unico elemento della lista per uniformare il grafico al file di riferimento
    colori = ['blue']

    # calcolo i limiti temporali di ciascuna run per poter disegnare le linee divisorie
    run_boundaries = []

    # aggiungo una variabile di controllo per inserire l'etichetta una sola volta nella legenda
    etichetta_aggiunta = False

    for idx, run in enumerate(RUN):
        if run in tempi_assoluti and tempi_assoluti[run]:
            # calcolo i minuti trascorsi dal mio primissimo scatto assoluto
            minuti_trascorsi = [(t - tempo_zero_globale).sec / 60.0 for t in tempi_assoluti[run]]

            # trovo il tempo massimo di questa run per tracciare il confine
            t_end = max(minuti_trascorsi)
            run_boundaries.append((run, t_end))

            colore = colori[0]

            # imposto l'etichetta solo se non l'ho già aggiunta in precedenza
            etichetta_legenda = "ADU value per pixel" if not etichetta_aggiunta else ""

            # uso i miei nuovi dati calcolati per creare la curva della run
            plt.plot(minuti_trascorsi, somme_pixel[run], marker='o', markersize=1,
                     linestyle='-', linewidth=.5, color=colore, label=etichetta_legenda)

            # aggiorno la variabile per non ripetere l'etichetta nei cicli successivi
            etichetta_aggiunta = True

    # disegno le linee divisorie per evidenziare le run e aggiungo i testi dimensionati
    for r_idx, (run_num, t_end) in enumerate(run_boundaries):
        plt.axvline(x=t_end, color='gray', linestyle='--', alpha=0.6)
        # aggiungo il testo richiesto accanto alle linee divisorie
        plt.text(t_end, 0.95, f'End of run {run_num}', color='gray', ha='right', va='top', rotation=90,
                 transform=plt.gca().get_xaxis_transform(), fontsize=12)

    # applico la mia formattazione grafica personalizzata con font adatti alla stampa
    plt.xlabel("Time from the start of run 1 (minutes)", fontsize=14)
    plt.ylabel("Total pixel sum (ADU)", fontsize=14)

    # dimensiono i valori numerici sugli assi
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # ottimizzo i margini per la figura
    plt.tight_layout()

    file_grafico = "andamento_pixel_totali_run_di_prova.png"

    # salvo assicurandomi che non vengano tagliate le etichette
    plt.savefig(file_grafico, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato con successo: {file_grafico}")
    # plt.show()