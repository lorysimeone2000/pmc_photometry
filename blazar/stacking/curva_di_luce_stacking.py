import numpy as np
import os
import sys
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import warnings
from astropy.wcs import FITSFixedWarning
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# importo i moduli per la fotometria, il ritaglio, il calcolo del tempo e la statistica
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.time import Time
from astropy.stats import sigma_clipped_stats

warnings.filterwarnings('ignore', category=FITSFixedWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI
# =============================================================================
def trova_cartella_base(nome_target="Lorenzo"):
    # cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent


# trovo la cartella base del mio progetto
BASE_DIR = trova_cartella_base("Lorenzo")

# =============================================================================
# 1. RACCOLTA DATI (ULTIMI 3 GIORNI, DIVISI PER RUN)
# =============================================================================
print("--- INIZIO CREAZIONE CURVA DI LUCE (ULTIMI 3 GIORNI - PER RUN) ---")
dir_dati = BASE_DIR / "PMC_DATA_BLAZAR"

if not dir_dati.exists():
    print(f"ERRORE: Impossibile trovare la cartella dati {dir_dati}")
    exit()

# isolo tutti i giorni disponibili ordinati e prendo solo gli ultimi 3
giorni_totali = sorted([d for d in dir_dati.iterdir() if d.is_dir()])
ultimi_3_giorni = giorni_totali

print(f"Giorni selezionati per l'analisi: {[d.name for d in ultimi_3_giorni]}")

# creo un dizionario per raggruppare i file per ogni singola run
file_per_run = {}

for giorno_dir in ultimi_3_giorni:
    giorno_nome = giorno_dir.name

    for run_dir in sorted([d for d in giorno_dir.iterdir() if d.is_dir()]):
        run_nome = run_dir.name
        estensioni_valide = ['.fit', '.fits']
        file_run = sorted([str(f) for f in run_dir.rglob('*') if f.suffix.lower() in estensioni_valide and f.is_file()])

        # salto la prima e le ultime due immagini della run per sicurezza
        if len(file_run) > 3:
            # creo un'etichetta univoca "Giorno - Run"
            etichetta_run = f"{giorno_nome}\n{run_nome}"
            file_per_run[etichetta_run] = file_run[1:-2]

if not file_per_run:
    print("ERRORE: Nessuna run valida trovata negli ultimi 3 giorni.")
    exit()

# =============================================================================
# 2. DEFINIZIONE DEL SISTEMA DI RIFERIMENTO E RITAGLIO
# =============================================================================
# prendo la prima immagine della prima run per creare il canvas
primo_file_assoluto = list(file_per_run.values())[0][0]
hdu_ref = fits.open(primo_file_assoluto)[0]
target_header_full = hdu_ref.header.copy()
target_wcs_full = WCS(target_header_full, relax=True)

# imposto le coordinate del blazar Mrk 421
coord_mrk421 = SkyCoord('11h04m27.31s', '+38d12m31.8s', frame='icrs')
dimensione_riquadro = u.Quantity((1.6, 1.6), u.arcmin)

# eseguo il ritaglio
ritaglio_ref = Cutout2D(hdu_ref.data, coord_mrk421, dimensione_riquadro, wcs=target_wcs_full, mode='partial')
target_wcs = ritaglio_ref.wcs
target_shape = ritaglio_ref.shape

# =============================================================================
# 3. STACKING PER SINGOLA RUN E FOTOMETRIA SUL CENTRO
# =============================================================================
tempi_medi_run = []
flussi_medi_run = []

# preparo le coordinate del centro esatto del riquadro (dove si trova il blazar)
centro_x = target_shape[1] / 2.0
centro_y = target_shape[0] / 2.0

# definisco un'apertura circolare di 5 pixel di raggio
apertura_blazar = CircularAperture((centro_x, centro_y), r=5.0)

for etichetta, lista_file in file_per_run.items():
    print(f"\nElaborazione {etichetta.replace(chr(10), ' ')} ({len(lista_file)} immagini)...")

    final_image_sum_run = np.zeros(target_shape)
    coverage_map_run = np.zeros(target_shape)

    # inizializzo una lista per raccogliere i tempi esatti di ogni singola foto della run
    tempi_scatti_jd = []

    for percorso_file_fits in tqdm(lista_file, desc="Stacking", unit="img"):
        try:
            with fits.open(percorso_file_fits) as hdu_list:
                data_sub = hdu_list[0].data
                header = hdu_list[0].header

                # calcolo la mediana sull'intera immagine grezza e la sottraggo per pulire il fondo
                _, mediana, _ = sigma_clipped_stats(data_sub, sigma=3.0)
                data_sub = data_sub - mediana

                # estraggo il tempo di scatto dall'header e lo salvo
                t_obs = Time(header['DATE-OBS'])
                tempi_scatti_jd.append(t_obs.jd)

                wcs_input = WCS(header, relax=True)

                # riproietto l'immagine corrente sul riquadro
                array_reprojected, footprint = reproject_interp(
                    (data_sub, wcs_input),
                    target_wcs,
                    shape_out=target_shape
                )

                final_image_sum_run += np.nan_to_num(array_reprojected, nan=0.0)
                coverage_map_run += np.nan_to_num(footprint, nan=0.0)

        except Exception as e:
            pass

    # calcolo il tempo medio della run
    if tempi_scatti_jd:
        media_jd = np.mean(tempi_scatti_jd)
        tempo_medio = Time(media_jd, format='jd').to_datetime()
        tempi_medi_run.append(tempo_medio)
    else:
        continue  # salto se per qualche motivo non ci sono tempi validi

    # CALCOLO DELLA MEDIA: divido la somma per il numero di coperture pixel per pixel
    with np.errstate(divide='ignore', invalid='ignore'):
        fattore_scala = np.where(coverage_map_run > 0, 1.0 / coverage_map_run, 0.0)

    immagine_run_media = final_image_sum_run * fattore_scala

    # misuro il flusso dentro l'apertura tramite photutils sull'immagine media
    tabella_fotometria = aperture_photometry(immagine_run_media, apertura_blazar)
    flusso_totale_medio = tabella_fotometria['aperture_sum'][0]

    flussi_medi_run.append(flusso_totale_medio)
    print(f"Flusso medio registrato: {flusso_totale_medio:.2f} counts (Ora media: {tempo_medio.strftime('%H:%M:%S')})")

# =============================================================================
# 4. PLOTTING DELLA CURVA DI LUCE PER RUN
# =============================================================================
plt.figure(figsize=(14, 7))

# traccio la curva di luce per singola run usando i tempi medi
plt.plot(tempi_medi_run, flussi_medi_run, marker='o', linestyle='-', color='black', linewidth=1, markersize=0.5)

plt.title("Curva di Luce Markarian 421 (dettaglio per Run)", fontsize=16)
plt.xlabel("Tempo medio di Osservazione (Ora)", fontsize=14)
plt.ylabel("Flusso strumentale medio per singola posa (Counts)", fontsize=14)

# imposto il limite dell'asse y da 0 al valore massimo (aggiungo un piccolo 5% di margine per non tagliare il punto più alto)
plt.ylim(0, max(flussi_medi_run) * 1.05)

# formatto l'asse x per mostrare chiaramente la data e l'ora
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate()

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# salvo il grafico
output_dir = BASE_DIR / 'pmc_photometry' / 'blazar' / 'analisi'
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'curva_di_luce_mrk421_ultimi3giorni_per_run.png', dpi=300)
plt.show()