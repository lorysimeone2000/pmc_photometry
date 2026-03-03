import numpy as np
import os
import sys
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales  # aggiunto per calcolare la scala dei pixel
import warnings
from astropy.wcs import FITSFixedWarning
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# importo i moduli per la fotometria, la statistica e il calcolo del tempo
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
# 1. RACCOLTA DATI (ULTIMI 3 GIORNI)
# =============================================================================
print("--- INIZIO CREAZIONE CURVA DI LUCE AD ALTA RISOLUZIONE (SINGOLE IMMAGINI) ---")
dir_dati = BASE_DIR / "PMC_DATA_BLAZAR"

if not dir_dati.exists():
    print(f"ERRORE: Impossibile trovare la cartella dati {dir_dati}")
    exit()

# isolo tutti i giorni disponibili ordinati e prendo solo gli ultimi 3
giorni_totali = sorted([d for d in dir_dati.iterdir() if d.is_dir()])
ultimi_3_giorni = giorni_totali[-3:]

print(f"Giorni selezionati per l'analisi: {[d.name for d in ultimi_3_giorni]}")

lista_file_validi = []

# esploro le cartelle per raccogliere tutti i file
for giorno_dir in ultimi_3_giorni:
    for run_dir in sorted([d for d in giorno_dir.iterdir() if d.is_dir()]):
        estensioni_valide = ['.fit', '.fits']
        file_run = sorted([str(f) for f in run_dir.rglob('*') if f.suffix.lower() in estensioni_valide and f.is_file()])

        # salto la prima e le ultime due immagini della run per sicurezza
        if len(file_run) > 3:
            lista_file_validi.extend(file_run[1:-2])

if not lista_file_validi:
    print("ERRORE: Nessuna immagine valida trovata negli ultimi 3 giorni.")
    exit()

# =============================================================================
# 2. FOTOMETRIA SU OGNI SINGOLA IMMAGINE CON SOTTRAZIONE DEL FONDO
# =============================================================================
# imposto le coordinate fisse del blazar Mrk 421
coord_mrk421 = SkyCoord('11h04m27.31s', '+38d12m31.8s', frame='icrs')

tempi_osservazione = []
flussi_singoli = []

print(f"\nEstrazione dei flussi da {len(lista_file_validi)} immagini in corso...")

for percorso_file in tqdm(lista_file_validi, desc="Fotometria", unit="img"):
    try:
        with fits.open(percorso_file) as hdu_list:
            data = hdu_list[0].data
            header = hdu_list[0].header

            # estraggo il tempo di osservazione dall'header
            t_obs = Time(header['DATE-OBS'])

            # leggo il wcs dell'immagine corrente
            wcs = WCS(header, relax=True)

            # calcolo la dimensione di un pixel in gradi
            scala_pixel_gradi = np.mean(proj_plane_pixel_scales(wcs))

            # imposto un diametro di 1.2 arcmin, quindi il raggio è 0.6 arcmin (convertito in gradi)
            raggio_gradi = 0.6 / 60.0
            raggio_pixel = raggio_gradi / scala_pixel_gradi

            # converto le coordinate celesti del blazar nei pixel di QUESTA specifica immagine
            x_pix, y_pix = wcs.world_to_pixel(coord_mrk421)

            # verifico che il blazar sia caduto all'interno del sensore in questo scatto
            ny, nx = data.shape
            if 0 <= x_pix < nx and 0 <= y_pix < ny:
                # ricavo la mediana dell'immagine usando sigma-clip e la sottraggo ai dati
                _, mediana, _ = sigma_clipped_stats(data, sigma=3.0)
                data_sottratta = data - mediana

                # definisco l'apertura dinamica in pixel e calcolo il flusso effettivo sui dati puliti dal fondo
                apertura = CircularAperture((x_pix, y_pix), r=raggio_pixel)
                tabella_fotometria = aperture_photometry(data_sottratta, apertura)
                flusso = tabella_fotometria['aperture_sum'][0]

                tempi_osservazione.append(t_obs)
                flussi_singoli.append(flusso)

    except Exception as e:
        # ignoro i file corrotti o in cui mancano dati
        pass

# =============================================================================
# 3. ELABORAZIONE DEL TEMPO E PLOTTING
# =============================================================================
# ordino cronologicamente i dati
dati_ordinati = sorted(zip(tempi_osservazione, flussi_singoli), key=lambda x: x[0].jd)
tempi_ordinati, flussi_ordinati = zip(*dati_ordinati)

# calcolo i minuti trascorsi prendendo come zero la primissima immagine
t0 = tempi_ordinati[0]
minuti_trascorsi = [(t - t0).to_value('jd') * 1440.0 for t in tempi_ordinati]  # 1 giorno = 1440 minuti

plt.figure(figsize=(14, 7))

# traccio i punti della curva di luce
plt.plot(minuti_trascorsi, flussi_ordinati, marker='.', linestyle='-', markersize=3, alpha=0.6)

plt.title("Curva di Luce Markarian 421 (Alta Risoluzione - Ultimi 3 giorni)", fontsize=16)
plt.xlabel("Tempo trascorso dalla prima osservazione (Minuti)", fontsize=14)
plt.ylabel("Flusso strumentale netto (Counts)", fontsize=14)

# imposto il limite dell'asse y da 0 al valore massimo con un margine del 5%
plt.ylim(0, max(flussi_ordinati) * 1.05)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# salvo il grafico
output_dir = BASE_DIR / 'pmc_photometry' / 'blazar' / 'analisi'
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'curva_di_luce_mrk421_singole_immagini_minuti.png', dpi=300)
plt.show()

# =============================================================================
# 4. SALVATAGGIO VALORI IN CSV
# =============================================================================
# creo un DataFrame con i dati dell'asse X e dell'asse Y
df_dati = pd.DataFrame({
    'Tempo_trascorso_minuti': minuti_trascorsi,
    'Flusso_netto': flussi_ordinati
})

# individuo la cartella esatta in cui si trova questo script
cartella_script = Path(__file__).resolve().parent

# salvo il file csv al suo interno
percorso_csv = cartella_script / 'curva_di_luce_mrk421_valori.csv'
df_dati.to_csv(percorso_csv, index=False)

print(f"Dati salvati con successo nel file: {percorso_csv.name}")