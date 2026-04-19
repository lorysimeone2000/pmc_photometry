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

# =============================================================================
# 1. LETTURA DEL FILE ORIGINALE E PREPARAZIONE TARGET
# =============================================================================

# individuo il file CSV originale
percorso_csv = cerca_file_nel_progetto(BASE_DIR, "candidati_frame.csv")

# estraggo la tabella pandas dal file
df = pd.read_csv(percorso_csv)

# =============================================================================
# 2. ORDINAMENTO TEMPORALE, CALCOLO MEDIE RUN E PLOT
# =============================================================================

# converto la colonna delle date nel formato datetime di pandas
df['DATE-OBS'] = pd.to_datetime(df['DATE-OBS'])

# ricavo tutti i label unici presenti nel dataframe
labels_presenti = df['label'].unique()

# cerco la mia cartella di destinazione una sola volta prima del ciclo
cartella = cerca_cartella_nel_progetto(BASE_DIR, 'sorgenti catalogate')

# verifico di aver trovato la cartella per evitare errori, altrimenti uso la cartella base come ripiego
if cartella is not None:
    output_dir = cartella / "curve_di_luce_senza_tagli"
else:
    output_dir = BASE_DIR / "curve_di_luce_senza_tagli"

# creo la mia cartella di output se non esiste
output_dir.mkdir(parents=True, exist_ok=True)

# avvio il ciclo su ogni singolo label per generare il grafico
for label in tqdm(labels_presenti):

    # filtro il dataframe per il label corrente, lo ordino e resetto l'indice
    df_label = df[df['label'] == label].sort_values(by='DATE-OBS').reset_index(drop=True)

    # ricavo il tempo zero basandomi sulla prima immagine della prima run
    t_zero = df_label.loc[0, 'DATE-OBS']

    # calcolo i giorni trascorsi dal tempo zero per ogni frame
    df_label['giorni_da_t0'] = (df_label['DATE-OBS'] - t_zero).dt.total_seconds() / 86400.0

    # creo la figura per il grafico ottimizzando le dimensioni per 0.90\textwidth in LaTeX
    plt.figure(figsize=(16, 6))

    i = 0

    # introduco una variabile per memorizzare il tempo medio della run precedente
    x_medio_precedente = None

    # itero sulle singole run sfruttando il RUN_ID originale
    for run_id, df_run in df_label.groupby('RUN_ID'):

        i += 1

        # calcolo il centro temporale della run in giorni
        x_medio = df_run['giorni_da_t0'].mean()

        # calcolo la magnitudine estratta media per questa run
        mag_media = df_run['Mag_estratta'].mean()

        # calcolo l'errore medio per questa run
        err_medio = df_run['err_Mag_estratta'].mean()

        # controllo se sono passati più di 10 giorni rispetto alla run precedente
        if x_medio_precedente is not None and (x_medio - x_medio_precedente) > 10:
            # evidenzio l'area vuota del grafico con uno sfondo giallo trasparente
            plt.axvspan(x_medio_precedente, x_medio, color='yellow', alpha=0.1)

        # traccio il singolo punto medio con la sua barra di errore
        plt.errorbar(x_medio, mag_media, yerr=err_medio, fmt='o', color='black',
                     markersize=2, capsize=1, ecolor='darkgray', linewidth=1.5, label='Run average magnitude')

        # ridimensiono la legenda per renderla proporzionata e visibile su LaTeX
        if i == 1: plt.legend(fontsize=24)

        # aggiorno il tempo medio precedente per l'iterazione successiva
        x_medio_precedente = x_medio

    # inverto l'asse y una sola volta per tutto il grafico fuori dal ciclo
    plt.gca().invert_yaxis()

    # ridimensiono i tick degli assi per renderli leggibili
    plt.tick_params(axis='both', which='major', labelsize=24)

    # aggiungo le etichette agli assi, formattate per visibilità su A4 a 0.90\textwidth
    plt.xlabel('Days from first observation', fontsize=28)
    plt.ylabel('Extracted magnitude', fontsize=28)

    # ottimizzo la disposizione degli elementi nel grafico PRIMA di salvare l'immagine
    plt.tight_layout()

    # salvo il grafico nella mia cartella definita prima del ciclo con parametri per l'alta risoluzione
    plt.savefig(output_dir / f'curva_{label}.png', dpi=300, bbox_inches='tight')

    # chiudo la figura per mantenere pulita la memoria durante le iterazioni
    plt.close()