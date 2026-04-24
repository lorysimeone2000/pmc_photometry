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
percorso_csv = cerca_file_nel_progetto(BASE_DIR, "stelle_tesi_frame.csv")

# estraggo la tabella pandas dal file
df = pd.read_csv(percorso_csv)

# =============================================================================
# 2. ORDINAMENTO TEMPORALE, COMPRESSIONE GAP E PLOT
# =============================================================================

# converto la colonna delle date nel formato datetime di pandas
df['DATE-OBS'] = pd.to_datetime(df['DATE-OBS'])

# ricavo tutti i label unici presenti nel dataframe
labels_presenti = df['label'].unique()

# cerco la mia cartella di destinazione una sola volta prima del ciclo
output_dir = cerca_cartella_nel_progetto(BASE_DIR, 'curve_di_luce_con_tagli')

# creo la mia cartella di output se non esiste
output_dir.mkdir(parents=True, exist_ok=True)

# avvio il ciclo su ogni singolo label per generare il grafico
for label in tqdm(labels_presenti):

    # filtro il dataframe per il label corrente, lo ordino e resetto l'indice
    df_label = df[df['label'] == label].sort_values(by='DATE-OBS').reset_index(drop=True)

    # estraggo il momento esatto della prima osservazione assoluta per questa curva
    t_inizio_assoluto = df_label.loc[0, 'DATE-OBS']

    # inizializzo le liste per costruire il mio asse X fittizio
    asse_x_compresso = [0.0]
    posizioni_etichette = [0.0]

    # la prima etichetta è sempre 0 (giorno iniziale)
    testi_etichette = ['0']
    ultimo_valore_scritto = 0

    posizioni_linee_rosse = []

    # inizializzo il tracciamento della run per separare i plot
    run_ids = [0]
    run_corrente = 0

    # decido quanto deve essere largo visivamente il gap sul grafico (120 unità fittizie)
    spazio_fisso_gap = 120.0

    # ciclo tra i punti per calcolare le distanze e verificare i cambi di run
    for i in range(1, len(df_label)):
        t_prec = df_label.loc[i - 1, 'DATE-OBS']
        t_corr = df_label.loc[i, 'DATE-OBS']

        # estraggo il RUN_ID per il punto precedente e quello corrente
        run_prec = df_label.loc[i - 1, 'RUN_ID']
        run_corr = df_label.loc[i, 'RUN_ID']

        # calcolo i secondi reali trascorsi dal punto precedente
        dt = (t_corr - t_prec).total_seconds()

        # se il RUN_ID cambia, applico la compressione e cambio run
        if run_prec != run_corr:
            run_corrente += 1
            # salvo la posizione centrale per la linea rossa
            posizioni_linee_rosse.append(asse_x_compresso[-1] + spazio_fisso_gap / 2)

            # avanzo nel mio grafico solo dello spazio fisso
            nuovo_x = asse_x_compresso[-1] + spazio_fisso_gap
            asse_x_compresso.append(nuovo_x)

            # registro la posizione del cambio run per il tick
            posizioni_etichette.append(nuovo_x)

            # calcolo i giorni interi trascorsi
            giorni_trascorsi = (t_corr - t_inizio_assoluto).days

            # aggiungo il testo solo se il valore è cambiato rispetto all'ultima etichetta mostrata
            if giorni_trascorsi != ultimo_valore_scritto:
                testi_etichette.append(str(giorni_trascorsi))
                ultimo_valore_scritto = giorni_trascorsi
            else:
                # inserisco una stringa vuota per mantenere il tick senza testo ripetuto
                testi_etichette.append("")
        else:
            # se fa parte della stessa run, mantengo la distanza temporale proporzionale reale
            asse_x_compresso.append(asse_x_compresso[-1] + dt)

        # associo il punto alla run corrente
        run_ids.append(run_corrente)

    # assegno il nuovo asse X e l'ID della run al dataframe
    df_label['x_plot'] = asse_x_compresso
    df_label['run_id'] = run_ids

    mag_vera = df_label['Mag_vera'].iloc[0]

    # creo la figura per il grafico
    plt.figure(figsize=(12, 6))

    plt.axhline(y=mag_vera, color='red', linewidth=1.5, label = 'Catalogued magnitude')

    # itero sulle singole run per disconnettere la linea nei gap temporali
    for run_id, df_run in df_label.groupby('run_id'):
        df_run = df_run.reset_index(drop=True)

        # disegno la banda continua dell'errore
        plt.fill_between(df_run['x_plot'],
                         df_run['Mag_estratta'] - df_run['err_Mag_estratta'],
                         df_run['Mag_estratta'] + df_run['err_Mag_estratta'],
                         color='black', alpha=0.15, edgecolor='none')

        # se c'è solo un punto nella run, lo plotto come marcatore singolo
        if len(df_run) == 1:
            p = df_run.iloc[0]
            plt.plot(p['x_plot'], p['Mag_estratta'], marker='o', color='black', markersize=3)
            continue

        # disegno la linea continua per la run corrente
        plt.plot(df_run['x_plot'], df_run['Mag_estratta'], linestyle='-', color='black', linewidth=0.5)

    contatore_linee = 0

    # disegno le linee rosse tratteggiate
    for pos in posizioni_linee_rosse:
        contatore_linee += 1
        if contatore_linee == 1:
            # uso il rosso per la prima riga per far coincidere il colore con la legenda
            plt.axvline(x=pos, color='red', linestyle='--', alpha=0.6, linewidth=0.5, label='Run change')
        else:
            plt.axvline(x=pos, color='red', linestyle='--', alpha=0.6, linewidth=0.5)

    # aggiungo la legenda se presente modificando il font per adattarlo a LaTeX
    if contatore_linee > 0:
        plt.legend(fontsize=16)

    # inverto l'asse y
    plt.gca().invert_yaxis()

    # applico le etichette dei giorni solo se univoche, rimpicciolendo il font per compensare il resize in LaTeX
    plt.xticks(posizioni_etichette, testi_etichette, rotation=0, fontsize=12)
    plt.yticks(fontsize=14)

    # aggiungo le etichette agli assi in inglese britannico con font leggibili
    plt.xlabel('Days from first observation', fontsize=18)
    plt.ylabel('Extracted magnitude', fontsize=18)

    # ottimizzo la disposizione
    plt.tight_layout()

    # salvo il grafico
    plt.savefig(output_dir / f'curva_con_tagli_{label}.png')

    # chiudo la figura
    plt.close()