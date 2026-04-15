import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def trova_cartella_base(nome_target="Lorenzo"):
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

if percorso_csv is None:
    print("Errore: file 'candidati_frame.csv' non trovato.")
    sys.exit()

# estraggo la tabella pandas dal file
df = pd.read_csv(percorso_csv)

# =============================================================================
# 2. ORDINAMENTO TEMPORALE E PLOT
# =============================================================================

# converto la colonna delle date nel formato datetime di pandas
df['DATE-OBS'] = pd.to_datetime(df['DATE-OBS'])

# ricavo tutti i label unici presenti nel dataframe
labels_presenti = df['label'].unique()

# cerco la mia cartella di destinazione una sola volta prima del ciclo
cartella = cerca_cartella_nel_progetto(BASE_DIR, 'presenza_consecutiva_multirun')

# verifico di aver trovato la cartella per evitare errori, altrimenti uso la cartella base come ripiego
if cartella is not None:
    output_dir = cartella / "curve_di_luce"
else:
    output_dir = BASE_DIR / "curve_di_luce"

# creo la mia cartella di output se non esiste
output_dir.mkdir(parents=True, exist_ok=True)

# avvio il ciclo su ogni singolo label per generare il grafico
for label in tqdm(labels_presenti, desc="Generazione Curve di Luce"):

    # filtro il dataframe per il label corrente, lo ordino e resetto l'indice
    df_label = df[df['label'] == label].sort_values(by='DATE-OBS').reset_index(drop=True)

    # calcolo la differenza temporale tra i punti in secondi
    df_label['dt'] = df_label['DATE-OBS'].diff().dt.total_seconds()

    # identifico un cambio di run quando il divario temporale supera i 300 secondi (5 minuti)
    df_label['is_new_run'] = df_label['dt'] > 300
    df_label['run_id'] = df_label['is_new_run'].cumsum()

    # creo la figura per il grafico
    plt.figure(figsize=(12, 6))

    # ciclo sulle singole run identificate per lasciare lo spazio vuoto tra di esse
    for run_id, run_df in df_label.groupby('run_id'):
        run_df = run_df.reset_index(drop=True)

        # disegno la banda dell'errore isolata per questa run
        plt.fill_between(run_df['DATE-OBS'],
                         run_df['Mag_estratta'] - run_df['err_Mag_estratta'],
                         run_df['Mag_estratta'] + run_df['err_Mag_estratta'],
                         color='black', alpha=0.15, edgecolor='none')

        # se la run ha un solo punto, lo stampo direttamente per non perderlo visivamente
        if len(run_df) == 1:
            colore_punto = 'darkgray' if not run_df.loc[0, 'segmentazione_trovata'] else 'black'
            plt.plot(run_df['DATE-OBS'], run_df['Mag_estratta'], marker='o', color=colore_punto, markersize=3)
            continue

        # --- Calcolo soglia per buchi infra-run ---
        # Calcoliamo il tempo di esposizione tipico della run (mediana dei dt validi)
        # Escludiamo il primo punto che ha dt NaN
        dt_validi = run_df['dt'].dropna()
        if not dt_validi.empty:
            esposizione_tipica = dt_validi.median()
            # Definiamo un buco se il salto è > 1.5 volte l'esposizione tipica
            soglia_buco = esposizione_tipica * 1.5
        else:
            soglia_buco = np.inf  # Nessun buco rilevabile se c'è un solo dt

        # ciclo sui singoli segmenti per applicare lo stile appropriato
        for i in range(len(run_df) - 1):
            t1, t2 = run_df.loc[i, 'DATE-OBS'], run_df.loc[i + 1, 'DATE-OBS']
            y1, y2 = run_df.loc[i, 'Mag_estratta'], run_df.loc[i + 1, 'Mag_estratta']

            f1 = run_df.loc[i, 'segmentazione_trovata']
            f2 = run_df.loc[i + 1, 'segmentazione_trovata']

            # Recuperiamo il dt del punto successivo (corrisponde all'intervallo t1-t2)
            dt_segmento = run_df.loc[i + 1, 'dt']

            # --- Identificazione e Plot dei buchi Infrarun ---
            # Se l'intervallo temporale supera la soglia definita per la run
            if dt_segmento > soglia_buco:
                # Disegniamo una riga verticale ciano, sottile, a metà dell'intervallo
                t_buco = t1 + (t2 - t1) / 2
                plt.axvline(x=t_buco, color='cyan', linestyle='-', linewidth=0.5, alpha=0.8)

            # --- Plot dei segmenti di magnitudine ---
            # se il segmento coinvolge un punto con segmentazione_trovata == False, lo disegno tratteggiato e grigio
            if not f1 or not f2:
                plt.plot([t1, t2], [y1, y2], color='darkgray', linestyle='--', linewidth=1)
            else:
                plt.plot([t1, t2], [y1, y2], color='black', linestyle='-', linewidth=1)

    # formato le date in asse x nel formato richiesto
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S'))
    plt.xticks(rotation=45, fontsize=10)

    # aggiungo i titoli e le etichette agli assi
    plt.title(f'Magnitude trend over time for object {label}', fontsize=16, pad=15)
    plt.xlabel('Observation date', fontsize=14)
    plt.ylabel('Extracted magnitude', fontsize=14)

    # ottimizzo la disposizione degli elementi nel grafico PRIMA di salvare l'immagine
    plt.tight_layout()

    # salvo il grafico nella mia cartella definita prima del ciclo
    plt.savefig(output_dir / f'curva_{label}.png')

    # chiudo la figura per mantenere pulita la memoria durante le iterazioni
    plt.close()

print(f"Elaborazione completata. Grafici salvati in: {output_dir}")