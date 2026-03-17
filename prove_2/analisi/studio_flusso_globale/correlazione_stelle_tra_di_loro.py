import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.table import Table
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning

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

# --- PARAMETRI CONFIGURAZIONE ---

run_list = [1, 2, 3]  # definisco la lista delle run da analizzare
base_path = BASE_DIR / "tabelle/tabelle_unite"

# definisco i flussi base da analizzare
flussi_base = [
    'kron_manuale_aper',
    'kron_manuale_seg',
    'somma_apertura_ultimo_pixel',
    'flusso_fisso_max_run',
    'flusso_raggio_fisso_doppio',
    'flusso_intera_segmentazione',
    'flusso_kron_intera_segmentazione'
]

# --- FASE 1: RACCOLTA DI TUTTI I DATI ---

print(f"--- FASE 1: Lettura dei dati dalle Run {run_list} ---")

lista_df = []
totale_immagini = 0

for run in run_list:
    cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

    if not os.path.exists(cartella_csv):
        print(f"ATTENZIONE: La cartella {cartella_csv} non esiste. Salto questa run.")
        continue

    file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])

    if not file_csv:
        print(f"Nessun file CSV trovato in Run {run}.")
        continue

    for n, f in enumerate(file_csv):
        percorso_file = os.path.join(cartella_csv, f)
        try:
            df_temp = pd.read_csv(percorso_file, comment='#')

            # filtro subito solo le stelle catalogate ('SI...') per garantire ID stabili
            if 'Corrispondenza' in df_temp.columns:
                mask_si = df_temp['Corrispondenza'].astype(str).str.startswith('SI')
                df_cat = df_temp[mask_si].copy()

                # creo un indice temporale unico per ogni immagine elaborata
                df_cat['time_idx'] = f"R{run}_{n:03d}"

                lista_df.append(df_cat)
            totale_immagini += 1
        except Exception as e:
            print(f"Errore nella lettura del file {f}: {e}")

print(f"Elaborate {totale_immagini} immagini totali.")

# unisco tutti i dataframe in uno solo grande
big_df = pd.concat(lista_df, ignore_index=True)

# --- FASE 2: SELEZIONE DELLE STELLE STABILI ---

# conto in quante immagini compare ogni ID
conteggi_stelle = big_df['ID'].value_counts()

# imposto una soglia di presenza: tengo solo le stelle che compaiono in almeno l'80% dei frame
soglia_presenza = int(0.8 * totale_immagini)
stelle_valide = conteggi_stelle[conteggi_stelle >= soglia_presenza].index

print(f"--- FASE 2: Filtro Stelle Stabili ---")
print(f"Trovate {len(stelle_valide)} stelle presenti in almeno l'80% delle immagini ({soglia_presenza} frame minimi).")

big_df_valido = big_df[big_df['ID'].isin(stelle_valide)].copy()

# rimuovo eventuali duplicati dello stesso ID all'interno della stessa immagine
# causati da oggetti molto vicini matchati alla stessa stella di catalogo
big_df_valido = big_df_valido.drop_duplicates(subset=['time_idx', 'ID'], keep='first')

# --- FASE 3: STUDIO DELLE CROSS-CORRELAZIONI ---

print(f"\n--- FASE 3: Calcolo delle correlazioni Pairwise per i flussi ---")

for flusso in flussi_base:
    if flusso not in big_df_valido.columns:
        print(f"Attenzione: Colonna {flusso} non trovata nei dati. Salto.")
        continue

    # mi assicuro che la colonna sia numerica
    big_df_valido[flusso] = pd.to_numeric(big_df_valido[flusso], errors='coerce')

    # creo una Pivot Table: righe = istante temporale, colonne = ID stella, valori = flusso
    df_pivot = big_df_valido.pivot(index='time_idx', columns='ID', values=flusso)

    # calcolo la matrice di correlazione di Pearson per tutte le combinazioni di stelle
    matrice_corr = df_pivot.corr(method='pearson')

    # estraggo solo i valori del triangolo superiore della matrice escludendo la diagonale (correlazione 1 di ogni stella con se stessa)
    triangolo_sup = matrice_corr.where(np.triu(np.ones(matrice_corr.shape), k=1).astype(bool))

    # appiattisco in un array 1D e rimuovo i NaN
    valori_correlazione = triangolo_sup.stack().dropna().values

    if len(valori_correlazione) == 0:
        print(f"Dati insufficienti per calcolare la correlazione per {flusso}")
        continue

    media_corr = np.mean(valori_correlazione)
    mediana_corr = np.median(valori_correlazione)

    print(f"-> {flusso}:")
    print(f"   Coppie di stelle confrontate: {len(valori_correlazione)}")
    print(f"   Correlazione Media: {media_corr:.4f}")

    # --- CREAZIONE DEL GRAFICO ---
    plt.figure(figsize=(9, 6))

    # creo un istogramma per visualizzare la distribuzione delle correlazioni
    n, bins, patches = plt.hist(valori_correlazione, bins=50, color='steelblue', edgecolor='black', alpha=0.8)

    # aggiungo linee verticali per la media e lo zero per riferimento
    plt.axvline(media_corr, color='red', linestyle='dashed', linewidth=2, label=f'Media: {media_corr:.2f}')
    plt.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Zero (nessuna correlazione)')

    # formatto il grafico
    plt.title(f"Distribuzione Correlazione Incrociata tra Stelle\n[{flusso}]", fontsize=14)
    plt.xlabel("Coefficiente di Correlazione di Pearson (r)", fontsize=12)
    plt.ylabel("Numero di coppie stellari", fontsize=12)
    plt.xlim(-1.1, 1.1)

    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # salvo il grafico
    nome_out = f"distribuzione_correlazioni_{flusso}.jpg"
    plt.savefig(nome_out, dpi=300)
    plt.show()

print("\nTutti i grafici delle correlazioni incrociate sono stati generati e salvati.")