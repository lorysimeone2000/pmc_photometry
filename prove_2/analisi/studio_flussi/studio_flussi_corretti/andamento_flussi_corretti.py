import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from astropy.table import Table
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning

# Gestisco i warning ignorandoli
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

KRON_TARGET = 300

# Indice del file nella lista da usare come riferimento per trovare l'ID della stella
INDICE_IMMAGINE_RIFERIMENTO = 35
INDICE_RUN_DI_RIFERIMENTO = 1

cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{INDICE_RUN_DI_RIFERIMENTO}")

# Verifica esistenza cartella
if not os.path.exists(cartella_csv):
    print(f"Errore: La cartella {cartella_csv} non esiste.")
    exit()

# Lista file ordinata
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

if not lista_percorsi_csv:
    print("Nessun file CSV trovato.")
    exit()

# --- FASE 1: IDENTIFICAZIONE STELLA TARGET ---

print(f"--- FASE 1: Ricerca stella con Kron ~ {KRON_TARGET} nel file #{INDICE_IMMAGINE_RIFERIMENTO} ---")

# Gestione indice fuori range
if INDICE_IMMAGINE_RIFERIMENTO >= len(lista_percorsi_csv):
    INDICE_IMMAGINE_RIFERIMENTO = 0
    print("Indice riferimento fuori range, uso il primo file.")

path_ref = lista_percorsi_csv[INDICE_IMMAGINE_RIFERIMENTO]
df_ref = pd.read_csv(path_ref, comment='#')
tbl_ref = Table.from_pandas(df_ref)

# Filtriamo solo le stelle che hanno una corrispondenza nel catalogo ('SI...')
# Convertiamo in stringa per sicurezza prima di fare startswith

mask_si = np.char.startswith(tbl_ref['Corrispondenza'].astype(str), 'SI')
tbl_catalogate_ref = tbl_ref[mask_si]

if len(tbl_catalogate_ref) == 0:
    print("Nessuna stella catalogata trovata nel file di riferimento.")
    exit()

# Calcola la differenza assoluta tra i flussi trovati e il target

differenze = np.abs(tbl_catalogate_ref['kron_flux'] - KRON_TARGET)

# Trova l'indice della differenza minima

idx_min = np.argmin(differenze)
stella_ref = tbl_catalogate_ref[idx_min]

# Salva l'ID univoco da cercare negli altri file

id_stella_target = stella_ref['ID']

print(f"--- ANALISI MULTI-RUN (Run: {run_list}) ---")
print(f"Target ID: {id_stella_target}")

# --- STRUTTURE DATI GLOBALI ---

# definisco i flussi base da analizzare estraendoli dalla lista delle colonne fornite
flussi_base = [
    'kron_manuale_aper',
    'kron_manuale_seg',
    'somma_apertura_ultimo_pixel',
    'flusso_fisso_max_run',
    'flusso_raggio_fisso_doppio',
    'flusso_intera_segmentazione',
    'flusso_kron_intera_segmentazione'
]

# accumulo tutti i dati qui per fare i plot iterativi
all_data = {}
for base in flussi_base:
    all_data[base] = []
    all_data[base + '_CORRETTO_Normalizzazione_Moltiplicativa'] = []
    all_data[base + '_CORRETTO_Correzione_Additiva_dell_Apertura'] = []

all_times = []

# variabile per il tempo iniziale globale (t=0 alla prima immagine della prima run)
t0_global = None

# colori per distinguere le run nel grafico (per le bande verticali)
run_boundaries = []

# --- CICLO SULLE RUN ---

for run in run_list:
    print(f"\n>>> Elaborazione RUN {run}...")

    cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

    # verifico esistenza cartella
    if not os.path.exists(cartella_csv):
        print(f"ATTENZIONE: La cartella {cartella_csv} non esiste. Salto questa run.")
        continue

    # creo lista file ordinata
    file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
    lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

    if not lista_percorsi_csv:
        print(f"Nessun file CSV trovato in Run {run}.")
        continue

    # --- ESTRAZIONE DATI FRAME PER FRAME ---

    start_idx = len(all_times)  # salvo l'indice di inizio di questa run nei dati globali

    for n, percorso_csv in enumerate(lista_percorsi_csv):
        try:
            # leggo i dati
            dataframe = pd.read_csv(percorso_csv, comment='#')
            header_dal_csv = leggi_header_da_csv(percorso_csv)
            tbl_frame = Table.from_pandas(dataframe)

            # gestisco il tempo per renderlo continuo tra le run
            t_curr = header_dal_csv.get('TSTART', 0)

            if t0_global is None:
                t0_global = t_curr  # imposto il tempo zero assoluto

            # calcolo il tempo relativo in secondi
            tempo_relativo = (t_curr - t0_global) / 1000.0 if t0_global is not None else 0

            # cerco la stella target
            mask_target = tbl_frame['ID'] == id_stella_target
            stella_nel_frame = tbl_frame[mask_target]

            if len(stella_nel_frame) > 0:
                # se trovo la stella, ciclo sui flussi base per estrarre i dati
                for base in flussi_base:
                    if base in stella_nel_frame.colnames:
                        all_data[base].append(stella_nel_frame[base][0])
                    else:
                        all_data[base].append(np.nan)

                    col_molt = base + '_CORRETTO_Normalizzazione_Moltiplicativa'
                    if col_molt in stella_nel_frame.colnames:
                        all_data[col_molt].append(stella_nel_frame[col_molt][0])
                    else:
                        all_data[col_molt].append(np.nan)

                    col_add = base + '_CORRETTO_Correzione_Additiva_dell_Apertura'
                    if col_add in stella_nel_frame.colnames:
                        all_data[col_add].append(stella_nel_frame[col_add][0])
                    else:
                        all_data[col_add].append(np.nan)
            else:
                # inserisco nan per mantenere l'allineamento temporale se non trovo la stella
                for base in flussi_base:
                    all_data[base].append(np.nan)
                    all_data[base + '_CORRETTO_Normalizzazione_Moltiplicativa'].append(np.nan)
                    all_data[base + '_CORRETTO_Correzione_Additiva_dell_Apertura'].append(np.nan)

            all_times.append(tempo_relativo)

        except Exception as e:
            print(f"Errore nel file {os.path.basename(percorso_csv)}: {e}")
            pass

        # feedback di caricamento
        if n == len(lista_percorsi_csv) - 1:
            print(f"  Elaborati {len(lista_percorsi_csv)} file...")

    # memorizzo dove finisce questa run per disegnarla graficamente
    if len(all_times) > start_idx:
        end_time = all_times[-1]
        run_boundaries.append((run, end_time))

# --- CONVERSIONE E GRAFICI ITERATIVI ---

times_arr = np.array(all_times)

def calc_stats(arr, mask):
    """Calcolo le statistiche escludendo i nan e gli zeri."""
    if np.sum(mask) > 0:
        vals = arr[mask]
        return np.mean(vals), np.std(vals)
    return 0.0, 0.0

print("\n=== GENERAZIONE GRAFICI ===")

# ciclo per iterare su tutti i flussi base e generare un grafico per ognuno
for base in flussi_base:
    arr_base = np.array(all_data[base])
    col_molt = base + '_CORRETTO_Normalizzazione_Moltiplicativa'
    col_add = base + '_CORRETTO_Correzione_Additiva_dell_Apertura'
    arr_molt = np.array(all_data[col_molt])
    arr_add = np.array(all_data[col_add])

    # filtro per le statistiche (escludo gli zeri e i nan)
    mask_base = (arr_base > 0) & (~np.isnan(arr_base))
    mask_molt = (arr_molt > 0) & (~np.isnan(arr_molt))
    mask_add = (arr_add > 0) & (~np.isnan(arr_add))

    # se non ho dati validi per questo flusso base, lo salto
    if np.sum(mask_base) == 0:
        print(f"Nessun dato valido per {base}, salto il plot.")
        continue

    media_base, std_base = calc_stats(arr_base, mask_base)
    media_molt, std_molt = calc_stats(arr_molt, mask_molt)
    media_add, std_add = calc_stats(arr_add, mask_add)

    plt.figure(figsize=(12, 7))

    # plotto la curva originale
    plt.plot(times_arr[mask_base], arr_base[mask_base],
             marker='o', linestyle='-', linewidth=0.8, markersize=3, alpha=0.7, color='blue',
             label=rf"{base} (Avg: {media_base:.0f}, $\sigma$: {(std_base / media_base * 100):.2f}%)")

    # plotto la curva corretta (Moltiplicativa) se presente
    if np.sum(mask_molt) > 0:
        plt.plot(times_arr[mask_molt], arr_molt[mask_molt],
                 marker='o', linestyle='-', linewidth=0.8, markersize=3, alpha=0.7, color='orange',
                 label=rf"Corr. Moltiplicativa (Avg: {media_molt:.0f}, $\sigma$: {(std_molt / media_molt * 100):.2f}%)")

    # plotto la curva corretta (Additiva) se presente
    if np.sum(mask_add) > 0:
        plt.plot(times_arr[mask_add], arr_add[mask_add],
                 marker='o', linestyle='-', linewidth=0.8, markersize=3, alpha=0.7, color='green',
                 label=rf"Corr. Additiva (Avg: {media_add:.0f}, $\sigma$: {(std_add / media_add * 100):.2f}%)")

    # aggiungo linee verticali per separare le Run (estetica)
    for r_idx, (run_num, t_end) in enumerate(run_boundaries):
        # non disegno la linea alla fine dell'ultima run
        if r_idx < len(run_boundaries):
            plt.axvline(x=t_end, color='gray', linestyle='--', alpha=0.5)
            # etichetta Run
            t_start_run = 0 if r_idx == 0 else run_boundaries[r_idx - 1][1]
            t_center = (t_start_run + t_end) / 2

            # calcolo il centro dell'asse Y per posizionare la scritta
            y_min, y_max = plt.ylim()
            y_mid = (y_min + y_max) / 2

            # scritta centrata
            plt.text(t_end, y_mid, f"Fine Run {run_num}",
                     rotation=90,
                     horizontalalignment='center',
                     verticalalignment='top',
                     color='#333333',
                     fontsize=8,
                     fontweight='bold')

    plt.title(f'Andamento {base} e Correzioni\nAnalisi Multi-Run (1, 2, 3) - ID {id_stella_target}\nTarget ~ {KRON_TARGET} ADU')
    plt.xlabel("Tempo dall'inizio della Run 1 (secondi)")
    plt.ylabel('Flusso (ADU)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim(0, None)
    plt.legend()
    plt.tight_layout()

    file_grafico = f'andamento_{base}_{KRON_TARGET}.png'
    plt.savefig(file_grafico, dpi=300)
    plt.show()

    print(f"Salvato grafico: {file_grafico}")