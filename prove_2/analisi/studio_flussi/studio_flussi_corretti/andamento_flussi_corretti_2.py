import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
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

KRON_TARGET = 350

# indice del file nella lista da usare come riferimento per trovare l'ID della stella
INDICE_IMMAGINE_RIFERIMENTO = 26
INDICE_RUN_DI_RIFERIMENTO = 1

cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{INDICE_RUN_DI_RIFERIMENTO}")

# verifico l'esistenza della cartella
if not os.path.exists(cartella_csv):
    print(f"Errore: La cartella {cartella_csv} non esiste.")
    exit()

# creo la lista dei file ordinata
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

if not lista_percorsi_csv:
    print("Nessun file CSV trovato.")
    exit()

# --- FASE 0.5: IDENTIFICAZIONE ID COMUNI A TUTTE LE RUN ---

print("--- FASE 0: Ricavo gli ID comuni presenti in tutte le run ---")
id_comuni = None

# ciclo su ogni run per estrarre gli ID univoci
for run in tqdm(run_list):
    cartella_run = os.path.join(base_path, f"tabelle_unite_run_{run}")
    if not os.path.exists(cartella_run):
        continue

    file_csv_run = sorted([os.path.join(cartella_run, f) for f in os.listdir(cartella_run) if f.endswith('.csv')])
    id_run_corrente = set()

    for f_csv in file_csv_run:
        try:
            # leggo solo la colonna ID per massimizzare le prestazioni
            df_tmp = pd.read_csv(f_csv, comment='#', usecols=['ID'])
            id_run_corrente.update(df_tmp['ID'].dropna().unique())
        except Exception:
            pass

    # interseco i set per mantenere solo gli ID che compaiono in tutte le run analizzate
    if id_comuni is None:
        id_comuni = id_run_corrente
    else:
        id_comuni = id_comuni.intersection(id_run_corrente)

if not id_comuni:
    print("Errore: nessun ID comune trovato in tutte le run indicate.")
    exit()

print(f"Trovati {len(id_comuni)} ID che compaiono in tutte le {len(run_list)} run.")

# --- FASE 1: IDENTIFICAZIONE STELLA TARGET ---

print(f"--- FASE 1: Ricerca stella con Kron ~ {KRON_TARGET} nel file #{INDICE_IMMAGINE_RIFERIMENTO} ---")

# gestisco l'indice fuori range
if INDICE_IMMAGINE_RIFERIMENTO >= len(lista_percorsi_csv):
    INDICE_IMMAGINE_RIFERIMENTO = 0
    print("Indice riferimento fuori range, uso il primo file.")

path_ref = lista_percorsi_csv[INDICE_IMMAGINE_RIFERIMENTO]
df_ref = pd.read_csv(path_ref, comment='#')
tbl_ref = Table.from_pandas(df_ref)

# filtro solo le stelle che hanno una corrispondenza nel catalogo ('SI...')
mask_si = np.char.startswith(tbl_ref['Corrispondenza'].astype(str), 'SI')

# mi assicuro che la stella compaia almeno una volta in tutte e tre le run
mask_comuni = np.isin(tbl_ref['ID'], list(id_comuni))

# unisco le due condizioni
mask_valida = mask_si & mask_comuni
tbl_catalogate_ref = tbl_ref[mask_valida]

if len(tbl_catalogate_ref) == 0:
    print("Nessuna stella catalogata e comune a tutte le run trovata nel file di riferimento.")
    exit()

# calcolo la differenza assoluta tra i flussi trovati e il target
differenze = np.abs(tbl_catalogate_ref['kron_flux'] - KRON_TARGET)

# trovo l'indice della differenza minima
idx_min = np.argmin(differenze)
stella_ref = tbl_catalogate_ref[idx_min]

# salvo l'ID univoco da cercare negli altri file
id_stella_target = stella_ref['ID']

print(f"--- ANALISI MULTI-RUN (Run: {run_list}) ---")
print(f"Target ID: {id_stella_target}")

# --- STRUTTURE DATI GLOBALI ---

# definisco le categorie di stringhe per comporre tutti i nomi delle colonne
flussi_base = [
    'kron_manuale_aper',
    'kron_manuale_seg',
    'somma_apertura_ultimo_pixel',
    'flusso_fisso_max_run',
    'flusso_raggio_fisso_doppio'
]

varianti_correzione = [
    '',
    '_CORRETTO_Correzione_Additiva_dell_Apertura',
    '_FONDO_SOTTRATTO'
]

varianti_decorrelazione = [
    '',
    '_DECORRELAZIONE_LINEARE',
    '_DECORRELAZIONE_STELLE',
    '_DECORRELAZIONE_LINEARE_DECORRELAZIONE_STELLE',
    '_DECORRELAZIONE_STELLE_GLOBALE',
    '_DECORRELAZIONE_LINEARE_DECORRELAZIONE_STELLE_GLOBALE'
]

# preparo il dizionario iterando dinamicamente per generare tutte le chiavi per flusso
all_data = {}
for base in flussi_base:
    for var in varianti_correzione:
        for dec in varianti_decorrelazione:
            nome_colonna = base + var + dec
            all_data[nome_colonna] = []

all_times = []
t0_global = None
run_boundaries = []

# --- CICLO SULLE RUN ---

for run in run_list:
    print(f"\n>>> Elaborazione RUN {run}...")

    cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

    if not os.path.exists(cartella_csv):
        print(f"ATTENZIONE: La cartella {cartella_csv} non esiste. Salto questa run.")
        continue

    file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
    lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

    if not lista_percorsi_csv:
        print(f"Nessun file CSV trovato in Run {run}.")
        continue

    # --- ESTRAZIONE DATI FRAME PER FRAME ---

    start_idx = len(all_times)

    for n, percorso_csv in enumerate(lista_percorsi_csv):
        try:
            dataframe = pd.read_csv(percorso_csv, comment='#')
            header_dal_csv = leggi_header_da_csv(percorso_csv)
            tbl_frame = Table.from_pandas(dataframe)

            t_curr = header_dal_csv.get('TSTART', 0)

            if t0_global is None:
                t0_global = t_curr

            tempo_relativo = (t_curr - t0_global) / 1000.0 if t0_global is not None else 0

            mask_target = tbl_frame['ID'] == id_stella_target
            stella_nel_frame = tbl_frame[mask_target]

            if len(stella_nel_frame) > 0:
                # ciclo su tutti i nomi di colonna generati per estrarre il dato se esiste
                for col_name in all_data.keys():
                    if col_name in stella_nel_frame.colnames:
                        try:
                            # converto in float in caso di artefatti testuali
                            val = float(stella_nel_frame[col_name][0])
                        except (ValueError, TypeError):
                            val = np.nan
                        all_data[col_name].append(val)
                    else:
                        all_data[col_name].append(np.nan)
            else:
                for col_name in all_data.keys():
                    all_data[col_name].append(np.nan)

            all_times.append(tempo_relativo)

        except Exception as e:
            print(f"Errore nel file {os.path.basename(percorso_csv)}: {e}")
            pass

        if n == len(lista_percorsi_csv) - 1:
            print(f"  Elaborati {len(lista_percorsi_csv)} file...")

    if len(all_times) > start_idx:
        end_time = all_times[-1]
        run_boundaries.append((run, end_time))

# --- FUNZIONI DI SUPPORTO PER I GRAFICI ---

times_arr = np.array(all_times)


def calc_stats(arr, mask):
    # calcolo media e deviazione standard escludendo gli zeri e i nan
    if np.sum(mask) > 0:
        vals = arr[mask]
        return np.mean(vals), np.std(vals)
    return 0.0, 0.0


print("\n=== GENERAZIONE GRAFICI A PANNELLI 2x3 ===")

titoli_subplot = [
    "Dati Originali (Nessuna Decorrelazione)",
    "Decorrelazione Lineare (Fondo)",
    "Decorrelazione Stelle (Locale)",
    "Dec. Lineare + Dec. Stelle (Locale)",
    "Decorrelazione Stelle (Globale)",
    "Dec. Lineare + Dec. Stelle (Globale)"
]

colori_plot = ['blue', 'orange', 'green', 'red']
etichette_legenda = ['Base', 'Additiva', 'Fondo Sott.']

for base in flussi_base:
    # controllo se esistono dati validi prima di creare il grafico
    arr_verifica = np.array(all_data[base])
    if np.sum((arr_verifica > 0) & (~np.isnan(arr_verifica))) == 0:
        print(f"Nessun dato valido per {base}, salto il plot.")
        continue

    # creo la figura a griglia 2x3
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    for idx_dec, dec in enumerate(varianti_decorrelazione):
        ax = axes[idx_dec]

        # plotto le varianti di correzione in ogni rispettivo quadrante
        for var, color, lbl in zip(varianti_correzione, colori_plot, etichette_legenda):
            nome_colonna = base + var + dec

            arr = np.array(all_data[nome_colonna])
            mask = (arr > 0) & (~np.isnan(arr))

            if np.sum(mask) > 0:
                media, std = calc_stats(arr, mask)
                # calcolo l'errore percentuale per la legenda
                err_pct = (std / media * 100) if media != 0 else 0

                ax.plot(times_arr[mask], arr[mask],
                        marker='o', linestyle='-', linewidth=1.0, markersize=3, alpha=0.7, color=color,
                        label=rf"{lbl} ($\sigma$: {err_pct:.2f}%)")

        # disegno le linee divisorie per evidenziare le run
        for r_idx, (run_num, t_end) in enumerate(run_boundaries):
            ax.axvline(x=t_end, color='gray', linestyle='--', alpha=0.5)

        # formatto il singolo quadrante
        ax.set_title(titoli_subplot[idx_dec], fontsize=12, fontweight='bold')
        ax.set_xlabel("Tempo dall'inizio della Run 1 (secondi)")
        ax.set_ylabel("Flusso (ADU)")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize='small', loc='best')

    # imposto il titolo globale del file
    fig.suptitle(
        f'Evoluzione del Detrending per [{base}]\nAnalisi Multi-Run - ID {id_stella_target} (Target Kron ~ {KRON_TARGET})',
        fontsize=16, fontweight='bold')

    # imposto il layout per non sovrapporre il titolo ai grafici
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    file_grafico = f'analisi_detrending_completa_{base}_{KRON_TARGET}.jpg'
    plt.savefig(file_grafico, dpi=300)
    plt.show()
    plt.close()

    print(f"Salvato grafico a pannelli: {file_grafico}")

print("\n--- TUTTI I GRAFICI SONO STATI GENERATI ---")