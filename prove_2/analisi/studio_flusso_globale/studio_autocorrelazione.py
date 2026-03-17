import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf
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

KRON_TARGET = 300

# indice del file nella lista da usare come riferimento per trovare l'ID della stella
INDICE_IMMAGINE_RIFERIMENTO = 26
INDICE_RUN_DI_RIFERIMENTO = 1

cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{INDICE_RUN_DI_RIFERIMENTO}")

# verifico esistenza cartella
if not os.path.exists(cartella_csv):
    print(f"Errore: La cartella {cartella_csv} non esiste.")
    exit()

# creo la lista dei file ordinata
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

if not lista_percorsi_csv:
    print("Nessun file CSV trovato.")
    exit()

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
tbl_catalogate_ref = tbl_ref[mask_si]

if len(tbl_catalogate_ref) == 0:
    print("Nessuna stella catalogata trovata nel file di riferimento.")
    exit()

# calcolo la differenza assoluta tra i flussi trovati e il target
differenze = np.abs(tbl_catalogate_ref['kron_flux'] - KRON_TARGET)

# ordino le stelle partendo da quella con il flusso più vicino al target
indici_ordinati = np.argsort(differenze)

id_stella_target = None
stella_ref = None

print("Cerco una stella che sia presente almeno una volta in tutte le run...")

# scorro le candidate per trovare la prima valida presente in tutte le run
for idx in indici_ordinati:
    candidata = tbl_catalogate_ref[idx]
    cand_id = candidata['ID']

    presente_in_tutte = True

    # verifico la presenza della candidata in ogni singola run
    for run in run_list:
        cartella_run = os.path.join(base_path, f"tabelle_unite_run_{run}")
        if not os.path.exists(cartella_run):
            presente_in_tutte = False
            break

        files_run = sorted([f for f in os.listdir(cartella_run) if f.endswith('.csv')])
        percorsi_run = [os.path.join(cartella_run, f) for f in files_run]

        trovata_in_run = False

        # cerco l'ID in almeno un file della run corrente
        for percorso in percorsi_run:
            try:
                # leggo solo la colonna ID per velocizzare la ricerca
                df_temp = pd.read_csv(percorso, usecols=['ID'], comment='#')
                if cand_id in df_temp['ID'].values:
                    trovata_in_run = True
                    break
            except Exception:
                pass

        # se non la trovo nella run attuale, scarto la stella e interrompo il ciclo
        if not trovata_in_run:
            presente_in_tutte = False
            break

    # se la trovo in tutte le run, salvo il suo ID e mi fermo
    if presente_in_tutte:
        id_stella_target = cand_id
        stella_ref = candidata
        print(f"Stella target trovata: ID {id_stella_target} (scostamento dal Kron target: {differenze[idx]:.2f})")
        break

if id_stella_target is None:
    print("Errore: nessuna stella trovata che sia presente in tutte le run.")
    exit()

print(f"--- ANALISI MULTI-RUN (Run: {run_list}) ---")
print(f"Target ID selezionato: {id_stella_target}")

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

all_times = []

# imposto la variabile per il tempo iniziale globale (t=0 alla prima immagine della prima run)
t0_global = None

# definisco i limiti delle run nel grafico (per le bande verticali)
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
            else:
                # inserisco nan per mantenere l'allineamento temporale se non trovo la stella
                for base in flussi_base:
                    all_data[base].append(np.nan)

            all_times.append(tempo_relativo)

        except Exception as e:
            print(f"Errore nel file {os.path.basename(percorso_csv)}: {e}")
            pass

        # aggiungo il feedback di caricamento
        if n == len(lista_percorsi_csv) - 1:
            print(f"  Elaborati {len(lista_percorsi_csv)} file...")

    # memorizzo dove finisce questa run per disegnarla graficamente
    if len(all_times) > start_idx:
        end_time = all_times[-1]
        run_boundaries.append((run, end_time))

# --- STUDIO AUTOCORRELAZIONE PER OGNI FLUSSO ---

times_arr = np.array(all_times)

# carico il file contenente i risultati del fondo
df_fondo = pd.read_csv('risultati_somma_pixel.csv')

# filtro il dataframe per includere tutte le run della lista e lo ordino
df_run = df_fondo[df_fondo['Run'].isin(run_list)].sort_values(['Run', 'Tempo_UTC']).copy()

# estraggo l'array dei valori del fondo per pixel
fondo_originale = df_run['fondo_per_pixel'].values

print("\n\n=========================================================")
print("RISULTATI AUTOCORRELAZIONE (Lag 1) SUI FLUSSI DECORRELATI")
print("=========================================================\n")
print("Nota: Un valore vicino a 0 indica l'assenza di sistematiche (puro rumore stocastico).")
print("Un valore alto indica che la decorrelazione non ha appiattito del tutto il trend.\n")

risultati_acf = {}

for nome_flusso in flussi_base:
    # estraggo l'array del flusso corrente
    flusso_stella = np.array(all_data[nome_flusso])

    # verifico che le lunghezze coincidano (in caso di file mancanti), troncando alla lunghezza minore
    lunghezza_minima = min(len(fondo_originale), len(flusso_stella))
    fondo = fondo_originale[:lunghezza_minima]
    flusso_troncato = flusso_stella[:lunghezza_minima]

    # creo una maschera booleana per rimuovere i NaN
    maschera_validi = ~np.isnan(fondo) & ~np.isnan(flusso_troncato)
    fondo_valido = fondo[maschera_validi]
    flusso_valido = flusso_troncato[maschera_validi]

    # controllo se ho dati a sufficienza per questo flusso
    if len(flusso_valido) < 10:
        print(f"[{nome_flusso}]: Dati insufficienti per l'analisi.")
        continue

    # applico la decorrelazione lineare per correggere il flusso
    z = np.polyfit(fondo_valido, flusso_valido, 1)
    m_pendenza = z[0]
    fondo_medio = np.mean(fondo_valido)
    flusso_corretto = flusso_valido - m_pendenza * (fondo_valido - fondo_medio)

    # calcolo l'autocorrelazione (fino a 50 lag o metà della serie se troppo corta)
    lags_max = min(50, len(flusso_corretto) // 2)
    autocorrelazione = acf(flusso_corretto, nlags=lags_max)

    # estraggo il valore a lag 1 (il primo dopo lo 0 che è sempre 1)
    acf_lag1 = autocorrelazione[1]
    risultati_acf[nome_flusso] = acf_lag1

    print(f"-> {nome_flusso}: ACF(Lag 1) = {acf_lag1:.4f}")

    # preparo la figura per il grafico dell'autocorrelazione
    plt.figure(figsize=(10, 5))

    # creo il grafico a barre per visualizzare l'autocorrelazione
    plt.bar(range(len(autocorrelazione)), autocorrelazione, color='steelblue', alpha=0.8)

    # aggiungo la linea dello zero
    plt.axhline(0, color='black', linestyle='-', linewidth=1)

    # calcolo e aggiungo le bande di confidenza per il rumore bianco (livello 95%)
    livello_confidenza = 1.96 / np.sqrt(len(flusso_corretto))
    plt.axhline(livello_confidenza, color='red', linestyle='--', label='Intervallo di confidenza 95%')
    plt.axhline(-livello_confidenza, color='red', linestyle='--')

    # formatto le etichette e il titolo
    plt.title(f"Autocorrelazione Flusso Decorrelato\n[{nome_flusso}]", fontsize=14)
    plt.xlabel("Lag temporale", fontsize=12)
    plt.ylabel("Coefficiente ACF", fontsize=12)
    plt.ylim(-0.5, 1.0)  # fisso la scala Y per confrontare ad occhio i grafici più facilmente
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # salvo la figura con il nome specifico del flusso
    nome_file_out = f"autocorrelazione_{nome_flusso}.jpg"
    plt.savefig(nome_file_out, dpi=300)
    plt.close()  # chiudo la figura per evitare che si sovrappongano tra un ciclo e l'altro

print("\nI grafici dell'autocorrelazione sono stati salvati nella directory di lavoro.")