import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
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
    all_data[base + '_CORRETTO_Normalizzazione_Moltiplicativa'] = []
    all_data[base + '_CORRETTO_Correzione_Additiva_dell_Apertura'] = []
    all_data[base + '_FONDO_SOTTRATTO'] = []

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

                    col_fondo = base + '_FONDO_SOTTRATTO'
                    if col_fondo in stella_nel_frame.colnames:
                        all_data[col_fondo].append(stella_nel_frame[col_fondo][0])
                    else:
                        all_data[col_fondo].append(np.nan)
            else:
                # inserisco nan per mantenere l'allineamento temporale se non trovo la stella
                for base in flussi_base:
                    all_data[base].append(np.nan)
                    all_data[base + '_CORRETTO_Normalizzazione_Moltiplicativa'].append(np.nan)
                    all_data[base + '_CORRETTO_Correzione_Additiva_dell_Apertura'].append(np.nan)
                    all_data[base + '_FONDO_SOTTRATTO'].append(np.nan)

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

# --- CONVERSIONE E GRAFICI ITERATIVI ---

times_arr = np.array(all_times)

# carico il file contenente i risultati del fondo
df_fondo = pd.read_csv('risultati_somma_pixel.csv')

# filtro il dataframe per includere tutte le run della lista e lo ordino
df_run = df_fondo[df_fondo['Run'].isin(run_list)].sort_values(['Run', 'Tempo_UTC']).copy()

# estraggo l'array dei valori del fondo per pixel
fondo = df_run['fondo_per_pixel'].values

# estraggo l'array del flusso richiesto unendo tutti i dati della stella
flusso_stella = np.array(all_data['flusso_fisso_max_run'])

# verifico che le lunghezze coincidano (in caso di file mancanti), troncando alla lunghezza minore
lunghezza_minima = min(len(fondo), len(flusso_stella))
fondo = fondo[:lunghezza_minima]
flusso_stella = flusso_stella[:lunghezza_minima]

# creo una maschera booleana per rimuovere i NaN che impedirebbero il calcolo di Pearson
maschera_validi = ~np.isnan(fondo) & ~np.isnan(flusso_stella)
fondo_valido = fondo[maschera_validi]
flusso_valido = flusso_stella[maschera_validi]

# estraggo l'array dei tempi corrispondente ai dati validi per tracciare la curva di luce
tempi_validi = times_arr[:lunghezza_minima][maschera_validi]

# calcolo il fit lineare (z[0] è la pendenza m, z[1] è l'intercetta q)
z = np.polyfit(fondo_valido, flusso_valido, 1)
m_pendenza = z[0]

# calcolo il valore medio del fondo cielo
fondo_medio = np.mean(fondo_valido)

# applico la decorrelazione lineare per correggere il flusso
flusso_corretto = flusso_valido - m_pendenza * (fondo_valido - fondo_medio)

# preparo la figura per visualizzare la correzione sulla curva di luce
plt.figure(figsize=(12, 6))

# creo il grafico con i flussi originali e corretti
plt.plot(tempi_validi, flusso_valido, marker='.', linewidth=1, linestyle='-', alpha=0.5, color='red',
         label='Flusso Originale (non corretto)')
plt.plot(tempi_validi, flusso_corretto, marker='.', linewidth=1, linestyle='-', alpha=0.8, color='green',
         label='Flusso Corretto (detrending fondo)')

# formatto le etichette e il titolo del grafico
plt.title(f"Confronto Curva di Luce - Flusso_fisso_max_run (ID {id_stella_target})", fontsize=14)
plt.xlabel("Tempo relativo (s)", fontsize=12)
plt.ylabel("Flusso stellare", fontsize=12)
plt.ylim(0, None)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

# aggiungo linee verticali per separare visivamente le run se desiderato
for run, t_end in run_boundaries[:-1]:
    plt.axvline(x=t_end, color='black', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("confronto_curva_di_luce_corretta.jpg", dpi=300)

# mostro il grafico a schermo
plt.show()

# calcolo il nuovo coefficiente di correlazione tra il fondo e il flusso corretto
correlazione_corretta, p_value_corretto = pearsonr(fondo_valido, flusso_corretto)

# stampo i risultati a terminale
print(f"\nAnalisi post-correzione per le Run {run_list}:")
print(f"Nuovo coefficiente di correlazione di Pearson (r): {correlazione_corretta:.4f}")
print(f"Nuovo P-value: {p_value_corretto:.4e}")

# preparo la figura per il nuovo grafico di dispersione
plt.figure(figsize=(8, 6))

# creo il grafico a dispersione con i dati corretti
plt.scatter(fondo_valido, flusso_corretto, alpha=0.7, color='green', edgecolor='k')

# calcolo e aggiungo la linea di tendenza (fit lineare di grado 1)
z_corr = np.polyfit(fondo_valido, flusso_corretto, 1)
p_corr = np.poly1d(z_corr)
plt.plot(fondo_valido, p_corr(fondo_valido), "r--", linewidth=2, label=f'Trend lineare (r={correlazione_corretta:.4f})')

# formatto le etichette e il titolo del grafico
plt.title("Correlazione post-detrending tra Fondo Cielo e Flusso Corretto", fontsize=14)
plt.xlabel("Fondo medio per pixel", fontsize=12)
plt.ylabel("Flusso singola stella (corretto)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("correlazione_flusso_corretto.jpg", dpi=300)

# mostro il grafico a schermo
plt.show()