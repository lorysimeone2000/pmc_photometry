import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.table import Table
import warnings
from astropy.wcs import FITSFixedWarning

# Sopprime il warning FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)


# --- FUNZIONI DI UTILITÀ ---

def converti_valore(valore):
    """Converte una stringa nel tipo di dato appropriato."""
    valore = str(valore).strip()
    if not valore: return valore
    try:
        return int(valore)
    except ValueError:
        pass
    try:
        return float(valore)
    except ValueError:
        pass
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False
    return valore


def leggi_header_da_csv(filename):
    """Legge l'header FITS salvato nelle prime righe del file CSV."""
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':
                break
    return header_dict


# --- PARAMETRI CONFIGURAZIONE ---

run_list = [1, 2, 3]  # Lista delle run da analizzare
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"

# ID Stella Target (Hardcoded come richiesto)
id_stella_target = 134270845394639028
KRON_TARGET = 300  # Solo per riferimento nel titolo

print(f"--- ANALISI MULTI-RUN (Run: {run_list}) ---")
print(f"Target ID: {id_stella_target}")

# --- STRUTTURE DATI GLOBALI ---
# Accumuleremo tutti i dati qui per fare un plot unico
all_times = []
all_flux_fisso = []
all_flux_auto = []
all_flux_manuale = []

# Variabile per il tempo iniziale globale (t=0 alla prima immagine della prima run)
t0_global = None

# Colori per distinguere le run nel grafico (opzionale, per le bande verticali)
run_boundaries = []

# --- CICLO SULLE RUN ---

for run in run_list:
    print(f"\n>>> Elaborazione RUN {run}...")

    cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

    # Verifica esistenza cartella
    if not os.path.exists(cartella_csv):
        print(f"ATTENZIONE: La cartella {cartella_csv} non esiste. Salto questa run.")
        continue

    # Lista file ordinata
    file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
    lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

    if not lista_percorsi_csv:
        print(f"Nessun file CSV trovato in Run {run}.")
        continue

    # Rilevamento colonne (solo dal primo file della run per controllo)
    try:
        df_header = pd.read_csv(lista_percorsi_csv[0], comment='#', nrows=0)
        colonne = df_header.columns.tolist()
        # print(f"  File trovati: {len(lista_percorsi_csv)}")
    except Exception as e:
        print(f"Errore lettura header Run {run}: {e}")
        continue

    # --- ESTRAZIONE DATI FRAME PER FRAME ---

    start_idx = len(all_times)  # Indice di inizio di questa run nei dati globali

    for n, percorso_csv in enumerate(lista_percorsi_csv):
        try:
            # Lettura dati
            dataframe = pd.read_csv(percorso_csv, comment='#')
            header_dal_csv = leggi_header_da_csv(percorso_csv)
            tbl_frame = Table.from_pandas(dataframe)

            # Gestione Tempo (Continuo tra le run)
            t_curr = header_dal_csv.get('TSTART', 0)

            if t0_global is None:
                t0_global = t_curr  # Setta il tempo zero assoluto

            # Calcolo tempo relativo in secondi
            tempo_relativo = (t_curr - t0_global) / 1000.0 if t0_global is not None else 0

            # Cerca la stella target
            mask_target = tbl_frame['ID'] == id_stella_target
            stella_nel_frame = tbl_frame[mask_target]

            val_fisso = 0.0
            val_auto = 0.0
            val_manuale = 0.0

            if len(stella_nel_frame) > 0:
                # Stella trovata, estrai i flussi
                # Usa .get o check colonne per evitare crash se manca una colonna in una run specifica
                if 'flusso_fisso_max_run' in stella_nel_frame.colnames:
                    val_fisso = stella_nel_frame['flusso_fisso_max_run'][0]
                if 'kron_flux' in stella_nel_frame.colnames:
                    val_auto = stella_nel_frame['kron_flux'][0]
                if 'kron_manuale_aper' in stella_nel_frame.colnames:
                    val_manuale = stella_nel_frame['kron_manuale_aper'][0]

            # Append ai dati globali
            all_times.append(tempo_relativo)
            all_flux_fisso.append(val_fisso)
            all_flux_auto.append(val_auto)
            all_flux_manuale.append(val_manuale)

        except Exception as e:
            print(f"Errore nel file {os.path.basename(percorso_csv)}: {e}")
            # Inseriamo NaN per mantenere l'allineamento temporale se critico, o saltiamo
            pass

        # Feedback
        if n==len(lista_percorsi_csv):
            print(f"  Elaborati {len(lista_percorsi_csv)} file...")

    # Memorizza dove finisce questa run per disegnarla graficamente
    if len(all_times) > start_idx:
        end_time = all_times[-1]
        run_boundaries.append((run, end_time))

# --- CONVERSIONE E PULIZIA ---

times_arr = np.array(all_times)
flux_fisso_arr = np.array(all_flux_fisso)
flux_auto_arr = np.array(all_flux_auto)
flux_manual_arr = np.array(all_flux_manuale)

# Filtro per le statistiche GLOBALI (escludo gli zeri)
mask_valid_fisso = (flux_fisso_arr > 0) & (~np.isnan(flux_fisso_arr))
mask_valid_auto = (flux_auto_arr > 0) & (~np.isnan(flux_auto_arr))
mask_valid_manual = (flux_manual_arr > 0) & (~np.isnan(flux_manual_arr))


# Calcolo Statistiche Globali
def calc_stats(arr, mask):
    if np.sum(mask) > 0:
        vals = arr[mask]
        return np.mean(vals), np.std(vals)
    return 0.0, 0.0


media_fisso, std_fisso = calc_stats(flux_fisso_arr, mask_valid_fisso)
media_auto, std_auto = calc_stats(flux_auto_arr, mask_valid_auto)
media_manual, std_manual = calc_stats(flux_manual_arr, mask_valid_manual)


# ------------------------ questa parte è per partire con un kron target --------------------------------------

'''KRON_TARGET = 1000

# Indice del file nella lista da usare come riferimento per trovare l'ID della stella
INDICE_IMMAGINE_RIFERIMENTO = 35

# Percorsi
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

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

id_stella_target = stella_ref['ID']'''

# ---------------------------------------------------------------------------------------------------

print("\n=== RISULTATI GLOBALI (Run 1+2+3) ===")
print(f"Totale punti temporali: {len(times_arr)}")
print(f"Fisso   -> Media: {media_fisso:.2f}, Std: {std_fisso:.2f} ({std_fisso / media_fisso * 100:.2f}%)")
print(f"Auto    -> Media: {media_auto:.2f},  Std: {std_auto:.2f}  ({std_auto / media_auto * 100:.2f}%)")
print(f"Manuale -> Media: {media_manual:.2f}, Std: {std_manual:.2f} ({std_manual / media_manual * 100:.2f}%)")

# --- GRAFICO ---

plt.figure(figsize=(12, 7))

# Plot Curve
plt.plot(times_arr[mask_valid_fisso], flux_fisso_arr[mask_valid_fisso],
         marker='o', linestyle='-', linewidth=0.8, markersize=3, alpha=0.7, color='blue',
         # NOTA: aggiunto 'r' prima di f"..." qui sotto
         label=rf"Kron Fisso (Avg: {media_fisso:.0f}, $\sigma$: {(std_fisso / media_fisso * 100):.2f}%)")

plt.plot(times_arr[mask_valid_auto], flux_auto_arr[mask_valid_auto],
         marker='o', linestyle='-', linewidth=0.8, markersize=3, alpha=0.7, color='orange',
         # NOTA: aggiunto 'r' qui sotto
         label=rf"Kron Auto (Avg: {media_auto:.0f}, $\sigma$: {(std_auto / media_auto * 100):.2f}%)")

plt.plot(times_arr[mask_valid_manual], flux_manual_arr[mask_valid_manual],
         marker='o', linestyle='-', linewidth=0.8, markersize=3, alpha=0.7, color='green',
         # NOTA: aggiunto 'r' qui sotto
         label=rf"Kron Manuale (Avg: {media_manual:.0f}, $\sigma$: {(std_manual / media_manual * 100):.2f}%)")

# Aggiunta linee verticali per separare le Run (estetica)
for r_idx, (run_num, t_end) in enumerate(run_boundaries):
    # Non disegniamo la linea alla fine dell'ultima run
    if r_idx < len(run_boundaries):
        plt.axvline(x=t_end, color='gray', linestyle='--', alpha=0.5)
        # Etichetta Run
        t_start_run = 0 if r_idx == 0 else run_boundaries[r_idx - 1][1]
        t_center = (t_start_run + t_end) / 2

        # Calcolo il centro dell'asse Y per posizionare la scritta
        y_min, y_max = plt.ylim()
        y_mid = (0 + y_max) / 2

        # Scritta centrata
        plt.text(t_end, y_mid, f"Fine Run {run_num}",
                 rotation=90,
                 horizontalalignment='center',  # centro la stringa lungo l'altezza del grafico
                 verticalalignment='top',  # metto il testo a destra, 'bottom' lo mette a sinistra
                 color='#333333',
                 fontsize=8,
                 fontweight='bold') # grassetto

plt.title(f'Analisi Multi-Run (1, 2, 3) - ID {id_stella_target}\nTarget ~ {KRON_TARGET} ADU')
plt.xlabel('Tempo dall\'inizio della Run 1 (secondi)')
plt.ylabel('Flusso (ADU)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.ylim(0, None)
plt.legend()
plt.tight_layout()

plt.savefig(f'andamento_kron_{KRON_TARGET}', dpi=300)
plt.show()