import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.table import Table
import warnings
from astropy.wcs import FITSFixedWarning
import matplotlib.cm as cm  # Import necessario per i colori dinamici

# Sopprime il warning FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

# --- FUNZIONI DI UTILITÀ ---
def converti_valore(valore):
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

# --- CONFIGURAZIONE VISIVA GLOBALE ---
LINE_WIDTH = 0.5  # Spessore linea
MARKER_SIZE = 2  # Grandezza punti
MARKER_STYLE = 'o'  # <--- FORZA UNICO MARKER PER TUTTI ('o'=cerchio, 's'=quadrato, etc.)

# --- PARAMETRI CONFIGURAZIONE ---

run = 1
KRON_TARGET = 500
INDICE_IMMAGINE_RIFERIMENTO = 35

# Percorsi
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

if not os.path.exists(cartella_csv):
    print(f"Errore: La cartella {cartella_csv} non esiste.")
    exit()

file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

if not lista_percorsi_csv:
    print("Nessun file CSV trovato.")
    exit()

# === 1. RILEVAMENTO DINAMICO DEI FLUSSI (SENZA SCRIVERE NOMI A MANO) ===
try:
    # Leggiamo solo l'intestazione del primo file
    df_header = pd.read_csv(lista_percorsi_csv[0], comment='#', nrows=0)
    colonne = df_header.columns.tolist()

    # Cerca le colonne sentinella in modo flessibile
    idx_start = -1
    if 'saturazione' in colonne:
        idx_start = colonne.index('saturazione') + 1
    elif 'Satura' in colonne:
        idx_start = colonne.index('Satura') + 1

    idx_end = -1
    if 'RA_centroid' in colonne: idx_end = colonne.index('RA_centroid')

    if idx_start != -1 and idx_end != -1 and idx_end > idx_start:
        FLUX_TYPES = colonne[idx_start:idx_end]
        print(f"✅ Flussi rilevati automaticamente: {FLUX_TYPES}")
    else:
        # Se qualcosa va storto, fallback di sicurezza
        print("⚠️ Impossibile rilevare i flussi tra Satura e RA_centroid.")
        FLUX_TYPES = ['kron_flux']  # Almeno uno di base

except Exception as e:
    print(f"⚠️ Errore nel rilevamento flussi: {e}")
    exit()

# --- FASE 1: IDENTIFICAZIONE STELLA TARGET ---

print(f"--- FASE 1: Ricerca stella con Kron ~ {KRON_TARGET} nel file #{INDICE_IMMAGINE_RIFERIMENTO} ---")

if INDICE_IMMAGINE_RIFERIMENTO >= len(lista_percorsi_csv):
    INDICE_IMMAGINE_RIFERIMENTO = 0

path_ref = lista_percorsi_csv[INDICE_IMMAGINE_RIFERIMENTO]
df_ref = pd.read_csv(path_ref, comment='#')
tbl_ref = Table.from_pandas(df_ref)

mask_si = np.char.startswith(tbl_ref['Corrispondenza'].astype(str), 'SI')
tbl_catalogate_ref = tbl_ref[mask_si]

if len(tbl_catalogate_ref) == 0:
    print("Nessuna stella catalogata trovata.")
    exit()

differenze = np.abs(tbl_catalogate_ref['kron_flux'] - KRON_TARGET)
idx_min = np.argmin(differenze)
stella_ref = tbl_catalogate_ref[idx_min]
id_stella_target = stella_ref['ID']

print(f"Stella selezionata ID: {id_stella_target}")
print("-" * 50)

# --- FASE 2: ESTRAZIONE DATI ---

print(f"--- FASE 2: Estrazione curve di luce... ---")

flux_data = {ft: [] for ft in FLUX_TYPES}
times = []
t0 = None

for n, percorso_csv in enumerate(lista_percorsi_csv):
    try:
        dataframe = pd.read_csv(percorso_csv, comment='#')
        header_dal_csv = leggi_header_da_csv(percorso_csv)
        tbl_frame = Table.from_pandas(dataframe)
    except Exception:
        continue

    t_curr = header_dal_csv.get('TSTART', 0)
    if n == 0:
        t0 = t_curr
        times.append(0.0)
    else:
        times.append((t_curr - t0) / 1000.0 if t0 is not None else 0)

    mask_target = tbl_frame['ID'] == id_stella_target
    stella_nel_frame = tbl_frame[mask_target]

    if len(stella_nel_frame) > 0:
        for ft in FLUX_TYPES:
            if ft in stella_nel_frame.colnames:
                flux_data[ft].append(stella_nel_frame[ft][0])
            else:
                flux_data[ft].append(np.nan)
    else:
        for ft in FLUX_TYPES:
            flux_data[ft].append(0.0)

    if (n + 1) % 50 == 0:
        print(f"Elaborati {n + 1}/{len(lista_percorsi_csv)} file...")

times = np.array(times)
for ft in FLUX_TYPES:
    flux_data[ft] = np.abs(np.array(flux_data[ft]))

# --- STATISTICHE E PLOTTING COMPLETAMENTE AUTOMATICO ---

print(f"\n=== RISULTATI CONFRONTO (ID: {id_stella_target}) ===")

plt.figure(figsize=(12, 7))

# 1. Preparazione Colori e Marker
# Genera una lista di colori distinti lunga quanto la lista dei flussi
num_flussi = len(FLUX_TYPES)
# Usa 'tab10' (10 colori) o 'tab20' (20 colori) se hai tantissimi flussi
colors = plt.cm.tab10(np.linspace(0, 1, min(num_flussi, 10)))
if num_flussi > 10:  # Fallback se hai più di 10 flussi, cicla i colori
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

markers_list = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # Lista marker da ciclare

# 2. Ciclo di Plotting
for i, ft in enumerate(FLUX_TYPES):

    data_arr = flux_data[ft]
    mask_validi = (data_arr > 0) & (~np.isnan(data_arr))
    dati_puliti = data_arr[mask_validi]
    tempi_puliti = times[mask_validi]

    if len(dati_puliti) > 0:
        media = np.mean(dati_puliti)
        std = np.std(dati_puliti)
        perc_err = (std / media) * 100 if media != 0 else 0

        print(f"\n> {ft}:")
        print(f"   Media: {media:.2f}")
        print(f"   Std:   {std:.2f} ({perc_err:.2f}%)")

        # --- ASSEGNAZIONE STILE AUTOMATICA ---
        # Colore basato sull'indice i (modulo la lunghezza della palette per sicurezza)
        colore_dinamico = colors[i % len(colors)]

        # Marker basato sull'indice i
        marker_dinamico = markers_list[i % len(markers_list)]

        # Label pulita (toglie underscore e mette maiuscole)
        label_pulita = ft.replace('_', ' ').replace('manuale', '(Man)').title()

        plt.plot(tempi_puliti, dati_puliti,
                 marker=marker_dinamico,
                 linestyle='-',
                 linewidth=LINE_WIDTH,
                 markersize=MARKER_SIZE,
                 alpha=0.8,
                 color=colore_dinamico,
                 label=f"{label_pulita} (Media: {media:.0f}, std: {(std/media*100):.2f}%)")

    else:
        print(f"\n> {ft}: Nessun dato valido.")

# Configurazione Grafico
plt.title(f'Confronto Flussi Relativi e Assoluti | ID Stella: {id_stella_target}\nTarget: ~{KRON_TARGET}')
plt.xlabel('Tempo (secondi)')
plt.ylabel('Flusso (Valore Assoluto)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, None)
plt.tight_layout()

plt.show()