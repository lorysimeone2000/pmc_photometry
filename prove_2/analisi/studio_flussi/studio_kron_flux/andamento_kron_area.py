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


def normalizza_01(data):
    """
    Normalizza un array numpy tra 0 e 1.
    Gestisce il caso di array costante (evita divisione per zero).
    """
    if len(data) == 0:
        return data
    d_min = np.min(data)
    d_max = np.max(data)

    if d_max == d_min:
        # Se tutti i valori sono uguali, restituisce un array di 1 (o 0)
        return np.ones_like(data)

    return (data - d_min) / (d_max - d_min)


# --- PARAMETRI CONFIGURAZIONE ---

run = 1
# Flusso target approssimativo per cercare la stella
KRON_TARGET = 1000
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

print(f"Stella selezionata:")
print(f"  ID: {id_stella_target}")
print(f"  Kron Flux nel file ref: {stella_ref['kron_flux']:.2f} (Target: {KRON_TARGET})")
print(f"  Area: {stella_ref['area']}")
print("-" * 50)

# --- FASE 2: TRACCIAMENTO TEMPORALE ---

print(f"--- FASE 2: Estrazione curva di luce per ID {id_stella_target} ---")

kron_flux_array = []
area_array = []  # NUOVO ARRAY PER L'AREA
times = []
bool_time = False
t0 = None

# Taglio della lista (primi 2 e ultimi 2 esclusi)
# lista_percorsi_csv = lista_percorsi_csv[2:-2]

for n, percorso_csv in enumerate(lista_percorsi_csv):

    n += 1
    # Lettura dati
    dataframe = pd.read_csv(percorso_csv, comment='#')
    header_dal_csv = leggi_header_da_csv(percorso_csv)
    tbl_frame = Table.from_pandas(dataframe)

    # Cerca la stella target specifica in questo frame
    mask_target = tbl_frame['ID'] == id_stella_target
    stella_nel_frame = tbl_frame[mask_target]

    if len(stella_nel_frame) > 0:
        # Stella trovata
        valore_flux = stella_nel_frame['kron_flux'][0]
        valore_area = stella_nel_frame['area'][0]  # ESTRAZIONE AREA

        # Salviamo i valori
        kron_flux_array.append(valore_flux)
        area_array.append(valore_area)
    else:
        # Stella persa in questo frame
        kron_flux_array.append(0.0)
        area_array.append(0.0)

    # Gestione Tempo
    t_curr = header_dal_csv.get('TSTART', 0)
    if not bool_time:
        bool_time = True
        t0 = t_curr
        times.append(0.0)
    else:
        # Tempo in secondi dall'inizio
        times.append((t_curr - t0) / 1000.0 if t0 is not None else 0)

    # Feedback progressivo
    if n % 50 == 0:
        print(f"Elaborati {n + 1}/{len(lista_percorsi_csv)} file...")

# Conversione in array numpy
kron_flux_array = np.array(kron_flux_array)
area_array = np.array(area_array)
times = np.array(times)

# Filtro per le statistiche (escludo gli zeri dove la stella non è stata trovata)
valori_validi = kron_flux_array[kron_flux_array > 0]
area_valida = area_array[area_array > 0]

print(f"\n=== RISULTATI FINALI ===")
if len(valori_validi) > 0:
    media = np.mean(valori_validi)
    std = np.std(valori_validi)
    media_area = np.mean(area_valida)
    std_area = np.std(area_valida)

    print(f"ID Target: {id_stella_target}")
    print(f"File totali: {len(lista_percorsi_csv)}")
    print(f"Rilevamenti validi: {len(valori_validi)}")
    print(f"Media Kron Flux: {media:.2f} +/- {std:.2f}")
    print(f"Media Area: {media_area:.2f} +/- {std_area:.2f}")
else:
    print("Nessun dato valido trovato per il grafico.")

# --- GRAFICO NORMALIZZATO ---

plt.figure(figsize=(12, 7))

# 1. Definiamo la maschera dei dati validi (dove la stella è stata trovata)
mask_plot = kron_flux_array > 0

# 2. Estraiamo i dati validi
t_valid = times[mask_plot]
flux_valid = kron_flux_array[mask_plot]
area_valid = area_array[mask_plot]

# 3. Normalizziamo solo i dati validi
flux_norm = normalizza_01(flux_valid)
area_norm = normalizza_01(area_valid)

# 4. Plot
plt.plot(t_valid, flux_norm, marker='o', ls='-', lw=1.0, ms=3, alpha=0.7, color='blue', label='Kron Flux (Norm)')
plt.plot(t_valid, area_norm, marker='.', ls='--', lw=1.0, ms=3, alpha=0.7, color='red', label='Area (Norm)')

# Abbellimenti
plt.title(f'Confronto Normalizzato [0-1]: Flusso vs Area\nID {id_stella_target} (Target Flux ~{KRON_TARGET})')
plt.xlabel('Tempo (secondi)')
plt.ylabel('Valore Normalizzato (a.u.)')
plt.legend(loc='best', framealpha=0.8)
plt.grid(True, linestyle=':', alpha=0.6)

# Non serve impostare ylim specifici perché i dati sono sempre tra 0 e 1
plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()