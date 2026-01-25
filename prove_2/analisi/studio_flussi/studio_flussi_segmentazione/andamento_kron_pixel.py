import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.table import Table
import warnings
from astropy.wcs import FITSFixedWarning

# Set up wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.wcsapi import SlicedLowLevelWCS
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table, vstack
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
        return np.ones_like(data)

    return (data - d_min) / (d_max - d_min)


# --- PARAMETRI CONFIGURAZIONE ---

run = 1
KRON_TARGET = 1000
INDICE_IMMAGINE_RIFERIMENTO = 35

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

# --- FASE 1: IDENTIFICAZIONE STELLA TARGET ---

print(f"--- FASE 1: Ricerca stella con Kron ~ {KRON_TARGET} nel file #{INDICE_IMMAGINE_RIFERIMENTO} ---")

if INDICE_IMMAGINE_RIFERIMENTO >= len(lista_percorsi_csv):
    INDICE_IMMAGINE_RIFERIMENTO = 0
    print("Indice riferimento fuori range, uso il primo file.")

path_ref = lista_percorsi_csv[INDICE_IMMAGINE_RIFERIMENTO]
df_ref = pd.read_csv(path_ref, comment='#')
tbl_ref = Table.from_pandas(df_ref)

mask_si = np.char.startswith(tbl_ref['Corrispondenza'].astype(str), 'SI')
tbl_catalogate_ref = tbl_ref[mask_si]

if len(tbl_catalogate_ref) == 0:
    print("Nessuna stella catalogata trovata nel file di riferimento.")
    exit()

differenze = np.abs(tbl_catalogate_ref['kron_flux'] - KRON_TARGET)
idx_min = np.argmin(differenze)
stella_ref = tbl_catalogate_ref[idx_min]
id_stella_target = stella_ref['ID']

print(f"Stella selezionata:")
print(f"  ID: {id_stella_target}")
print(f"  Kron Flux nel file ref: {stella_ref['kron_flux']:.2f}")
print(f"  Coordinate Ref: ({stella_ref['xcentroid']:.2f}, {stella_ref['ycentroid']:.2f})")
print("-" * 50)

# --- FASE 2: TRACCIAMENTO TEMPORALE ---

print(f"--- FASE 2: Estrazione curva di luce per ID {id_stella_target} ---")

kron_flux_array = []
x_centroid = []
y_centroid = []
spostamento = []  # NUOVO ARRAY PER LA DISTANZA
times = []
bool = False
t0 = None

# Variabili per memorizzare la posizione iniziale (X0, Y0)
x_start = None
y_start = None

# Slicing della lista file
# lista_percorsi_csv = lista_percorsi_csv[2:-2]

for n, percorso_csv in enumerate(lista_percorsi_csv):

    n += 1
    dataframe = pd.read_csv(percorso_csv, comment='#')
    header_dal_csv = leggi_header_da_csv(percorso_csv)
    tbl_frame = Table.from_pandas(dataframe)

    mask_target = tbl_frame['ID'] == id_stella_target
    stella_nel_frame = tbl_frame[mask_target]

    # Variabili temporanee per questo frame
    x_curr = 0.0
    y_curr = 0.0

    if len(stella_nel_frame) > 0:
        valore_flux = stella_nel_frame['kron_flux'][0]
        x_curr = stella_nel_frame['xcentroid'][0]
        y_curr = stella_nel_frame['ycentroid'][0]

        kron_flux_array.append(valore_flux)
        x_centroid.append(x_curr)
        y_centroid.append(y_curr)
    else:
        kron_flux_array.append(0.0)
        x_centroid.append(0.0)
        y_centroid.append(0.0)

    # Gestione Tempo e Posizione Iniziale
    t_curr = header_dal_csv.get('TSTART', 0)

    if not bool:
        bool = True
        t0 = t_curr
        # Fissiamo la posizione iniziale al primo frame valido della serie tagliata
        x_start = x_curr
        y_start = y_curr

        times.append(0.0)
        spostamento.append(0.0)
    else:
        times.append((t_curr - t0) / 1000.0 if t0 is not None else 0)

        # Calcolo distanza Euclidea (Teorema di Pitagora) dal punto iniziale
        if x_start is not None:
            dist = np.sqrt((x_curr - x_start) ** 2 + (y_curr - y_start) ** 2)
            spostamento.append(dist)
        else:
            spostamento.append(0.0)

    if n % 50 == 0:
        print(f"Elaborati {n + 1}/{len(lista_percorsi_csv)} file...")

# Conversione in array numpy
kron_flux_array = np.array(kron_flux_array)
x_centroid = np.array(x_centroid)
y_centroid = np.array(y_centroid)
spostamento = np.array(spostamento)
times = np.array(times)

valori_validi = kron_flux_array[kron_flux_array > 0]

print(f"\n=== RISULTATI FINALI ===")
if len(valori_validi) > 0:
    media = np.mean(valori_validi)
    std = np.std(valori_validi)
    print(f"ID Target: {id_stella_target}")
    print(f"File totali: {len(lista_percorsi_csv)}")
    print(f"Media Kron Flux: {media:.2f}")
    print(f"Std Dev: {std:.2f}")
else:
    print("Nessun dato valido trovato.")

# --- GRAFICO NORMALIZZATO ---

plt.figure(figsize=(12, 7))

# 1. Definiamo la maschera dei dati validi
mask_plot = kron_flux_array > 0

# 2. Estraiamo i dati validi
t_valid = times[mask_plot]
flux_valid = kron_flux_array[mask_plot]
x_valid = x_centroid[mask_plot]
y_valid = y_centroid[mask_plot]
spost_valid = spostamento[mask_plot]  # Dati validi spostamento

# 3. Normalizziamo
flux_norm = normalizza_01(flux_valid)
x_norm = normalizza_01(x_valid)
y_norm = normalizza_01(y_valid)
spost_norm = normalizza_01(spost_valid)

# 4. Plot
plt.plot(t_valid, flux_norm, marker='o', ls='-', lw=1.0, ms=3, alpha=0.62, color='blue', label='Kron Flux (Norm)')
plt.plot(t_valid, x_norm, marker='.', ls='--', lw=1.0, ms=3, alpha=0.6, color='red', label='X Centroid (Norm)')
plt.plot(t_valid, y_norm, marker='.', ls='--', lw=1.0, ms=3, alpha=0.6, color='green', label='Y Centroid (Norm)')
plt.plot(t_valid, spost_norm, marker='.', ls='--', lw=1.0, ms=3, alpha=0.6, color='#FFC300', label='Spostamento (Norm)')

# Abbellimenti
plt.title(
    f'Confronto Normalizzato [0-1]\nID {id_stella_target} (Target Flux ~{KRON_TARGET})')
plt.xlabel('Tempo (secondi)')
plt.ylabel('Valore Normalizzato (a.u.)')
plt.legend(loc='lower left', framealpha=0.5)
plt.grid(True, linestyle=':', alpha=0.6)

plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()