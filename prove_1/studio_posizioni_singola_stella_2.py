import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from matplotlib.colors import LogNorm
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from photutils.aperture import CircularAperture
# IMPORTANTE: Questa è la funzione che sostituisce il doppio passaggio manuale
from astropy.wcs.utils import pixel_to_pixel

# Soppressione warning
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- FUNZIONI DI UTILITÀ (Invariate) ---
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
                    if key == 'PERCORSO_FILE':
                        pass
                    elif ' / ' in value:
                        value = value.split(' / ')[0].strip()
                    elif '/' in value and not value.strip().startswith('/'):
                        value = value.split('/')[0].strip()
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':
                break
    return header_dict


# --- PARAMETRI ---
run = 1
KRON_TARGET = 27000
INDICE_IMMAGINE_RIFERIMENTO = 35
max_sep = 2.0  # In pixel (circa), usato solo per il disegno dell'apertura

# --- SETUP E CARICAMENTO IMMAGINE DI RIFERIMENTO ---
# (Questa parte serve per avere il WCS di destinazione e l'immagine di sfondo)
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

# Identificazione file di riferimento
path_ref_csv = lista_percorsi_csv[INDICE_IMMAGINE_RIFERIMENTO]
header_ref = leggi_header_da_csv(path_ref_csv)
path_fits_ref = header_ref.get('PERCORSO_FILE', '')

# Caricamento WCS Riferimento (DESTINAZIONE)
try:
    with fits.open(path_fits_ref) as hdu_ref:
        image_data = hdu_ref[0].data
        header_image = hdu_ref[0].header
        wcs_ref = WCS(header_image)  # Questo è il WCS di destinazione
except Exception as e:
    print(f"Errore critico caricamento riferimento: {e}")
    exit()

# Trova ID stella target (invariato)
df_ref = pd.read_csv(path_ref_csv, comment='#')
tbl_ref = Table.from_pandas(df_ref)
mask_si = np.char.startswith(tbl_ref['Corrispondenza'].astype(str), 'SI')
tbl_catalogate_ref = tbl_ref[mask_si]
idx_min = np.argmin(np.abs(tbl_catalogate_ref['kron_flux'] - KRON_TARGET))
stella_ref = tbl_catalogate_ref[idx_min]
id_stella_target = stella_ref['ID']
print(f"Tracking ID: {id_stella_target}")

# --- FASE 2: ESTRAZIONE E TRASFORMAZIONE DIRETTA (PIXEL -> PIXEL) ---
print(f"--- FASE 2: Trasformazione coordinate Pixel-to-Pixel ---")

ref_pixel_x_list = []
ref_pixel_y_list = []
times = []

t0 = None
bool_time = False

for i, percorso_csv in enumerate(lista_percorsi_csv):

    # 1. Lettura dati
    header_tmp = leggi_header_da_csv(percorso_csv)
    path_fits_curr = header_tmp.get('PERCORSO_FILE', '')

    try:
        df = pd.read_csv(percorso_csv, comment='#')
        tbl_frame = Table.from_pandas(df)
    except:
        continue

    # 2. Gestione Tempo
    t_curr = header_tmp.get('TSTART', 0)
    if not bool_time:
        bool_time = True
        t0 = t_curr
        times.append(0.0)
    else:
        times.append((t_curr - t0) / 1000 if t0 else 0)

    # 3. Estrazione coordinate e trasformazione
    stella = tbl_frame[tbl_frame['ID'] == id_stella_target]

    if len(stella) > 0:
        x_pix_curr, y_pix_curr = stella['xcentroid'][0], stella['ycentroid'][0]

        # Apriamo il FITS corrente SOLO per ottenere il WCS di partenza
        try:
            with fits.open(path_fits_curr) as hdu_curr:
                wcs_curr = WCS(hdu_curr[0].header)

                # --- IL PUNTO CHIAVE: pixel_to_pixel ---
                # Trasforma direttamente (x,y)_curr in (x,y)_ref
                # input: wcs partenza, wcs arrivo, x, y
                x_ref, y_ref = pixel_to_pixel(wcs_curr, wcs_ref, x_pix_curr, y_pix_curr)

                ref_pixel_x_list.append(x_ref)
                ref_pixel_y_list.append(y_ref)

        except Exception as e:
            # Se fallisce il WCS o il file
            ref_pixel_x_list.append(np.nan)
            ref_pixel_y_list.append(np.nan)
    else:
        ref_pixel_x_list.append(np.nan)
        ref_pixel_y_list.append(np.nan)

# Pulizia dati
ref_x_arr = np.array(ref_pixel_x_list)
ref_y_arr = np.array(ref_pixel_y_list)
time_arr = np.array(times)
mask = ~np.isnan(ref_x_arr) & ~np.isnan(ref_y_arr)
x_final, y_final, t_final = ref_x_arr[mask], ref_y_arr[mask], time_arr[mask]

print(f"Punti validi: {len(x_final)}")

# --- FASE 3: PLOT ---
plt.figure(figsize=(10, 10))

# Sfondo
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
plt.imshow(image_data, cmap="grey_r", norm=LogNorm(), interpolation='nearest', origin='lower')

# Traccia
sc = plt.scatter(x_final, y_final, c=t_final, cmap='viridis', s=40, edgecolor='white', linewidth=0.5)
plt.plot(x_final, y_final, c='cyan', alpha=0.3, linestyle='--')

# Start/End
if len(x_final) > 0:
    plt.plot(x_final[0], y_final[0], 'rx', markersize=12, markeredgewidth=2, label='Start')
    plt.plot(x_final[-1], y_final[-1], 'gx', markersize=12, markeredgewidth=2, label='End')

# Apertura di riferimento (dove dovrebbe essere la stella nel frame di riferimento)
if INDICE_IMMAGINE_RIFERIMENTO < len(lista_percorsi_csv):
    # Nota: se usiamo pixel_to_pixel, la posizione nel frame di riferimento è... se stessa.
    # Quindi possiamo prendere la coordinata calcolata all'indice del riferimento.
    # Cerchiamo l'indice nell'array pulito che corrisponde al tempo del riferimento (o approssimato)
    # Metodo semplice: ricalcoliamo la posizione della stella nel frame di riferimento
    stella_target_ref = tbl_catalogate_ref[tbl_catalogate_ref['ID'] == id_stella_target]
    if len(stella_target_ref) > 0:
        xref_static = stella_target_ref['xcentroid'][0]
        yref_static = stella_target_ref['ycentroid'][0]
        ap = CircularAperture((xref_static, yref_static), r=max_sep)
        ap.plot(color='yellow', lw=2, label='Posizione Ref')

plt.colorbar(sc, label='Tempo (s)')
plt.legend()
plt.title(f"Traccia (Pixel-to-Pixel) ID {id_stella_target}")
plt.xlabel("Pixel X (Ref)")
plt.ylabel("Pixel Y (Ref)")

# Zoom
pad = 50
if len(x_final) > 0:
    plt.xlim(min(x_final) - pad, max(x_final) + pad)
    plt.ylim(min(y_final) - pad, max(y_final) + pad)

plt.show()