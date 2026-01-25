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
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from photutils.aperture import CircularAperture

# Set up matplotlib
import matplotlib.pyplot as plt

# Soppressione warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=UserWarning)


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
KRON_TARGET = 150
INDICE_IMMAGINE_RIFERIMENTO = 35
max_sep = 0.003349 * u.deg

# --- SETUP STELLE CATALOGATE ---
file_csv_catalogate = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run}/run_1_stelle_catalogate_immagine_{INDICE_IMMAGINE_RIFERIMENTO:03d}.csv"

if not os.path.exists(file_csv_catalogate):
    print(f"Errore: File catalogate non trovato: {file_csv_catalogate}")
    exit()

dataframe2 = pd.read_csv(file_csv_catalogate, comment='#')
tbl_catalogate = Table.from_pandas(dataframe2)
header_csv = leggi_header_da_csv(file_csv_catalogate)

percorso_file_fits = header_csv.get('PERCORSO_FILE', '')
if not os.path.exists(percorso_file_fits):
    print(f"Attenzione: FITS originale non trovato in {percorso_file_fits}")

# Caricamento FITS di sfondo
path_image_file = percorso_file_fits
try:
    hdu_list = fits.open(path_image_file)
    image_data = hdu_list[0].data
    header_image = hdu_list[0].header
    wcs_image_ref = WCS(header_image)
except Exception as e:
    print(f"Errore caricamento immagine sfondo: {e}")
    image_data = np.zeros((100, 100))
    wcs_image_ref = None

posizioni_vere_celesti = SkyCoord(ra=tbl_catalogate['RAJ2000'], dec=tbl_catalogate['DEJ2000'], unit='deg', frame='icrs')

if os.path.exists(percorso_file_fits):
    hdu_list = fits.open(percorso_file_fits)
    header = hdu_list[0].header
    w = WCS(header)
    posizioni_vere_pixel_tuple = w.world_to_pixel(posizioni_vere_celesti)
    posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel_tuple[0], posizioni_vere_pixel_tuple[1]))
else:
    posizioni_vere_pixel = np.zeros((len(tbl_catalogate), 2))

magnitudini = tbl_catalogate['Mag']
cmap_mag = plt.cm.viridis_r
norm_mag = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())
colors = cmap_mag(norm_mag(magnitudini))

# --- GESTIONE LISTA FILE CSV ---
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

# --- FASE 1: IDENTIFICAZIONE TARGET ---
print(f"--- FASE 1: Ricerca stella con Kron ~ {KRON_TARGET} ---")
path_ref = lista_percorsi_csv[INDICE_IMMAGINE_RIFERIMENTO]
df_ref = pd.read_csv(path_ref, comment='#')
tbl_ref = Table.from_pandas(df_ref)

mask_si = np.char.startswith(tbl_ref['Corrispondenza'].astype(str), 'SI')
tbl_catalogate_ref = tbl_ref[mask_si]
if len(tbl_catalogate_ref) == 0:
    print("Nessuna stella catalogata trovata.")
    exit()

idx_min = np.argmin(np.abs(tbl_catalogate_ref['kron_flux'] - KRON_TARGET))
stella_ref = tbl_catalogate_ref[idx_min]
id_stella_target = stella_ref['ID']
# Definiamo la posizione corretta per CircularAperture: una lista di tuple o array (N,2)
position_ref = np.array([[stella_ref['xcentroid'], stella_ref['ycentroid']]])
print(f"Tracking ID: {id_stella_target}")

# --- FASE 2: ESTRAZIONE DATI ---
print(f"--- FASE 2: Estrazione dati ---")
ra_list, dec_list, times = [], [], []
n = 0
t0 = None
bool_time = False

for percorso_csv in lista_percorsi_csv:
    n +=1

    header_dal_csv_ = leggi_header_da_csv(percorso_csv)
    image_file = header_dal_csv_['PERCORSO_FILE']
    w_curr = None
    try:
        with fits.open(image_file) as hdu_list:
            fits_header = hdu_list[0].header
            w_curr = WCS(fits_header)
    except Exception as e:
        # Gestisci l'errore se il file non si apre
        pass

    header_tmp = leggi_header_da_csv(percorso_csv)
    try:
        df = pd.read_csv(percorso_csv, comment='#')
    except:
        continue
    tbl_frame = Table.from_pandas(df)

    t_curr = header_tmp.get('TSTART', 0)
    if not bool_time:
        bool_time = True
        t0 = t_curr
        times.append(0.0)
    else:
        times.append((t_curr - t0) / 1000 if t0 else 0)

    stella = tbl_frame[tbl_frame['ID'] == id_stella_target]
    if len(stella) > 0:
        x_pix, y_pix = stella['xcentroid'][0], stella['ycentroid'][0]
        # fits_header = fits.Header(header_tmp)
        try:
            w_curr = WCS(fits_header)
            sky = w_curr.pixel_to_world(x_pix, y_pix)
            ra_list.append(sky.ra.deg)
            dec_list.append(sky.dec.deg)
        except:
            ra_list.append(np.nan);
            dec_list.append(np.nan)

        if n == INDICE_IMMAGINE_RIFERIMENTO:
            scales = proj_plane_pixel_scales(wcs_image_ref)
            pixel_scale_deg = np.mean(scales)
            r_in_pixels = max_sep.to(u.deg).value / pixel_scale_deg
            position_ref_ = np.array([x_pix, y_pix])
            # Salviamo in una variabile che NON viene sovrascritta
            aperture = CircularAperture(position_ref_, r=r_in_pixels)

    else:
        ra_list.append(np.nan);
        dec_list.append(np.nan)

ra_arr = np.array(ra_list)
dec_arr = np.array(dec_list)
time_arr = np.array(times)
mask_valid = ~np.isnan(ra_arr) & ~np.isnan(dec_arr)
ra_clean = ra_arr[mask_valid]
dec_clean = dec_arr[mask_valid]
time_clean = time_arr[mask_valid]

print(f"Punti validi: {len(ra_clean)}")

'''# --- FASE 3: GRAFICO RA/DEC ---
if len(ra_clean) > 0:
    plt.figure(figsize=(10, 8))
    # NOTA: Rimosso cmap dove c=colors (array esplicito)
    sc = plt.scatter(ra_clean, dec_clean, c=time_clean, cmap='grey_r', s=50, alpha=1, edgecolor='k', zorder=10)
    plt.scatter(posizioni_vere_celesti.ra.deg, posizioni_vere_celesti.dec.deg,
                c=colors, s=36, alpha=1, edgecolor='magenta', label='Stelle Catalogate', zorder=1)

    plt.plot(ra_clean, dec_clean, c='gray', alpha=0.3, linestyle='--')
    plt.plot(ra_clean[0], dec_clean[0], 'r^', markersize=12, label='Inizio')
    plt.plot(ra_clean[-1], dec_clean[-1], 'g^', markersize=12, label='Fine')

    cbar = plt.colorbar(sc)
    cbar.set_label('Tempo (s)')
    plt.gca().invert_xaxis()
    plt.xlabel('RA [deg]');
    plt.ylabel('Dec [deg]')
    plt.title(f'Moto RA/Dec stella di kron circa {KRON_TARGET}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ticklabel_format(useOffset=False, style='plain')
    pad_deg = 0.005
    plt.xlim(max(ra_clean) + pad_deg, min(ra_clean) - pad_deg)
    plt.ylim(min(dec_clean) - pad_deg, max(dec_clean) + pad_deg)
    plt.show()'''

# --- FASE 4: GRAFICO IMMAGINE (CORRETTO) ---

# Proiezione traccia su pixel
if wcs_image_ref and len(ra_clean) > 0:
    ref_x, ref_y = wcs_image_ref.world_to_pixel_values(ra_clean, dec_clean)
else:
    ref_x, ref_y = [], []

# --- RIMOSSO np.transpose(ref_x, ref_y) CHE CAUSAVA ERRORE ---

plt.figure(figsize=(12, 10))

# Immagine sfondo
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
plt.imshow(image_data, cmap="grey_r", norm=LogNorm(), interpolation='nearest', origin='lower')
plt.colorbar(label='Conteggi')

# Plot
if len(ref_x) > 0:
    # 1. Stelle Catalogate (Sfondo)
    plt.scatter(posizioni_vere_pixel[:, 0], posizioni_vere_pixel[:, 1],
                c=colors, s=36, alpha=1, edgecolor='magenta', label='Catalogate')
    cmap = plt.cm.viridis_r
    norm = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Magnitudini catalogate')

    # 2. Traccia Centroide (Primo piano)
    sc = plt.scatter(ref_x, ref_y, c=time_clean, cmap='viridis', s=40, alpha=0.9, edgecolor='white', linewidth=0.5,
                     zorder=10, label='Centroidi')
    plt.plot(ref_x, ref_y, c='cyan', alpha=0.3, linewidth=1, linestyle='--', zorder=10)
    plt.plot(ref_x[0], ref_y[0], 'rx', markersize=15, markeredgewidth=2, label='Inizio', zorder=11)
    plt.plot(ref_x[-1], ref_y[-1], 'gx', markersize=15, markeredgewidth=2, label='Fine', zorder=11)

    # 3. Aperture (Cerchi gialli) attorno alla stella trovata nel frame di riferimento

    if wcs_image_ref:
        aperture.plot(color='yellow', lw=1.5, alpha=0.8, label='Regione di correlazione')

    cbar_time = plt.colorbar(sc, pad=0.08)
    cbar_time.set_label('Tempo (s)')
    plt.legend(loc='upper right')

    # --- FIX CRUCIALE PER LO ZOOM ---
    # pad_pix = np.sqrt(stella_ref['area'])/2
    pad_pix = 50
    lim_x_min = min(ref_x) - pad_pix
    lim_x_max = max(ref_x) + pad_pix
    lim_y_min = min(ref_y) - pad_pix
    lim_y_max = max(ref_y) + pad_pix

    plt.xlim(lim_x_min, lim_x_max)
    plt.ylim(lim_y_min, lim_y_max)
    # --------------------------------

plt.title(f"Traccia stella di kron circa {KRON_TARGET} su frame #{INDICE_IMMAGINE_RIFERIMENTO}")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")

# --- SALVATAGGIO ---
nome_file_output = f"traccia_finale_stella_di_kron{KRON_TARGET}_immagine_di_riferimento_{INDICE_IMMAGINE_RIFERIMENTO}_run{run}.png"
# Correzione slash iniziale nel percorso
path_output = os.path.join('/home/lorysimeone/tesi_magistrale/prove_2/analisi/studio_posizioni', nome_file_output)

# Crea la cartella se non esiste
os.makedirs(os.path.dirname(path_output), exist_ok=True)

plt.savefig(path_output, dpi=300, bbox_inches='tight')
print(f"Immagine salvata correttamente in: {path_output}")

plt.show()