import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from astropy.table import Table
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from astropy.wcs import WCS, FITSFixedWarning
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from astropy.visualization import simple_norm
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


def analisi_image_segmentation(data, std_esterna=None):  # Aggiunto parametro opzionale
    """
    Returns:
    tbl_filtrato, parametri, segment_map (Nuovo output!)
    """

    # Se fornisco una std esterna (calcolata su tutta l'immagine), uso quella.
    # Altrimenti la calcolo sui dati passati (rischioso se è un cutout piccolo)
    if std_esterna is not None:
        mean, median, std = 0, 0, std_esterna
    else:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # ... [Lettura parametri e Convoluzione rimangono uguali] ...
    # Lettura parametri
    parametri = {}
    with open('/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt', 'r') as file:
        next(file)
        for riga in file:
            riga = riga.strip()
            if riga and not riga.startswith('#'):
                parametro, valore = riga.split()
                parametri[parametro] = float(valore) if '.' in valore else int(valore)

    fwhm = parametri['fwhm']
    size = parametri['size']
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)

    # Sourcefinder
    # threshold = parametri['threshold_sigma'] * std # ESEMPIO: Se volessi usare la std
    threshold = parametri['threshold_assoluta']  # Tu usi quella assoluta, quindi la std impatta meno, ma è bene averla
    n = parametri['pixel']

    finder = SourceFinder(npixels=n, progress_bar=False)  # Progress bar false per non intasare il terminale nel loop
    segment_map = finder(convolved_data, threshold)

    # Se non trova nulla, restituisci None
    if segment_map is None:
        return None, parametri, None

    # Catalogo sorgenti
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    # ... [Il tuo codice di filtraggio rimane uguale] ...
    # (Ometto per brevità le righe di filtraggio che hai già scritto, copiale qui)
    # ...

    # Creazione tabella filtrata (copia la tua logica finale)
    tbl_filtrato = tbl  # (o tbl[indici_validi] se hai applicato il filtro)

    # RESTITUISCI ANCHE segment_map
    return tbl_filtrato, parametri, segment_map

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
KRON_TARGET = 27000
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

# --- CONFIGURAZIONE ---
# Creiamo una cartella dedicata per non inondare la directory corrente
output_dir = os.path.join(os.getcwd(), f"video_stessa_stella_di_kron_circa_{KRON_TARGET}")
os.makedirs(output_dir, exist_ok=True)
print(f"Le immagini verranno salvate in: {output_dir}")

# Calcolo del padding base (fisso per tutta la run basato sulla reference)
# Nota: Moltiplico per un fattore (es. 5) per vedere un po' di contesto attorno alla stella
# Altrimenti il crop sarebbe troppo stretto (esattamente sui pixel della stella).
base_pad = (np.sqrt(stella_ref['area']) * 2)
zoom_factor = 1.  # Fattore estetico: 1 = solo la stella, 5 = stella + contesto
final_pad = base_pad * zoom_factor

print(f"Dimensione finestra di zoom: +/- {final_pad:.1f} pixel dal centroide")

magnitudini = tbl_catalogate['Mag']
cmap_mag = plt.cm.viridis_r
norm_mag = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())
colors = cmap_mag(norm_mag(magnitudini))
cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# --- CICLO SULLA RUN ---
for i, path_csv in enumerate(lista_percorsi_csv):

    i+=1

    # 1. Leggi il CSV corrente
    try:
        df_curr = pd.read_csv(path_csv, comment='#')
        tbl_curr = Table.from_pandas(df_curr)
        # Leggi l'header per trovare il percorso FITS
        header_info = leggi_header_da_csv(path_csv)
        path_fits = header_info.get('PERCORSO_FILE', '')
        with fits.open(path_fits) as hdu_list:
            header = hdu_list[0].header
            w = WCS(header)
    except Exception as e:
        print(f"Skipping frame {i}: Errore lettura CSV ({e})")
        continue

    # 2. Trova la stella target in questo frame
    # Cerchiamo per ID univoco
    row_stella = df_curr[df_curr['ID'] == id_stella_target]
    stella = tbl_curr[tbl_curr['ID'] == id_stella_target]

    if len(row_stella) == 0:
        print(f"Frame {i}: Stella target non trovata (persa o non rilevata).")
        continue

    # Estrai coordinate correnti
    cur_x = row_stella.iloc[0]['xcentroid']
    cur_y = row_stella.iloc[0]['ycentroid']

    # 3. Carica immagine FITS
    if not os.path.exists(path_fits):
        print(f"Frame {i}: File FITS non trovato ({path_fits})")
        continue

    with fits.open(path_fits) as hdu:
        data = hdu[0].data
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        data = data - median

    # 4. Plotting
    plt.figure(figsize=(6, 6))

    # Normalizzazione Logaritmica per vedere meglio la struttura
    plt.imshow(data, cmap="grey_r", norm=LogNorm(),
               interpolation='nearest')  # genero l'immagine con scala di colori bianco e nero
    plt.gca().invert_yaxis()

    # --- APPLICAZIONE ZOOM ---
    # Centrato sul centroide corrente
    # Dimensione fissa basata sulla stella di riferimento (area)
    lim_x_min = cur_x - final_pad
    lim_x_max = cur_x + final_pad
    lim_y_min = cur_y - final_pad
    lim_y_max = cur_y + final_pad

    plt.xlim(lim_x_min, lim_x_max)
    plt.ylim(lim_y_min, lim_y_max)

    posizioni_vere_celesti = SkyCoord(ra=tbl_catalogate['RAJ2000'],
                                      dec=tbl_catalogate['DEJ2000'],
                                      unit='deg',
                                      frame='icrs')

    posizioni_vere_pixel = w.world_to_pixel(posizioni_vere_celesti)  # converto da celesti a pixel
    posizioni_vere_pixel = np.column_stack((posizioni_vere_pixel[0], posizioni_vere_pixel[1]))

    scatter_cat = plt.scatter(posizioni_vere_pixel[:, 0], posizioni_vere_pixel[:, 1],
                c=colors, s=36, alpha=1, edgecolor='magenta', label='Catalogate')

    # Estetica
    plt.title(f"Frame {i:03d} \n ID: {id_stella_target} , flusso circa {KRON_TARGET}")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")

    # disegno la regione di correlazione

    scales = proj_plane_pixel_scales(w)
    pixel_scale_deg = np.mean(scales)
    r_in_pixels = max_sep.to(u.deg).value / pixel_scale_deg
    position_ref_ = np.column_stack((tbl_curr['xcentroid'], tbl_curr['ycentroid']))
    aperture = CircularAperture(position_ref_, r=r_in_pixels)
    cerchio = Line2D([0], [0],
                             marker='o',  # Forma circolare
                             color='w',  # Colore linea (invisibile/bianco per non vederla)
                             label='Regione di correlazione',
                             markerfacecolor='none',  # Nessun riempimento
                             markeredgecolor='yellow',  # Bordo giallo
                             markeredgewidth=1.5,
                             markersize=10)  # Grandezza del cerchio nella legenda
    aperture.plot(color='yellow', lw=1.5, alpha=0.8, label='Regione di correlazione')

    scatter_centroide = plt.scatter(position_ref_[0], position_ref_[1], s=r_in_pixels, color='yellow', facecolors='none', lw=1.5, edgecolors='yellow', alpha=1, label='Regione di correlazione')

    plt.tight_layout()
    plt.colorbar(sm, ax=plt.gca(), label='Magnitudine V')
    plt.legend(handles=[scatter_cat, cerchio], loc='upper right')

    # 5. Salvataggio
    nome_file_out = f"frame_{i:03d}.png"
    path_out = os.path.join(output_dir, nome_file_out)
    plt.savefig(path_out, dpi=100)
    plt.close()  # Importante per liberare memoria

    # Feedback a video ogni 10 frame
    if (i) % 10 == 0:
        print(f"Salvato frame {i}...")

print(f"\nGenerazione completata. Immagini salvate in: {output_dir}")