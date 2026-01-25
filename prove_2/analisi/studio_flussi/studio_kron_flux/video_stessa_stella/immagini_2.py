import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from astropy.table import Table
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from astropy.wcs import WCS, FITSFixedWarning
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from astropy.visualization import simple_norm
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from photutils.aperture import CircularAperture
from photutils.segmentation import make_2dgaussian_kernel, SourceCatalog, SourceFinder
from astropy.convolution import convolve

# Soppressione warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- FUNZIONE DI ANALISI (Modificata per ritornare la mappa) ---
def analisi_image_segmentation(data, std_esterna=None):
    """
    Esegue la segmentazione.
    Returns: tbl_filtrato, parametri, segment_map
    """
    # Se fornisco una std esterna (calcolata su tutta l'immagine), uso quella.
    if std_esterna is not None:
        mean, median, std = 0, 0, std_esterna
    else:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # Lettura parametri (Hardcoded path per brevità, assicurati sia corretto)
    parametri = {}
    path_param = '/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt'
    if os.path.exists(path_param):
        with open(path_param, 'r') as file:
            next(file)
            for riga in file:
                riga = riga.strip()
                if riga and not riga.startswith('#'):
                    parametro, valore = riga.split()
                    parametri[parametro] = float(valore) if '.' in valore else int(valore)
    else:
        # Valori di fallback se il file non c'è
        parametri = {'fwhm': 3.0, 'size': 5, 'threshold_assoluta': 3.61, 'pixel': 5}

    fwhm = parametri.get('fwhm', 3.0)
    size = int(parametri.get('size', 5))
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)

    threshold = parametri.get('threshold_assoluta', 3.0)
    n = int(parametri.get('pixel', 5))

    finder = SourceFinder(npixels=n, progress_bar=False)
    segment_map = finder(convolved_data, threshold)

    if segment_map is None:
        return None, parametri, None

    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    # Filtraggio base (semplificato rispetto al tuo script originale per brevità, ma funzionale)
    indici_validi = []
    soglia_assoluta = 2.5
    soglia_relativa = 0.05
    bordo = 2  # ridotto per i cutout
    ny, nx = data.shape

    for i, sorgente in enumerate(tbl):
        label = sorgente['label']
        mask_sorgente = (segment_map.data == label)
        valori_originali = data[mask_sorgente]
        xcentroid = sorgente['xcentroid']
        ycentroid = sorgente['ycentroid']

        dentro_riquadro = (xcentroid >= bordo) and (xcentroid < nx - bordo) and \
                          (ycentroid >= bordo) and (ycentroid < ny - bordo)

        if not dentro_riquadro: continue

        pixel_sopra_soglia_assoluta = np.sum(valori_originali > soglia_assoluta)
        pixel_sopra_soglia_relativa = np.sum(valori_originali > soglia_relativa * sorgente['max_value'])

        if pixel_sopra_soglia_assoluta >= 3 and pixel_sopra_soglia_relativa >= 3:
            indici_validi.append(i)

    tbl_filtrato = tbl[indici_validi]
    if len(tbl_filtrato) > 0:
        tbl_filtrato['label'] = np.arange(1, len(tbl_filtrato) + 1)

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
KRON_TARGET = 1000
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
posizioni_vere_celesti = SkyCoord(ra=tbl_catalogate['RAJ2000'], dec=tbl_catalogate['DEJ2000'], unit='deg', frame='icrs')

magnitudini = tbl_catalogate['Mag']
cmap_mag = plt.cm.viridis_r
norm_mag = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())
colors = cmap_mag(norm_mag(magnitudini))
cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=magnitudini.min(), vmax=magnitudini.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

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
print(f"Tracking ID: {id_stella_target}")

# --- CONFIGURAZIONE ---
output_dir = os.path.join(os.getcwd(), f"video_segmentazione_kron_{KRON_TARGET}")
os.makedirs(output_dir, exist_ok=True)
print(f"Le immagini verranno salvate in: {output_dir}")

base_pad = (np.sqrt(stella_ref['area']) * 2)
zoom_factor = 2.0  # Un po' più di zoom per vedere bene i pixel
final_pad = base_pad * zoom_factor

print(f"Dimensione finestra di zoom: +/- {final_pad:.1f} pixel dal centroide")

# --- CICLO SULLA RUN ---
for i, path_csv in enumerate(lista_percorsi_csv):
    i += 1

    # 1. Lettura CSV e FITS Header
    try:
        df_curr = pd.read_csv(path_csv, comment='#')
        tbl_curr = Table.from_pandas(df_curr)
        header_info = leggi_header_da_csv(path_csv)
        path_fits = header_info.get('PERCORSO_FILE', '')

        if not os.path.exists(path_fits): continue

        with fits.open(path_fits) as hdu_list:
            header = hdu_list[0].header
            w = WCS(header)
            full_data = hdu_list[0].data  # Leggiamo l'immagine intera
    except Exception as e:
        print(f"Skipping frame {i}: {e}")
        continue

    # 2. Trova la stella target in questo frame
    row_stella = df_curr[df_curr['ID'] == id_stella_target]
    if len(row_stella) == 0:
        print(f"Frame {i}: Stella persa.")
        continue

    # Estrai coordinate CENTROIDE globali
    cur_x = row_stella.iloc[0]['xcentroid']
    cur_y = row_stella.iloc[0]['ycentroid']

    # 3. Preparazione Dati (Sottrazione fondo globale)
    mean, median, std_globale = sigma_clipped_stats(full_data, sigma=3.0)
    data_sub = full_data - median

    # 4. CREAZIONE CUTOUT (Zoom fisico sulla matrice)
    # Convertiamo i limiti in interi per lo slicing
    y_min_cut = int(max(0, cur_y - final_pad))
    y_max_cut = int(min(data_sub.shape[0], cur_y + final_pad))
    x_min_cut = int(max(0, cur_x - final_pad))
    x_max_cut = int(min(data_sub.shape[1], cur_x + final_pad))

    # Estraggo solo la porzione di immagine che ci interessa
    cutout_data = data_sub[y_min_cut:y_max_cut, x_min_cut:x_max_cut]

    # 5. ESECUZIONE SEGMENTAZIONE (Solo sul Cutout)
    # Passiamo std_globale per usare la stessa soglia dell'immagine intera
    tbl_seg, params, seg_map_cutout = analisi_image_segmentation(cutout_data, std_esterna=std_globale)

    # 6. PLOTTING
    plt.figure(figsize=(6, 6))

    # Disegniamo il CUTOUT (Coordinate partono da 0,0 locale)
    plt.imshow(cutout_data, cmap="grey_r", norm=LogNorm(), interpolation='nearest', origin='lower')

    # --- VISUALIZZAZIONE SEGMENTAZIONE (Contorni) ---
    if seg_map_cutout is not None:
        # Disegna contorni Ciano attorno ai pixel segmentati
        # levels=[0.5] traccia il confine tra 0 (fondo) e >0 (sorgente)
        plt.contour(seg_map_cutout.data, levels=[0.5], colors='cyan', linewidths=2, alpha=0.9)

        # Opzionale: Creiamo un oggetto per la legenda
        linea_ciano = Line2D([0], [0], color='cyan', lw=2, label='Pixel Segmentati')
    else:
        linea_ciano = Line2D([0], [0], color='cyan', lw=2, label='Nessuna Segm.')

    # --- GESTIONE COORDINATE (Traslazione Globale -> Locale) ---
    # Dobbiamo sottrarre x_min_cut e y_min_cut a tutte le coordinate globali per sovrapporle al cutout

    # A. Stelle Catalogate
    posizioni_vere_pixel_globali = w.world_to_pixel(posizioni_vere_celesti)
    pix_x_cat = posizioni_vere_pixel_globali[0] - x_min_cut
    pix_y_cat = posizioni_vere_pixel_globali[1] - y_min_cut

    # Filtriamo quelle fuori dal cutout per pulizia (opzionale ma consigliato)
    mask_in_cutout = (pix_x_cat >= 0) & (pix_x_cat < cutout_data.shape[1]) & \
                     (pix_y_cat >= 0) & (pix_y_cat < cutout_data.shape[0])

    scatter_cat = plt.scatter(pix_x_cat[mask_in_cutout], pix_y_cat[mask_in_cutout],
                              c=colors[mask_in_cutout], s=36, alpha=1, edgecolor='magenta', label='Catalogate')

    # B. Centroide (quello trovato in precedenza nel CSV)
    centroide_x_loc = cur_x - x_min_cut
    centroide_y_loc = cur_y - y_min_cut

    # C. Regione di Correlazione (Cerchio giallo)
    scales = proj_plane_pixel_scales(w)
    pixel_scale_deg = np.mean(scales)
    r_in_pixels = max_sep.to(u.deg).value / pixel_scale_deg

    # Cerchio giallo (Regione di ricerca usata per il match)
    # Nota: Usiamo la posizione del centroide corrente come centro approssimativo per visualizzazione
    # Oppure ricalcoliamo la posizione 'prevista' se avessimo i dati.
    # Qui usiamo il centroide trovato per coerenza visiva.
    scatter_region = plt.scatter(centroide_x_loc, centroide_y_loc, s=r_in_pixels,
                                 facecolors='none', edgecolors='yellow', lw=1.5, label='Regione corr.')

    # Cerchio Rosso (Centroide puntuale)
    scatter_centroide = plt.scatter(centroide_x_loc, centroide_y_loc, s=10, c='red', marker='+', label='Centroide')

    # Estetica
    plt.title(f"Frame {i:03d}\nID: {id_stella_target} | Flux: {row_stella.iloc[0]['kron_flux']:.0f}")
    plt.xlabel("X Pixel (Locali)")
    plt.ylabel("Y Pixel (Locali)")

    # Legenda Personalizzata
    simbolo_cerchio_giallo = Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                    markeredgecolor='yellow', markeredgewidth=1.5, markersize=10, label='Regione Corr.')

    plt.legend(handles=[scatter_cat, simbolo_cerchio_giallo, linea_ciano], loc='upper right')
    plt.colorbar(sm, ax=plt.gca(), label='Magnitudine V')
    plt.tight_layout()

    # 5. Salvataggio
    nome_file_out = f"seg_zoom_{i:03d}.png"
    path_out = os.path.join(output_dir, nome_file_out)
    plt.savefig(path_out, dpi=100)
    plt.close()

    if i % 10 == 0:
        print(f"Salvato frame {i}...")

print(f"\nGenerazione completata. Immagini salvate in: {output_dir}")