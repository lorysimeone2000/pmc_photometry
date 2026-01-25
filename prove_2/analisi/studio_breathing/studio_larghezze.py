import pandas as pd
import numpy as np
import os
import warnings
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from astropy.io import fits
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from photutils.segmentation import make_2dgaussian_kernel, SourceCatalog, SourceFinder
from astropy.convolution import convolve
from tqdm import tqdm

# --- SOPPRESSIONE WARNING ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='No sources were found')


# --- FUNZIONI (Analisi, Helper, ecc. invariate tranne analisi_stella che diventa silenziosa) ---

def analisi_image_segmentation(data, std_esterna=None, parametri_esterni=None):
    # ... (Codice invariato, uguale a prima) ...
    if std_esterna is not None:
        mean, median, std = 0, 0, std_esterna
    else:
        mean, median, std = sigma_clipped_stats(data[::5, ::5], sigma=3.0)

    if parametri_esterni is not None:
        parametri = parametri_esterni
    else:
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

    if len(tbl) > 0:
        soglia_assoluta = 2.5
        soglia_relativa = 0.05
        bordo = 2
        ny, nx = data.shape
        mask_bordo = (tbl['xcentroid'] >= bordo) & (tbl['xcentroid'] < nx - bordo) & \
                     (tbl['ycentroid'] >= bordo) & (tbl['ycentroid'] < ny - bordo)
        indici_validi = []
        for i, prop in enumerate(cat):
            if not mask_bordo[i]: continue
            slices = prop.slices
            cutout_data = data[slices]
            cutout_seg = segment_map.data[slices]
            mask_stella = (cutout_seg == prop.label)
            vals = cutout_data[mask_stella]
            p_abs = np.sum(vals > soglia_assoluta)
            p_rel = np.sum(vals > soglia_relativa * prop.max_value)
            if p_abs >= 3 and p_rel >= 3:
                indici_validi.append(i)
        tbl_filtrato = tbl[indici_validi]
        if len(tbl_filtrato) > 0:
            tbl_filtrato['label'] = np.arange(1, len(tbl_filtrato) + 1)
    else:
        tbl_filtrato = tbl

    return tbl_filtrato, parametri, segment_map


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


def somma_magnitudini(series_mags):
    mags = series_mags.dropna()
    if len(mags) == 0: return np.nan
    flussi = 10 ** (-0.4 * np.array(mags))
    return -2.5 * np.log10(np.sum(flussi)) if np.sum(flussi) > 0 else np.nan


def leggi_header_da_csv(filename):
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    if 'PERCORSO_FILE' in key:
                        pass
                    elif '/' in value:
                        value = value.split('/')[0].strip()
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':
                break
    return header_dict


# --- FUNZIONE ANALISI STELLA (SILENZIOSA) ---
def analisi_stella(stella_ref):
    """Versione silenziosa ottimizzata per loop esterno."""
    id_stella_target = stella_ref['ID']
    base_pad = (np.sqrt(stella_ref['area']) * 2)
    final_pad = int(base_pad * 2.0)

    # Parametri pre-caricati (hardcoded o letti una volta sola fuori se preferisci)
    parametri_seg = {'fwhm': 3.0, 'size': 5, 'threshold_assoluta': 3.61, 'pixel': 5}

    lista_risultati = []

    # Niente print qui dentro!

    for i, path_csv in enumerate(lista_percorsi_csv):
        idx_frame = i + 1
        distanza_max_current = np.nan

        try:
            # Lettura ottimizzata solo colonne necessarie
            df_curr = pd.read_csv(path_csv, comment='#',
                                  usecols=lambda x: x in ['ID', 'xcentroid', 'ycentroid', 'kron_flux', 'Mag'])

            # Recupero percorso FITS dall'header (veloce)
            header_info = leggi_header_da_csv(path_csv)
            path_fits = header_info.get('PERCORSO_FILE', '')

            if not os.path.exists(path_fits):
                lista_risultati.append({'Dist_Max_Pixel': np.nan})
                continue

            # Check veloce se la stella esiste nel CSV prima di aprire il FITS pesante
            row_stella = df_curr[df_curr['ID'] == id_stella_target]
            if len(row_stella) == 0:
                lista_risultati.append({'Dist_Max_Pixel': np.nan})
                continue

            cur_x = row_stella.iloc[0]['xcentroid']
            cur_y = row_stella.iloc[0]['ycentroid']

            # Apertura FITS ottimizzata
            with fits.open(path_fits, memmap=False) as hdu_list:
                full_data = hdu_list[0].data
                # Statistiche veloci (subsampling)
                _, _, std_globale = sigma_clipped_stats(full_data[::10, ::10], sigma=3.0)
                median_val = np.nanmedian(full_data[::10, ::10])

            # Cutout e sottrazione
            y_min = int(max(0, cur_y - final_pad))
            y_max = int(min(full_data.shape[0], cur_y + final_pad))
            x_min = int(max(0, cur_x - final_pad))
            x_max = int(min(full_data.shape[1], cur_x + final_pad))

            cutout_data = full_data[y_min:y_max, x_min:x_max] - median_val

            # Segmentazione
            tbl_seg, _, seg_map = analisi_image_segmentation(cutout_data, std_esterna=std_globale,
                                                             parametri_esterni=parametri_seg)

            # Calcolo Distanza Max
            tgt_x_loc = cur_x - x_min
            tgt_y_loc = cur_y - y_min

            if seg_map is not None and tbl_seg is not None and len(tbl_seg) > 0:
                dists = np.sqrt((tbl_seg['xcentroid'] - tgt_x_loc) ** 2 + (tbl_seg['ycentroid'] - tgt_y_loc) ** 2)
                idx_best = np.argmin(dists)

                if dists[idx_best] < 10:
                    label_target = tbl_seg[idx_best]['label']
                    cy, cx = np.where(seg_map.data == label_target)
                    if len(cx) > 0:
                        pix_dists = np.sqrt((cx - tgt_x_loc) ** 2 + (cy - tgt_y_loc) ** 2)
                        distanza_max_current = np.max(pix_dists)

        except Exception:
            pass  # Skip silenzioso errori frame singolo

        lista_risultati.append({'Dist_Max_Pixel': distanza_max_current})

    return lista_risultati


# --- PARAMETRI E SETUP ---
run = 1
KRON_TARGET = 1000
INDICE_IMMAGINE_RIFERIMENTO = 35

base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

# Recupero tabella di riferimento
path_ref = lista_percorsi_csv[INDICE_IMMAGINE_RIFERIMENTO]
df_ref = pd.read_csv(path_ref, comment='#')

# Filtro 'SI'
mask_corr = df_ref['Corrispondenza'].astype(str).str.startswith('SI')
df_ref = df_ref[mask_corr]

# Raggruppamento (Identificazione Stelle Uniche)
df_raggruppato = df_ref.groupby('label').agg(
    ID=('ID', 'first'),
    Mag_Integrata=('Mag', somma_magnitudini),
    area=('area', 'first'),
    # Aggiungi altre colonne se servono per analisi_stella
).reset_index()

tbl_raggruppato = Table.from_pandas(df_raggruppato)

print(f"Stelle uniche da analizzare: {len(tbl_raggruppato)}")
print("Avvio elaborazione...")

risultati_aggregati = []

# --- CICLO CON BARRA DI PROGRESSO (TQDM) ---
# tqdm avvolge l'iterabile tbl_raggruppato
for row in tqdm(tbl_raggruppato, desc="Analisi Stelle", unit="stella"):

    lista_risultati = analisi_stella(row)

    if not lista_risultati:
        continue

    # Estrazione distanze
    array_distanze_massime = np.array([item.get('Dist_Max_Pixel', np.nan) for item in lista_risultati], dtype=float)
    valori_validi = np.count_nonzero(~np.isnan(array_distanze_massime))

    if valori_validi >= 25:
        mag = row['Mag_Integrata']
        risultati_aggregati.append((
            mag,
            np.nanmax(array_distanze_massime),
            np.nanmin(array_distanze_massime),
            np.nanmean(array_distanze_massime),
            np.nanstd(array_distanze_massime)
        ))

# --- SALVATAGGIO ---
if risultati_aggregati:
    magnitudine, d_max, d_min, d_mean, d_std = map(list, zip(*risultati_aggregati))

    df_export = pd.DataFrame({
        'Magnitudine': magnitudine,
        'Distanza_max': d_max,
        'Distanza_min': d_min,
        'Distanza_media': d_mean,
        'Distanza_std': d_std
    })

    df_export.sort_values(by='Magnitudine', ascending=True, inplace=True)
    df_export.insert(0, 'label', range(1, len(df_export) + 1))

    nome_file_csv = 'statistiche_distanze_ordinate.csv'
    df_export.to_csv(nome_file_csv, index=False)
    print(f"\nSalvataggio completato: {len(df_export)} stelle valide.")
    print(df_export.head())
else:
    print("\nNessuna stella valida trovata con > 25 rilevazioni.")