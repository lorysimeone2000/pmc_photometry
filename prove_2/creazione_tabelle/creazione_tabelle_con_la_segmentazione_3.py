import pandas as pd
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
from photutils.aperture import aperture_photometry, CircularAperture
from astropy.stats import sigma_clipped_stats
from tqdm import tqdm
import warnings
from astropy.wcs import FITSFixedWarning

# Soppressione warning
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CONFIGURAZIONE ---
run = 1
base_input_path = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}"
# Se vuoi sovrascrivere i file originali, metti uguale a base_input_path
output_path = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite/tabelle_unite_run_{run}_aggiornate"

if not os.path.exists(output_path):
    os.makedirs(output_path)


# --- FUNZIONI HELPER ---
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
    return valore


def leggi_header_da_csv(filename):
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            else:
                break
    return header_dict


def salva_csv_con_header(df, header_dict, output_file):
    """Salva il DataFrame mantenendo l'header originale."""
    with open(output_file, 'w') as f:
        f.write("# Header FITS (Aggiornato):\n")
        for k, v in header_dict.items():
            f.write(f"# {k}: {v}\n")
        f.write("#\n")
        df.to_csv(f, index=False)


# =============================================================================
# FASE 1: CALCOLO DEL RAGGIO MASSIMO PER OGNI STELLA SU TUTTA LA RUN
# =============================================================================
print("--- FASE 1: Mappatura Raggi Massimi Globali ---")

file_csv_list = sorted([os.path.join(base_input_path, f) for f in os.listdir(base_input_path) if f.endswith('.csv')])

# Liste per accumulare i dati necessari al calcolo del max
all_ids = []
all_radii = []

for file_csv in tqdm(file_csv_list, desc="Scansione Raggi"):
    # Leggiamo solo ID e raggio per velocità
    try:
        df_temp = pd.read_csv(file_csv, comment='#', usecols=['ID', 'raggio_kron_aper'])
        # Filtriamo eventuali NaN o ID non validi
        df_temp = df_temp.dropna(subset=['ID', 'raggio_kron_aper'])
        all_ids.append(df_temp['ID'].values)
        all_radii.append(df_temp['raggio_kron_aper'].values)
    except Exception as e:
        print(f"Skipping {os.path.basename(file_csv)}: {e}")

# Concatenazione veloce
if len(all_ids) > 0:
    big_ids = np.concatenate(all_ids)
    big_radii = np.concatenate(all_radii)

    # Creiamo un DataFrame temporaneo globale
    df_global = pd.DataFrame({'ID': big_ids, 'R': big_radii})

    # Raggruppiamo per ID e prendiamo il MASSIMO raggio visto nella run
    # Aggiungiamo un piccolo buffer di sicurezza? (Es. 0.5 pixel) Facoltativo.
    # Qui prendiamo il massimo puro.
    # Invece di .max(), usa quantile
    map_raggi_max = df_global.groupby('ID')['R'].quantile(0.95).to_dict()

    print(f"\nMappate {len(map_raggi_max)} stelle uniche.")
else:
    print("Nessun dato trovato.")
    exit()

# =============================================================================
# FASE 2: RICALCOLO FLUSSI CON APERTURA FISSA (SPECIFICA PER STELLA)
# =============================================================================

print("\n--- FASE 2: Fotometria con Apertura Fissa (Max Run) ---")

for file_csv in tqdm(file_csv_list, desc="Ricalcolo Flussi"):

    # 1. Caricamento Dati
    df_frame = pd.read_csv(file_csv, comment='#')
    header_info = leggi_header_da_csv(file_csv)
    path_fits = header_info.get('PERCORSO_FILE', '')

    if not os.path.exists(path_fits):
        print(f"Warning: FITS non trovato: {path_fits}")
        continue

    # 2. Caricamento Immagine
    with fits.open(path_fits, memmap=False) as hdu:
        data = hdu[0].data
        _, median_bg, _ = sigma_clipped_stats(data[::10, ::10], sigma=3.0)
        data_sub = data - median_bg

    # 3. Preparazione Raggi
    raggi_fissi = []
    ids_presenti = df_frame['ID'].values

    # 4. Esecuzione Fotometria (CICLO SICURO PER EVITARE ERRORI VETTORIALI)
    # CircularAperture con array può fallire se ci sono NaN o versioni vecchie.
    flussi_calcolati = []

    for i, star_id in enumerate(ids_presenti):
        # A. Recupero Raggio
        r_globale = map_raggi_max.get(star_id, np.nan)

        if np.isnan(r_globale) or r_globale <= 0:
            # Fallback locale
            idx = df_frame.index[i]
            r_globale = df_frame.at[idx, 'raggio_kron_aper']

        # Salvataggio raggio usato
        raggi_fissi.append(r_globale)

        # B. Calcolo Fotometria Singola
        if r_globale > 0 and not np.isnan(r_globale):
            # Posizione singola
            pos = (df_frame.at[i, 'xcentroid'], df_frame.at[i, 'ycentroid'])
            aper = CircularAperture(pos, r=r_globale)

            # Aperture photometry su singola stella
            phot = aperture_photometry(data_sub, aper)
            flusso = phot['aperture_sum'][0]
            flussi_calcolati.append(flusso)
        else:
            flussi_calcolati.append(np.nan)

    # 5. Aggiornamento DataFrame
    df_frame['kron_fisso_max_run'] = flussi_calcolati
    df_frame['raggio_fisso_max_run'] = raggi_fissi

    # Formattazione
    df_frame['kron_fisso_max_run'] = df_frame['kron_fisso_max_run'].map(
        lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')
    df_frame['raggio_fisso_max_run'] = df_frame['raggio_fisso_max_run'].map(
        lambda x: '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')

    # 6. Salvataggio
    nome_file = os.path.basename(file_csv)
    path_out = os.path.join(output_path, nome_file)
    salva_csv_con_header(df_frame, header_info, path_out)

print(f"\nElaborazione completata. File salvati in: {output_path}")