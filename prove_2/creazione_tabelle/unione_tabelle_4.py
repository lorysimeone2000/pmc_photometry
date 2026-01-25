import pandas as pd
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from pathlib import Path
import warnings
from astropy.wcs import FITSFixedWarning

# Sopprimo warning non critici
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
            if line.startswith('#'):
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            else:
                break
    return header_dict


def salva_csv_con_header_fits(dataframe, header_fits, filename):
    with open(filename, 'w') as f:
        f.write("# Header FITS:\n")
        f.write(
            "# DESCRIZIONE: Questo file csv contiene la tabella di tutte le sorgenti trovate con image segmentation insieme alle informazioni dell'eventuale stella corrispondente dei cataloghi (Rank 1, 2, 3 per luminosita')\n")
        for key, value in header_fits.items():
            f.write(f"# {key}: {value}\n")
        f.write("#\n")
        dataframe.to_csv(f, index=False)


# --- INPUT UTENTE E INIZIALIZZAZIONE ---

try:
    run = int(input("Quale run vuoi elaborare: "))
except ValueError:
    print("Input non valido.")
    exit()

cartella_base = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle"
cartella_csv = os.path.join(cartella_base, f"sorgenti_catalogate_run/sorgenti_catalogate_run_{run}")
cartella_csv_ = os.path.join(cartella_base, f"sorgenti_trovate_run/sorgenti_trovate_run_{run}")
output_dir = os.path.join(cartella_base, f"tabelle_unite/tabelle_unite_run_{run}")

Path(output_dir).mkdir(parents=True, exist_ok=True)

file_csv_catalogate = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv_stelle_catalogate = [os.path.join(cartella_csv, file) for file in file_csv_catalogate]

file_csv_trovate = sorted([f for f in os.listdir(cartella_csv_) if f.endswith('.csv')])
lista_percorsi_csv_stelle_trovate = [os.path.join(cartella_csv_, file) for file in file_csv_trovate]

if len(lista_percorsi_csv_stelle_trovate) != len(lista_percorsi_csv_stelle_catalogate):
    print("ATTENZIONE: Il numero di file trovati e catalogati non corrisponde.")
    exit()

soglia_correlazione = 0.003349 * u.deg

# --- LOOP PRINCIPALE OTTIMIZZATO ---

for n in range(len(lista_percorsi_csv_stelle_catalogate)):
    percorso_csv_stelle_trovate = lista_percorsi_csv_stelle_trovate[n]
    percorso_csv_stelle_catalogate = lista_percorsi_csv_stelle_catalogate[n]

    # 1. Lettura dati e Header
    header_dal_csv = leggi_header_da_csv(percorso_csv_stelle_trovate)
    percorso_file_fits = header_dal_csv.get('PERCORSO_FILE', '')
    print(f"\nElaborando file {n + 1} di {len(lista_percorsi_csv_stelle_catalogate)}:")
    print(percorso_file_fits)

    df_trovate = pd.read_csv(percorso_csv_stelle_trovate, comment='#')
    df_catalogate = pd.read_csv(percorso_csv_stelle_catalogate, comment='#')

    # --- LOGICA DI SELEZIONE COLONNE DINAMICA (Corretta e Completa) ---
    all_cols = df_trovate.columns.tolist()
    cols_base = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']

    try:
        # Cerca l'indice di 'saturazione'
        if 'saturazione' in all_cols:
            idx_satura = all_cols.index('saturazione')
            cols_extra = all_cols[idx_satura:]  # Prende Satura e TUTTO ciò che segue
        else:
            # Se manca Satura, prova a cercare da kron_flux
            idx_start = all_cols.index('kron_flux') if 'kron_flux' in all_cols else len(all_cols)
            cols_extra = all_cols[idx_start:]

        cols_dinamiche = []

        # 1. Aggiungi Satura (se c'è)
        if 'saturazione' in cols_extra:
            cols_dinamiche.append('saturazione')

        # 2. Aggiungi kron_flux (se c'è) subito dopo
        if 'kron_flux' in all_cols:
            cols_dinamiche.append('kron_flux')

        # 3. Aggiungi TUTTO il resto che era in cols_extra
        for c in cols_extra:
            if c not in cols_dinamiche and c not in cols_base:
                cols_dinamiche.append(c)

    except ValueError:
        # Fallback totale se qualcosa va storto con gli indici
        cols_dinamiche = []
        if 'kron_flux' in all_cols: cols_dinamiche.append('kron_flux')

    cols_finali = cols_base + cols_dinamiche

    # Filtriamo il dataframe mantenendo solo le colonne desiderate nell'ordine corretto
    # Usiamo intersection per evitare errori se per caso una colonna base manca
    cols_presenti = [c for c in cols_finali if c in df_trovate.columns]
    df_trovate = df_trovate[cols_presenti]
    # -----------------------------------------------------------

    # Calcolo coordinate celesti (Vettoriale con WCS)
    try:
        # Usa memmap=False per sicurezza con header complessi
        with fits.open(percorso_file_fits, memmap=False) as hdu_list:
            w = WCS(hdu_list[0].header)
        coords_trovate = w.pixel_to_world(df_trovate['xcentroid'], df_trovate['ycentroid'])

        # Aggiungi le nuove colonne al dataframe filtrato
        # Nota: Pandas avvisa se modifichi una slice, usiamo .copy() implicito o loc se serve,
        # ma qui df_trovate è già un nuovo oggetto dopo il filtro colonne
        df_trovate = df_trovate.copy()
        df_trovate['RA_centroid'] = coords_trovate.ra.deg
        df_trovate['DEC_centroid'] = coords_trovate.dec.deg

    except Exception as e:
        print(f"Errore WCS/FITS: {e}")
        continue

    # Preparazione coordinate catalogate
    if 'RAJ2000' in df_catalogate.columns:
        coords_catalogate = SkyCoord(ra=df_catalogate['RAJ2000'].values * u.deg,
                                     dec=df_catalogate['DEJ2000'].values * u.deg)
    else:
        continue

    print(f"Cercata correlazione in {len(coords_trovate)} stelle")

    # --- 2. MATCHING VELOCE (search_around_sky) ---
    idx_trovate, idx_catalogate, d2d, _ = coords_catalogate.search_around_sky(coords_trovate, soglia_correlazione)

    matches = pd.DataFrame({
        'idx_t': idx_trovate,
        'idx_c': idx_catalogate,
        'dist': d2d.deg,
        'mag': df_catalogate.iloc[idx_catalogate]['Mag'].values
    })

    # --- 3. LOGICA RANK ---
    matches.sort_values(by=['idx_t', 'mag'], inplace=True)
    matches['rank'] = matches.groupby('idx_t').cumcount() + 1
    matches['Corrispondenza'] = 'SI (Rank ' + matches['rank'].astype(str) + ')'

    # --- 4. COSTRUZIONE TABELLA FINALE ---

    # A. Match SI
    part_trovate = df_trovate.iloc[matches['idx_t']].reset_index(drop=True)
    part_catalogate = df_catalogate.iloc[matches['idx_c']].reset_index(drop=True)
    part_rank = matches[['Corrispondenza']].reset_index(drop=True)

    df_si = pd.concat([part_trovate, part_rank, part_catalogate], axis=1)

    # B. Match NO
    all_indices = set(range(len(df_trovate)))
    matched_indices = set(matches['idx_t'].unique())
    unmatched_indices = list(all_indices - matched_indices)

    if unmatched_indices:
        df_no = df_trovate.iloc[unmatched_indices].copy()
        df_no['Corrispondenza'] = 'NO'

        for col in df_catalogate.columns:
            if col == 'Catalogo':
                df_no[col] = 'N/A'
            elif pd.api.types.is_integer_dtype(df_catalogate[col]):
                df_no[col] = -999
            elif pd.api.types.is_float_dtype(df_catalogate[col]):
                df_no[col] = np.nan
            else:
                df_no[col] = 'N/A'

        # Allinea colonne
        # Assicuriamoci che df_no abbia tutte le colonne di df_si nello stesso ordine
        # Le colonne mancanti (quelle del catalogo) sono state appena create, quindi dovrebbe combaciare
        df_no = df_no.reindex(columns=df_si.columns, fill_value=np.nan)

    else:
        df_no = pd.DataFrame(columns=df_si.columns)

    # C. Unione
    df_finale = pd.concat([df_si, df_no], ignore_index=True)

    if 'label' in df_finale.columns:
        df_finale.sort_values('label', inplace=True)

    # --- 5. RIORDINAMENTO FINALE (Catalogo prima di ID) ---
    colonne = df_finale.columns.tolist()
    if 'ID' in colonne and 'Catalogo' in colonne:
        colonne.remove('Catalogo')
        pos_id = colonne.index('ID')
        colonne.insert(pos_id, 'Catalogo')
        df_finale = df_finale[colonne]

    # Salvataggio
    filename = os.path.join(output_dir, f'run_{run}_stelle_trovate_e_catalogate_immagine_{n + 1:03d}.csv')
    salva_csv_con_header_fits(df_finale, header_dal_csv, filename)

print("Elaborazione completata.")