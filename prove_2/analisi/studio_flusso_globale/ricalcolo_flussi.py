import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from photutils.aperture import CircularAperture


# Cerco la mia cartella base del progetto
def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent


# Imposto la mia directory base e le cartelle di lavoro
BASE_DIR = trova_cartella_base("Lorenzo")
cartella_tabelle_unite = BASE_DIR / "tabelle" / "tabelle_unite"
percorso_somma_pixel = BASE_DIR / "risultati_somma_pixel.csv"

# Imposto il mio numero totale di pixel della mia immagine
N_tot = 3072 * 2048

# Costante k per la mia correzione additiva dell'apertura
k_additivo = 1

# Leggo il mio file contenente i risultati della somma dei pixel
df_somma = pd.read_csv(percorso_somma_pixel)
col_run = 'Run' if 'Run' in df_somma.columns else df_somma.columns[0]
col_img = 'Immagine' if 'Immagine' in df_somma.columns else df_somma.columns[1]
col_somma = 'Somma' if 'Somma' in df_somma.columns else df_somma.columns[-1]

# Estraggo il mio valore S_ref stabile di riferimento (Run 1, Immagine 35)
S_ref = df_somma[(df_somma[col_run] == 1) & (df_somma[col_img] == 35)][col_somma].values[0]

# Trovo tutti i miei file CSV generati
lista_file_csv = list(cartella_tabelle_unite.rglob("run_*_stelle_trovate_e_catalogate_immagine_*.csv"))

# Identifico le mie run univoche da elaborare
run_disponibili = set()
for file_csv in lista_file_csv:
    match = re.search(r'run_(\d+).*immagine_(\d+)', file_csv.name)
    if match:
        run_disponibili.add(int(match.group(1)))

# Processo ogni mia run copiando esattamente la tua logica di estrazione dei raggi
for run in run_disponibili:
    # Isolo i miei file della run corrente
    file_csv_run = [f for f in lista_file_csv if re.search(f'run_{run}.*immagine', f.name)]
    file_csv_run = sorted(file_csv_run)

    # =========================================================================
    # Ricopio il mio metodo esatto per estrarre il 95° percentile globale
    # =========================================================================
    all_ids = []
    all_radii = []

    for file_csv in file_csv_run:
        try:
            df_temp = pd.read_csv(file_csv, comment='#', usecols=['ID', 'raggio_kron_aper'])
            df_temp = df_temp.dropna(subset=['ID', 'raggio_kron_aper'])
            all_ids.append(df_temp['ID'].values)
            all_radii.append(df_temp['raggio_kron_aper'].values)
        except Exception:
            pass

    if len(all_ids) > 0:
        big_ids = np.concatenate(all_ids)
        big_radii = np.concatenate(all_radii)
        df_global = pd.DataFrame({'ID': big_ids, 'R': big_radii})
        map_raggi_max = df_global.groupby('ID')['R'].quantile(0.95).to_dict()
    else:
        map_raggi_max = {}

    # =========================================================================
    # Ricalcolo le mie aree usando CircularAperture e applico le correzioni
    # =========================================================================
    for file_csv in file_csv_run:
        match = re.search(r'run_(\d+).*immagine_(\d+)', file_csv.name)
        img_corrente = int(match.group(2))

        # Trovo il mio S(t) per l'immagine corrente
        s_t_riga = df_somma[(df_somma[col_run] == run) & (df_somma[col_img] == img_corrente)]
        if s_t_riga.empty:
            continue
        S_t = s_t_riga[col_somma].values[0]

        # Leggo il mio header commentato
        header_lines = []
        with open(file_csv, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    header_lines.append(line)
                else:
                    break

        # Leggo il mio dataframe
        df = pd.read_csv(file_csv, comment='#')
        colonne_attuali = df.columns.tolist()

        ids_presenti = df['ID'].values

        # Preparo i miei array di aree per ogni singola stella
        aree_calcolate = {
            'kron_manuale_aper': [],
            'somma_apertura_ultimo_pixel': [],
            'flusso_fisso_max_run': [],
            'flusso_raggio_fisso_doppio': [],
            'kron_flux': []
        }

        for idx_star, star_id in enumerate(ids_presenti):

            # 1. Ricopio la mia logica esatta per ottenere il raggio globale
            r_globale = map_raggi_max.get(star_id, np.nan)
            if np.isnan(r_globale) or r_globale <= 0:
                if 'raggio_kron_aper' in df.columns:
                    r_globale = df.at[idx_star, 'raggio_kron_aper']
                else:
                    r_globale = np.nan

            # 2. Prendo il mio raggio kron locale della singola immagine
            r_locale = df.at[idx_star, 'raggio_kron_aper'] if 'raggio_kron_aper' in df.columns else np.nan

            # 3. Uso esattamente il metodo di photutils (CircularAperture.area)
            if pd.notnull(r_globale) and r_globale > 0:
                area_fisso = CircularAperture((0, 0), r=r_globale).area
                area_doppio = CircularAperture((0, 0), r=r_globale * 2).area
            else:
                area_fisso = np.nan
                area_doppio = np.nan

            if pd.notnull(r_locale) and r_locale > 0:
                area_locale = CircularAperture((0, 0), r=r_locale).area
            else:
                area_locale = np.nan

            aree_calcolate['flusso_fisso_max_run'].append(area_fisso)
            aree_calcolate['flusso_raggio_fisso_doppio'].append(area_doppio)
            aree_calcolate['kron_manuale_aper'].append(area_locale)
            aree_calcolate['somma_apertura_ultimo_pixel'].append(area_locale)
            aree_calcolate['kron_flux'].append(area_locale)

        # Popolo le mie aree di segmentazione prendendole direttamente dal valore esatto dei pixel (area)
        if 'area' in df.columns:
            aree_calcolate['kron_manuale_seg'] = df['area'].values
            aree_calcolate['flusso_intera_segmentazione'] = df['area'].values
            aree_calcolate['flusso_kron_intera_segmentazione'] = df['area'].values

        # Eseguo le mie correzioni matematiche aggiungendo le nuove colonne
        for flusso, array_aree in aree_calcolate.items():
            if flusso in df.columns:
                array_aree = np.array(array_aree)

                # Normalizzazione Moltiplicativa
                nome_colonna_molt = f"{flusso}_CORRETTO_Normalizzazione_Moltiplicativa"
                df[nome_colonna_molt] = df[flusso] * (S_ref / S_t)

                # Correzione Additiva dell'Apertura
                nome_colonna_add = f"{flusso}_CORRETTO_Correzione_Additiva_dell_Apertura"
                df[nome_colonna_add] = df[flusso] - (k_additivo * array_aree * (S_t / N_tot))

                # Riorganizzo le mie colonne per inserirle di fianco al flusso originale
                if nome_colonna_molt not in colonne_attuali:
                    idx_flusso = colonne_attuali.index(flusso)
                    colonne_attuali.insert(idx_flusso + 1, nome_colonna_molt)
                    colonne_attuali.insert(idx_flusso + 2, nome_colonna_add)

        # Riassegno il mio DataFrame sovrascrivendo l'originale
        df = df[colonne_attuali]
        with open(file_csv, 'w') as f:
            for line in header_lines:
                f.write(line)
            df.to_csv(f, index=False)

        print(f"Completato: {file_csv.name}")