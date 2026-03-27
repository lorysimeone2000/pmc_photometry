import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.optimize import curve_fit
import warnings
from pathlib import Path
from tqdm import tqdm
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import FITSFixedWarning
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


def trova_cartella_base(nome_target="pmc_photometry"):
    # cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")

PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

from funzioni.utilita import *
from funzioni.astrometria import *


# =============================================================================
# 0. FUNZIONI DI UTILITÀ E CONFIGURAZIONE
# =============================================================================


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # cerco la cartella specificata nel percorso del progetto
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


def modello_lineare(mag, m, q):
    # definisco il mio modello: log10(Flux) = m * Mag + q
    return m * mag + q


def ottieni_magnitudini_vizier_con_coordinate(ra, dec, id_stella, raggio_arcsec=2.0):
    """
    Ottiene le magnitudini da VizieR usando le coordinate celesti.
    """
    # dizionario di default con tutti i valori NaN
    dati_stella = {
        'ID': id_stella,
        'gmag': np.nan,
        'rmag': np.nan,
        'imag': np.nan,
        'zmag': np.nan,
        'ymag': np.nan
    }

    try:
        # configuro Vizier per il catalogo Pan-STARRS
        vizier_ps1 = Vizier(
            catalog="II/389/ps1_dr2",
            columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
            row_limit=-1
        )

        # cerco nella regione intorno alle coordinate
        result = vizier_ps1.query_region(
            SkyCoord(ra=ra, dec=dec, unit=u.deg),
            radius=raggio_arcsec * u.arcsec
        )

        if len(result) > 0:
            tabella_res = result[0]

            # converto l'ID in intero per confronto
            try:
                id_intero = int(float(id_stella))
            except (ValueError, TypeError):
                id_intero = None

            # cerco la riga con l'ID corrispondente
            if id_intero is not None and 'objID' in tabella_res.colnames:
                for riga in tabella_res:
                    try:
                        if int(riga['objID']) == id_intero:
                            for banda in ['gmag', 'rmag', 'imag', 'zmag', 'ymag']:
                                if banda in riga.colnames and not pd.isna(riga[banda]):
                                    dati_stella[banda] = riga[banda]
                            return dati_stella
                    except (ValueError, TypeError):
                        continue

            # Se non trovo per ID, prendo la stella più vicina
            riga = tabella_res[0]
            for banda in ['gmag', 'rmag', 'imag', 'zmag', 'ymag']:
                if banda in riga.colnames and not pd.isna(riga[banda]):
                    dati_stella[banda] = riga[banda]

    except Exception as e:
        print(f"  Attenzione: Errore per stella {id_stella} a coordinate ({ra:.6f}, {dec:.6f}): {e}")

    return dati_stella


# configuro le mie impostazioni di base
RUN_TO_ANALYZE = [1, 2, 3]

# definisco il flusso esatto che voglio analizzare applicando la correzione additiva e la decorrelazione globale
FLUSSI_DA_ANALIZZARE = [
    "flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE"
]

# =============================================================================
# 1. CARICAMENTO DATI SINTESI FLIR
# =============================================================================

lista_dfs_sintesi = []

cols_needed_sintesi = ['label', 'ID', 'Corrispondenza', 'Mag', 'saturazione', 'RAJ2000', 'DEJ2000', 'RA_centroid',
                       'DEC_centroid', 'Catalogo']
for flusso in FLUSSI_DA_ANALIZZARE:
    cols_needed_sintesi.extend([f"media_{flusso}", f"std_{flusso}"])

print("=== CARICAMENTO DATI SINTESI FLIR (tabelle) ===")
for run in RUN_TO_ANALYZE:
    nome_cartella = f"tabelle_unite_run_{run}"
    path_cartella = cerca_cartella_nel_progetto(BASE_DIR / "tabelle", nome_cartella)

    if path_cartella is None:
        print(f"Attenzione: Cartella {nome_cartella} non trovata.")
        continue
    else:
        print(f"cartella trovata in {path_cartella}")

    files_csv = sorted(list(path_cartella.glob("*.csv")))
    print(f"Run {run}: Trovati {len(files_csv)} file. Caricamento in corso...")

    for f in tqdm(files_csv, leave=False):
        try:
            df_temp = pd.read_csv(f, comment='#')
            cols_disponibili = [c for c in cols_needed_sintesi if c in df_temp.columns]
            df_temp = df_temp[cols_disponibili]
            df_temp['run_origin'] = run
            lista_dfs_sintesi.append(df_temp)
        except Exception as e:
            pass

if not lista_dfs_sintesi:
    print("ERRORE: Nessun dato caricato per la sintesi.")
    exit()

df_sintesi = pd.concat(lista_dfs_sintesi, ignore_index=True)
print(f"Totale righe sintesi caricate: {len(df_sintesi)}")

# =============================================================================
# 2. CARICAMENTO DATI GMAG PURA
# =============================================================================

lista_dfs_gmag = []

cols_needed_gmag = ['label', 'ID', 'Corrispondenza', 'Mag', 'saturazione', 'RAJ2000', 'DEJ2000', 'RA_centroid',
                    'DEC_centroid', 'Catalogo']
for flusso in FLUSSI_DA_ANALIZZARE:
    cols_needed_gmag.extend([f"media_{flusso}", f"std_{flusso}"])

print("\n=== CARICAMENTO DATI GMAG PURA (tabelle_gmag) ===")
for run in RUN_TO_ANALYZE:
    nome_cartella = f"tabelle_unite_run_{run}"
    path_cartella = cerca_cartella_nel_progetto(BASE_DIR / "tabelle_gmag", nome_cartella)

    if path_cartella is None:
        print(f"Attenzione: Cartella {nome_cartella} non trovata.")
        continue
    else:
        print(f"cartella trovata in {path_cartella}")

    files_csv = sorted(list(path_cartella.glob("*.csv")))
    print(f"Run {run}: Trovati {len(files_csv)} file. Caricamento in corso...")

    for f in tqdm(files_csv, leave=False):
        try:
            df_temp = pd.read_csv(f, comment='#')
            cols_disponibili = [c for c in cols_needed_gmag if c in df_temp.columns]
            df_temp = df_temp[cols_disponibili]
            df_temp['run_origin'] = run
            lista_dfs_gmag.append(df_temp)
        except Exception as e:
            pass

if not lista_dfs_gmag:
    print("ERRORE: Nessun dato caricato per gmag.")
    exit()

df_gmag = pd.concat(lista_dfs_gmag, ignore_index=True)
print(f"Totale righe gmag caricate: {len(df_gmag)}")

# =============================================================================
# 3. PREPARAZIONE DATI PER LA LOGICA ORIGINALE (24 RIGHE)
# =============================================================================

# deduplico i dati per entrambi i dataset
df_sintesi_unique = df_sintesi.drop_duplicates(subset=['label'], keep='first').copy()
df_gmag_unique = df_gmag.drop_duplicates(subset=['label'], keep='first').copy()

print(f"\nOggetti UNICI sintesi: {len(df_sintesi_unique)}")
print(f"Oggetti UNICI gmag: {len(df_gmag_unique)}")

# separo matchati e non matchati per la sintesi
mask_match_sintesi = df_sintesi_unique['Corrispondenza'].astype(str).str.startswith('SI', na=False)
df_match_sintesi = df_sintesi_unique[mask_match_sintesi].copy()
df_no_match_sintesi = df_sintesi_unique[~mask_match_sintesi].copy()

# separo matchati e non matchati per gmag
mask_match_gmag = df_gmag_unique['Corrispondenza'].astype(str).str.startswith('SI', na=False)
df_match_gmag = df_gmag_unique[mask_match_gmag].copy()
df_no_match_gmag = df_gmag_unique[~mask_match_gmag].copy()

print(f"\nSintesi - Match: {len(df_match_sintesi)}, No Match: {len(df_no_match_sintesi)}")
print(f"GMAG - Match: {len(df_match_gmag)}, No Match: {len(df_no_match_gmag)}")

# =============================================================================
# 4. LOGICA ORIGINALE: stelle non matchate in gmag ma matchate in sintesi
# =============================================================================

# estraggo le label delle stelle non matchate in gmag e in sintesi
label_no_gmag = set(df_no_match_gmag['label'])
label_no_sintesi = set(df_no_match_sintesi['label'])

# identifico le stelle non matchate in gmag ma matchate in sintesi
label_target = label_no_gmag - label_no_sintesi
print(f"\nStelle target (non matchate in gmag ma matchate in sintesi): {len(label_target)}")

if len(label_target) > 0:
    print(f"\nLista completa delle label delle stelle target:")
    for i, label in enumerate(sorted(list(label_target))):
        print(f"  {i + 1:3d}. {label}")

    # =========================================================================
    # 5. RECUPERO I DATI COMPLETI PER QUESTE STELLE DAL DATASET SINTESI
    # =========================================================================

    print("\n=== RECUPERO DATI COMPLETI DAL DATASET SINTESI ===")

    # filtro il dataframe sintesi per le label target (prendo solo matchate)
    df_target_sintesi = df_match_sintesi[df_match_sintesi['label'].isin(label_target)].copy()

    # mantengo solo la prima occorrenza per ogni label
    df_target_unico = df_target_sintesi.drop_duplicates(subset=['label'], keep='first').copy()

    print(f"\nStelle target uniche recuperate: {len(df_target_unico)}")

    # Mostro la tabella COMPLETA senza salti
    print("\n" + "=" * 100)
    print("TABELLA COMPLETA DELLE STELLE TARGET CON I LORO DATI")
    print("=" * 100)

    # Imposto le opzioni di pandas per mostrare tutte le righe e colonne
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Mostro tutte le colonne disponibili
    colonne_da_mostrare = ['label', 'ID', 'Mag']
    if 'Catalogo' in df_target_unico.columns:
        colonne_da_mostrare.append('Catalogo')
    if 'RAJ2000' in df_target_unico.columns:
        colonne_da_mostrare.append('RAJ2000')
    if 'DEJ2000' in df_target_unico.columns:
        colonne_da_mostrare.append('DEJ2000')
    if 'RA_centroid' in df_target_unico.columns:
        colonne_da_mostrare.append('RA_centroid')
    if 'DEC_centroid' in df_target_unico.columns:
        colonne_da_mostrare.append('DEC_centroid')

    print(df_target_unico[colonne_da_mostrare].to_string(index=True))
    print("=" * 100)

    # =========================================================================
    # 6. QUERY VIZIER CON COORDINATE PER OGNI STELLA TARGET
    # =========================================================================

    print("\n=== RICERCA MAGNITUDINI SU VIZIER PER STELLE TARGET ===")

    risultati_vizier = []

    # imposto Vizier per query coordinate
    Vizier.ROW_LIMIT = -1

    for index, row in tqdm(df_target_unico.iterrows(), total=len(df_target_unico), desc="Query VizieR"):
        id_stella = row['ID']
        label_stella = row['label']

        # CERCO DI RECUPERARE LE COORDINATE (priorità: RAJ2000/DEJ2000, poi RA_centroid/DEC_centroid)
        ra = None
        dec = None

        if 'RAJ2000' in row and pd.notna(row['RAJ2000']):
            try:
                ra = float(row['RAJ2000'])
                dec = float(row['DEJ2000']) if 'DEJ2000' in row and pd.notna(row['DEJ2000']) else None
            except (ValueError, TypeError):
                pass

        if ra is None or dec is None:
            if 'RA_centroid' in row and pd.notna(row['RA_centroid']):
                try:
                    ra = float(row['RA_centroid'])
                    dec = float(row['DEC_centroid']) if 'DEC_centroid' in row and pd.notna(
                        row['DEC_centroid']) else None
                except (ValueError, TypeError):
                    pass

        if ra is None or dec is None:
            print(f"  Attenzione: Coordinate non disponibili per stella {id_stella} (label: {label_stella})")
            risultati_vizier.append({
                'label': label_stella,
                'ID': id_stella,
                'gmag': np.nan,
                'rmag': np.nan,
                'imag': np.nan,
                'zmag': np.nan,
                'ymag': np.nan
            })
            continue

        # ottengo le magnitudini usando le coordinate
        dati_mag = ottieni_magnitudini_vizier_con_coordinate(ra, dec, id_stella, raggio_arcsec=2.0)
        dati_mag['label'] = label_stella
        risultati_vizier.append(dati_mag)

    # =========================================================================
    # 7. CREAZIONE TABELLA FINALE
    # =========================================================================

    # converto i risultati in DataFrame
    df_risultati = pd.DataFrame(risultati_vizier)

    # preparo le colonne per il merge
    colonne_merge = ['label']
    if 'ID' in df_target_unico.columns:
        colonne_merge.append('ID')
    if 'Mag' in df_target_unico.columns:
        colonne_merge.append('Mag')
    if 'Catalogo' in df_target_unico.columns:
        colonne_merge.append('Catalogo')
    if 'RAJ2000' in df_target_unico.columns:
        colonne_merge.append('RAJ2000')
    if 'DEJ2000' in df_target_unico.columns:
        colonne_merge.append('DEJ2000')

    # unisco con i dati originali
    df_finale = df_target_unico[colonne_merge].merge(
        df_risultati[['label', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']],
        on='label',
        how='left'
    )

    # creo la tabella Astropy
    from astropy.table import Table

    # Imposto le opzioni di stampa per Astropy per mostrare tutte le righe
    from astropy.table import conf

    conf.max_lines = -1  # nessun limite di righe
    conf.max_width = -1  # nessun limite di larghezza

    print("\n" + "=" * 100)
    print("STELLE NON MATCHATE IN GMAG MA MATCHATE IN SINTESI - DATI COMPLETI")
    print("=" * 100)

    tabella_astropy_target = Table.from_pandas(df_finale)
    print(tabella_astropy_target)
    print("=" * 100)

    # =========================================================================
    # 8. STATISTICHE RIASSUNTIVE
    # =========================================================================

    print("\n" + "=" * 60)
    print("STATISTICHE RIASSUNTIVE")
    print("=" * 60)

    # conto quante stelle hanno magnitudini valide
    for banda in ['gmag', 'rmag', 'imag', 'zmag', 'ymag']:
        valide = df_finale[banda].notna().sum()
        percentuale = (valide / len(df_finale)) * 100 if len(df_finale) > 0 else 0
        print(f"Stelle con {banda} valida: {valide} / {len(df_finale)} ({percentuale:.1f}%)")

    # Mostro le stelle che hanno gmag valida in formato esteso
    df_con_gmag = df_finale[df_finale['gmag'].notna()]
    if len(df_con_gmag) > 0:
        print(f"\nStelle con gmag valida ({len(df_con_gmag)}):")
        print("=" * 80)
        colonne_output = ['label', 'ID', 'Mag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']
        colonne_output = [c for c in colonne_output if c in df_con_gmag.columns]
        print(df_con_gmag[colonne_output].to_string(index=True))
        print("=" * 80)

    # Mostro anche le stelle senza gmag valida
    df_senza_gmag = df_finale[df_finale['gmag'].isna()]
    if len(df_senza_gmag) > 0:
        print(f"\nStelle SENZA gmag valida ({len(df_senza_gmag)}):")
        print("=" * 80)
        colonne_output = ['label', 'ID', 'Mag']
        colonne_output = [c for c in colonne_output if c in df_senza_gmag.columns]
        print(df_senza_gmag[colonne_output].to_string(index=True))
        print("=" * 80)

    # Salvo i risultati in un file CSV
    output_file = BASE_DIR / "tabelle" / "stelle_non_matchate_gmag_con_magnitudini.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_finale.to_csv(output_file, index=False)
    print(f"\nRisultati salvati in: {output_file}")

    # Salvo anche una versione con tutte le coordinate
    output_file_coord = BASE_DIR / "tabelle" / "stelle_target_con_coordinate.csv"
    df_target_unico.to_csv(output_file_coord, index=False)
    print(f"Coordinate stelle target salvate in: {output_file_coord}")

else:
    print("\nNessuna stella rispetta i criteri richiesti.")

print("\n=== ELABORAZIONE COMPLETATA ===")