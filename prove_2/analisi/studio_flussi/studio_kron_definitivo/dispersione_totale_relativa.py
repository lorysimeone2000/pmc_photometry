import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path
from scipy.stats import norm
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import FITSFixedWarning
from scipy.stats import norm, chisquare

# gestisco i warning ignorandoli
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


def trova_cartella_base(nome_target="pmc_photometry"):
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
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


# configuro le mie impostazioni di base
RUN_TO_ANALYZE = [1, 2, 3]

# definisco il flusso esatto che voglio analizzare applicando la correzione additiva e la decorrelazione globale
FLUSSI_DA_ANALIZZARE = [
    "flusso_fisso_max_run_senza_correzioni",
    "flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE"
]

# =============================================================================
# 1. CARICAMENTO DATI (TUTTE LE RUN)
# =============================================================================

print(f"--- Caricamento dati per Fit Globale ---")
lista_dfs = []
t0_global = None

# aggiungo alla mia lista tutte le colonne dei flussi che mi servono successivamente per il filtro, inclusa la colonna base
cols_needed = ['label', 'ID', 'Corrispondenza', 'Mag', 'saturazione']
for flusso in FLUSSI_DA_ANALIZZARE:
    cols_needed.extend([flusso, f"media_{flusso}", f"std_{flusso}"])

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
            df_temp = pd.read_csv(f, comment='#', usecols=lambda c: c in cols_needed)
            df_temp['run_origin'] = run

            # estraggo il nome del file per poter identificare il singolo scatto
            df_temp['file_name'] = f.name

            # leggo l'header per calcolare il tempo relativo
            header_dal_csv = leggi_header_da_csv(f)
            t_curr = header_dal_csv.get('TSTART', 0)

            if t0_global is None:
                t0_global = t_curr

            df_temp['tempo_relativo'] = (t_curr - t0_global) / 1000.0 if t0_global is not None else 0

            lista_dfs.append(df_temp)
        except Exception as e:
            pass

if not lista_dfs:
    print("ERRORE: Nessun dato caricato.")
    exit()

df_total = pd.concat(lista_dfs, ignore_index=True)
print(f"Totale righe caricate: {len(df_total)}")

# =============================================================================
# 2. CALCOLO DELLE VARIAZIONI RELATIVE
# =============================================================================

# preparo un dizionario in cui memorizzo l'array delle variazioni per ciascun flusso
dati_flussi_istogramma = {}

for flusso in FLUSSI_DA_ANALIZZARE:
    colonna_media = f"media_{flusso}"

    # rimuovo ogni riga priva di valore valido per lo specifico flusso
    df_valid_tmp = df_total.dropna(subset=[flusso, colonna_media])

    # conservo unicamente il dato con media maggiore di zero per evitare la divisione per zero
    df_valid_tmp = df_valid_tmp[df_valid_tmp[colonna_media] > 0].copy()

    # calcolo la variazione relativa
    var_rel = (df_valid_tmp[flusso] - df_valid_tmp[colonna_media]) / df_valid_tmp[colonna_media]

    # scarto l'anomalia estrema per mantenere leggibile il grafico
    mask_outliers = (var_rel > -1) & (var_rel < 1)

    # salvo l'array filtrato nel dizionario
    dati_flussi_istogramma[flusso] = var_rel[mask_outliers].values

    # mantengo la variabile df_valid impostata sull'ultimo flusso elaborato affinché il grafico dell'Opzione 3 continui a funzionare sul flusso corretto
    if flusso == FLUSSI_DA_ANALIZZARE[1]:
        df_valid = df_valid_tmp
        df_valid['variazione_relativa'] = var_rel

# ripristino gli array usati dalle altre sezioni basandomi sull'ultimo flusso analizzato
arr_valid = df_valid['variazione_relativa'].values
runs_valid = df_valid['run_origin'].values

output_dir = "dispersione_relativa"
cartella_corrente = Path.cwd()
nuova_sottocartella = cartella_corrente / "dispersione_relativa"
nuova_sottocartella.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. OPZIONE 1: CREAZIONE ISTOGRAMMI SOVRAPPOSTI
# =============================================================================

plt.figure(figsize=(12, 8))

# definisco il colore chiaro per l'istogramma e quello scuro per il fit
colori_hist = ['dodgerblue', 'tomato']
colori_fit = ['navy', 'darkred']
nomi_legenda = ['Senza Correzioni', 'Corretto']

# inizializzo i limiti dell'asse x per uniformare il fit gaussiano
xmin_globale, xmax_globale = 0, 0

# avvio il ciclo per estrarre ed elaborare il dato di ciascun flusso
for i, flusso in enumerate(FLUSSI_DA_ANALIZZARE):
    dati = dati_flussi_istogramma[flusso]

    # ricavo la media e la deviazione standard sull'intero set di dati
    media_var = np.mean(dati)
    std_var = np.std(dati)

    # isolo i dati nell'intervallo compreso tra -1 e +1 deviazione standard rispetto alla media
    mask_fit = (dati >= media_var - std_var) & (dati <= media_var + std_var)
    dati_per_fit = dati[mask_fit]

    # calcolo i nuovi parametri del fit basandomi solo sui dati ristretti
    mu_fit, sigma_fit = norm.fit(dati_per_fit)

    # disegno le due linee verticali tratteggiate per evidenziare i limiti del fit
    plt.axvline(media_var - std_var, color=colori_fit[i], linestyle='--', alpha=0.4, linewidth=0.8)
    plt.axvline(media_var + std_var, color=colori_fit[i], linestyle='--', alpha=0.4, linewidth=0.8)

    # ottengo il conteggio osservato e definisco il bin sull'intero set di dati
    conteggi, bordi_bin = np.histogram(dati, bins='auto')
    centri_bin = (bordi_bin[:-1] + bordi_bin[1:]) / 2
    larghezza_bin = np.diff(bordi_bin)

    # calcolo il conteggio atteso teorico usando i parametri del fit ristretto
    conteggi_attesi = len(dati) * larghezza_bin * norm.pdf(centri_bin, mu_fit, sigma_fit)

    # applico il test del chi quadro
    mask_chi = conteggi > 0
    osservati = conteggi[mask_chi]
    attesi = conteggi_attesi[mask_chi]
    chi_quadro, p_value = chisquare(f_obs=osservati, f_exp=attesi, ddof=2)

    # ricavo il grado di libertà e il chi quadro ridotto
    gradi_liberta = len(osservati) - 3
    chi_quadro_ridotto = chi_quadro / gradi_liberta if gradi_liberta > 0 else np.nan

    # genero l'istogramma a gradini per impedire che le barre piene si coprano a vicenda senza l'uso di trasparenze
    plt.hist(dati, bins='auto', density=True, histtype='step', linewidth=1, alpha=1.0,
             color=colori_hist[i], label=f"Dati: {nomi_legenda[i]}")

    # aggiorno il limite orizzontale per coprire correttamente l'estensione di entrambi i flussi
    xmin_curr, xmax_curr = plt.xlim()
    xmin_globale = min(xmin_globale, xmin_curr) if i > 0 else xmin_curr
    xmax_globale = max(xmax_globale, xmax_curr) if i > 0 else xmax_curr

    # preparo il dato per il fit gaussiano coprendo tutta l'estensione dei bordi
    x = np.linspace(bordi_bin[0], bordi_bin[-1], 200)
    p = norm.pdf(x, mu_fit, sigma_fit)

    # disegno il fit con il colore scuro su tutti i dati
    plt.plot(x, p, color=colori_fit[i], linewidth=1, linestyle='-',
             label=f"Fit {nomi_legenda[i]} (mu={mu_fit:.4f}, sigma={sigma_fit:.4f}, chi_quadro_ridotto ={chi_quadro_ridotto:.2f})")

# inserisco il titolo comprensivo di tutti i riferimenti
plt.title("Distribuzione delle Variazioni Relative: Senza Correzioni vs Con Correzioni",
          fontsize=13, fontweight='bold')
plt.xlabel("Variazione Relativa: (Flusso - Media) / Media")
plt.ylabel("Densità di Probabilità")

# traccio la linea centrale
plt.axvline(x=0, color='black', linestyle=':', linewidth=1.5, label='Centro (0)')

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()

plt.savefig(nuova_sottocartella / "dispersione_istogrammi_sovrapposti.jpg", dpi=300)
# plt.show()

# =============================================================================
# 5. OPZIONE 2: Dispersione Mobile nel Tempo (Medie per Scatto)
# =============================================================================

plt.figure(figsize=(12, 6))

# raggruppo i dati includendo anche il tempo relativo per calcolare la media in quel preciso istante
df_temporale = df_valid.groupby(['run_origin', 'file_name', 'tempo_relativo'])[
    'variazione_relativa'].mean().reset_index()

# mi assicuro che l'ordine temporale sia corretto
df_temporale = df_temporale.sort_values(by=['tempo_relativo']).reset_index(drop=True)

# estraggo la serie temporale delle medie e calcolo la deviazione standard mobile
finestra_mobile = 10
serie_medie_scatto = df_temporale['variazione_relativa']
rolling_std = serie_medie_scatto.rolling(window=finestra_mobile, center=True).std() * 100

# estraggo i tempi in secondi per l'asse X
tempi_scatti = df_temporale['tempo_relativo'].values

# ricavo i confini delle run posizionandoli al tempo finale di ciascuna run
run_boundaries_temporali = []
for r in RUN_TO_ANALYZE:
    mask_run = df_temporale['run_origin'] == r
    if np.sum(mask_run) > 0:
        tempo_finale_run = df_temporale.loc[mask_run, 'tempo_relativo'].iloc[-1]
        run_boundaries_temporali.append((r, tempo_finale_run))

# traccio la curva
plt.plot(tempi_scatti, rolling_std, color='darkred', linewidth=2,
         label=f"Dev. Std Mobile (finestra={finestra_mobile} scatti)")

# inserisco le divisioni per le run
for r_idx, (run_num, t_end) in enumerate(run_boundaries_temporali):
    plt.axvline(x=t_end, color='gray', linestyle='--', alpha=0.6)

plt.title("Opzione 3: Dispersione Mobile delle Medie per Scatto\n(Valori Normalizzati, Media Globale ~0)",
          fontsize=12, fontweight='bold')
plt.xlabel("Tempo dall'inizio della Run 1 (secondi)")
plt.ylabel("Deviazione Standard Locale (%)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(nuova_sottocartella / "dispersione_rolling_std_globale.jpg", dpi=300)

# plt.show()

print("\n--- ELABORAZIONE DISPERSIONE COMPLETATA ---")

# =============================================================================
# 6. PREPARAZIONE DATI UNICI PER ANALISI SUCCESSIVE
# =============================================================================

# ordino i miei dati
df_total_sorted = df_total.sort_values(by=['label', 'Mag'], ascending=[True, True])

# deduplico i miei dati considerando la stessa stella solo all'interno dello stesso file
df_unique = df_total_sorted.drop_duplicates(subset=['label', 'file_name'], keep='first').copy()
print(f"Righe totali mantenendo l'evoluzione temporale: {len(df_unique)}")