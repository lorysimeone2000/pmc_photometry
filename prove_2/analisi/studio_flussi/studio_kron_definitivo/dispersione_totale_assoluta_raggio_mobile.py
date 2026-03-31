import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path
from scipy.stats import norm, chisquare
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import FITSFixedWarning

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

# definisco i flussi esatti che voglio analizzare
FLUSSI_DA_ANALIZZARE = [
    "flusso_fisso_max_run_senza_correzioni",
    "flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE"
]

cartella_corrente = Path.cwd()
nuova_sottocartella = cartella_corrente / "dispersione_assoluta"
nuova_sottocartella.mkdir(parents=True, exist_ok=True)

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

            # calcolo il tempo relativo
            df_temp['tempo_relativo'] = (t_curr - t0_global) / 1000.0 if t0_global is not None else 0

            # appendo il dataframe alla mia lista
            lista_dfs.append(df_temp)
        except Exception as e:
            pass

if not lista_dfs:
    print("ERRORE: Nessun dato caricato.")
    exit()

# unisco i miei dataframe
df_total = pd.concat(lista_dfs, ignore_index=True)
print(f"Totale righe caricate: {len(df_total)}")

# =============================================================================
# 2. CALCOLO DELLE DIFFERENZE ASSOLUTE
# =============================================================================

# rimuovo le righe senza valori validi per tutti i flussi che mi interessano
df_valid = df_total.copy()
for flusso in FLUSSI_DA_ANALIZZARE:
    col_media = f"media_{flusso}"
    df_valid = df_valid.dropna(subset=[flusso, col_media])
    # mantengo solo le righe dove la media è maggiore di zero
    df_valid = df_valid[df_valid[col_media] > 0]

# calcolo la differenza assoluta per i due flussi
flusso_senza_corr = FLUSSI_DA_ANALIZZARE[0]
flusso_corretto = FLUSSI_DA_ANALIZZARE[1]

df_valid['diff_ass_senza_correzioni'] = df_valid[flusso_senza_corr] - df_valid[f"media_{flusso_senza_corr}"]
df_valid['differenza_assoluta'] = df_valid[flusso_corretto] - df_valid[f"media_{flusso_corretto}"]

# preparo gli array globali che mi servono per i primi grafici
arr_valid_senza_corr = df_valid['diff_ass_senza_correzioni'].values
arr_valid = df_valid['differenza_assoluta'].values
runs_valid = df_valid['run_origin'].values

# =============================================================================
# 3. PREPARAZIONE DATI UNICI PER ANALISI SUCCESSIVE E APPLICAZIONE TFA
# =============================================================================

# ordino i miei dati
df_total_sorted = df_valid.sort_values(by=['label', 'Mag'], ascending=[True, True])

# deduplico i miei dati considerando la stessa stella solo all'interno dello stesso file
df_unique = df_total_sorted.drop_duplicates(subset=['label', 'file_name'], keep='first').copy()
print(f"Righe totali mantenendo l'evoluzione temporale: {len(df_unique)}")

print("\n--- AVVIO TREND FILTERING ALGORITHM (TFA) ---")

# creo una matrice dei residui usando il nome del file come indice temporale primario
df_pivot = df_unique.pivot_table(index='file_name', columns='label', values='differenza_assoluta', aggfunc='mean')

# riempio i valori mancanti con 0
df_pivot = df_pivot.fillna(0)

# seleziono le stelle più osservate e stabili come template set
conteggi_stelle = df_unique.groupby('label')['differenza_assoluta'].count()
num_templates = min(150, max(10, len(conteggi_stelle) // 4))
template_labels = conteggi_stelle.nlargest(num_templates).index.tolist()

print(f"Ho selezionato {len(template_labels)} stelle template per la costruzione del filtro TFA.")

# costruisco la matrice dei template base
X_full = df_pivot[template_labels].values

# inizializzo il mio dizionario per i dati corretti
dati_tfa_corretti = {}

# applico il TFA iterando stella per stella
for stella in tqdm(df_pivot.columns, desc="Applicazione Minimi Quadrati (TFA)"):
    y = df_pivot[stella].values

    # se la stella target è nel set di template, la escludo categoricamente per evitare il self-fitting
    if stella in template_labels:
        idx_stella = template_labels.index(stella)
        X = np.delete(X_full, idx_stella, axis=1)
    else:
        X = X_full

    # calcolo i coefficienti ottimali tramite minimi quadrati
    c, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # costruisco il filtro sistematico globale e lo sottraggo
    filtro_sistematico = np.dot(X, c)
    y_corretto = y - filtro_sistematico

    # salvo il dato corretto
    dati_tfa_corretti[stella] = y_corretto

# ricostruisco il DataFrame corretto
df_corretto_pivot = pd.DataFrame(dati_tfa_corretti, index=df_pivot.index)

# riporto il DataFrame nel formato lungo originario
df_tfa_long = df_corretto_pivot.reset_index().melt(id_vars='file_name', var_name='label',
                                                   value_name='differenza_assoluta_tfa')

# unisco i dati corretti TFA al dataframe unique originale
df_unique = df_unique.merge(df_tfa_long, on=['file_name', 'label'], how='left')

# estraggo gli array validi post-TFA
arr_valid_tfa = df_unique['differenza_assoluta_tfa'].dropna().values
runs_valid_tfa = df_unique.dropna(subset=['differenza_assoluta_tfa'])['run_origin'].values

# =============================================================================
# 4. OPZIONE 1: CREAZIONE ISTOGRAMMI SOVRAPPOSTI (TUTTI I FLUSSI)
# =============================================================================

# preparo il mio dizionario in cui memorizzo l'array delle differenze
dati_flussi_istogramma = {}

# scarto le anomalie estreme all'1% e 99% per il flusso senza correzioni
limite_inf_sc = np.percentile(arr_valid_senza_corr, 1)
limite_sup_sc = np.percentile(arr_valid_senza_corr, 99)
mask_outliers_sc = (arr_valid_senza_corr > limite_inf_sc) & (arr_valid_senza_corr < limite_sup_sc)
dati_flussi_istogramma['Senza Correzioni'] = arr_valid_senza_corr[mask_outliers_sc]

# scarto le anomalie estreme all'1% e 99% per il flusso pre-TFA
limite_inf = np.percentile(arr_valid, 1)
limite_sup = np.percentile(arr_valid, 99)
mask_outliers = (arr_valid > limite_inf) & (arr_valid < limite_sup)
dati_flussi_istogramma['Pre-TFA'] = arr_valid[mask_outliers]

# scarto le anomalie estreme all'1% e 99% per il flusso post-TFA
limite_inf_tfa = np.percentile(arr_valid_tfa, 1)
limite_sup_tfa = np.percentile(arr_valid_tfa, 99)
mask_outliers_tfa = (arr_valid_tfa > limite_inf_tfa) & (arr_valid_tfa < limite_sup_tfa)
dati_flussi_istogramma['Post-TFA'] = arr_valid_tfa[mask_outliers_tfa]

plt.figure(figsize=(12, 8))

# definisco i colori per gli istogrammi e i fit
colori_hist = ['darkgray', 'dodgerblue', 'tomato']
colori_fit = ['black', 'navy', 'darkred']
nomi_legenda = ['Senza Correzioni', 'Pre-TFA', 'Post-TFA']
chiavi_flussi = ['Senza Correzioni', 'Pre-TFA', 'Post-TFA']

# inizializzo i limiti dell'asse x per uniformare il fit gaussiano
xmin_globale, xmax_globale = 0, 0

# avvio il ciclo per estrarre ed elaborare il dato di ciascun flusso
for i, chiave in enumerate(chiavi_flussi):
    dati = dati_flussi_istogramma[chiave]

    # ricavo la media e la deviazione standard sull'intero set di dati
    media_var = np.mean(dati)
    std_var = np.std(dati)

    # preparo il range di raggi in termini di deviazioni standard (da 0.01 a 5 sigma)
    raggi_test = np.linspace(0.01 * std_var, 5.0 * std_var, 200)

    # inizializzo le variabili per conservare i parametri del fit migliore
    best_chi_ridotto = np.inf
    best_mu = media_var
    best_sigma = std_var
    best_raggio = std_var

    # ottengo i bin una sola volta sull'intero set per calcolare coerentemente il chi quadro
    conteggi, bordi_bin = np.histogram(dati, bins='auto')
    centri_bin = (bordi_bin[:-1] + bordi_bin[1:]) / 2
    larghezza_bin = np.diff(bordi_bin)

    # ciclo attraverso tutti i raggi di test per trovare il fit ottimale
    for raggio in raggi_test:
        # isolo i dati nell'intervallo mobile
        mask_fit = (dati >= media_var - raggio) & (dati <= media_var + raggio)
        dati_per_fit = dati[mask_fit]

        # evito il fit se non ho abbastanza dati nel raggio ristretto
        if len(dati_per_fit) < 10:
            continue

        # calcolo i nuovi parametri del fit basandomi solo sui dati ristretti dal raggio corrente
        mu_fit, sigma_fit = norm.fit(dati_per_fit)

        # calcolo il conteggio atteso teorico usando i parametri appena trovati
        conteggi_attesi = len(dati) * larghezza_bin * norm.pdf(centri_bin, mu_fit, sigma_fit)

        # applico il test del chi quadro
        mask_chi = (conteggi > 0) & (conteggi_attesi > 0)
        osservati = conteggi[mask_chi]
        attesi = conteggi_attesi[mask_chi]

        # salto l'iterazione se i bin validi sono troppo pochi per un calcolo corretto dei gradi di libertà
        if len(osservati) <= 3:
            continue

        try:
            chi_quadro, p_value = chisquare(f_obs=osservati, f_exp=attesi, ddof=2)
            gradi_liberta = len(osservati) - 3
            chi_quadro_ridotto = chi_quadro / gradi_liberta if gradi_liberta > 0 else np.inf

            # aggiorno i valori se trovo un chi quadro ridotto più piccolo
            if chi_quadro_ridotto < best_chi_ridotto:
                best_chi_ridotto = chi_quadro_ridotto
                best_mu = mu_fit
                best_sigma = sigma_fit
                best_raggio = raggio
        except Exception:
            pass

    # disegno le due linee verticali tratteggiate per evidenziare i limiti del fit migliore
    plt.axvline(media_var - best_raggio, color=colori_fit[i], linestyle='--', alpha=0.4, linewidth=0.8)
    plt.axvline(media_var + best_raggio, color=colori_fit[i], linestyle='--', alpha=0.4, linewidth=0.8)

    # genero l'istogramma a gradini per impedire che le barre piene si coprano a vicenda
    plt.hist(dati, bins='auto', density=True, histtype='step', linewidth=1.5, alpha=1.0,
             color=colori_hist[i], label=f"Dati: {nomi_legenda[i]}")

    # aggiorno il limite orizzontale per coprire correttamente l'estensione di tutti i flussi
    xmin_curr, xmax_curr = plt.xlim()
    xmin_globale = min(xmin_globale, xmin_curr) if i > 0 else xmin_curr
    xmax_globale = max(xmax_globale, xmax_curr) if i > 0 else xmax_curr

    # preparo il dato per il fit gaussiano finale (che ha ottenuto il chi quadro minore) coprendo tutta l'estensione
    x = np.linspace(bordi_bin[0], bordi_bin[-1], 200)
    p = norm.pdf(x, best_mu, best_sigma)

    # disegno il fit con il colore scuro su tutti i dati
    label_fit = f"Fit {nomi_legenda[i]} (mu={best_mu:.4f}, sigma={best_sigma:.4f}, raggio={best_raggio / std_var:.1f}σ, chi2_red={best_chi_ridotto:.2f})"
    plt.plot(x, p, color=colori_fit[i], linewidth=2, linestyle='-', label=label_fit)

# inserisco il titolo comprensivo di tutti i riferimenti aggiornato con intervallo migliore
plt.title("Distribuzione delle Differenze Assolute: Senza Correzioni vs Pre-TFA vs Post-TFA\n (Intervallo Migliore)",
          fontsize=13, fontweight='bold')
plt.xlabel("Differenza Assoluta: Flusso - Media")
plt.ylabel("Densità di Probabilità")

# traccio la linea centrale
plt.axvline(x=0, color='black', linestyle=':', linewidth=1.5, label='Centro (0)')

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()

plt.savefig(nuova_sottocartella / "dispersione_istogrammi_sovrapposti_assoluta_raggio_mobile.jpg", dpi=300)
# plt.show()

# =============================================================================
# 6. OPZIONE 3: Dispersione Mobile nel Tempo (Medie per Scatto, Pre-TFA)
# =============================================================================

plt.figure(figsize=(12, 6))

# raggruppo i dati includendo anche il tempo relativo per calcolare la media in quel preciso istante
df_temporale = df_valid.groupby(['run_origin', 'file_name', 'tempo_relativo'])[
    'differenza_assoluta'].mean().reset_index()

# mi assicuro che l'ordine temporale sia corretto
df_temporale = df_temporale.sort_values(by=['tempo_relativo']).reset_index(drop=True)

# estraggo la serie temporale delle medie e calcolo la deviazione standard mobile
finestra_mobile = 10
serie_medie_scatto = df_temporale['differenza_assoluta']
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
         label=f"Dev. Std Mobile Pre-TFA (finestra={finestra_mobile} scatti)")

# inserisco le divisioni per le run
for r_idx, (run_num, t_end) in enumerate(run_boundaries_temporali):
    plt.axvline(x=t_end, color='gray', linestyle='--', alpha=0.6)

plt.title("Opzione 3: Dispersione Mobile delle Medie per Scatto (Pre-TFA)",
          fontsize=12, fontweight='bold')
plt.xlabel("Tempo dall'inizio della Run 1 (secondi)")
plt.ylabel("Deviazione Standard Locale")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()

plt.savefig(nuova_sottocartella / "dispersione_rolling_std_globale_assoluta_pre_tfa.jpg", dpi=300)
# plt.show()

# --- Rolling Std TFA ---
plt.figure(figsize=(12, 6))

# raggruppo i dati temporalmente per il TFA
df_temporale_tfa = df_unique.dropna(subset=['differenza_assoluta_tfa', 'tempo_relativo']).groupby(
    ['run_origin', 'file_name', 'tempo_relativo'])['differenza_assoluta_tfa'].mean().reset_index()
df_temporale_tfa = df_temporale_tfa.sort_values(by=['tempo_relativo']).reset_index(drop=True)

# calcolo la deviazione standard mobile per il TFA
serie_medie_scatto_tfa = df_temporale_tfa['differenza_assoluta_tfa']
rolling_std_tfa = serie_medie_scatto_tfa.rolling(window=finestra_mobile, center=True).std() * 100

# estraggo i tempi per l'asse X
tempi_scatti_tfa = df_temporale_tfa['tempo_relativo'].values

# ricavo i confini delle run per il TFA
run_boundaries_temporali_tfa = []
for r in RUN_TO_ANALYZE:
    mask_run = df_temporale_tfa['run_origin'] == r
    if np.sum(mask_run) > 0:
        tempo_finale_run = df_temporale_tfa.loc[mask_run, 'tempo_relativo'].iloc[-1]
        run_boundaries_temporali_tfa.append((r, tempo_finale_run))

# traccio la curva TFA
plt.plot(tempi_scatti_tfa, rolling_std_tfa, color='forestgreen', linewidth=2,
         label=f"Dev. Std Mobile TFA (finestra={finestra_mobile} scatti)")

# inserisco le divisioni
for r_idx, (run_num, t_end) in enumerate(run_boundaries_temporali_tfa):
    plt.axvline(x=t_end, color='gray', linestyle='--', alpha=0.6)

plt.title("Opzione 3: Dispersione Mobile delle Medie per Scatto (Post-TFA) \n fit su un raggio di 1 sigma",
          fontsize=12, fontweight='bold')
plt.xlabel("Tempo dall'inizio della Run 1 (secondi)")
plt.ylabel("Deviazione Standard Locale")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()

plt.savefig(nuova_sottocartella / "dispersione_rolling_std_globale_tfa.jpg", dpi=300)
# plt.show()

print("\n--- ELABORAZIONE TFA E GRAFICI COMPLETATA ---")