import pandas as pd
import matplotlib.pyplot as plt
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
# 2. CALCOLO DELLE DIFFERENZE ASSOLUTE
# =============================================================================

colonna_flusso = FLUSSI_DA_ANALIZZARE[0]
colonna_media = f"media_{colonna_flusso}"

# rimuovo le righe senza valori validi
df_valid = df_total.dropna(subset=[colonna_flusso, colonna_media])

# mantengo solo le righe dove la media è maggiore di zero per evitare divisioni per zero e creo una copia sicura
df_valid = df_valid[df_valid[colonna_media] > 0].copy()

# calcolo la differenza assoluta per ogni singola misurazione e la salvo in una nuova colonna
df_valid['differenza_assoluta'] = df_valid[colonna_flusso] - df_valid[colonna_media]

# preparo gli array globali che mi servono per i primi due grafici
arr_valid = df_valid['differenza_assoluta'].values
runs_valid = df_valid['run_origin'].values

# =============================================================================
# 3. OPZIONE 1: CREAZIONE ISTOGRAMMA CENTRATO SU 0
# =============================================================================

plt.figure(figsize=(10, 6))

# rimuovo gli outlier estremi calcolando i percentili all'1% e al 99% per mantenere leggibile l'istogramma
limite_inf = np.percentile(arr_valid, 1)
limite_sup = np.percentile(arr_valid, 99)
mask_outliers = (arr_valid > limite_inf) & (arr_valid < limite_sup)
dati_istogramma = arr_valid[mask_outliers]

# calcolo la media e la deviazione standard delle variazioni
media_var = np.mean(dati_istogramma)
std_var = np.std(dati_istogramma)

# calcolo il chi quadro ridotto sui conteggi per mantenere il rigore statistico
conteggi, bordi_bin = np.histogram(dati_istogramma, bins=100)
centri_bin = (bordi_bin[:-1] + bordi_bin[1:]) / 2
larghezza_bin = np.diff(bordi_bin)
conteggi_attesi = len(dati_istogramma) * larghezza_bin * norm.pdf(centri_bin, media_var, std_var)

# filtro i bin con conteggi attesi nulli per evitare divisioni per zero
mask_chi = conteggi_attesi > 0
chi_quadro = np.sum(((conteggi[mask_chi] - conteggi_attesi[mask_chi]) ** 2) / conteggi_attesi[mask_chi])
gradi_liberta = np.sum(mask_chi) - 3
chi_quadro_ridotto = chi_quadro / gradi_liberta if gradi_liberta > 0 else np.nan

# creo l'istogramma normalizzato
n, bins, patches = plt.hist(dati_istogramma, bins=100, density=True, alpha=0.6, color='steelblue', edgecolor='black',
                            label='Differenze Assolute Sperimentali')

# sovrappongo il fit gaussiano teorico
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = norm.pdf(x, media_var, std_var)
plt.plot(x, p, 'k', linewidth=2, label=rf"Fit Normale (Teorico, $\chi^2_\nu={chi_quadro_ridotto:.2e}$)")

# inserisco le statistiche nel grafico
plt.text(xmax * 0.45, np.max(p) * 0.8,
         f"$\\mu={media_var:.4f}$\n$\\sigma={std_var:.4f}$",
         fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

plt.title("Opzione 1: Distribuzione delle Differenze Assolute del Flusso\n(Tutte le Stelle, Tutte le Run)",
          fontsize=12, fontweight='bold')
plt.xlabel("Differenza Assoluta: Flusso - Media")
plt.ylabel("Densità di Probabilità")

# traccio la linea centrale sullo 0
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Centro (0)')

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("dispersione_istogramma_globale_assoluta.jpg", dpi=300)
plt.show()

# =============================================================================
# 4. OPZIONE 2: Boxplot per singola Run (con Legenda)
# =============================================================================

plt.figure(figsize=(10, 6))

dati_per_run = []
labels_run = []

for r in RUN_TO_ANALYZE:
    mask_r = (runs_valid == r)
    if np.sum(mask_r) > 0:
        dati_per_run.append(arr_valid[mask_r])
        labels_run.append(f"Run {r}")

# creo il boxplot
bp = plt.boxplot(dati_per_run, labels=labels_run, patch_artist=True)

# personalizzo i colori e aggiungo etichette per la legenda
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
bp['boxes'][0].set_label('Distribuzione Centrale (IQR)')
bp['medians'][0].set_label('Mediana')
bp['fliers'][0].set_label('Outliers (Fuori Baffi)')

plt.title("Opzione 2: Dispersione Statistica delle Differenze Assolute per Singola Run\n(Tutte le Stelle)",
          fontsize=12, fontweight='bold')
plt.ylabel("Differenza Assoluta: Flusso - Media")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.savefig("dispersione_boxplot_globale_assoluta.jpg", dpi=300)
plt.show()

# =============================================================================
# 5. OPZIONE 3: Dispersione Mobile nel Tempo (Medie per Scatto)
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
         label=f"Dev. Std Mobile (finestra={finestra_mobile} scatti)")

# inserisco le divisioni per le run
for r_idx, (run_num, t_end) in enumerate(run_boundaries_temporali):
    plt.axvline(x=t_end, color='gray', linestyle='--', alpha=0.6)

plt.title("Opzione 3: Dispersione Mobile delle Medie per Scatto\n(Valori Assoluti)",
          fontsize=12, fontweight='bold')
plt.xlabel("Tempo dall'inizio della Run 1 (secondi)")
plt.ylabel("Deviazione Standard Locale")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("dispersione_rolling_std_globale_assoluta.jpg", dpi=300)
plt.show()

print("\n--- ELABORAZIONE DISPERSIONE COMPLETATA ---")

# =============================================================================
# 6. PREPARAZIONE DATI UNICI PER ANALISI SUCCESSIVE
# =============================================================================

# ordino i miei dati
# Utilizzo df_valid invece di df_total per mantenere la colonna calcolata in precedenza
df_total_sorted = df_valid.sort_values(by=['label', 'Mag'], ascending=[True, True])

# deduplico i miei dati considerando la stessa stella solo all'interno dello stesso file
df_unique = df_total_sorted.drop_duplicates(subset=['label', 'file_name'], keep='first').copy()
print(f"Righe totali mantenendo l'evoluzione temporale: {len(df_unique)}")

# =============================================================================
# 7. APPLICAZIONE TREND FILTERING ALGORITHM (TFA)
# =============================================================================
print("\n--- AVVIO TREND FILTERING ALGORITHM (TFA) ---")

# Creo una matrice [Tempo (file_name) x Stelle (label)] dei residui (differenza assoluta).
# Uso 'file_name' come indice temporale primario in quanto id univoco dello scatto.
df_pivot = df_unique.pivot_table(index='file_name', columns='label', values='differenza_assoluta', aggfunc='mean')

# Riempio i valori mancanti con 0.
# Poiché i dati sono differenze dalla media (quindi a media zero), 0 è il valore neutro.
df_pivot = df_pivot.fillna(0)

# Seleziono le stelle più osservate e stabili (max osservazioni valide) come template set.
conteggi_stelle = df_unique.groupby('label')['differenza_assoluta'].count()
num_templates = min(150, max(10, len(conteggi_stelle) // 4))  # Fino a 150 template
template_labels = conteggi_stelle.nlargest(num_templates).index.tolist()

print(f"Ho selezionato {len(template_labels)} stelle template per la costruzione del filtro TFA.")

# Costruisco la matrice dei template base
X_full = df_pivot[template_labels].values

dati_tfa_corretti = {}

# Applico il TFA iterando stella per stella
for stella in tqdm(df_pivot.columns, desc="Applicazione Minimi Quadrati (TFA)"):
    y = df_pivot[stella].values

    # Se la stella target è nel set di template, la escludo categoricamente per evitare il self-fitting
    if stella in template_labels:
        idx_stella = template_labels.index(stella)
        X = np.delete(X_full, idx_stella, axis=1)
    else:
        X = X_full

    # Calcolo i coefficienti ottimali (c) tramite minimi quadrati: (X^T X)^{-1} X^T y
    c, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # Costruisco il filtro sistematico globale e lo sottraggo
    filtro_sistematico = np.dot(X, c)
    y_corretto = y - filtro_sistematico

    dati_tfa_corretti[stella] = y_corretto

# Ricostruisco il DataFrame corretto
df_corretto_pivot = pd.DataFrame(dati_tfa_corretti, index=df_pivot.index)

# Riporto il DataFrame nel formato lungo (long format) originario
df_tfa_long = df_corretto_pivot.reset_index().melt(id_vars='file_name', var_name='label',
                                                   value_name='differenza_assoluta_tfa')

# Unisco i dati corretti TFA al dataframe unique originale
df_unique = df_unique.merge(df_tfa_long, on=['file_name', 'label'], how='left')

arr_valid_tfa = df_unique['differenza_assoluta_tfa'].dropna().values
runs_valid_tfa = df_unique.dropna(subset=['differenza_assoluta_tfa'])['run_origin'].values

# =============================================================================
# 8. GRAFICI POST-TFA
# =============================================================================
print("\n--- GENERAZIONE GRAFICI POST-TFA ---")

# --- Opzione 1: Istogramma TFA ---
plt.figure(figsize=(10, 6))

limite_inf_tfa = np.percentile(arr_valid_tfa, 1)
limite_sup_tfa = np.percentile(arr_valid_tfa, 99)
mask_outliers_tfa = (arr_valid_tfa > limite_inf_tfa) & (arr_valid_tfa < limite_sup_tfa)
dati_istogramma_tfa = arr_valid_tfa[mask_outliers_tfa]

media_var_tfa = np.mean(dati_istogramma_tfa)
std_var_tfa = np.std(dati_istogramma_tfa)

conteggi_tfa, bordi_bin_tfa = np.histogram(dati_istogramma_tfa, bins=100)
centri_bin_tfa = (bordi_bin_tfa[:-1] + bordi_bin_tfa[1:]) / 2
larghezza_bin_tfa = np.diff(bordi_bin_tfa)
conteggi_attesi_tfa = len(dati_istogramma_tfa) * larghezza_bin_tfa * norm.pdf(centri_bin_tfa, media_var_tfa,
                                                                              std_var_tfa)

mask_chi_tfa = conteggi_attesi_tfa > 0
chi_quadro_tfa = np.sum(
    ((conteggi_tfa[mask_chi_tfa] - conteggi_attesi_tfa[mask_chi_tfa]) ** 2) / conteggi_attesi_tfa[mask_chi_tfa])
gradi_liberta_tfa = np.sum(mask_chi_tfa) - 3
chi_quadro_ridotto_tfa = chi_quadro_tfa / gradi_liberta_tfa if gradi_liberta_tfa > 0 else np.nan

plt.hist(dati_istogramma_tfa, bins=100, density=True, alpha=0.6, color='mediumseagreen', edgecolor='black',
         label='Differenze Assolute Sperimentali (TFA)')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = norm.pdf(x, media_var_tfa, std_var_tfa)
plt.plot(x, p, 'k', linewidth=2, label=rf"Fit Normale (Teorico, $\chi^2_\nu={chi_quadro_ridotto_tfa:.2e}$)")

plt.text(xmax * 0.45, np.max(p) * 0.8,
         f"$\\mu={media_var_tfa:.4f}$\n$\\sigma={std_var_tfa:.4f}$",
         fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

plt.title("Opzione 1: Distribuzione Differenze Assolute del Flusso (TFA Applicato)\n(Tutte le Stelle, Tutte le Run)",
          fontsize=12, fontweight='bold')
plt.xlabel("Differenza Assoluta: Flusso - Media")
plt.ylabel("Densità di Probabilità")
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Centro (0)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("dispersione_istogramma_globale_tfa.jpg", dpi=300)
plt.show()

# --- Opzione 2: Boxplot TFA ---
plt.figure(figsize=(10, 6))

dati_per_run_tfa = []
labels_run_tfa = []

for r in RUN_TO_ANALYZE:
    mask_r = (runs_valid_tfa == r)
    if np.sum(mask_r) > 0:
        dati_per_run_tfa.append(arr_valid_tfa[mask_r])
        labels_run_tfa.append(f"Run {r}")

bp = plt.boxplot(dati_per_run_tfa, labels=labels_run_tfa, patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
bp['boxes'][0].set_label('Distribuzione Centrale (IQR)')
bp['medians'][0].set_label('Mediana')
bp['fliers'][0].set_label('Outliers (Fuori Baffi)')

plt.title("Opzione 2: Dispersione Statistica Differenze Assolute per Singola Run (TFA Applicato)\n(Tutte le Stelle)",
          fontsize=12, fontweight='bold')
plt.ylabel("Differenza Assoluta: Flusso - Media")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.savefig("dispersione_boxplot_globale_tfa.jpg", dpi=300)
plt.show()

# --- Opzione 3: Rolling Std TFA ---
plt.figure(figsize=(12, 6))

df_temporale_tfa = df_unique.dropna(subset=['differenza_assoluta_tfa', 'tempo_relativo']).groupby(
    ['run_origin', 'file_name', 'tempo_relativo'])['differenza_assoluta_tfa'].mean().reset_index()
df_temporale_tfa = df_temporale_tfa.sort_values(by=['tempo_relativo']).reset_index(drop=True)

serie_medie_scatto_tfa = df_temporale_tfa['differenza_assoluta_tfa']
rolling_std_tfa = serie_medie_scatto_tfa.rolling(window=finestra_mobile, center=True).std() * 100

tempi_scatti_tfa = df_temporale_tfa['tempo_relativo'].values

run_boundaries_temporali_tfa = []
for r in RUN_TO_ANALYZE:
    mask_run = df_temporale_tfa['run_origin'] == r
    if np.sum(mask_run) > 0:
        tempo_finale_run = df_temporale_tfa.loc[mask_run, 'tempo_relativo'].iloc[-1]
        run_boundaries_temporali_tfa.append((r, tempo_finale_run))

plt.plot(tempi_scatti_tfa, rolling_std_tfa, color='forestgreen', linewidth=2,
         label=f"Dev. Std Mobile TFA (finestra={finestra_mobile} scatti)")

for r_idx, (run_num, t_end) in enumerate(run_boundaries_temporali_tfa):
    plt.axvline(x=t_end, color='gray', linestyle='--', alpha=0.6)

plt.title("Opzione 3: Dispersione Mobile delle Medie per Scatto (TFA Applicato)\n(Valori Assoluti)",
          fontsize=12, fontweight='bold')
plt.xlabel("Tempo dall'inizio della Run 1 (secondi)")
plt.ylabel("Deviazione Standard Locale")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("dispersione_rolling_std_globale_tfa.jpg", dpi=300)
plt.show()

print("\n--- ELABORAZIONE TFA COMPLETATA ---")