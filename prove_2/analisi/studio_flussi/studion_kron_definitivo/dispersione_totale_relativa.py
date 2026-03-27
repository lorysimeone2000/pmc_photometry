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
# 2. CALCOLO DELLE VARIAZIONI RELATIVE
# =============================================================================

colonna_flusso = FLUSSI_DA_ANALIZZARE[0]
colonna_media = f"media_{colonna_flusso}"

# rimuovo le righe senza valori validi
df_valid = df_total.dropna(subset=[colonna_flusso, colonna_media])

# mantengo solo le righe dove la media è maggiore di zero per evitare divisioni per zero e creo una copia sicura
df_valid = df_valid[df_valid[colonna_media] > 0].copy()

# calcolo la variazione relativa per ogni singola misurazione e la salvo in una nuova colonna
df_valid['variazione_relativa'] = (df_valid[colonna_flusso] - df_valid[colonna_media]) / df_valid[colonna_media]

# preparo gli array globali che mi servono per i primi due grafici
arr_valid = df_valid['variazione_relativa'].values
runs_valid = df_valid['run_origin'].values

# =============================================================================
# 3. OPZIONE 1: CREAZIONE ISTOGRAMMA CENTRATO SU 0
# =============================================================================

plt.figure(figsize=(10, 6))

# filtro eventuali anomalie estreme per mantenere leggibile l'istogramma
mask_outliers = (arr_valid > -1) & (arr_valid < 1)
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
                            label='Variazioni Relative Sperimentali')

# sovrappongo il fit gaussiano teorico
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = norm.pdf(x, media_var, std_var)
plt.plot(x, p, 'k', linewidth=2, label=rf"Fit Normale (Teorico, $\chi^2_\nu={chi_quadro_ridotto:.2f}$)")

# inserisco le statistiche nel grafico
plt.text(xmax * 0.45, np.max(p) * 0.8,
         f"$\\mu={media_var:.4f}$\n$\\sigma={std_var:.4f}$",
         fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

plt.title("Opzione 1: Distribuzione delle Variazioni Relative del Flusso\n(Tutte le Stelle, Tutte le Run)",
          fontsize=12, fontweight='bold')
plt.xlabel("Variazione Relativa: (Flusso - Media) / Media")
plt.ylabel("Densità di Probabilità")

# traccio la linea centrale sullo 0
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Centro (0)')

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("dispersione_istogramma_globale.jpg", dpi=300)
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

plt.title("Opzione 2: Dispersione Statistica delle Variazioni per Singola Run\n(Tutte le Stelle)",
          fontsize=12, fontweight='bold')
plt.ylabel("Variazione Relativa: (Flusso - Media) / Media")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.savefig("dispersione_boxplot_globale.jpg", dpi=300)
plt.show()

# =============================================================================
# 5. OPZIONE 3: Dispersione Mobile nel Tempo (Medie per Scatto)
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
plt.savefig("dispersione_rolling_std_globale.jpg", dpi=300)
plt.show()

print("\n--- ELABORAZIONE DISPERSIONE COMPLETATA ---")

# =============================================================================
# 6. PREPARAZIONE DATI UNICI PER ANALISI SUCCESSIVE
# =============================================================================

# ordino i miei dati
df_total_sorted = df_total.sort_values(by=['label', 'Mag'], ascending=[True, True])

# deduplico i miei dati considerando la stessa stella solo all'interno dello stesso file
df_unique = df_total_sorted.drop_duplicates(subset=['label', 'file_name'], keep='first').copy()
print(f"Righe totali mantenendo l'evoluzione temporale: {len(df_unique)}")