import pandas as pd
import matplotlib
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

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


def trova_cartella_base(nome_target="pmc_photometry"):
    # cerco la mia cartella base risalendo l'albero delle directory
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
    # cerco la cartella specificata nel percorso del mio progetto
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


def modello_lineare(mag, m, q):
    # definisco il mio modello: log10(Flux) = m * Mag + q
    return m * mag + q


# configuro le mie impostazioni di base
RUN_TO_ANALYZE = [1]

# definisco il flusso esatto che voglio analizzare applicando la correzione additiva e la decorrelazione globale
FLUSSI_DA_ANALIZZARE = [
    "flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE"
]

# =============================================================================
# 1. CARICAMENTO DATI (TUTTE LE RUN)
# =============================================================================

print(f"--- Caricamento dati per Fit Globale ---")
lista_dfs = []

# aggiungo alla mia lista tutte le colonne dei flussi e le coordinate dei centroidi per il calcolo delle distanze
cols_needed = ['label', 'ID', 'Corrispondenza', 'Mag', 'saturazione', 'xcentroid', 'ycentroid']
for flusso in FLUSSI_DA_ANALIZZARE:
    cols_needed.extend([f"media_{flusso}", f"std_{flusso}"])

for run in RUN_TO_ANALYZE:
    nome_cartella = f"tabelle_unite_run_{run}"
    path_cartella = cerca_cartella_nel_progetto(BASE_DIR / "tabelle_alleggerite", nome_cartella)

    if path_cartella is None:
        print(f"Attenzione: Cartella {nome_cartella} non trovata.")
        continue
    else:
        print(f"cartella trovata in {path_cartella}")

    # estraggo i miei file parquet seguendo il pattern richiesto
    files_parquet = sorted(list(path_cartella.glob(f"run_{run}_stelle_trovate_e_catalogate_immagine_*.parquet")))
    print(f"Run {run}: Trovati {len(files_parquet)} file parquet. Caricamento in corso...")

    for f in tqdm(files_parquet, leave=False):
        try:
            # leggo il mio dataset parquet
            df_temp = pd.read_parquet(f)
            # mantengo solamente le colonne che mi servono e che sono effettivamente presenti
            colonne_valide = [c for c in cols_needed if c in df_temp.columns]
            df_temp = df_temp[colonne_valide].copy()
            df_temp['run_origin'] = run
            lista_dfs.append(df_temp)
        except Exception as e:
            pass

if not lista_dfs:
    print("ERRORE: Nessun dato caricato.")
    exit()

df_total = pd.concat(lista_dfs, ignore_index=True)
print(f"Totale righe caricate: {len(df_total)}")

# =============================================================================
# 2. PREPARAZIONE DATI PER IL FIT E IL GRAFICO
# =============================================================================

# separo preliminarmente le mie stelle catalogate da quelle non catalogate
mask_match_globale = df_total['Corrispondenza'] == True
df_match_all = df_total[mask_match_globale].copy()
df_no_match_all = df_total[~mask_match_globale].copy()

# deduplico i miei dati catalogati ordinandoli per label e poi per magnitudine (crescente)
df_match_sorted = df_match_all.sort_values(by=['label', 'Mag'], ascending=[True, True])

# tengo esclusivamente la prima occorrenza per ogni label (quella con Mag minore)
df_match = df_match_sorted.drop_duplicates(subset=['label'], keep='first').copy()

# rimuovo i duplicati dai miei dati non catalogati basandomi unicamente sulla label
df_no_match = df_no_match_all.drop_duplicates(subset=['label'], keep='first').copy()

print(f"Oggetti Catalogati UNICI: {len(df_match)}")
print(f"Oggetti NON Catalogati UNICI: {len(df_no_match)}")

# isolo le mie stelle sature lavorando sul dataframe delle catalogate
if 'saturazione' in df_match.columns:
    mask_sature = df_match['saturazione'] == True
    df_sature = df_match[mask_sature].copy()
    # mantengo i miei dati non saturi come potenziali candidati per il fit
    df_fit_potential = df_match[~mask_sature].copy()
else:
    df_sature = pd.DataFrame()
    df_fit_potential = df_match.copy()

# applico il mio filtro di validità numerica base per assicurarmi che esista una magnitudine
mask_valid_base = df_fit_potential['Mag'].notna()
df_fit_valid = df_fit_potential[mask_valid_base].copy()

# applico la mia soglia di magnitudine limite per pulire i dati del fit
SOGLIA_MAG_FIT = 10.0
df_fit_clean = df_fit_valid[df_fit_valid['Mag'] <= SOGLIA_MAG_FIT].copy()

if not df_sature.empty:
    df_sature = df_sature[df_sature['Mag'] <= SOGLIA_MAG_FIT].copy()

print(f"Oggetti di base analizzabili per il fit (Match SI, No Saturi, Mag <= {SOGLIA_MAG_FIT}): {len(df_fit_clean)}")

# =============================================================================
# 3. FIT LINEARE PER AREE DI DISTANZA DAL BORDO E PLOTTING
# =============================================================================

# definisco le mie dimensioni del sensore per il calcolo dei bordi
W_IMG, H_IMG = 3072, 2048

for flusso in FLUSSI_DA_ANALIZZARE:
    col_media = f"media_{flusso}"
    col_std = f"std_{flusso}"

    print(f"\n===========================================================")
    print(f"ANALISI: {flusso}")
    print(f"===========================================================")

    if col_media not in df_fit_clean.columns or col_std not in df_fit_clean.columns:
        print(f"Le colonne per '{flusso}' non sono presenti nel dataset. Salto l'analisi.")
        continue

    # isolo solo i miei elementi che hanno flussi, errori e coordinate validi
    mask_flusso_valido = (
            (df_fit_clean[col_media].notna()) & (df_fit_clean[col_media] > 0) &
            (df_fit_clean[col_std].notna()) & (df_fit_clean[col_std] > 0) &
            (df_fit_clean['xcentroid'].notna()) & (df_fit_clean['ycentroid'].notna())
    )
    df_fit_curr = df_fit_clean[mask_flusso_valido].copy()

    if len(df_fit_curr) == 0:
        continue

    # calcolo la mia distanza dal bordo più vicino per ogni stella
    df_fit_curr['dist_bordo'] = np.minimum.reduce([
        df_fit_curr['xcentroid'],
        df_fit_curr['ycentroid'],
        W_IMG - df_fit_curr['xcentroid'],
        H_IMG - df_fit_curr['ycentroid']
    ])

    # divido le mie distanze in 5 aree uguali e ordino gli intervalli
    df_fit_curr['area_bordo'] = pd.cut(df_fit_curr['dist_bordo'], bins=5)
    aree = sorted(df_fit_curr['area_bordo'].unique())

    # filtro i miei validi anche per i saturi (solo per plottarli)
    if not df_sature.empty and col_media in df_sature.columns:
        mask_sature_valido = (df_sature[col_media].notna()) & (df_sature[col_media] > 0)
        df_sature_plot = df_sature[mask_sature_valido].copy()
    else:
        df_sature_plot = pd.DataFrame()

    # filtro i miei validi anche per i non matchati (solo per plottarli)
    if not df_no_match.empty and col_media in df_no_match.columns:
        mask_no_match_valido = (df_no_match[col_media].notna()) & (df_no_match[col_media] > 0)
        df_no_match_plot = df_no_match[mask_no_match_valido].copy()
    else:
        df_no_match_plot = pd.DataFrame()

    # preparo la mia palette di colori dinamica basata sulla distanza dal centro
    # mappo linearmente l'indice dell'area sull'intero range di viridis [0, 1]
    cmap = plt.cm.viridis
    colori_aree = [cmap(i / (len(aree) - 1)) for i in range(len(aree))]

    # inizio la costruzione della mia figura
    plt.figure(figsize=(14, 10))

    # itero sulle mie 5 aree per calcolare e disegnare i fit separatamente
    for idx_area, area in enumerate(aree):
        df_area = df_fit_curr[df_fit_curr['area_bordo'] == area]

        if len(df_area) < 3:
            print(f"Area {idx_area + 1} saltata: non ho abbastanza punti validi.")
            continue

        nome_area = f"{area.left:.0f}-{area.right:.0f} px"
        colore = colori_aree[idx_area]

        # estraggo i miei dati originali X, Y e relativi errori per l'area corrente
        X = df_area['Mag'].values
        Y_flux = df_area[col_media].values
        sigma_flux = df_area[col_std].values

        # definisco il mio numero di bin specifico per quest'area
        n_bins = max(5, int(np.sqrt(len(X))))

        # ordino i miei array per magnitudine prima di spezzarli
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        Y_sorted = Y_flux[sort_idx]
        Err_sorted = sigma_flux[sort_idx]

        X_chunks = np.array_split(X_sorted, n_bins)
        Y_chunks = np.array_split(Y_sorted, n_bins)
        Err_chunks = np.array_split(Err_sorted, n_bins)

        X_binned, Y_binned, Err_binned = [], [], []

        # assemblo i miei bin
        for x_bin, y_bin, err_bin in zip(X_chunks, Y_chunks, Err_chunks):
            if len(x_bin) > 0:
                y_media_semplice = np.mean(y_bin)
                if len(y_bin) > 1:
                    y_errore_semplice = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin))
                    if y_errore_semplice == 0:
                        y_errore_semplice = np.mean(err_bin)
                else:
                    y_errore_semplice = err_bin[0]

                X_binned.append(np.mean(x_bin))
                Y_binned.append(y_media_semplice)
                Err_binned.append(y_errore_semplice)

        X_binned = np.array(X_binned)
        Y_binned = np.array(Y_binned)
        Err_binned = np.array(Err_binned)

        # applico il mio logaritmo ai dati binnati e propago l'errore
        Y_log_binned = np.log10(Y_binned)
        sigma_log_binned = (1 / np.log(10)) * (Err_binned / Y_binned)

        try:
            # eseguo il mio fit lineare sull'area specifica
            popt, pcov = curve_fit(modello_lineare, X_binned, Y_log_binned, sigma=sigma_log_binned, absolute_sigma=True)
            m_fit, q_fit = popt
            err_m, err_q = np.sqrt(np.diag(pcov))

            y_model_binned = modello_lineare(X_binned, m_fit, q_fit)
            dof = len(X_binned) - 2
            chi2_red = (np.sum(((Y_log_binned - y_model_binned) / sigma_log_binned) ** 2) / dof) if dof > 0 else 0

            print(
                f"Area {idx_area + 1} [{nome_area}]: m = {m_fit:.4f}±{err_m:.4f}, q = {q_fit:.4f}±{err_q:.4f}, Chi2_R = {chi2_red:.2f}")

            # ---------------------------------------------------------
            # DISEGNO GLI ELEMENTI PER L'AREA CORRENTE
            # ---------------------------------------------------------

            # disegno i miei punti originali (trasparenti) omettendo la label
            plt.errorbar(
                X, Y_flux, yerr=sigma_flux,
                fmt='o', markersize=2, color=colore, alpha=0.3
            )

            # disegno i miei bin usati per il fit (Verdi)
            plt.errorbar(
                X_binned, Y_binned, yerr=Err_binned,
                fmt='o', markersize=2, color=colore, ecolor=colore, capsize=1, alpha=0.6,
                label=f'Bin Fit', zorder=15
            )

            # definisco i limiti per il disegno della mia retta coprendo solo il range delle catalogate
            x_min_plot = df_fit_curr['Mag'].min()
            x_max_plot = df_fit_curr['Mag'].max()

            if not df_sature_plot.empty:
                x_min_plot = min(x_min_plot, df_sature_plot['Mag'].min())
                x_max_plot = max(x_max_plot, df_sature_plot['Mag'].max())

            x_plot = np.linspace(x_min_plot, x_max_plot, 100)
            y_plot_log = modello_lineare(x_plot, m_fit, q_fit)

            label_fit = rf'Fit {nome_area}: log(F)=({m_fit:.2f})M + ({q_fit:.2f})'
            plt.plot(x_plot, 10 ** y_plot_log, linestyle='--', linewidth=2, color=colore, label=label_fit, zorder=16)

        except Exception as e:
            print(f"Errore durante l'esecuzione del fit per l'area {nome_area}: {e}")

    # =============================================================================
    # DISEGNO GLI ELEMENTI GLOBALI (Saturi e Non Catalogati)
    # =============================================================================

    if not df_sature_plot.empty:
        # disegno le mie stelle sature omettendo la label
        plt.scatter(
            df_sature_plot['Mag'], df_sature_plot[col_media],
            s=40, c='red', marker='x', linewidth=1,
            zorder=20
        )

    if not df_no_match_plot.empty:
        # disegno i miei oggetti senza corrispondenza omettendo la label
        mag_fittizia = np.min(df_fit_curr['Mag']) - 1.5 if len(df_fit_curr) > 0 else 4.0
        plt.scatter(
            np.full(len(df_no_match_plot), mag_fittizia), df_no_match_plot[col_media],
            s=40, c='orange', marker='D', edgecolors='black', alpha=0.8,
            zorder=10
        )

        plt.annotate("Fictitious mag",
                     xy=(mag_fittizia, np.mean(df_no_match_plot[col_media])),
                     xytext=(mag_fittizia, np.max(df_no_match_plot[col_media]) * 1.5),
                     arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')

    # configuro il mio grafico finale
    plt.yscale('log')
    plt.gca().invert_xaxis()

    # imposto le etichette in UK English e ingrandisco i font per LaTeX
    plt.xlabel("Catalogue magnitude (Mag)", fontsize=24)
    plt.ylabel("Instrumental flux (ADU)", fontsize=24)

    plt.grid(True, which="both", ls="-", alpha=0.2)

    # dispongo la mia legenda normalmente all'interno del grafico
    plt.legend(fontsize=20, loc='best')
    plt.tight_layout()

    # salvo il mio grafico
    plt.savefig(f"fit_bin_aree_bordo.png")

    # mostro il mio grafico a schermo
    #plt.show()
    plt.close()

print("\n--- ELABORAZIONE COMPLETATA CON SUCCESSO ---")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

# configuro le mie dimensioni del sensore
W_IMG, H_IMG = 3072, 2048

# calcolo la mia distanza massima dal bordo
max_dist = min(W_IMG / 2, H_IMG / 2)

# definisco la mia mappa di colori dinamica
# mappo linearmente l'indice dell'area sull'intero range di viridis [0, 1]
cmap = plt.cm.viridis
colori_aree = [cmap(i / 4) for i in range(5)]

# inizio la costruzione della mia figura
fig, ax = plt.subplots(figsize=(10, 6))

# calcolo la mia ampiezza per ciascuna delle 5 aree
step = max_dist / 5

# disegno i miei rettangoli partendo dall'esterno verso l'interno per sovrapporli correttamente
for i in range(5):
    d = i * step
    w = W_IMG - 2 * d
    h = H_IMG - 2 * d

    # creo il mio rettangolo corrente con il colore corrispondente alla distanza calcolata
    rect = patches.Rectangle((d, d), w, h, linewidth=0, facecolor=colori_aree[i])
    ax.add_patch(rect)

# configuro i miei limiti degli assi per inquadrare esattamente il sensore
ax.set_xlim(0, W_IMG)
ax.set_ylim(0, H_IMG)

# mantengo le mie proporzioni reali tra gli assi
ax.set_aspect('equal')

# imposto le mie etichette
ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)

# creo la mia legenda personalizzata per mantenere coerenza coi colori dinamici
handles = [patches.Patch(color=colori_aree[i]) for i in range(5)]

# ottimizzo i miei margini
plt.tight_layout()

# salvo il mio rettangolo
plt.savefig("aree_uguali.png", dpi=300, bbox_inches='tight')

# chiudo la mia figura
plt.close()