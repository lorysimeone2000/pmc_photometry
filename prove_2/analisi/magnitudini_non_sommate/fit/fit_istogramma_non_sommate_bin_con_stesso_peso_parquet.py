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

# aggiungo alla mia lista tutte le colonne dei flussi che mi servono successivamente per il filtro
cols_needed = ['label', 'ID', 'Corrispondenza', 'Mag', 'saturazione']
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
    files_parquet = sorted(list(path_cartella.glob(f"run_{run}_immagine_*.parquet")))
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

# deduplico i miei dati ordinandoli per etichetta e magnitudine (crescente)
df_total_sorted = df_total.sort_values(by=['label', 'Mag'], ascending=[True, True])

# ora che il valore di Mag più basso è il primo di ogni gruppo, tengo la prima occorrenza per ogni label
df_unique = df_total_sorted.drop_duplicates(subset=['label'], keep='first').copy()
print(f"Oggetti UNICI totali (Catalogati + Non): {len(df_unique)}")

# isolo i miei non catalogati e rimuovo i duplicati per mantenere un solo elemento per label
# applico la mia maschera considerando che la colonna corrispondenza è un booleano
mask_match = df_unique['Corrispondenza'] == True
df_no_match = df_unique[~mask_match].copy()
df_no_match = df_no_match.drop_duplicates(subset=['label'])

# isolo i miei matchati
df_match_raw = df_unique[mask_match].copy()

# preparo il mio dizionario di aggregazione per raggruppare i dati per label
# imposto la magnitudine massima come valore di riferimento per il gruppo
agg_dict = {
    'Mag': 'min',
    'Corrispondenza': 'first',
    'ID': 'first'
}
if 'saturazione' in df_match_raw.columns:
    agg_dict['saturazione'] = 'first'
for flusso in FLUSSI_DA_ANALIZZARE:
    if f"media_{flusso}" in df_match_raw.columns:
        agg_dict[f"media_{flusso}"] = 'first'
    if f"std_{flusso}" in df_match_raw.columns:
        agg_dict[f"std_{flusso}"] = 'first'

# eseguo il mio raggruppamento per label applicando la magnitudine massima
df_match = df_match_raw.groupby('label').agg(agg_dict).reset_index()

# isolo le mie stelle sature usando il mio dataframe raggruppato
if 'saturazione' in df_match.columns:
    mask_sature = df_match['saturazione'] == True
    df_sature = df_match[mask_sature].copy()
    # tengo i miei non saturi per il fit
    df_fit_potential = df_match[~mask_sature].copy()
else:
    df_sature = pd.DataFrame()
    df_fit_potential = df_match.copy()

# applico il mio filtro di validità numerica base (Mag esistente)
mask_valid_base = df_fit_potential['Mag'].notna()
df_fit_valid = df_fit_potential[mask_valid_base].copy()

# applico il mio filtro sulla magnitudine per il fit
SOGLIA_MAG_FIT = 10.0
df_fit_clean = df_fit_valid[df_fit_valid['Mag'] <= SOGLIA_MAG_FIT].copy()

if not df_sature.empty:
    df_sature = df_sature[df_sature['Mag'] <= SOGLIA_MAG_FIT].copy()

print(f"Oggetti di base analizzabili (Match SI, No Saturi, Mag <= {SOGLIA_MAG_FIT}): {len(df_fit_clean)}")

# =============================================================================
# 3. FIT LINEARE E PLOTTING CICLICO
# =============================================================================

# inizio il mio ciclo sui tipi di flussi da analizzare
for flusso in FLUSSI_DA_ANALIZZARE:
    col_media = f"media_{flusso}"
    col_std = f"std_{flusso}"

    print(f"\n===========================================================")
    print(f"ANALISI: {flusso}")
    print(f"===========================================================")

    if col_media not in df_fit_clean.columns or col_std not in df_fit_clean.columns:
        print(f"Le colonne per '{flusso}' non sono presenti nel dataset. Salto l'analisi.")
        continue

    # isolo solo i miei elementi che hanno flussi ed errori positivi validi
    mask_flusso_valido = (
            (df_fit_clean[col_media].notna()) & (df_fit_clean[col_media] > 0) &
            (df_fit_clean[col_std].notna()) & (df_fit_clean[col_std] > 0)
    )
    df_fit_curr = df_fit_clean[mask_flusso_valido].copy()

    # filtro i miei validi anche per i saturi
    if not df_sature.empty and col_media in df_sature.columns:
        mask_sature_valido = (df_sature[col_media].notna()) & (df_sature[col_media] > 0)
        df_sature_plot = df_sature[mask_sature_valido].copy()
    else:
        df_sature_plot = pd.DataFrame()

    # filtro i miei validi anche per i non matchati
    if not df_no_match.empty and col_media in df_no_match.columns:
        mask_no_match_valido = (df_no_match[col_media].notna()) & (df_no_match[col_media] > 0)
        df_no_match_plot = df_no_match[mask_no_match_valido].copy()
    else:
        df_no_match_plot = pd.DataFrame()

    if len(df_fit_curr) > 2:
        # estraggo i miei dati originali X, Y e relativi errori
        X = df_fit_curr['Mag'].values
        Y_flux = df_fit_curr[col_media].values
        sigma_flux = df_fit_curr[col_std].values

        # definisco il mio numero di bin
        n_bins = max(5, int(np.sqrt(len(X))))

        # ordino i miei array per magnitudine prima di spezzarli in chunk
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        Y_sorted = Y_flux[sort_idx]
        Err_sorted = sigma_flux[sort_idx]

        # divido i miei array in blocchi con lo stesso numero di stelle
        X_chunks = np.array_split(X_sorted, n_bins)
        Y_chunks = np.array_split(Y_sorted, n_bins)
        Err_chunks = np.array_split(Err_sorted, n_bins)

        X_binned = []
        Y_binned = []
        Err_binned = []

        # assemblo i miei bin
        for x_bin, y_bin, err_bin in zip(X_chunks, Y_chunks, Err_chunks):
            if len(x_bin) > 0:
                # calcolo la mia media semplice del flusso
                y_media_semplice = np.mean(y_bin)

                # calcolo la mia deviazione standard della media
                if len(y_bin) > 1:
                    y_errore_semplice = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin))
                    if y_errore_semplice == 0:
                        y_errore_semplice = np.mean(err_bin)
                else:
                    y_errore_semplice = err_bin[0]

                x_mean = np.mean(x_bin)

                X_binned.append(x_mean)
                Y_binned.append(y_media_semplice)
                Err_binned.append(y_errore_semplice)

        X_binned = np.array(X_binned)
        Y_binned = np.array(Y_binned)
        Err_binned = np.array(Err_binned)

        # applico il mio logaritmo ai dati binnati
        Y_log_binned = np.log10(Y_binned)

        # propago il mio errore sui dati binnati
        sigma_log_binned = (1 / np.log(10)) * (Err_binned / Y_binned)

        try:
            # eseguo il mio fit lineare usando i miei dati binnati
            popt, pcov = curve_fit(modello_lineare, X_binned, Y_log_binned, sigma=sigma_log_binned, absolute_sigma=True)
            m_fit, q_fit = popt
            err_m, err_q = np.sqrt(np.diag(pcov))

            # calcolo il mio Chi Quadro ridotto
            y_model_binned = modello_lineare(X_binned, m_fit, q_fit)
            chi2 = np.sum(((Y_log_binned - y_model_binned) / sigma_log_binned) ** 2)
            dof = len(X_binned) - 2
            chi2_red = chi2 / dof if dof > 0 else 0

            print(f"Risultati Fit Binnato per {flusso}:")
            print(f"m = {m_fit:.4f} ± {err_m:.4f}")
            print(f"q = {q_fit:.4f} ± {err_q:.4f}")
            print(f"Chi2 Ridotto = {chi2_red:.2f}")

            # =============================================================================
            # 4. PLOTTING (STILE AGGIORNATO)
            # =============================================================================
            plt.figure(figsize=(12, 9))

            # disegno i miei punti originali (Blu con barre errore chiare)
            plt.errorbar(
                X, Y_flux, yerr=sigma_flux,
                fmt='o', markersize=1, color='blue', ecolor='lightblue', alpha=0.7,
                label=f'Catalogati Validi ({len(X)})'
            )

            # disegno i miei bin usati per il fit (Verdi)
            plt.errorbar(
                X_binned, Y_binned, yerr=Err_binned,
                fmt='o', markersize=5, color='green', ecolor='green', capsize=3, alpha=0.9,
                label=f'Bin Fit ({len(X_binned)})', zorder=15
            )

            # disegno le mie stelle sature (X Rosse)
            if not df_sature_plot.empty:
                plt.scatter(
                    df_sature_plot['Mag'],
                    df_sature_plot[col_media],
                    s=40, c='red', marker='x', linewidth=1,
                    label=f'Sature (Escluse) ({len(df_sature_plot)})', zorder=20
                )

            # disegno i miei oggetti senza corrispondenza (Rombi Arancioni)
            if not df_no_match_plot.empty:
                mag_fittizia = np.min(X) - 1.5 if len(X) > 0 else 4.0
                plt.scatter(
                    np.full(len(df_no_match_plot), mag_fittizia),
                    df_no_match_plot[col_media],
                    s=40, c='orange', marker='D', edgecolors='black', alpha=0.8,
                    label=f'NON Catalogati ({len(df_no_match_plot)})', zorder=10
                )

                plt.annotate("Mag Fittizia",
                             xy=(mag_fittizia, np.mean(df_no_match_plot[col_media])),
                             xytext=(mag_fittizia, np.max(df_no_match_plot[col_media]) * 1.5),
                             arrowprops=dict(facecolor='black', arrowstyle='->'),
                             ha='center')

                x_min_plot = min(np.min(X), mag_fittizia - 0.5)
            else:
                x_min_plot = np.min(X) - 0.5

            # disegno la mia retta di fit
            x_max_plot = np.max(X) + 0.5
            x_plot = np.linspace(x_min_plot, x_max_plot, 100)
            y_plot_log = modello_lineare(x_plot, m_fit, q_fit)

            label_fit = (rf'Fit Binnato: log(F)=({m_fit:.2f}$\pm${err_m:.2f})M + ({q_fit:.2f}$\pm${err_q:.2f})'
                         rf'$\chi^2_R$={chi2_red:.2f}')

            plt.plot(x_plot, 10 ** y_plot_log, 'g--', linewidth=2, label=label_fit, zorder=16)

            # configuro il mio grafico
            plt.yscale('log')
            plt.gca().invert_xaxis()
            plt.xlabel("Magnitudine Catalogo (Mag)", fontsize=12)
            plt.ylabel(f"Media {flusso} (ADU)", fontsize=12)
            plt.title(f"Calibrazione Fotometrica Globale (Run {RUN_TO_ANALYZE}) - {flusso} \n metodo bin uguali", fontsize=14)
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend(fontsize=11, loc='best')
            plt.tight_layout()

            # salvo il mio grafico
            plt.savefig(f"fit_bin_pesi_uguali.png")

            # mostro il mio grafico a schermo
            plt.show()

            plt.close()

        except Exception as e:
            print(f"Errore durante l'esecuzione del fit per {flusso}: {e}")
    else:
        print(f"Non ho abbastanza punti validi per eseguire il fit di {flusso}.")

print("\n--- ELABORAZIONE COMPLETATA CON SUCCESSO ---")