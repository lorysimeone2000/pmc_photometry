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
    files_parquet = sorted(list(path_cartella.glob(f"run_{run}_stelle_trovate_e_catalogate_immagine_*.parquet")))
    print(f"Run {run}: Trovati {len(files_parquet)} file parquet. Caricamento in corso...")

    for f in tqdm(files_parquet, leave=False):
        try:
            df_temp = pd.read_parquet(f)
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
# 2. PREPARAZIONE DATI GLOBALI PER IL GRAFICO DI SFONDO
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

print(f"Oggetti Catalogati UNICI Globali: {len(df_match)}")
print(f"Oggetti NON Catalogati UNICI Globali: {len(df_no_match)}")

# isolo le mie stelle sature lavorando sul dataframe delle catalogate
if 'saturazione' in df_match.columns:
    mask_sature = df_match['saturazione'] == True
    df_sature = df_match[mask_sature].copy()
    # mantengo i miei dati non saturi come potenziali candidati
    df_fit_potential = df_match[~mask_sature].copy()
else:
    df_sature = pd.DataFrame()
    df_fit_potential = df_match.copy()

# applico il mio filtro di validità numerica base per assicurarmi che esista una magnitudine
mask_valid_base = df_fit_potential['Mag'].notna()
df_fit_valid = df_fit_potential[mask_valid_base].copy()

# applico la mia soglia di magnitudine limite per pulire i dati
SOGLIA_MAG_FIT = 10.0
df_fit_clean = df_fit_valid[df_fit_valid['Mag'] <= SOGLIA_MAG_FIT].copy()

if not df_sature.empty:
    df_sature = df_sature[df_sature['Mag'] <= SOGLIA_MAG_FIT].copy()

print(f"Oggetti di base analizzabili globali (Match SI, No Saturi, Mag <= {SOGLIA_MAG_FIT}): {len(df_fit_clean)}")

# =============================================================================
# 3. FIT LINEARI SEPARATI E PLOTTING CICLICO
# =============================================================================

# inizio il mio ciclo sui tipi di flussi da analizzare
for flusso in FLUSSI_DA_ANALIZZARE:
    col_media = f"media_{flusso}"
    col_std = f"std_{flusso}"

    print(f"\n===========================================================")
    print(f"ANALISI: {flusso}")
    print(f"===========================================================")

    # mi assicuro che le mie colonne esistano prima di procedere
    if col_media not in df_fit_clean.columns or col_std not in df_fit_clean.columns:
        print(f"Le colonne per '{flusso}' non sono presenti nel dataset. Salto l'analisi.")
        continue

    # isolo solo i miei elementi che hanno flussi ed errori positivi validi per questo specifico parametro
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
        # estraggo i miei dati X e Y globali per i limiti del plot
        X_globale = df_fit_curr['Mag'].values

        try:
            # =============================================================================
            # 4. PLOTTING E CALCOLO FIT PER SINGOLA RUN
            # =============================================================================
            # creo il mio grafico impostando dimensioni ottimali per 0.45\textwidth
            plt.figure(figsize=(4.66, 3.5))

            # disegno le mie stelle sature (X Rosse) omettendo la label
            # mi assicuro di avere i miei dati validi per il plot
            if not df_sature_plot.empty:
                plt.scatter(
                    df_sature_plot['Mag'],
                    df_sature_plot[col_media],
                    s=15, c='red', marker='x', linewidth=1,
                    zorder=20
                )

            # disegno i miei oggetti senza corrispondenza (Rombi Arancioni) omettendo la label
            if not df_no_match_plot.empty:
                # calcolo la mia magnitudine fittizia per posizionarli a sinistra
                mag_fittizia = np.min(X_globale) - 1.5 if len(X_globale) > 0 else 4.0

                plt.scatter(
                    np.full(len(df_no_match_plot), mag_fittizia),
                    df_no_match_plot[col_media],
                    s=20, c='orange', marker='D', edgecolors='black', alpha=0.8,
                    zorder=10
                )

                # aggiungo la mia annotazione formattata ridimensionandola per un grafico più piccolo
                plt.annotate("Dummy magnitude",
                             xy=(mag_fittizia, np.max(df_no_match_plot[col_media]) * 1.1),
                             xytext=(mag_fittizia, np.max(df_no_match_plot[col_media]) * 2.0),
                             arrowprops=dict(facecolor='black', arrowstyle='->'),
                             ha='center', fontsize=8)

                # aggiorno i miei limiti del plot se necessario
                x_min_plot = min(np.min(X_globale), mag_fittizia - 0.5)
            else:
                x_min_plot = np.min(X_globale) - 0.5

            x_max_plot = np.max(X_globale) + 0.5
            x_plot = np.linspace(x_min_plot, x_max_plot, 100)

            # definisco il mio dizionario dei colori per i fit richiesti
            colori_fit = {1: 'blue', 2: 'violet', 3: 'green'}

            # eseguo il mio ciclo per calcolare e tracciare un fit normale per ogni run
            for run_id in RUN_TO_ANALYZE:
                # isolo i miei dati per la run corrente dal dataframe grezzo
                df_run = df_total[df_total['run_origin'] == run_id].copy()

                # filtro e deduplico i miei dati specifici per la singola run
                mask_match_run = df_run['Corrispondenza'] == True
                df_match_run = df_run[mask_match_run].copy()

                df_match_run_sorted = df_match_run.sort_values(by=['label', 'Mag'], ascending=[True, True])
                df_match_run_dedup = df_match_run_sorted.drop_duplicates(subset=['label'], keep='first').copy()

                # rimuovo i miei dati saturi per la singola run
                if 'saturazione' in df_match_run_dedup.columns:
                    mask_sature_run = df_match_run_dedup['saturazione'] == True
                    df_run_fit_potential = df_match_run_dedup[~mask_sature_run].copy()
                else:
                    df_run_fit_potential = df_match_run_dedup.copy()

                # applico i miei filtri numerici
                mask_run_valid_base = df_run_fit_potential['Mag'].notna()
                df_run_fit_valid = df_run_fit_potential[mask_run_valid_base].copy()
                df_run_fit_clean = df_run_fit_valid[df_run_fit_valid['Mag'] <= SOGLIA_MAG_FIT].copy()

                mask_run_flusso = (
                        (df_run_fit_clean[col_media].notna()) & (df_run_fit_clean[col_media] > 0) &
                        (df_run_fit_clean[col_std].notna()) & (df_run_fit_clean[col_std] > 0)
                )
                df_run_final = df_run_fit_clean[mask_run_flusso].copy()

                if len(df_run_final) > 2:
                    # estraggo i miei dati della singola run per il fit normale
                    X_run = df_run_final['Mag'].values
                    Y_flux_run = df_run_final[col_media].values
                    sigma_flux_run = df_run_final[col_std].values

                    colore_linea = colori_fit.get(run_id, 'black')

                    # disegno i miei punti della singola run omettendo la label per ripulire la legenda
                    plt.errorbar(
                        X_run, Y_flux_run, yerr=sigma_flux_run,
                        fmt='o', markersize=.3, color=colore_linea, ecolor=colore_linea, alpha=0.7,
                        zorder=15
                    )

                    # calcolo il mio logaritmo e propago il mio errore
                    Y_log_run = np.log10(Y_flux_run)
                    sigma_log_run = (1 / np.log(10)) * (sigma_flux_run / Y_flux_run)

                    # eseguo il mio fit lineare
                    popt, pcov = curve_fit(modello_lineare, X_run, Y_log_run, sigma=sigma_log_run, absolute_sigma=True)
                    m_fit, q_fit = popt
                    err_m, err_q = np.sqrt(np.diag(pcov))

                    # calcolo il mio Chi Quadro ridotto
                    y_model_run = modello_lineare(X_run, m_fit, q_fit)
                    chi2 = np.sum(((Y_log_run - y_model_run) / sigma_log_run) ** 2)
                    dof = len(X_run) - 2
                    chi2_red = chi2 / dof if dof > 0 else 0

                    print(f"Risultati per {flusso} - Run {run_id}:")
                    print(f"m = {m_fit:.4f} ± {err_m:.4f}")
                    print(f"q = {q_fit:.4f} ± {err_q:.4f}")
                    print(f"Chi2 Ridotto = {chi2_red:.2f}")

                    # calcolo la mia funzione logaritmica da plottare
                    y_plot_log = modello_lineare(x_plot, m_fit, q_fit)

                    # formatto le mie stringhe di errore per omettere il valore 0.00
                    str_err_m = f"$\\pm${err_m:.2f}" if f"{err_m:.2f}" != "0.00" else ""
                    str_err_q = f"$\\pm${err_q:.2f}" if f"{err_q:.2f}" != "0.00" else ""

                    # aggiungo il mio errore sui parametri nella label
                    label_fit = rf'Fit Run {run_id}: log(F)=({m_fit:.2f}{str_err_m})M + ({q_fit:.2f}{str_err_q})'

                    # disegno la mia retta di fit (non tratteggiata, spessore 1, colore dedicato)
                    plt.plot(x_plot, 10 ** y_plot_log, color=colore_linea, linestyle='-', linewidth=.3, label=label_fit,
                             zorder=16)

                else:
                    print(f"Non ho abbastanza punti validi per eseguire il fit di {flusso} per la Run {run_id}.")

            # configuro il mio grafico scalando i testi per adattarli alle nuove dimensioni
            plt.yscale('log')
            plt.gca().invert_xaxis()

            # dimensiono i tick degli assi scalandoli per LaTeX
            plt.tick_params(axis='both', which='major', labelsize=8)

            # dimensiono le label degli assi scalandole per LaTeX
            plt.xlabel("Catalogue magnitude (mag)", fontsize=10)
            plt.ylabel("Instrumental flux (ADU)", fontsize=10)
            plt.grid(True, which="both", ls="-", alpha=0.2)

            # dimensiono la grandezza della legenda scalandola per LaTeX
            plt.legend(fontsize=8, loc='best')
            plt.tight_layout()

            # salvo il mio grafico ad alta risoluzione
            out_file = f"fit_normale_confronto_tra_run.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            print(f"Grafico salvato: {out_file}")

            # plt.show()

            # chiudo la mia figura per evitare che si aprano troppe finestre in simultanea
            plt.close()

        except Exception as e:
            print(f"Errore durante l'esecuzione del fit per {flusso}: {e}")
    else:
        print(f"Non ho abbastanza punti validi per eseguire il fit di {flusso}.")

print("\n--- ELABORAZIONE COMPLETATA CON SUCCESSO ---")