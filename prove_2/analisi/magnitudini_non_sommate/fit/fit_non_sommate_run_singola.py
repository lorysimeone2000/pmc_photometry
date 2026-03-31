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

# deduplico i miei dati ordinandoli
df_total_sorted = df_total.sort_values(by=['label', 'Mag'], ascending=[True, True])
# prendo solo i miei primi valori per ID
df_unique = df_total_sorted.drop_duplicates(subset=['ID'], keep='first').copy()
print(f"Oggetti UNICI totali (Catalogati + Non): {len(df_unique)}")

# isolo i miei non catalogati e rimuovo i duplicati per mantenere un solo elemento per label
mask_match = df_unique['Corrispondenza'].astype(str).str.startswith('SI')
df_no_match = df_unique[~mask_match].copy()
df_no_match = df_no_match.drop_duplicates(subset=['label'])

# isolo i miei matchati
df_match_raw = df_unique[mask_match].copy()

# preparo il mio dizionario di aggregazione per raggruppare i dati per label
# imposto la magnitudine minima come valore di riferimento per il mio gruppo
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

# eseguo il mio raggruppamento per label applicando le direttive del dizionario
df_match = df_match_raw.groupby('label').agg(agg_dict).reset_index()

# isolo le mie stelle sature usando il mio dataframe raggruppato
if 'saturazione' in df_match.columns:
    mask_sature = df_match['saturazione'].astype(str).str.startswith('SI')
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
        # estraggo i miei dati X e Y
        X = df_fit_curr['Mag'].values
        Y_flux = df_fit_curr[col_media].values
        Y_log = np.log10(Y_flux)

        # estraggo i miei errori
        sigma_flux = df_fit_curr[col_std].values

        # propago il mio errore sul logaritmo
        sigma_log = (1 / np.log(10)) * (sigma_flux / Y_flux)

        try:
            # eseguo il mio fit
            popt, pcov = curve_fit(modello_lineare, X, Y_log, sigma=sigma_log, absolute_sigma=True)
            m_fit, q_fit = popt
            err_m, err_q = np.sqrt(np.diag(pcov))

            # calcolo il mio Chi Quadro ridotto
            y_model = modello_lineare(X, m_fit, q_fit)
            chi2 = np.sum(((Y_log - y_model) / sigma_log) ** 2)
            dof = len(X) - 2
            chi2_red = chi2 / dof if dof > 0 else 0

            print(f"Risultati per {flusso}:")
            print(f"m = {m_fit:.4f} ± {err_m:.4f}")
            print(f"q = {q_fit:.4f} ± {err_q:.4f}")
            print(f"Chi2 Ridotto = {chi2_red:.2f}")

            # =============================================================================
            # 4. PLOTTING (STILE AGGIORNATO)
            # =============================================================================
            # creo il mio grafico aggiornato
            plt.figure(figsize=(12, 9))

            # disegno i miei punti usati per il fit (Blu con barre errore chiare)
            plt.errorbar(
                X, Y_flux, yerr=sigma_flux,
                fmt='o', markersize=1, color='blue', ecolor='lightblue', alpha=0.7,
                label=f'Catalogati Validi ({len(X)})'
            )

            # disegno le mie stelle sature (X Rosse)
            # mi assicuro di avere i miei dati validi per il plot
            if not df_sature_plot.empty:
                plt.scatter(
                    df_sature_plot['Mag'],
                    df_sature_plot[col_media],
                    s=40, c='red', marker='x', linewidth=1,
                    label=f'Sature (Escluse) ({len(df_sature_plot)})', zorder=20
                )

            # disegno i miei oggetti senza corrispondenza (Rombi Arancioni)
            if not df_no_match_plot.empty:
                # calcolo la mia magnitudine fittizia per posizionarli a sinistra
                mag_fittizia = np.min(X) - 1.5 if len(X) > 0 else 4.0

                plt.scatter(
                    np.full(len(df_no_match_plot), mag_fittizia),
                    df_no_match_plot[col_media],
                    s=40, c='orange', marker='D', edgecolors='black', alpha=0.8,
                    label=f'NON Catalogati ({len(df_no_match_plot)})', zorder=10
                )

                # aggiungo la mia annotazione
                plt.annotate("Mag Fittizia",
                             xy=(mag_fittizia, np.mean(df_no_match_plot[col_media])),
                             xytext=(mag_fittizia, np.max(df_no_match_plot[col_media]) * 1.5),
                             arrowprops=dict(facecolor='black', arrowstyle='->'),
                             ha='center')

                # aggiorno i miei limiti del plot se necessario
                x_min_plot = min(np.min(X), mag_fittizia - 0.5)
            else:
                x_min_plot = np.min(X) - 0.5

            # disegno la mia retta di fit (Nera tratteggiata)
            x_max_plot = np.max(X) + 0.5
            x_plot = np.linspace(x_min_plot, x_max_plot, 100)
            y_plot_log = modello_lineare(x_plot, m_fit, q_fit)

            # aggiungo il mio errore sui parametri nella label
            # uso rf anche sulla seconda stringa per correggere il SyntaxWarning
            label_fit = (rf'Fit: log(F)=({m_fit:.2f}$\pm${err_m:.2f})M + ({q_fit:.2f}$\pm${err_q:.2f})'
                         rf'$\chi^2_R$={chi2_red:.2f}')

            plt.plot(x_plot, 10 ** y_plot_log, 'k--', linewidth=2, label=label_fit)

            # configuro il mio grafico
            plt.yscale('log')
            plt.gca().invert_xaxis()
            plt.xlabel("Magnitudine Catalogo (Mag)", fontsize=12)
            plt.ylabel(f"Media {flusso} (ADU)", fontsize=12)
            plt.title(f"Calibrazione Fotometrica Globale (Run {RUN_TO_ANALYZE}) - {flusso}", fontsize=14)
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend(fontsize=11, loc='best')
            plt.tight_layout()

            # salvo il mio grafico
            out_file = f"fit_normale_run_1.png"
            plt.savefig(out_file, dpi=300)
            print(f"Grafico salvato: {out_file}")

            # plt.show()

            # chiudo la mia figura per evitare che si aprano troppe finestre in simultanea
            plt.close()

        except Exception as e:
            print(f"Errore durante l'esecuzione del fit per {flusso}: {e}")
    else:
        print(f"Non ho abbastanza punti validi per eseguire il fit di {flusso}.")

print("\n--- ELABORAZIONE COMPLETATA CON SUCCESSO ---")