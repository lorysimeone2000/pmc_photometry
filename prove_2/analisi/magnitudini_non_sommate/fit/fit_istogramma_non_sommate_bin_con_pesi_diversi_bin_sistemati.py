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

        # =============================================================================
        # METODO 1: FUSIONE DEI BIN CON 0 O 1 STELLA
        # =============================================================================
        print("\nEsecuzione Metodo 1: Fusione bin sparsi...")

        # definisco i miei bin iniziali di larghezza uguale
        n_bins_iniziali = max(5, int(np.sqrt(len(X))))
        bins_iniziali = np.linspace(X.min(), X.max(), n_bins_iniziali + 1)

        bin_data = []

        # raggruppo i miei dati nei bin iniziali
        for i in range(n_bins_iniziali):
            if i == n_bins_iniziali - 1:
                mask = (X >= bins_iniziali[i]) & (X <= bins_iniziali[i + 1])
            else:
                mask = (X >= bins_iniziali[i]) & (X < bins_iniziali[i + 1])
            bin_data.append((X[mask], Y_flux[mask], sigma_flux[mask]))

        # unisco i miei bin che hanno 0 o 1 stella al precedente
        i = 1
        while i < len(bin_data):
            if len(bin_data[i][0]) <= 1:
                # fondo il mio bin con il precedente
                x_fuso = np.concatenate((bin_data[i - 1][0], bin_data[i][0]))
                y_fuso = np.concatenate((bin_data[i - 1][1], bin_data[i][1]))
                err_fuso = np.concatenate((bin_data[i - 1][2], bin_data[i][2]))
                bin_data[i - 1] = (x_fuso, y_fuso, err_fuso)
                bin_data.pop(i)
            else:
                i += 1

        # controllo il mio primo bin (se ha <= 1 stella lo unisco al successivo)
        if len(bin_data) > 1 and len(bin_data[0][0]) <= 1:
            x_fuso = np.concatenate((bin_data[0][0], bin_data[1][0]))
            y_fuso = np.concatenate((bin_data[0][1], bin_data[1][1]))
            err_fuso = np.concatenate((bin_data[0][2], bin_data[1][2]))
            bin_data[1] = (x_fuso, y_fuso, err_fuso)
            bin_data.pop(0)

        X_binned_1 = []
        Y_binned_1 = []
        Err_binned_1 = []

        # calcolo le mie medie finali per i bin fusi
        for x_bin, y_bin, err_bin in bin_data:
            if len(y_bin) > 0:
                y_media_semplice = np.mean(y_bin)
                if len(y_bin) > 1:
                    y_errore_semplice = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin))
                    if y_errore_semplice == 0:
                        y_errore_semplice = np.mean(err_bin)
                else:
                    y_errore_semplice = err_bin[0]

                X_binned_1.append(np.mean(x_bin))
                Y_binned_1.append(y_media_semplice)
                Err_binned_1.append(y_errore_semplice)

        X_binned_1 = np.array(X_binned_1)
        Y_binned_1 = np.array(Y_binned_1)
        Err_binned_1 = np.array(Err_binned_1)

        # applico il mio logaritmo ai dati binnati
        Y_log_binned_1 = np.log10(Y_binned_1)

        # propago il mio errore sui dati binnati
        sigma_log_binned_1 = (1 / np.log(10)) * (Err_binned_1 / Y_binned_1)

        try:
            # eseguo il mio fit lineare usando i miei dati binnati fusi
            popt_1, pcov_1 = curve_fit(modello_lineare, X_binned_1, Y_log_binned_1, sigma=sigma_log_binned_1,
                                       absolute_sigma=True)
            m_fit_1, q_fit_1 = popt_1
            err_m_1, err_q_1 = np.sqrt(np.diag(pcov_1))

            # calcolo il mio Chi Quadro ridotto
            y_model_binned_1 = modello_lineare(X_binned_1, m_fit_1, q_fit_1)
            chi2_1 = np.sum(((Y_log_binned_1 - y_model_binned_1) / sigma_log_binned_1) ** 2)
            dof_1 = len(X_binned_1) - 2
            chi2_red_1 = chi2_1 / dof_1 if dof_1 > 0 else 0

            print(f"Risultati Fit Metodo 1 (Fusione) per {flusso}:")
            print(f"m = {m_fit_1:.4f} ± {err_m_1:.4f}")
            print(f"q = {q_fit_1:.4f} ± {err_q_1:.4f}")
            print(f"Chi2 Ridotto = {chi2_red_1:.2f}")

            # disegno la mia prima figura
            plt.figure(figsize=(12, 9))

            # disegno i miei punti originali
            plt.errorbar(X, Y_flux, yerr=sigma_flux, fmt='o', markersize=1, color='blue', ecolor='lightblue', alpha=0.7,
                         label=f'Catalogati Validi ({len(X)})')

            # disegno i miei bin fusi usati per il fit
            plt.errorbar(X_binned_1, Y_binned_1, yerr=Err_binned_1, fmt='o', markersize=5, color='green',
                         ecolor='green', capsize=3, alpha=0.9, label=f'Bin Fit ({len(X_binned_1)})', zorder=15)

            # disegno le mie stelle sature
            if not df_sature_plot.empty:
                plt.scatter(df_sature_plot['Mag'], df_sature_plot[col_media], s=40, c='red', marker='x', linewidth=1,
                            label=f'Sature (Escluse) ({len(df_sature_plot)})', zorder=20)

            # disegno i miei oggetti senza corrispondenza
            if not df_no_match_plot.empty:
                mag_fittizia = np.min(X) - 1.5 if len(X) > 0 else 4.0
                plt.scatter(np.full(len(df_no_match_plot), mag_fittizia), df_no_match_plot[col_media], s=40, c='orange',
                            marker='D', edgecolors='black', alpha=0.8,
                            label=f'NON Catalogati ({len(df_no_match_plot)})', zorder=10)
                plt.annotate("Mag Fittizia", xy=(mag_fittizia, np.mean(df_no_match_plot[col_media])),
                             xytext=(mag_fittizia, np.max(df_no_match_plot[col_media]) * 1.5),
                             arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')
                x_min_plot = min(np.min(X), mag_fittizia - 0.5)
            else:
                x_min_plot = np.min(X) - 0.5

            # disegno la mia retta di fit
            x_max_plot = np.max(X) + 0.5
            x_plot = np.linspace(x_min_plot, x_max_plot, 100)
            y_plot_log_1 = modello_lineare(x_plot, m_fit_1, q_fit_1)

            label_fit_1 = (
                rf'Fit Binnato: log(F)=({m_fit_1:.2f}$\pm${err_m_1:.2f})M + ({q_fit_1:.2f}$\pm${err_q_1:.2f})' rf'\n$\chi^2_R$={chi2_red_1:.2f}')
            plt.plot(x_plot, 10 ** y_plot_log_1, 'g--', linewidth=2, label=label_fit_1, zorder=16)

            # configuro il mio grafico
            plt.yscale('log')
            plt.gca().invert_xaxis()
            plt.xlabel("Magnitudine Catalogo (Mag)", fontsize=12)
            plt.ylabel(f"Media {flusso} (ADU)", fontsize=12)
            plt.title(f"Calibrazione Fotometrica Globale (Run {RUN_TO_ANALYZE}) - {flusso} \n metodo bin con fusione",
                      fontsize=14)
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend(fontsize=11, loc='best')
            plt.tight_layout()

            # salvo il mio primo grafico
            plt.savefig(f"fit_binnato_{flusso}_fusione_style.png")
            plt.close()

        except Exception as e:
            print(f"Errore durante l'esecuzione del fit (Metodo 1) per {flusso}: {e}")

        # =============================================================================
        # METODO 2: RICERCA NUMERO BIN PER AVERE LARGHEZZA UGUALE E >= 3 STELLE
        # =============================================================================
        print("\nEsecuzione Metodo 2: Ottimizzazione bin uniformi (minimo 3 stelle)...")

        # cerco il mio numero di bin ottimale partendo da quello calcolato in precedenza
        n_bins_opt = n_bins_iniziali

        # scendo con il numero di bin finché tutti i miei bin hanno almeno 3 stelle
        while n_bins_opt > 1:
            bins_opt = np.linspace(X.min(), X.max(), n_bins_opt + 1)
            tutti_validi = True
            for i in range(n_bins_opt):
                if i == n_bins_opt - 1:
                    mask = (X >= bins_opt[i]) & (X <= bins_opt[i + 1])
                else:
                    mask = (X >= bins_opt[i]) & (X < bins_opt[i + 1])

                if np.sum(mask) < 3:
                    tutti_validi = False
                    break

            if tutti_validi:
                break
            n_bins_opt -= 1

        # preparo i miei nuovi bin ottimali
        bins_opt = np.linspace(X.min(), X.max(), n_bins_opt + 1)

        X_binned_2 = []
        Y_binned_2 = []
        Err_binned_2 = []

        # calcolo le mie medie per i bin uniformi ottimali
        for i in range(n_bins_opt):
            if i == n_bins_opt - 1:
                mask = (X >= bins_opt[i]) & (X <= bins_opt[i + 1])
            else:
                mask = (X >= bins_opt[i]) & (X < bins_opt[i + 1])

            if np.sum(mask) > 0:
                x_bin = X[mask]
                y_bin = Y_flux[mask]
                err_bin = sigma_flux[mask]

                y_media_semplice = np.mean(y_bin)

                if len(y_bin) > 1:
                    y_errore_semplice = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin))
                    if y_errore_semplice == 0:
                        y_errore_semplice = np.mean(err_bin)
                else:
                    y_errore_semplice = err_bin[0]

                X_binned_2.append(np.mean(x_bin))
                Y_binned_2.append(y_media_semplice)
                Err_binned_2.append(y_errore_semplice)

        X_binned_2 = np.array(X_binned_2)
        Y_binned_2 = np.array(Y_binned_2)
        Err_binned_2 = np.array(Err_binned_2)

        # applico il mio logaritmo ai dati binnati
        Y_log_binned_2 = np.log10(Y_binned_2)

        # propago il mio errore sui dati binnati
        sigma_log_binned_2 = (1 / np.log(10)) * (Err_binned_2 / Y_binned_2)

        try:
            # eseguo il mio fit lineare usando i miei dati binnati uniformi
            popt_2, pcov_2 = curve_fit(modello_lineare, X_binned_2, Y_log_binned_2, sigma=sigma_log_binned_2,
                                       absolute_sigma=True)
            m_fit_2, q_fit_2 = popt_2
            err_m_2, err_q_2 = np.sqrt(np.diag(pcov_2))

            # calcolo il mio Chi Quadro ridotto
            y_model_binned_2 = modello_lineare(X_binned_2, m_fit_2, q_fit_2)
            chi2_2 = np.sum(((Y_log_binned_2 - y_model_binned_2) / sigma_log_binned_2) ** 2)
            dof_2 = len(X_binned_2) - 2
            chi2_red_2 = chi2_2 / dof_2 if dof_2 > 0 else 0

            print(f"Risultati Fit Metodo 2 (Uniforme) per {flusso}:")
            print(f"m = {m_fit_2:.4f} ± {err_m_2:.4f}")
            print(f"q = {q_fit_2:.4f} ± {err_q_2:.4f}")
            print(f"Chi2 Ridotto = {chi2_red_2:.2f}")

            # disegno la mia seconda figura
            plt.figure(figsize=(12, 9))

            # disegno i miei punti originali
            plt.errorbar(X, Y_flux, yerr=sigma_flux, fmt='o', markersize=1, color='blue', ecolor='lightblue', alpha=0.7,
                         label=f'Catalogati Validi ({len(X)})')

            # disegno i miei bin uniformi usati per il fit
            plt.errorbar(X_binned_2, Y_binned_2, yerr=Err_binned_2, fmt='o', markersize=5, color='green',
                         ecolor='green', capsize=3, alpha=0.9, label=f'Bin Fit ({len(X_binned_2)})', zorder=15)

            # disegno le mie stelle sature
            if not df_sature_plot.empty:
                plt.scatter(df_sature_plot['Mag'], df_sature_plot[col_media], s=40, c='red', marker='x', linewidth=1,
                            label=f'Sature (Escluse) ({len(df_sature_plot)})', zorder=20)

            # disegno i miei oggetti senza corrispondenza
            if not df_no_match_plot.empty:
                mag_fittizia = np.min(X) - 1.5 if len(X) > 0 else 4.0
                plt.scatter(np.full(len(df_no_match_plot), mag_fittizia), df_no_match_plot[col_media], s=40, c='orange',
                            marker='D', edgecolors='black', alpha=0.8,
                            label=f'NON Catalogati ({len(df_no_match_plot)})', zorder=10)
                plt.annotate("Mag Fittizia", xy=(mag_fittizia, np.mean(df_no_match_plot[col_media])),
                             xytext=(mag_fittizia, np.max(df_no_match_plot[col_media]) * 1.5),
                             arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')
                x_min_plot = min(np.min(X), mag_fittizia - 0.5)
            else:
                x_min_plot = np.min(X) - 0.5

            # disegno la mia retta di fit
            x_max_plot = np.max(X) + 0.5
            x_plot = np.linspace(x_min_plot, x_max_plot, 100)
            y_plot_log_2 = modello_lineare(x_plot, m_fit_2, q_fit_2)

            label_fit_2 = (
                rf'Fit Binnato: log(F)=({m_fit_2:.2f}$\pm${err_m_2:.2f})M + ({q_fit_2:.2f}$\pm${err_q_2:.2f})' rf'\n$\chi^2_R$={chi2_red_2:.2f}')
            plt.plot(x_plot, 10 ** y_plot_log_2, 'g--', linewidth=2, label=label_fit_2, zorder=16)

            # configuro il mio grafico
            plt.yscale('log')
            plt.gca().invert_xaxis()
            plt.xlabel("Magnitudine Catalogo (Mag)", fontsize=12)
            plt.ylabel(f"Media {flusso} (ADU)", fontsize=12)
            plt.title(
                f"Calibrazione Fotometrica Globale (Run {RUN_TO_ANALYZE}) - {flusso} \n metodo bin uniforme (min 3 stelle)",
                fontsize=14)
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend(fontsize=11, loc='best')
            plt.tight_layout()

            # salvo il mio secondo grafico
            plt.savefig(f"fit_binnato_{flusso}_uniforme_style.png")
            plt.close()

        except Exception as e:
            print(f"Errore durante l'esecuzione del fit (Metodo 2) per {flusso}: {e}")

    else:
        print(f"Non ho abbastanza punti validi per eseguire i fit di {flusso}.")

print("\n--- ELABORAZIONE COMPLETATA CON SUCCESSO ---")