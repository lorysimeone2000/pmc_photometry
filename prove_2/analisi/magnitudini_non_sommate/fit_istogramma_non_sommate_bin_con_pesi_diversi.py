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
    # definisco il modello lineare per il fit
    return m * mag + q


# imposto la configurazione per le run
BASE_DIR = trova_cartella_base("Lorenzo")
RUN_TO_ANALYZE = [1, 2, 3]

# =============================================================================
# 1. CARICAMENTO DATI (TUTTE LE RUN)
# =============================================================================

print(f"--- Caricamento dati per Fit Globale ---")
lista_dfs = []

# ciclo attraverso le run e carico i file csv
for run in RUN_TO_ANALYZE:
    nome_cartella = f"tabelle_unite_run_{run}"
    path_cartella = cerca_cartella_nel_progetto(BASE_DIR / "tabelle", nome_cartella)

    if path_cartella is None:
        print(f"Attenzione: Cartella {nome_cartella} non trovata.")
        continue

    files_csv = sorted(list(path_cartella.glob("*.csv")))
    print(f"Run {run}: Trovati {len(files_csv)} file. Caricamento in corso...")

    for f in tqdm(files_csv, leave=False):
        try:
            # carico solo le colonne essenziali
            cols_needed = [
                'label', 'ID', 'Corrispondenza', 'Mag', 'saturazione',
                'media_flusso_fisso_max_run', 'std_flusso_fisso_max_run',
                'media_flusso_raggio_fisso_doppio', 'std_flusso_raggio_fisso_doppio',
            ]

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
# 2. PREPARAZIONE DATI PER IL FIT E I GRAFICI
# =============================================================================

# ordino i dati e rimuovo i duplicati, mantenendoli però separati per singola run
df_total_sorted = df_total.sort_values(by=['run_origin', 'label', 'Mag'], ascending=[True, True, True])
df_unique = df_total_sorted.drop_duplicates(subset=['run_origin', 'ID'], keep='first').copy()
print(f"Oggetti UNICI totali (Catalogati + Non) mantenendo le run: {len(df_unique)}")

# filtro solo le stelle che hanno una corrispondenza nel catalogo
mask_match = df_unique['Corrispondenza'].astype(str).str.startswith('SI')
df_match = df_unique[mask_match].copy()

# escludo gli oggetti saturi
if 'saturazione' in df_match.columns:
    mask_sature = df_match['saturazione'].astype(str).str.startswith('SI')
    df_fit_potential = df_match[~mask_sature].copy()
else:
    df_fit_potential = df_match.copy()

# verifico che i dati numerici siano validi
mask_valid = (
        (df_fit_potential['Mag'].notna()) &
        (df_fit_potential['media_flusso_fisso_max_run'] > 0) &
        (df_fit_potential['media_flusso_fisso_max_run'] > 0) &
        (df_fit_potential['media_flusso_raggio_fisso_doppio'] > 0) &
        (df_fit_potential['std_flusso_fisso_max_run'] > 0) &
        (df_fit_potential['std_flusso_raggio_fisso_doppio'] > 0) &
        (df_fit_potential['std_flusso_fisso_max_run'] > 0)
)
df_fit_valid = df_fit_potential[mask_valid].copy()

# applico una soglia limite di magnitudine
SOGLIA_MAG_FIT = 10.0
df_fit_clean = df_fit_valid[df_fit_valid['Mag'] <= SOGLIA_MAG_FIT].copy()

print(f"Oggetti validi per i Bin (Match SI, No Saturi, Mag <= {SOGLIA_MAG_FIT}): {len(df_fit_clean)}")

# preparo la lista dei flussi da analizzare
flussi_da_analizzare = [
    ('media_flusso_fisso_max_run', 'std_flusso_fisso_max_run', 'Media Flusso Fisso Max Run'),
    ('media_flusso_raggio_fisso_doppio', 'std_flusso_raggio_fisso_doppio', 'Media Flusso Raggio Fisso Doppio'),
    ('media_flusso_intera_segmentazione', 'std_flusso_intera_segmentazione', 'Media Flusso Intera Segmentazione'),
    ('media_flusso_kron_intera_segmentazione', 'std_flusso_kron_intera_segmentazione',
     'Media Flusso Kron Intera Segmentazione')
]

# =============================================================================
# 3. GRAFICO 1: BINNING E FIT GLOBALE (TUTTE LE RUN UNITE)
# =============================================================================

for col_flusso, col_err, nome_flusso in flussi_da_analizzare:
    print(f"\n--- ELABORAZIONE GRAFICO 1: FIT GLOBALE - {nome_flusso} ---")
    plt.figure(figsize=(12, 9))

    if len(df_fit_clean) > 2:
        X_all_g = df_fit_clean['Mag'].values
        Y_all_g = df_fit_clean[col_flusso].values
        Err_all_g = df_fit_clean[col_err].values

        n_bins_g = max(5, int(np.sqrt(len(X_all_g))))

        # creo bin di larghezza uguale su tutto l'intervallo di magnitudini
        bins_g = np.linspace(X_all_g.min(), X_all_g.max(), n_bins_g + 1)

        X_binned_g = []
        Y_binned_g = []
        Err_binned_g = []
        Popolazioni_binned_g = []

        # raggruppo le stelle all'interno di ciascun bin a larghezza fissa
        for i in range(n_bins_g):
            if i == n_bins_g - 1:
                mask = (X_all_g >= bins_g[i]) & (X_all_g <= bins_g[i + 1])
            else:
                mask = (X_all_g >= bins_g[i]) & (X_all_g < bins_g[i + 1])

            if np.sum(mask) > 0:
                x_bin = X_all_g[mask]
                y_bin = Y_all_g[mask]
                err_bin = Err_all_g[mask]

                # calcolo la media semplice del flusso
                y_media_semplice = np.mean(y_bin)

                # calcolo l'errore standard della media dipendente dalla popolazione N del bin
                if len(y_bin) > 1:
                    y_errore_semplice = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin))
                    if y_errore_semplice == 0:
                        y_errore_semplice = np.mean(err_bin)
                else:
                    y_errore_semplice = err_bin[0]

                x_mean = np.mean(x_bin)

                X_binned_g.append(x_mean)
                Y_binned_g.append(y_media_semplice)
                Err_binned_g.append(y_errore_semplice)
                Popolazioni_binned_g.append(len(y_bin))

        X_binned_g = np.array(X_binned_g)
        Y_binned_g = np.array(Y_binned_g)
        Err_binned_g = np.array(Err_binned_g)

        Y_log_g = np.log10(Y_binned_g)

        # propago l'errore (che essendo proporzionale a 1/sqrt(N) conferirà più peso ai bin più popolati)
        sigma_log_g = (1 / np.log(10)) * (Err_binned_g / Y_binned_g)

        popt_g, pcov_g = curve_fit(modello_lineare, X_binned_g, Y_log_g, sigma=sigma_log_g, absolute_sigma=True)
        m_fit_g, q_fit_g = popt_g
        err_m_g, err_q_g = np.sqrt(np.diag(pcov_g))

        y_model_g = modello_lineare(X_binned_g, m_fit_g, q_fit_g)
        chi2_g = np.sum(((Y_log_g - y_model_g) / sigma_log_g) ** 2)
        dof_g = len(X_binned_g) - 2
        chi2_red_g = chi2_g / dof_g if dof_g > 0 else 0

        print(f"Numero di bin effettivi usati (Globale): {len(X_binned_g)}")
        print(f"Popolazione media dei bin: {np.mean(Popolazioni_binned_g):.1f} stelle")
        print(f"m = {m_fit_g:.4f} ± {err_m_g:.4f}")
        print(f"q = {q_fit_g:.4f} ± {err_q_g:.4f}")
        print(f"Chi2 Ridotto = {chi2_red_g:.2f}")

        plt.errorbar(
            X_binned_g, Y_binned_g, yerr=Err_binned_g,
            fmt='o', markersize=4, color='blue', ecolor='lightblue', capsize=0,
            label=f'Valori Medi Binnati (Globale - {len(X_binned_g)} bin)'
        )

        x_min_plot_g = np.min(X_binned_g) - 0.1
        x_plot_g = np.linspace(x_min_plot_g, 10.0, 100)
        y_plot_log_g = modello_lineare(x_plot_g, m_fit_g, q_fit_g)

        label_fit_g = (rf'Fit Globale: log(F)=({m_fit_g:.2f}$\pm${err_m_g:.2f})M + ({q_fit_g:.2f}$\pm${err_q_g:.2f})'
                       f'\n$\chi^2_R$={chi2_red_g:.2f}')

        plt.plot(x_plot_g, 10 ** y_plot_log_g, 'k--', linewidth=2, label=label_fit_g)

        plt.yscale('log')
        plt.xlim(10, x_min_plot_g - 0.2)
        plt.xlabel("Magnitudine Catalogo (Mag)", fontsize=12)
        plt.ylabel(f"{nome_flusso} (ADU)", fontsize=12)
        plt.title(f"Calibrazione Fotometrica Globale Binnata ({nome_flusso})", fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(fontsize=11, loc='best')
        plt.tight_layout()

        out_file_globale = f"fit_globale_PESATI_binned_media_non_pesata_{col_flusso}.png"
        plt.savefig(out_file_globale, dpi=300)
        print(f"Grafico 1 salvato: {out_file_globale}")
        plt.show()

    else:
        print("Non abbastanza punti validi per eseguire il fit globale.")

# =============================================================================
# 4. GRAFICO 2: BINNING E FIT SEPARATO PER SINGOLA RUN
# =============================================================================

for col_flusso, col_err, nome_flusso in flussi_da_analizzare:
    print(f"\n--- ELABORAZIONE GRAFICO 2: FIT PER SINGOLA RUN - {nome_flusso} ---")
    plt.figure(figsize=(12, 9))

    colori = ['red', 'green', 'orange', 'purple', 'cyan', 'brown', 'magenta']
    x_min_globale_run = 10.0

    for idx_run, run_id in enumerate(sorted(df_fit_clean['run_origin'].unique())):
        df_run = df_fit_clean[df_fit_clean['run_origin'] == run_id]

        if len(df_run) > 2:
            X_all = df_run['Mag'].values
            Y_all = df_run[col_flusso].values
            Err_all = df_run[col_err].values

            n_bins = max(5, int(np.sqrt(len(X_all))))

            # creo i bin a larghezza fissa per questa run
            bins = np.linspace(X_all.min(), X_all.max(), n_bins + 1)

            X_binned = []
            Y_binned = []
            Err_binned = []
            Popolazioni_binned = []

            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (X_all >= bins[i]) & (X_all <= bins[i + 1])
                else:
                    mask = (X_all >= bins[i]) & (X_all < bins[i + 1])

                if np.sum(mask) > 0:
                    x_bin = X_all[mask]
                    y_bin = Y_all[mask]
                    err_bin = Err_all[mask]

                    # calcolo la media semplice
                    y_media_semplice = np.mean(y_bin)

                    # calcolo l'errore standard che dipendendo da N assegnerà dinamicamente il peso
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
                    Popolazioni_binned.append(len(y_bin))

            X_binned = np.array(X_binned)
            Y_binned = np.array(Y_binned)
            Err_binned = np.array(Err_binned)

            if len(X_binned) > 0 and X_binned.min() < x_min_globale_run:
                x_min_globale_run = X_binned.min()

            Y_log = np.log10(Y_binned)
            sigma_log = (1 / np.log(10)) * (Err_binned / Y_binned)

            popt, pcov = curve_fit(modello_lineare, X_binned, Y_log, sigma=sigma_log, absolute_sigma=True)
            m_fit, q_fit = popt
            err_m, err_q = np.sqrt(np.diag(pcov))

            y_model = modello_lineare(X_binned, m_fit, q_fit)
            chi2 = np.sum(((Y_log - y_model) / sigma_log) ** 2)
            dof = len(X_binned) - 2
            chi2_red = chi2 / dof if dof > 0 else 0

            print(f"\nRisultati Run {run_id}:")
            print(f"Numero bin usati: {len(X_binned)} (su {n_bins})")
            print(f"Popolazione media dei bin: {np.mean(Popolazioni_binned):.1f} stelle")
            print(f"m = {m_fit:.4f} ± {err_m:.4f}")
            print(f"q = {q_fit:.4f} ± {err_q:.4f}")
            print(f"Chi2 Ridotto = {chi2_red:.2f}")

            colore_run = colori[idx_run % len(colori)]

            plt.errorbar(
                X_binned, Y_binned, yerr=Err_binned,
                fmt='o', markersize=3, color=colore_run, ecolor=colore_run, capsize=0, alpha=0.7,
                label=f'Bin Run {run_id}'
            )

            x_min_plot = np.min(X_binned) - 0.1
            x_plot = np.linspace(x_min_plot, 10.0, 100)
            y_plot_log = modello_lineare(x_plot, m_fit, q_fit)

            label_fit = (rf'Fit Run {run_id}: log(F)=({m_fit:.2f}$\pm${err_m:.2f})M + ({q_fit:.2f}$\pm${err_q:.2f})'
                         f' [$\chi^2_R$={chi2_red:.2f}]')

            plt.plot(x_plot, 10 ** y_plot_log, linestyle='--', color=colore_run, linewidth=2, label=label_fit)

        else:
            print(f"Non abbastanza punti validi per eseguire il fit e il binning per la Run {run_id}.")

    plt.yscale('log')
    plt.xlim(10, x_min_globale_run - 0.2)
    plt.xlabel("Magnitudine Catalogo (Mag)", fontsize=12)
    plt.ylabel(f"{nome_flusso} (ADU)", fontsize=12)
    plt.title(f"Calibrazione Fotometrica Binnata per Singola Run ({nome_flusso})", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()

    out_file_run = f"fit_globale_binned_PESATI_per_run_media_non_pesata_{col_flusso}.png"
    plt.savefig(out_file_run, dpi=300)
    print(f"\nGrafico 2 salvato con successo come: {out_file_run}")
    plt.show()