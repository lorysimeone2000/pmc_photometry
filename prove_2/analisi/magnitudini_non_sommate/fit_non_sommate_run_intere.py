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
    """ Modello: log10(Flux) = m * Mag + q """
    return m * mag + q


# Configurazione
BASE_DIR = trova_cartella_base("pmc_photometry")
RUN_TO_ANALYZE = [1, 2, 3]  # Puoi specificare quali run includere nel fit globale

# =============================================================================
# 1. CARICAMENTO DATI (TUTTE LE RUN)
# =============================================================================

print(f"--- Caricamento dati per Fit Globale ---")
lista_dfs = []

for run in RUN_TO_ANALYZE:
    nome_cartella = f"tabelle_unite_run_{run}"
    path_cartella = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella)

    if path_cartella is None:
        print(f"Attenzione: Cartella {nome_cartella} non trovata.")
        continue

    files_csv = sorted(list(path_cartella.glob("*.csv")))
    print(f"Run {run}: Trovati {len(files_csv)} file. Caricamento in corso...")

    for f in tqdm(files_csv, leave=False):
        try:
            # Carico solo le colonne essenziali
            cols_needed = [
                'label', 'ID', 'Corrispondenza', 'Mag', 'saturazione',
                'media_flusso_fisso_max_run', 'std_flusso_fisso_max_run'
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
# 2. PREPARAZIONE DATI PER IL FIT E IL GRAFICO
# =============================================================================

# A. DEDUPLICAZIONE
df_total_sorted = df_total.sort_values(by=['label', 'Mag'], ascending=[True, True])  # magnitudini in ordine crescente
df_unique = df_total_sorted.drop_duplicates(subset=['ID'], keep='first').copy()  # prendo solo i primi valori
print(f"Oggetti UNICI totali (Catalogati + Non): {len(df_unique)}")

# B. SEPARAZIONE CATEGORIE (Per replicare lo stile grafico richiesto)
# Per avere le X rosse (saturi) e i rombi arancioni (no match), devo estrarli prima di filtrare tutto.

# 1. Non Catalogati (Corrispondenza != SI)
mask_match = df_unique['Corrispondenza'].astype(str).str.startswith('SI')
df_no_match = df_unique[~mask_match].copy()

# 2. Matchati (Base per analisi)
df_match = df_unique[mask_match].copy()

# 3. Saturi (Corrispondenza SI ma Saturazione SI)
if 'saturazione' in df_match.columns:
    mask_sature = df_match['saturazione'].astype(str).str.startswith('SI')
    df_sature = df_match[mask_sature].copy()
    # Tengo i non saturi per il fit
    df_fit_potential = df_match[~mask_sature].copy()
else:
    df_sature = pd.DataFrame()
    df_fit_potential = df_match.copy()

# 4. Filtro Validità Numerica (Mag esistente, Flussi positivi)
mask_valid = (
        (df_fit_potential['Mag'].notna()) &
        (df_fit_potential['media_flusso_fisso_max_run'] > 0) &
        (df_fit_potential['std_flusso_fisso_max_run'] > 0)
)
df_fit_valid = df_fit_potential[mask_valid].copy()

# 5. Filtro Magnitudine per il FIT (es. Mag <= 10)
SOGLIA_MAG_FIT = 10.0
df_fit_clean = df_fit_valid[df_fit_valid['Mag'] <= SOGLIA_MAG_FIT].copy()
df_sature = df_sature[df_sature['Mag'] <= SOGLIA_MAG_FIT].copy()

print(f"Oggetti validi per il FIT (Match SI, No Saturi, Mag <= {SOGLIA_MAG_FIT}): {len(df_fit_clean)}")

# =============================================================================
# 3. FIT LINEARE
# =============================================================================

if len(df_fit_clean) > 2:
    # Dati X e Y
    X = df_fit_clean['Mag'].values
    Y_flux = df_fit_clean['media_flusso_fisso_max_run'].values
    Y_log = np.log10(Y_flux)

    # Errori
    sigma_flux = df_fit_clean['std_flusso_fisso_max_run'].values
    # Propagazione errore sul logaritmo
    sigma_log = (1 / np.log(10)) * (sigma_flux / Y_flux)

    # Esecuzione Fit
    popt, pcov = curve_fit(modello_lineare, X, Y_log, sigma=sigma_log, absolute_sigma=True)
    m_fit, q_fit = popt
    err_m, err_q = np.sqrt(np.diag(pcov))

    # Calcolo Chi Quadro Ridotto
    y_model = modello_lineare(X, m_fit, q_fit)
    chi2 = np.sum(((Y_log - y_model) / sigma_log) ** 2)
    dof = len(X) - 2
    chi2_red = chi2 / dof if dof > 0 else 0

    print("\n--- RISULTATI FIT ---")
    print(f"m = {m_fit:.4f} ± {err_m:.4f}")
    print(f"q = {q_fit:.4f} ± {err_q:.4f}")
    print(f"Chi2 Ridotto = {chi2_red:.2f}")

    # =============================================================================
    # 4. PLOTTING (STILE AGGIORNATO)
    # =============================================================================
    plt.figure(figsize=(12, 9))

    # A. Punti usati per il fit (Blu con barre errore chiare)
    if len(X) > 0:
        plt.errorbar(
            X, Y_flux, yerr=sigma_flux,
            fmt='o', markersize=1, color='blue', ecolor='lightblue', alpha=0.7,
            label=f'Catalogati Validi ({len(X)})'
        )

    # B. Stelle Sature (X Rosse)
    if not df_sature.empty:
        # Mi assicuro che abbiano dati validi per il plot
        mask_sat_valid = (df_sature['media_flusso_fisso_max_run'] > 0) & (df_sature['Mag'].notna())
        df_sature_plot = df_sature[mask_sat_valid]

        if not df_sature_plot.empty:
            plt.scatter(
                df_sature_plot['Mag'],
                df_sature_plot['media_flusso_fisso_max_run'],
                s=40, c='red', marker='x', linewidth=1,
                label=f'Sature (Escluse) ({len(df_sature_plot)})', zorder=20
            )

    # C. Oggetti SENZA Corrispondenza (Rombi Arancioni)
    if not df_no_match.empty:
        # Calcolo una magnitudine fittizia per posizionarli a sinistra
        mag_fittizia = 4.0
        if len(X) > 0:
            mag_fittizia = np.min(X) - 1.5

        # Filtro quelli con flusso valido
        df_no_match_plot = df_no_match[df_no_match['media_flusso_fisso_max_run'] > 0]

        if not df_no_match_plot.empty:
            plt.scatter(
                np.full(len(df_no_match_plot), mag_fittizia),
                df_no_match_plot['media_flusso_fisso_max_run'],
                s=40, c='orange', marker='D', edgecolors='black', alpha=0.8,
                label=f'NON Catalogati ({len(df_no_match_plot)})', zorder=10
            )
            # Aggiungo annotazione
            plt.annotate("Mag Fittizia",
                         xy=(mag_fittizia, np.mean(df_no_match_plot['media_flusso_fisso_max_run'])),
                         xytext=(mag_fittizia, np.max(df_no_match_plot['media_flusso_fisso_max_run']) * 1.5),
                         arrowprops=dict(facecolor='black', arrowstyle='->'),
                         ha='center')

            # Aggiorno limiti plot se necessario
            x_min_plot = min(np.min(X), mag_fittizia - 0.5)
    else:
        x_min_plot = np.min(X) - 0.5

    # D. Retta di Fit (Nera tratteggiata)
    x_max_plot = np.max(X) + 0.5
    x_plot = np.linspace(x_min_plot, x_max_plot, 100)
    y_plot_log = modello_lineare(x_plot, m_fit, q_fit)

    # --- MODIFICA: AGGIUNTO ERRORE SUI PARAMETRI NELLA LABEL ---
    label_fit = (rf'Fit: log(F)=({m_fit:.2f}$\pm${err_m:.2f})M + ({q_fit:.2f}$\pm${err_q:.2f})'
                 f'\n$\chi^2_R$={chi2_red:.2f}')

    plt.plot(x_plot, 10 ** y_plot_log, 'k--', linewidth=2, label=label_fit)

    # Configurazione Grafico
    plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.xlabel("Magnitudine Catalogo (Mag)", fontsize=12)
    plt.ylabel("Media Flusso Fisso Max Run (ADU)", fontsize=12)
    plt.title(f"Calibrazione Fotometrica Globale (Run {RUN_TO_ANALYZE})", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()

    # Salvataggio
    out_file = "fit_globale_media_style.png"
    plt.savefig(out_file, dpi=300)
    print(f"Grafico salvato: {out_file}")
    plt.show()

else:
    print("Non abbastanza punti validi per eseguire il fit.")