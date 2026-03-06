import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from scipy.optimize import curve_fit
import warnings
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

def modello_lineare_generico(mag, m, q):
    """
    Modello invertito: log10(Flux) = m * Mag + q
    """
    return m * mag + q

# --- CONFIGURAZIONE ---
run = 1
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

n_immagine = 35
if n_immagine >= len(lista_percorsi_csv): n_immagine = len(lista_percorsi_csv) - 1
percorso_file_csv = lista_percorsi_csv[n_immagine]
print(f"Elaborazione file: {os.path.basename(percorso_file_csv)}")

df = pd.read_csv(percorso_file_csv, comment='#')

# --- PREPARAZIONE DATI ---

mask_no_corr = df['Corrispondenza'] == 'NO'
df_corr = df[~mask_no_corr].copy()

counts = df_corr['label'].value_counts()
df_corr['num_stelle'] = df_corr['label'].map(counts)

# Filtro Best Match
idx_best = df_corr.groupby('label')['Mag'].idxmin()
df_best_match = df_corr.loc[idx_best].copy()
df_scartate = df_corr.drop(df_best_match.index)

# Preparazione dati per il FIT
data_fit = df_best_match.dropna(subset=['flusso_fisso_max_run', 'Mag'])
data_fit = data_fit[data_fit['flusso_fisso_max_run'] > 0]

# --- INVERSIONE ASSI ---
# X = Magnitudine
# Y = Log10(Flusso)
X = data_fit['Mag'].values
Y = np.log10(data_fit['flusso_fisso_max_run'].values)

# --- CALCOLO FIT ---

# 1. Fit Preliminare
popt_prelim, _ = curve_fit(modello_lineare_generico, X, Y)
m_prelim, q_prelim = popt_prelim

# 2. Sigma Clipping
residui = Y - modello_lineare_generico(X, m_prelim, q_prelim)
std_residui = np.std(residui)
soglia_sigma = 2.5
mask_inliers = np.abs(residui) < (soglia_sigma * std_residui)

X_clean = X[mask_inliers]
Y_clean = Y[mask_inliers]
X_out = X[~mask_inliers]
Y_out = Y[~mask_inliers]

# 3. Fit Finale
popt, pcov = curve_fit(modello_lineare_generico, X_clean, Y_clean)
m_fit, q_fit = popt
err_m, err_q = np.sqrt(np.diag(pcov))

# 4. Chi Quadro
# Nota: sigma_y ora è l'errore sul log(flux), non sulla magnitudine.
# Se non noto, usiamo un valore fittizio per il calcolo relativo.
y_model = modello_lineare_generico(X_clean, m_fit, q_fit)
residui_finali = Y_clean - y_model
sigma_y_log_flux = 0.05
chi_squared = np.sum((residui_finali / sigma_y_log_flux) ** 2)
dof = len(X_clean) - 2
chi_reduced = chi_squared / dof

print(f"\n--- Risultati Fit (Assi Invertiti: Mag -> Flux) ---")
print(f"Pendenza (m): {m_fit:.4f} +/- {err_m:.4f}")
print(f"Intercetta (q): {q_fit:.4f} +/- {err_q:.4f}")
print(f"Chi-Quadro Ridotto: {chi_reduced:.4f}")

# --- VISUALIZZAZIONE ---

plt.figure(figsize=(12, 9))

# 1. Plot delle stelle "Secondarie" (scartate)
if not df_scartate.empty:
    # Attenzione: qui x=Mag, y=Flux
    plt.scatter(df_scartate['Mag'], df_scartate['flusso_fisso_max_run'],
                s=15, c='lightgray', alpha=0.8, marker='.',
                label=f'Secondarie Scartate ({len(df_scartate)})')

# 2. Plot dei dati usati (Best Match)
mask_singole_best = df_best_match['num_stelle'] == 1
mask_multiple_best = df_best_match['num_stelle'] > 1

plt.scatter(df_best_match[mask_singole_best]['Mag'], df_best_match[mask_singole_best]['kron_flux'],
            s=20, c='blue', alpha=0.7, label='Singole')

plt.scatter(df_best_match[mask_multiple_best]['Mag'], df_best_match[mask_multiple_best]['flusso_fisso_max_run'],
            s=20, c='red', alpha=0.7, label='Multiple (stella più luminosa)')

# 3. Outliers statistici
if len(X_out) > 0:
    # X_out è Mag, Y_out è log(flux), quindi plottiamo 10^Y_out
    plt.scatter(X_out, 10**Y_out,
                s=50, c='black', marker='x', linewidth=1.5,
                label=f'Esclusi Sigma Clip ({len(X_out)})', zorder=5)

# 4. Retta di Fit
# Creiamo un range di magnitudini (asse X)
x_plot = np.linspace(min(X), max(X), 100)
# Calcoliamo il log(flux) previsto
log_flux_model = modello_lineare_generico(x_plot, m_fit, q_fit)
# Convertiamo in flusso lineare per il plot
y_plot = 10**log_flux_model

label_fit = (f'Fit: log(F) = {m_fit:.2f}*Mag + {q_fit:.2f}\n'
             f'$\chi^2_{{red}}$ = {chi_reduced:.2f}')

plt.plot(x_plot, y_plot, 'k--', linewidth=2, label=label_fit)

plt.title(f'Calibrazione Invertita: Mag vs Kron Flux\n(Run {run} - Immagine {n_immagine})', fontsize=14)
plt.xlabel('Magnitudine Catalogata', fontsize=12)
plt.ylabel('Kron Flux fisso [Scala Log]', fontsize=12)

# --- IMPOTANTE: Settaggi Assi ---
plt.yscale('log')          # Scala Log su Y (Flusso)
plt.gca().invert_xaxis()   # Invertiamo X (Magnitudine): sinistra=brillanti, destra=deboli
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.legend(fontsize=11, loc='best', framealpha=0.9)
plt.show()