import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
import warnings
from pathlib import Path
from tqdm import tqdm  # Aggiunto per barra caricamento

# Ignora warning numerici (es. log(0))
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# 0. FUNZIONI DI GESTIONE PERCORSI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata. Uso la directory dello script.")
    return path_corrente.parent


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    """
    Cerca una CARTELLA ricorsivamente in tutte le sottocartelle di base_dir.
    """
    # Cerchiamo directory che matchano il nome
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]

    if not cartelle_trovate:
        return None

    # Ordiniamo per lunghezza percorso
    cartelle_trovate.sort(key=lambda p: len(str(p)))

    if len(cartelle_trovate) > 1:
        print(
            f"INFO: Trovate {len(cartelle_trovate)} cartelle '{nome_cartella_esatto}'. Uso la prima: {cartelle_trovate[0].relative_to(base_dir)}")

    return cartelle_trovate[0]


def modello_lineare_generico(mag, m, q):
    return m * mag + q


# =============================================================================
# 1. CONFIGURAZIONE E CARICAMENTO (MULTI-FILE)
# =============================================================================

BASE_DIR = trova_cartella_base("pmc_photometry")
print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")

# Input Utente per la Run
try:
    run = int(input("Quale run vuoi analizzare per il fit (es. 1)? "))
except ValueError:
    print("Errore: Inserire un numero intero valido.")
    exit()

# 1. Trova la cartella della run
nome_cartella = f"tabelle_unite_run_{run}"
path_cartella = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella)

if path_cartella is None:
    print(f"ERRORE CRITICO: La cartella '{nome_cartella}' non è stata trovata in nessuna sottocartella di {BASE_DIR}.")
    exit()

print(f"Cartella trovata: {path_cartella}")

# 2. Trova tutti i CSV nella cartella
files_csv = sorted(list(path_cartella.glob("*.csv")))

if not files_csv:
    print(f"ERRORE: Nessun file .csv trovato in {path_cartella}")
    exit()

print(f"Trovati {len(files_csv)} file CSV da elaborare.")

# 3. Caricamento e Concatenazione
lista_dfs = []
print("Caricamento file in corso...")
for f in tqdm(files_csv):
    try:
        # Legge il csv
        tmp_df = pd.read_csv(f, comment='#')
        lista_dfs.append(tmp_df)
    except Exception as e:
        print(f"Errore lettura file {f.name}: {e}")

if not lista_dfs:
    print("Nessun dato valido caricato.")
    exit()

df_total = pd.concat(lista_dfs, ignore_index=True)

# 4. Deduplicazione per ID
# Poiché le colonne 'media_' sono identiche per la stessa stella in tutti i file,
# dobbiamo ridurre il dataset a una riga per stella per fare il fit corretto.
# Usiamo 'ID' come identificativo univoco.
print("Unione dati e rimozione duplicati per oggetto...")
if 'ID' in df_total.columns:
    df = df_total.drop_duplicates(subset=['ID']).copy()
else:
    print("ATTENZIONE: Colonna 'ID' non trovata. Impossibile deduplicare correttamente. Uso tutti i dati.")
    df = df_total.copy()

# --- NOMI COLONNE ---
col_flux = 'media_flusso_fisso_max_run'
col_std = 'std_flusso_fisso_max_run'
col_mag = 'Mag_Brightest'
col_count = 'count_flusso_fisso_max_run'  # O 'ripetizioni_run_X' se hai cambiato nome, verifica nel CSV

# Verifica rapida esistenza colonne
if col_flux not in df.columns:
    print(f"ERRORE: Colonna flusso '{col_flux}' non trovata. Colonne disponibili: {list(df.columns)}")
    exit()

# =============================================================================
# 2. SEPARAZIONE DATI (LOGICA ORIGINALE)
# =============================================================================

# A. Identifichiamo i "Non Catalogati" sul DF originale
mask_no_match = (df[col_mag].isna()) | (df['ID'].astype(str).str.startswith('NOMATCH'))
df_no_match = df[mask_no_match].copy()

# B. Identifichiamo i "Saturi" (per il plot delle X rosse) sul DF originale
mask_sature_original = df['saturazione'].astype(str).str.startswith('SI')
# Prendiamo le sature che HANNO corrispondenza (per avere la Mag sull'asse X)
df_sature = df[mask_sature_original & ~mask_no_match].copy()

# C. Creazione DF Match (Base per il fit)
df_match = df[~mask_no_match].copy()

# D. Filtri sequenziali su df_match
# Filtriamo i saturi USANDO LA COLONNA DI df_match
mask_sature_match = df_match['saturazione'].astype(str).str.startswith('SI')
df_match = df_match[~mask_sature_match].copy()

# Filtriamo i deboli USANDO LA COLONNA DI df_match
mask_deboli_match = df_match[col_mag] >= 10
df_match = df_match[~mask_deboli_match].copy()

# --- DEBUG: STAMPA COSA HAI TROVATO ---
print(f"\n--- STATISTICHE DATI (Aggregati per Stella) ---")
print(f"Totale stelle uniche trovate: {len(df)}")
print(f"Oggetti CON corrispondenza (potenziali per fit): {len(df_match)}")
print(f"Oggetti SENZA corrispondenza: {len(df_no_match)}")
print(f"Oggetti SATURI (esclusi dal fit): {len(df_sature)}")

# =============================================================================
# 3. PREPARAZIONE DATI PER IL FIT (SOLO MATCHATI E NON SATURI)
# =============================================================================

# Filtro validità rigoroso SOLO per il fit
# Nota: col_count potrebbe non esistere nei nuovi file se si chiama 'ripetizioni_run_X'
# Se non esiste, usiamo 1 come count di fallback o cerchiamo la colonna giusta
cols_available = df_match.columns.tolist()
if col_count not in cols_available:
    # Tentativo di trovare la colonna ripetizioni corretta
    possibile_col = f'ripetizioni_run_{run}'
    if possibile_col in cols_available:
        col_count = possibile_col
    elif 'ripetizione_run' in cols_available:
        col_count = 'ripetizione_run'
    else:
        # Creiamo colonna fittizia se manca per non rompere il codice
        df_match[col_count] = 1

mask_valid_fit = (
        (df_match[col_flux] > 0) &
        (df_match[col_mag].notna()) &
        (df_match[col_std] > 0) &
        (df_match[col_count] > 0)
)
data_fit = df_match[mask_valid_fit].copy()

# Dati per il fit
X = data_fit[col_mag].values
Y_linear = data_fit[col_flux].values
Y_log = np.log10(Y_linear)

# Sigma_mean = Sigma_std / sqrt(N)
sigma_std = data_fit[col_std].values
counts = data_fit[col_count].values
sigma_mean_linear = sigma_std / np.sqrt(counts)

# --- PROPAGAZIONE ERRORE SUL LOGARITMO ---
# Sigma_log = (1/ln(10)) * (Sigma_mean_linear / Flux)
sigma_log = (1 / np.log(10)) * (sigma_mean_linear / Y_linear)

# =============================================================================
# 4. ESECUZIONE FIT
# =============================================================================

if len(X) > 2:
    popt, pcov = curve_fit(modello_lineare_generico, X, Y_log, sigma=sigma_log, absolute_sigma=True)
    m_fit, q_fit = popt

    # Chi Quadro
    y_model_log = modello_lineare_generico(X, m_fit, q_fit)
    residui_finali = Y_log - y_model_log
    chi_squared = np.sum((residui_finali / sigma_log) ** 2)
    chi_reduced = chi_squared / (len(X) - 2)

    print(f"\n--- Risultati Fit ---")
    print(f"m: {m_fit:.4f}, q: {q_fit:.4f}, Chi2_red: {chi_reduced:.4f}")
else:
    print("Non abbastanza punti per il fit.")
    m_fit, q_fit, chi_reduced = 0, 0, 0

# =============================================================================
# 5. VISUALIZZAZIONE
# =============================================================================

plt.figure(figsize=(12, 9))

# A. Dati del Fit (Blu)
if len(X) > 0:
    plt.errorbar(
        X, 10 ** Y_log, yerr=sigma_mean_linear,
        fmt='o', markersize=4, color='blue', ecolor='lightblue', alpha=0.7,
        label=f'Catalogati Validi ({len(X)})'
    )

# B. Stelle Sature (X Rossa)
if len(df_sature) > 0:
    mask_sat_valid = (df_sature[col_flux] > 0) & (df_sature[col_mag].notna())
    df_sature_plot = df_sature[mask_sat_valid]

    if len(df_sature_plot) > 0:
        plt.scatter(
            df_sature_plot[col_mag],
            df_sature_plot[col_flux],
            s=80, c='red', marker='x', linewidth=2,
            label=f'Sature (Escluse) ({len(df_sature_plot)})', zorder=20
        )

# C. OGGETTI SENZA CORRISPONDENZA (Arancione)
if len(df_no_match) > 0:
    mag_fittizia = 4.0
    if len(X) > 0:
        mag_fittizia = np.min(X) - 1.5

    df_no_match_plot = df_no_match[df_no_match[col_flux] > 0]

    if len(df_no_match_plot) > 0:
        plt.scatter(
            np.full(len(df_no_match_plot), mag_fittizia),
            df_no_match_plot[col_flux],
            s=40, c='orange', marker='D', edgecolors='black', alpha=0.8,
            label=f'NON Catalogati ({len(df_no_match_plot)})', zorder=10
        )

        plt.annotate("Mag Fittizia", xy=(mag_fittizia, np.mean(df_no_match_plot[col_flux])),
                     xytext=(mag_fittizia, np.max(df_no_match_plot[col_flux]) * 1.5),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     ha='center')

# D. Retta di Fit
if len(X) > 2:
    x_min_plot = min(np.min(X), mag_fittizia - 0.5)
    x_max_plot = max(X)
    x_plot = np.linspace(x_min_plot, x_max_plot, 100)
    y_plot_linear = 10 ** modello_lineare_generico(x_plot, m_fit, q_fit)

    label_fit = rf'Fit: log(F)={m_fit:.2f}M + {q_fit:.2f} ($\chi^2_R$={chi_reduced:.2f})'
    plt.plot(x_plot, y_plot_linear, 'k--', linewidth=2, label=label_fit)

# --- FORMATTAZIONE ---
plt.title(f'Calibrazione Fotometrica Run {run} (Dati Aggregati)', fontsize=14)
plt.xlabel('Magnitudine Catalogo (Brightest)', fontsize=12)
plt.ylabel('Media Flusso Fisso Max Run [ADU]', fontsize=12)
plt.yscale('log')
plt.gca().invert_xaxis()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=11, loc='best')
plt.tight_layout()

# Salva
output_img = f"fit_aggregato_run_{run}.png"
plt.savefig(output_img, dpi=300)
print(f"Grafico salvato in: {output_img}")
plt.show()