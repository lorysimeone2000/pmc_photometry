import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.table import Table
import warnings
from astropy.wcs import FITSFixedWarning

# Sopprimo i warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Per log10(0) se capita


# --- FUNZIONI MATEMATICHE ---

def somma_magnitudini(series_mags):
    """
    Riceve una serie di magnitudini, le converte in flusso,
    le somma e restituisce la magnitudine integrata.
    """
    # Rimuove eventuali NaN
    mags = series_mags.dropna()

    if len(mags) == 0:
        return np.nan

    # 1. Conversione Mag -> Flusso
    # F = 10^(-0.4 * m)
    flussi = 10 ** (-0.4 * np.array(mags))

    # 2. Somma dei flussi
    flusso_totale = np.sum(flussi)

    if flusso_totale <= 0:
        return np.nan

    # 3. Conversione Flusso Totale -> Mag
    # m = -2.5 * log10(F)
    mag_integrata = -2.5 * np.log10(flusso_totale)

    return mag_integrata


def converti_valore(valore):
    """Utility per convertire stringhe in numeri nel parsing header."""
    valore = str(valore).strip()
    if not valore: return valore
    try:
        return int(valore)
    except ValueError:
        pass
    try:
        return float(valore)
    except ValueError:
        pass
    return valore


# --- CONFIGURAZIONE ---

'''try:
    run = int(input("Quale run vuoi elaborare (es. 1, 2, 3): "))
except ValueError:
    print("Devi inserire un numero intero.")
    exit()'''

run = 1

# Definizione percorsi
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

# Verifica e selezione file
if not os.path.exists(cartella_csv):
    print(f"Errore: La cartella {cartella_csv} non esiste.")
    exit()

file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

if not lista_percorsi_csv:
    print("Nessun file CSV trovato.")
    exit()

# Selezione immagine
n_immagine = 35
if n_immagine >= len(lista_percorsi_csv):
    n_immagine = len(lista_percorsi_csv) - 1

percorso_file_csv = lista_percorsi_csv[n_immagine]
print(f"Elaborazione file: {os.path.basename(percorso_file_csv)}")

# --- CARICAMENTO E PREPARAZIONE DATI ---

# Caricamento DataFrame
df = pd.read_csv(percorso_file_csv, comment='#')

# Separazione tra correlate e non correlate
# Le "Non Correlate" hanno 'Corrispondenza' == 'NO'
mask_no_corr = df['Corrispondenza'] == 'NO'
df_no_corr = df[mask_no_corr].copy()
df_corr = df[~mask_no_corr].copy()

# --- INTEGRAZIONE MAGNITUDINI (Il cuore della modifica) ---

print("\nEseguo l'integrazione delle magnitudini per pallocchio...")

# Raggruppiamo per 'label' (che identifica il pallocchio univoco)
# Aggreghiamo i dati:
# - Mag: applichiamo la funzione somma_magnitudini
# - kron_flux, area, max_value: prendiamo il primo valore (sono identici per lo stesso label)
# - Corrispondenza: contiamo quante righe ci sono (per sapere quante stelle compongono il pallocchio)

df_raggruppato = df_corr.groupby('label').agg({
    'Mag': somma_magnitudini,  # La nostra funzione custom
    'kron_flux': 'first',  # Dato fotometrico (uguale per tutto il gruppo)
    'area': 'first',  # Dato fotometrico
    'max_value': 'first',  # Dato fotometrico
    'Corrispondenza': 'count'  # Conta quante stelle ci sono nel pallocchio
}).reset_index()

# Rinominiamo la colonna conteggio per chiarezza
df_raggruppato.rename(columns={'Corrispondenza': 'num_stelle', 'Mag': 'Mag_Integrata'}, inplace=True)

# Creiamo due sotto-dataset per il plot
df_singole = df_raggruppato[df_raggruppato['num_stelle'] == 1]
df_multiple = df_raggruppato[df_raggruppato['num_stelle'] > 1]

# Preparazione Non Correlate (prendiamo una riga per label univoco, tanto i dati fotometrici sono uguali)
df_no_corr_unique = df_no_corr.drop_duplicates(subset=['label'])
# Magnitudine fittizia per plot
mag_finta = 4.0
y_no_corr = np.full(len(df_no_corr_unique), mag_finta)

# --- STATISTICHE ---

print(f"\nRisultati Integrazione:")
print(f"- Pallocchi con 1 sola stella catalogata: {len(df_singole)}")
print(f"- Pallocchi con >1 stella (Magnitudine integrata): {len(df_multiple)}")
print(f"- Pallocchi non correlati: {len(df_no_corr_unique)}")
print(f"- Totale sorgenti rilevate (labels unici): {len(df_raggruppato) + len(df_no_corr_unique)}")

# --- VISUALIZZAZIONE ---

plt.figure(figsize=(10, 8))

plt.scatter(df_singole['kron_flux'], df_singole['Mag_Integrata'],
            s=15, c='blue', alpha=0.6, label=f'Singola Componente ({len(df_singole)})')

plt.scatter(df_multiple['kron_flux'], df_multiple['Mag_Integrata'],
            s=15, c='red', alpha=0.8, label=f'Multi-Componente (Integrata) {len(df_multiple)}')

plt.scatter(df_no_corr_unique['kron_flux'], y_no_corr,
            s=30, c='orange', marker='x', alpha=0.5, label=f'Non Correlate ({len(df_no_corr_unique)})')

plt.title(f'Focus: Kron Flux vs Magnitudine Integrata\n(Run {run} - {os.path.basename(percorso_file_csv)})',
          fontsize=14)
plt.xlabel('Kron Flux (log scale)', fontsize=12)
plt.ylabel('Magnitudine Integrata (Invertita)', fontsize=12)
plt.xscale('log')
plt.gca().invert_yaxis()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=12)

'''# Annotazione statistica sul grafico
info_text = (f"Singole: {len(df_singole)}\n"
             f"Multiple: {len(df_multiple)}\n"
             f"Ghost: {len(df_no_corr_unique)}")
plt.text(0.02, 0.05, info_text, transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))'''

plt.show()