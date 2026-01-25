import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
import warnings
from astropy.wcs import FITSFixedWarning

# Sopprimo i warning non critici
warnings.filterwarnings('ignore', category=FITSFixedWarning)


# --- FUNZIONI UTILITY ---

def converti_valore(valore):
    """
    Converte una stringa nel tipo di dato appropriato.
    """
    valore = str(valore).strip()  # Assicura che sia stringa prima dello strip
    if not valore: return valore
    try:
        return int(valore)
    except ValueError:
        pass
    try:
        return float(valore)
    except ValueError:
        pass
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False
    return valore


def leggi_header_da_csv(filename):
    """Legge l'header FITS dal file CSV"""
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            elif line.strip() == '#':
                break
    return header_dict


# --- INPUT E CONFIGURAZIONE ---

'''try:
    run = int(input("Quale run vuoi elaborare (es. 1, 2, 3): "))
except ValueError:
    print("Devi inserire un numero intero.")
    exit()'''

run = 1

# Definizione percorsi
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

# Verifica esistenza cartella
if not os.path.exists(cartella_csv):
    print(f"Errore: La cartella {cartella_csv} non esiste.")
    exit()

# Lista file
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

if not lista_percorsi_csv:
    print("Nessun file CSV trovato nella cartella.")
    exit()

# Selezione immagine (puoi cambiare questo indice o chiedere input)
n_immagine = 35

if n_immagine >= len(lista_percorsi_csv):
    print(f"Indice immagine {n_immagine} fuori range. Uso l'ultima disponibile.")
    n_immagine = len(lista_percorsi_csv) - 1

percorso_file_csv = lista_percorsi_csv[n_immagine]
print(f"Elaborazione file: {os.path.basename(percorso_file_csv)}")

# --- CARICAMENTO DATI ---

# Nota: comment='#' righe siano commenti/header.
# Se cambia il numero di righe di header, questo valore va aggiornato.
dataframe = pd.read_csv(percorso_file_csv, comment='#')
tbl = Table.from_pandas(dataframe)

print("Tabella: ",tbl)

# --- FILTRAGGIO DATI PER RANK ---

# Creazione delle maschere per i diversi Rank
tbl_rank1 = tbl[tbl['Corrispondenza'] == 'SI (Rank 1)']
tbl_rank2 = tbl[tbl['Corrispondenza'] == 'SI (Rank 2)']
tbl_rank3 = tbl[tbl['Corrispondenza'] == 'SI (Rank 3)']
rank_esclusi = ['SI (Rank 1)', 'SI (Rank 2)', 'SI (Rank 3)']
tbl_altri_rank = tbl[~np.isin(tbl['Corrispondenza'], rank_esclusi)]
tbl_no_corr = tbl[tbl['Corrispondenza'] == 'NO']

print(f"\nStatistiche sorgenti:")
print(f"- Corrispondenza Rank 1 (Principale): {len(tbl_rank1)}")
print(f"- Corrispondenza Rank 2 (Secondaria): {len(tbl_rank2)}")
print(f"- Corrispondenza Rank 3 (Terziaria):  {len(tbl_rank3)}")
print(f"- Corrispondenza altri Rank:  {len(tbl_altri_rank)}")
print(f"- Nessuna corrispondenza:             {len(tbl_no_corr)}")

# Preparazione dati fittizi per visualizzare le "non correlate"
mag_finta = 4.0
y_no_corr = np.full(len(tbl_no_corr), mag_finta)

# --- VISUALIZZAZIONE (TRE SUBPLOTS) ---

# Impostazioni grafiche
size = 20
alpha = 0.7

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Relazioni Magnitudine vs Parametri Photutils (Run {run} - Img {n_immagine})', fontsize=16)

# Lista di configurazione per il loop: (Nome Colonna, Asse, Titolo asse X)
plots_config = [
    ('area', axes[0], 'Area (pixel)'),
    ('flusso_fisso_max_run', axes[1], 'Kron Flux (conteggi)'),
    ('max_value', axes[2], 'Max Value (conteggi)')
]

for col_name, ax, x_label in plots_config:
    # 1. Rank 1 (Blu)
    ax.scatter(tbl_rank1['Mag'], tbl_rank1[col_name],
               s=size, c='blue', alpha=alpha, label='Rank 1 (magnitudine massima)')

    # 2. Rank 2 (Verde)
    ax.scatter(tbl_rank2['Mag'], tbl_rank2[col_name],
               s=size, c='green', alpha=alpha, label='Rank 2')

    # 3. Rank 3 (Rosso)
    ax.scatter(tbl_rank3['Mag'], tbl_rank3[col_name],
               s=size, c='red', alpha=alpha, label='Rank 3')

    # 4. Altri rank (giallo)
    ax.scatter(tbl_altri_rank['Mag'], tbl_altri_rank[col_name],
               s=size, c='yellow', alpha=alpha, label='Altri rank')

    # 5. Non Correlate (Arancione/Grigio)
    ax.scatter( y_no_corr, tbl_no_corr[col_name],
               s=size, c='orange', marker='x', alpha=0.5, label='Non Correlate')

    # Configurazione assi
    ax.set_title(f'{x_label} vs Mag')
    ax.set_ylabel(x_label)
    ax.set_xlabel('Magnitudine Catalogata')
    ax.invert_xaxis()  # Inversione asse X (Magnitudine)
    ax.set_yscale('log')  # Scala logaritmica su y
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='best', fontsize='small')

plt.tight_layout()
plt.show()

# --- VISUALIZZAZIONE DETTAGLIO (SOLO FLUSSO) ---

plt.figure(figsize=(10, 8))

# Riplottiamo tutto su un grafico singolo più grande per chiarezza
plt.scatter( tbl_rank1['Mag'], tbl_rank1['flusso_fisso_max_run'], s=15, c='blue', alpha=alpha, label='Rank 1 (magnitudine massima)')
plt.scatter(tbl_rank2['Mag'], tbl_rank2['flusso_fisso_max_run'], s=15, c='green', alpha=alpha, label='Rank 2')
plt.scatter(tbl_rank3['Mag'], tbl_rank3['flusso_fisso_max_run'], s=15, c='red', alpha=alpha, label='Rank 3')
plt.scatter(tbl_altri_rank['Mag'], tbl_altri_rank['flusso_fisso_max_run'], s=15, c='yellow', alpha=alpha, label='Altri rank')
plt.scatter(y_no_corr, tbl_no_corr['flusso_fisso_max_run'], s=15, c='orange', marker='x', label='Non Correlate')

plt.title(f'Dettaglio: Kron Flux vs Magnitudine\n(Run {run} - {os.path.basename(percorso_file_csv)})', fontsize=14)
plt.ylabel('Kron Flux (log scale)', fontsize=12)
plt.xlabel('Magnitudine Catalogata (Invertita)', fontsize=12)
plt.yscale('log')
plt.gca().invert_xaxis()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=12)

plt.show()