import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
from astropy.table import Table

# --- CONFIGURAZIONE VISIVA GLOBALE ---
LINE_WIDTH = 0.5  # Spessore linea
MARKER_SIZE = 2  # Grandezza punti
MARKER_STYLE = 'o'  # <--- FORZA UNICO MARKER PER TUTTI ('o'=cerchio, 's'=quadrato, etc.)

# Colori predefiniti per gestire N flussi automaticamente
default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

def get_distinct_colors(n):
    if n <= 10:
        return plt.cm.tab10(np.linspace(0, 1, 10))[:n]
    elif n <= 20:
        return plt.cm.tab20(np.linspace(0, 1, 20))[:n]
    else:
        # Se sono tantissimi, usa lo spettro completo HSV per massimizzare la distanza
        return plt.cm.hsv(np.linspace(0, 1, n+1))[:n] # +1 per evitare che il primo e l'ultimo siano uguali (rosso)

'''def get_style(flusso_name, index):
    """Restituisce uno stile (colore, label) per un dato flusso."""

    # Stili fissi manuali (SOLO COLORI E LABEL, IL MARKER VIENE IGNORATO DOPO)
    manual_styles = {
        'kron_flux': {'label': 'Kron Flux (Automatico)', 'color': 'blue'},
        'somma_apertura_ultimo_pixel': {'label': 'Somma Apertura (R Max)', 'color': 'red'},
        'kron_flux_manuale': {'label': 'Kron Flux (Manuale)', 'color': 'green'}
    }

    if flusso_name in manual_styles:
        return manual_styles[flusso_name]

    # Fallback automatico
    return {
        'label': flusso_name.replace('_', ' ').title(),
        'color': default_colors[index % len(default_colors)]
    }
'''

# --- PARAMETRI FILE E PERCORSI ---
run = 1
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
lista_percorsi_csv = [os.path.join(cartella_csv, file) for file in file_csv]

filename = "../dispersione_flussi/risultati_analisi_run_1.csv"

# --- RILEVAMENTO FLUSSI ---
df_header = pd.read_csv(lista_percorsi_csv[0], comment='#', nrows=0)
colonne = df_header.columns.tolist()
idx_start = colonne.index('saturazione') + 1
idx_end = colonne.index('RA_centroid')
tipi_flusso = colonne[idx_start:idx_end] # lista completa dei flussi
print(f"Tipi di flusso rilevati: {tipi_flusso}")


tipi_flusso = [col for col in tipi_flusso if col != 'raggio_kron_aper']
'''# Inserisci qui SOLO i nomi esatti dei flussi che vuoi graficare
tipi_flusso = [
    'flusso_fisso_max_run',
    'kron_manuale_aper',
    'kron_flux'
]'''



colors_list = get_distinct_colors(len(tipi_flusso))


def get_style(flusso_name, index):
    """Restituisce uno stile (colore, label) per un dato flusso."""

    # Stili fissi manuali (SOLO LABEL e override specifici se vuoi forzare un colore)
    # Rimuoviamo i colori fissi hardcoded per evitare conflitti, lasciamo decidere alla palette dinamica
    manual_labels = {
        'kron_flux': 'Kron Flux (Automatico)',
        'somma_apertura_ultimo_pixel': 'Somma Apertura (R Max)',
        'kron_flux_manuale': 'Kron Flux (Manuale)',
        'flusso_fisso_max_run': 'Flusso Fisso (Max Run)'
    }

    # Determina la label
    label = manual_labels.get(flusso_name, flusso_name.replace('_', ' ').title())

    # Assegna il colore dalla lista generata
    color = colors_list[index]

    return {
        'label': label,
        'color': color
    }

# --- CARICAMENTO DATI ---
if not os.path.exists(filename):
    print(f"ERRORE: Il file {filename} non esiste.")
    exit()

dataframe = pd.read_csv(filename, comment="#")
tbl = Table.from_pandas(dataframe)
print(f"Dati caricati: {len(tbl)} stelle.")

magnitudini_sommate = tbl['Mag_Integrata']
magnitudini_massime = tbl['Mag_Brightest']


def get_clean_data(x_arr, y_arr):
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr) & ~np.isinf(x_arr) & ~np.isinf(y_arr)
    return x_arr[mask], y_arr[mask]


# =============================================================================
# GRAFICO 1: Deviazione Standard (Assoluta) vs Flusso Medio
# =============================================================================
plt.figure(figsize=(10, 6))

for i, flusso in enumerate(tipi_flusso):
    col_media = f'media_{flusso}'
    col_std = f'std_{flusso}'

    if col_media not in tbl.colnames: continue

    media_val = tbl[col_media]
    std_val = tbl[col_std]
    x_clean, y_clean = get_clean_data(media_val, std_val)

    idx_sort = np.argsort(x_clean)
    x_sorted = x_clean[idx_sort]
    y_sorted = y_clean[idx_sort]

    style = get_style(flusso, i)

    plt.plot(x_sorted, y_sorted,
             marker=MARKER_STYLE,  # <--- USA LA VARIABILE GLOBALE
             linestyle='-',
             linewidth=LINE_WIDTH,
             markersize=MARKER_SIZE,
             alpha=0.7,
             color=style['color'],
             label=style['label'])

plt.title('Stabilità: Errore Assoluto vs Flusso Medio')
plt.xlabel('Flusso Medio (ADU) - Scala Log')
plt.ylabel('Deviazione Standard (ADU)')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# GRAFICO 2: Errore Percentuale vs Magnitudine Integrata
# =============================================================================
plt.figure(figsize=(10, 6))

for i, flusso in enumerate(tipi_flusso):
    col_media = f'media_{flusso}'
    col_std = f'std_{flusso}'
    if col_media not in tbl.colnames: continue

    media_val = tbl[col_media]
    std_val = tbl[col_std]
    with np.errstate(divide='ignore', invalid='ignore'):
        err_percent = (std_val / media_val) * 100

    x_clean, y_clean = get_clean_data(magnitudini_sommate, err_percent)
    idx_sort = np.argsort(x_clean)
    x_sorted = x_clean[idx_sort]
    y_sorted = y_clean[idx_sort]

    style = get_style(flusso, i)

    plt.plot(x_sorted, y_sorted,
             marker=MARKER_STYLE,  # <--- USA LA VARIABILE GLOBALE
             linestyle='-',
             linewidth=LINE_WIDTH,
             markersize=MARKER_SIZE,
             alpha=0.7,
             color=style['color'],
             label=style['label'])

plt.title('Stabilità: Errore Percentuale vs Magnitudine Integrata')
plt.xlabel('Magnitudine Integrata')
plt.ylabel('Deviazione Standard (%)')
plt.ylim(0, 100)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# GRAFICO 3: Errore Percentuale vs Magnitudine Massima
# =============================================================================
plt.figure(figsize=(10, 6))

for i, flusso in enumerate(tipi_flusso):
    col_media = f'media_{flusso}'
    col_std = f'std_{flusso}'
    if col_media not in tbl.colnames: continue

    media_val = tbl[col_media]
    std_val = tbl[col_std]
    with np.errstate(divide='ignore', invalid='ignore'):
        err_percent = (std_val / media_val) * 100

    x_clean, y_clean = get_clean_data(magnitudini_massime, err_percent)
    idx_sort = np.argsort(x_clean)
    x_sorted = x_clean[idx_sort]
    y_sorted = y_clean[idx_sort]

    style = get_style(flusso, i)

    plt.plot(x_sorted, y_sorted,
             marker=MARKER_STYLE,  # <--- USA LA VARIABILE GLOBALE
             linestyle='-',
             linewidth=LINE_WIDTH,
             markersize=MARKER_SIZE,
             alpha=0.7,
             color=style['color'],
             label=style['label'])

plt.title('Stabilità: Errore Percentuale vs Magnitudine Pixel Max')
plt.xlabel('Magnitudine Stella Massima (Brightest Pixel)')
plt.ylabel('Deviazione Standard (%)')
plt.ylim(0, 100)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.show()