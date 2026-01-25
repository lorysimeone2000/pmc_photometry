import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from astropy.table import Table
import warnings
from astropy.wcs import FITSFixedWarning

# Sopprime il warning FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)


# --- FUNZIONI DI UTILITÀ ---

def converti_valore(valore):
    """Converte una stringa nel tipo di dato appropriato."""
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
    if valore.upper() in ['T', 'TRUE', 'YES', 'Y']:
        return True
    elif valore.upper() in ['F', 'FALSE', 'NO', 'N']:
        return False
    return valore


'''# --- CONFIGURAZIONE ZOOM ---
base_pad = (np.sqrt(stella_ref['area']) * 2)
zoom_factor = 2.0  # Tienilo basso per vedere bene i pixel (es. 2 o 3)
final_pad = base_pad * zoom_factor
# --- CICLO SULLA RUN ---
for i, path_csv in enumerate(lista_percorsi_csv):

    # ... [Lettura CSV e caricamento FITS come prima] ...
    # (Assumo tu abbia già caricato 'data' col fondo sottratto e trovato 'cur_x', 'cur_y')

    # ESEMPIO RAPIDO DI RECUPERO DATI (per contesto):
    try:
        df_curr = pd.read_csv(path_csv, comment='#')
        row_stella = df_curr[df_curr['ID'] == id_stella_target]
        if len(row_stella) == 0: continue
        cur_x, cur_y = row_stella.iloc[0]['xcentroid'], row_stella.iloc[0]['ycentroid']

        header_info = leggi_header_da_csv(path_csv)
        path_fits = header_info.get('PERCORSO_FILE', '')
        with fits.open(path_fits) as hdu:
            full_data = hdu[0].data
            _, median_globale, std_globale = sigma_clipped_stats(full_data, sigma=3.0)
            full_data = full_data - median_globale
    except Exception:
        continue

    # --- 1. DEFINIZIONE DEL CUTOUT (RITAGLIO) ---
    # Convertiamo i limiti in interi per lo slicing di numpy
    # Numpy usa formato [y, x], quindi dobbiamo stare attenti

    y_min_cut = int(max(0, cur_y - final_pad))
    y_max_cut = int(min(full_data.shape[0], cur_y + final_pad))
    x_min_cut = int(max(0, cur_x - final_pad))
    x_max_cut = int(min(full_data.shape[1], cur_x + final_pad))

    # Creiamo la piccola immagine (Cutout)
    cutout_data = full_data[y_min_cut:y_max_cut, x_min_cut:x_max_cut]

    # --- 2. ESECUZIONE SEGMENTAZIONE SUL CUTOUT ---
    # Passiamo la std globale per evitare che la statistica impazzisca sul piccolo riquadro
    tbl_cutout, params, seg_map_cutout = analisi_image_segmentation(cutout_data, std_esterna=std_globale)

    # --- 3. PLOTTING ---
    plt.figure(figsize=(6, 6))

    # Mostriamo i dati del cutout
    plt.imshow(cutout_data, cmap="grey_r", norm=LogNorm(), interpolation='nearest', origin='lower')

    # --- 4. MARCATURA PIXEL (VISUALIZZAZIONE SEGMENTAZIONE) ---
    if seg_map_cutout is not None:
        # Metodo A: Contorni Colorati (Molto elegante)
        # Disegna una linea attorno ai segmenti. levels=[0.5] traccia il bordo tra 0 (fondo) e 1 (sorgente)
        plt.contour(seg_map_cutout.data, levels=[0.5], colors='cyan', linewidths=2, alpha=0.8)

        # Metodo B: Sovrapposizione semitrasparente (Opzionale, se preferisci riempire l'area)
        # masked_seg = np.ma.masked_where(seg_map_cutout.data == 0, seg_map_cutout.data)
        # plt.imshow(masked_seg, cmap='spring', alpha=0.3, origin='lower')

    # Estetica
    plt.title(f"Frame {i:03d} - Zoom & Segmentazione\nID: {id_stella_target}")

    # Nota: Poiché abbiamo fatto imshow del cutout, gli assi partono da 0,0 (coordinate locali).
    # Non serve settare xlim/ylim perché l'immagine è già ritagliata!

    # Se vuoi indicare il centroide calcolato nel cutout (sarà diverso da cur_x globale)
    # Le coordinate locali sono: x_globale - x_min_cut
    plt.scatter(cur_x - x_min_cut, cur_y - y_min_cut, s=100, facecolors='none', edgecolors='r',
                label='Centroide Catalogo')

    plt.tight_layout()

    # Salvataggio
    nome_file_out = f"seg_zoom_{i:03d}.png"
    plt.savefig(os.path.join(output_dir, nome_file_out), dpi=100)
    plt.close()

    if i % 10 == 0: print(f"Fatto frame {i}")
'''
def somma_magnitudini(series_mags):
    """Integra le magnitudini sommando i flussi."""
    mags = series_mags.dropna()
    if len(mags) == 0: return np.nan
    flussi = 10 ** (-0.4 * np.array(mags))
    flusso_totale = np.sum(flussi)
    if flusso_totale <= 0: return np.nan
    return -2.5 * np.log10(flusso_totale)

def leggi_header_da_csv(filename):
    """Legge l'header FITS salvato nelle prime righe del file CSV."""
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

# Imposta il percorso del file
filename = "/home/lorysimeone/tesi_magistrale/prove_2/analisi/magnitudini/studio_kron_flux/dispersione_flusso/risultati_analisi_run_1.csv"

# Lettura dati
dataframe = pd.read_csv(filename, comment="#")
tbl = Table.from_pandas(dataframe)
print(tbl)

# Estrazione colonne
medie_kron = tbl['media_kron']
std_kron = tbl['std_kron']
std_kron_percent = std_kron/medie_kron * 100
magnitudini_sommate = tbl['Mag_Integrata'] # Non usata nel plot attuale
magnitudini_massime = tbl['Mag_Brightest'] # Non usata nel plot attuale

# Ordinamento dati per il plot (estetico per la linea continua)
indici_ordinati = np.argsort(medie_kron)
x_sorted = medie_kron[indici_ordinati]
y_sorted = std_kron[indici_ordinati]

# --- PLOTTING ---
plt.figure(figsize=(10, 6)) # Opzionale: imposta una dimensione fissa per il grafico

plt.plot(x_sorted, y_sorted,
         marker='o', linestyle='-', linewidth=1, markersize=4, alpha=0.7)

# Titoli ed etichette
plt.title(f'Errore in funzione del kron medio di ogni stella')
plt.xlabel('kron medio nella run (Scala Logaritmica)')
plt.ylabel('deviazione standard sulla run')

# --- MODIFICA RICHIESTA: SCALA LOGARITMICA SU X ---
plt.xscale('log')

# Griglia e limiti
plt.grid(True, which="both", linestyle='--', alpha=0.5) # "which='both'" mostra griglia anche per le suddivisioni logaritmiche
plt.ylim(0, None)

plt.tight_layout()
plt.show()



# Ordinamento dati per il plot (estetico per la linea continua)
indici_ordinati = np.argsort(medie_kron)
x_sorted = medie_kron[indici_ordinati]
y_sorted = std_kron_percent[indici_ordinati]

# --- PLOTTING MAGNITUDINI SOMMATE---
plt.figure(figsize=(10, 6)) # Opzionale: imposta una dimensione fissa per il grafico

plt.plot(x_sorted, y_sorted,
         marker='o', linestyle='-', linewidth=1, markersize=4, alpha=0.7)

# Titoli ed etichette
plt.title(f'Errore in funzione del kron medio di ogni stella')
plt.xlabel('kron medio nella run (Scala Logaritmica)')
plt.ylabel('deviazione standard in percentuale sulla run')

# --- MODIFICA RICHIESTA: SCALA LOGARITMICA SU X ---
plt.xscale('log')

# Griglia e limiti
plt.grid(True, which="both", linestyle='--', alpha=0.5) # "which='both'" mostra griglia anche per le suddivisioni logaritmiche
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

# Ordinamento dati per il plot (estetico per la linea continua)
indici_ordinati = np.argsort(magnitudini_sommate)
x_sorted = magnitudini_sommate[indici_ordinati]
y_sorted = std_kron_percent[indici_ordinati]

# --- PLOTTING ---
plt.figure(figsize=(10, 6)) # Opzionale: imposta una dimensione fissa per il grafico

plt.plot(x_sorted, y_sorted,
         marker='o', linestyle='-', linewidth=1, markersize=4, alpha=0.7)

# Titoli ed etichette
plt.title(f'Errore in funzione della magnitudine integrata')
plt.xlabel('Magnitudine integrata')
plt.ylabel('deviazione standard in percentuale sulla run')

# Griglia e limiti
plt.grid(True, which="both", linestyle='--', alpha=0.5) # "which='both'" mostra griglia anche per le suddivisioni logaritmiche
plt.ylim(0, 100)

plt.tight_layout()
plt.gca().invert_xaxis()
plt.show()

# --- PLOTTING MAGNITUDINI MASSIME---

# Ordinamento dati per il plot (estetico per la linea continua)
indici_ordinati = np.argsort(magnitudini_massime)
x_sorted = magnitudini_massime[indici_ordinati]
y_sorted = std_kron_percent[indici_ordinati]

plt.figure(figsize=(10, 6)) # Opzionale: imposta una dimensione fissa per il grafico

plt.plot(x_sorted, y_sorted,
         marker='o', linestyle='-', linewidth=1, markersize=4, alpha=0.7)

# Titoli ed etichette
plt.title(f'Errore in funzione della magnitudine massima')
plt.xlabel('Magnitudine stella massima')
plt.ylabel('deviazione standard in percentuale sulla run')

# Griglia e limiti
plt.grid(True, which="both", linestyle='--', alpha=0.5) # "which='both'" mostra griglia anche per le suddivisioni logaritmiche
plt.ylim(0, 100)

plt.tight_layout()
plt.gca().invert_xaxis()
plt.show()