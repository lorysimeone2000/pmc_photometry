import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
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
# 1. FUNZIONI DI SUPPORTO GRAFICO
# =============================================================================

def freedman_diaconis_bins(data, num_images=1, max_bins=60):
    # riduco N al numero medio di stelle per immagine per non far esplodere la formula
    data_clean = data[~np.isnan(data)]
    n_effettivo = len(data_clean) / max(num_images, 1)
    if n_effettivo < 2: return 1

    iqr = np.percentile(data_clean, 75) - np.percentile(data_clean, 25)
    if iqr == 0: return 1

    bin_width = 2 * iqr / (n_effettivo ** (1 / 3))
    data_range = np.max(data_clean) - np.min(data_clean)
    bins = int(np.ceil(data_range / bin_width))

    # impongo un tetto massimo e un minimo di 1
    return min(max(bins, 1), max_bins)


# =============================================================================
# 2. RICERCA E RACCOLTA DATI (TUTTI I GIORNI E LE RUN DEL BLAZAR)
# =============================================================================

print("--- INIZIO CALCOLO COMPLETEZZA (DATASET BLAZAR) ---")

dir_unite = BASE_DIR / "tabelle_blazar" / "tabelle_unite"
dir_cataloghi = BASE_DIR / "tabelle_blazar" / "tabelle_cataloghi"

if not dir_unite.exists() or not dir_cataloghi.exists():
    print("ERRORE: Impossibile trovare le cartelle 'tabelle_unite' o 'tabelle_cataloghi' in blazar/tabelle/")
    exit()

# inizializzo le liste globali per accumulare i dati di tutte le immagini
tutti_mag_data = []
tutti_mag_cat_data = []
totale_perse = 0
totale_catalogate = 0
totale_correlate = 0
immagini_totali = 0

fwhm_usato = None
size_usato = None
nomi_run_processate = []

# esploro la cartella dei giorni
for giorno_dir in sorted([d for d in dir_unite.iterdir() if d.is_dir()]):
    giorno_nome = giorno_dir.name
    giorno_cat_dir = dir_cataloghi / giorno_nome

    if not giorno_cat_dir.exists():
        continue

    # esploro le run all'interno del giorno
    for run_dir in sorted([d for d in giorno_dir.iterdir() if d.is_dir()]):
        run_nome = run_dir.name
        run_cat_dir = giorno_cat_dir / run_nome

        if not run_cat_dir.exists():
            continue

        csv_unite_list = sorted(list(run_dir.glob("*.csv")))
        csv_cat_list = sorted(list(run_cat_dir.glob("*.csv")))

        num_file = min(len(csv_unite_list), len(csv_cat_list))
        if num_file == 0:
            continue

        nomi_run_processate.append(f"{giorno_nome}/{run_nome}")

        # analizzo ogni immagine della run corrente
        for i in tqdm(range(num_file), desc=f"Elaborazione {giorno_nome} - {run_nome}"):
            df_unite = pd.read_csv(csv_unite_list[i], comment='#')
            df_cat = pd.read_csv(csv_cat_list[i], comment='#')

            # salvo i parametri dalla prima immagine valida per scriverli nel titolo
            if fwhm_usato is None:
                header_dal_csv = leggi_header_da_csv(csv_unite_list[i])
                fwhm_usato = header_dal_csv.get('fwhm', header_dal_csv.get('FWHM'))
                size_usato = header_dal_csv.get('size', header_dal_csv.get('SIZE'))

            # preparo i dati (filtro usando pandas nativo)
            mask_si = df_unite['Corrispondenza'].astype(str).str.startswith('SI', na=False)
            df_si = df_unite[mask_si]

            ids_trovati_e_correlati = set(df_si['ID'])

            # conto le stelle non correlate (perse)
            for star_id in df_cat['ID']:
                if star_id not in ids_trovati_e_correlati:
                    totale_perse += 1

            # rimuovo i duplicati (una stella di catalogo assegnata a più sorgenti vicine)
            df_uniche = df_si.drop_duplicates(subset=['ID'])

            totale_catalogate += len(df_cat)
            totale_correlate += len(df_uniche)

            # estraggo i valori delle magnitudini puri (rimuovendo eventuali NaN)
            mags_correlate = df_uniche['Mag'].dropna().values
            mags_catalogo = df_cat['Mag'].dropna().values

            # aggiungo i dati dell'immagine corrente alle liste globali
            tutti_mag_data.extend(mags_correlate)
            tutti_mag_cat_data.extend(mags_catalogo)
            immagini_totali += 1

# blocco l'esecuzione se non ho caricato dati
if len(tutti_mag_cat_data) == 0:
    print("ERRORE: Nessun dato valido caricato.")
    exit()

print(f"\n--- RIEPILOGO GLOBALE ---")
print(f"Totale Immagini Elaborate: {immagini_totali}")
print(f"Stelle totali di catalogo (tutte le run): {totale_catalogate}")
print(f"Stelle correlate uniche (tutte le run): {totale_correlate}")
print(f"Stelle di catalogo NON correlate/perse: {totale_perse}")

# =============================================================================
# 3. ELABORAZIONE STATISTICA E PLOTTING
# =============================================================================

# 1. calcolo i bin comuni GLOBALI
dati_totali = np.concatenate((tutti_mag_data, tutti_mag_cat_data))
n_bin = freedman_diaconis_bins(dati_totali, num_images=immagini_totali)
hist_range = (np.min(dati_totali), np.max(dati_totali))
bins = np.histogram_bin_edges(dati_totali, bins=n_bin, range=hist_range)

# 2. calcolo i conteggi GLOBALI per il grafico
counts_cat, bin_edges = np.histogram(tutti_mag_cat_data, bins=bins)
counts_corr, _ = np.histogram(tutti_mag_data, bins=bins)

# 3. calcolo LA PERCENTUALE GLOBALE (COMPLETEZZA)
with np.errstate(divide='ignore', invalid='ignore'):
    percentuale_completezza = (counts_corr / counts_cat) * 100

# sostituisco i NaN e gli Inf con 0
percentuale_completezza = np.nan_to_num(percentuale_completezza)

# calcolo i centri dei bin
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 4. creo il grafico a linee
plt.figure(figsize=(14, 8))

plt.plot(bin_centers, percentuale_completezza,
         color='green',
         marker='o',
         markersize=4,
         linestyle='-',
         linewidth=2,
         label='Percentuale di rilevamento Globale')

# traccio le linee di riferimento al 100% e al 50%
plt.axhline(100, color='gray', linestyle='--', alpha=0.5, label='100% Completezza')
plt.axhline(50, color='red', linestyle=':', alpha=0.5, label='50% Completezza')

# imposto gli assi
plt.xlabel('Magnitudine (Centri dei Bin)', fontsize=12)
plt.ylabel('Percentuale stelle correlate / totali (%)', fontsize=12)

# costruisco il titolo dinamicamente
titolo = f'Funzione di Completezza Globale (Dataset Blazar - {len(nomi_run_processate)} Run Analizzate)\n'
titolo += f'Media: {totale_correlate / immagini_totali:.1f} match su {totale_catalogate / immagini_totali:.1f} catalogate per immagine'
if fwhm_usato and size_usato:
    titolo += f' (FWHM = {fwhm_usato}, size = {size_usato})'
plt.title(titolo, fontsize=14)

# inverto l'asse X (magnitudini astronomiche decrescenti: più basse a sinistra)
plt.gca().invert_xaxis()

# imposto il limite Y da 0 a poco più di 100 per mantenere il grafico pulito
plt.ylim(0, 110)

# aggiungo la griglia
plt.grid(True, which="both", linestyle='--', alpha=0.6)

# formatto i Tick sull'Asse X mostrando solo un tot di etichette per leggibilità
tick_labels = [f'{c:.2f}' for c in bin_centers]
step = 4
subset_ticks = bin_centers[::step]
subset_labels = tick_labels[::step]
plt.gca().set_xticks(subset_ticks)
plt.gca().set_xticklabels(subset_labels, rotation=45, ha='right')

plt.legend(fontsize=11)
plt.tight_layout()

# salvo il grafico nella cartella di analisi del blazar
output_dir_grafici = BASE_DIR / "pmc_photometry" / "blazar" / "analisi" / "sensibilità_magnitudini"
output_dir_grafici.mkdir(parents=True, exist_ok=True)
output_path = output_dir_grafici / "completezza_globale_blazar.png"

plt.savefig(output_path, dpi=300)
print(f"Grafico salvato in: {output_path.relative_to(BASE_DIR)}")

plt.savefig("completezza_globale_blazar.png", dpi=300)
plt.show()

# =============================================================================
# 4. GENERAZIONE GRAFICI PER SINGOLE IMMAGINI
# =============================================================================

print("\n--- INIZIO CALCOLO E PLOTTING SINGOLE IMMAGINI ---")

# definisco la directory base per i nuovi grafici
dir_output_singole = output_dir_grafici / "cumulativa_tutte_le_run_completezza"

# esploro nuovamente la cartella dei giorni per elaborare l'immagine singola
for giorno_dir in sorted([d for d in dir_unite.iterdir() if d.is_dir()]):
    giorno_nome = giorno_dir.name
    giorno_cat_dir = dir_cataloghi / giorno_nome

    if not giorno_cat_dir.exists():
        continue

    # esploro la run all'interno del giorno
    for run_dir in sorted([d for d in giorno_dir.iterdir() if d.is_dir()]):
        run_nome = run_dir.name
        run_cat_dir = giorno_cat_dir / run_nome

        if not run_cat_dir.exists():
            continue

        csv_unite_list = sorted(list(run_dir.glob("*.csv")))
        csv_cat_list = sorted(list(run_cat_dir.glob("*.csv")))

        num_file = min(len(csv_unite_list), len(csv_cat_list))
        if num_file == 0:
            continue

        # creo la sottocartella per il giorno e la run
        out_run_dir = dir_output_singole / giorno_nome / run_nome
        out_run_dir.mkdir(parents=True, exist_ok=True)

        # analizzo l'immagine per generare il suo grafico
        for i in tqdm(range(num_file), desc=f"Plotting {giorno_nome} - {run_nome}"):
            # procedo solo per l'ordine delle immagini divisibile per 20
            if i % 20 != 0:
                continue

            file_unite = csv_unite_list[i]
            file_cat = csv_cat_list[i]
            nome_base = file_unite.stem

            df_unite = pd.read_csv(file_unite, comment='#')
            df_cat = pd.read_csv(file_cat, comment='#')

            # filtro il match valido
            mask_si = df_unite['Corrispondenza'].astype(str).str.startswith('SI', na=False)
            df_si = df_unite[mask_si]
            df_uniche = df_si.drop_duplicates(subset=['ID'])

            # estraggo il vettore dell'immagine
            mags_correlate_singola = df_uniche['Mag'].dropna().values
            mags_catalogo_singola = df_cat['Mag'].dropna().values

            if len(mags_catalogo_singola) == 0 and len(mags_correlate_singola) == 0:
                continue

            # unisco il dato per calcolare il bin ottimale
            dati_singoli = np.concatenate((mags_correlate_singola, mags_catalogo_singola))
            if len(dati_singoli) < 2:
                continue

            n_bin_singola = freedman_diaconis_bins(dati_singoli, num_images=1)
            hist_range_singola = (np.min(dati_singoli), np.max(dati_singoli))
            bins_singola = np.histogram_bin_edges(dati_singoli, bins=n_bin_singola, range=hist_range_singola)

            # calcolo i conteggi specifici per la singola immagine
            counts_cat_singola, bin_edges_singola = np.histogram(mags_catalogo_singola, bins=bins_singola)
            counts_corr_singola, _ = np.histogram(mags_correlate_singola, bins=bins_singola)

            # calcolo la percentuale di completezza singola
            with np.errstate(divide='ignore', invalid='ignore'):
                percentuale_completezza_singola = (counts_corr_singola / counts_cat_singola) * 100

            percentuale_completezza_singola = np.nan_to_num(percentuale_completezza_singola)
            bin_centers_singola = (bin_edges_singola[:-1] + bin_edges_singola[1:]) / 2

            # genero il grafico a linee
            plt.figure(figsize=(14, 8))

            plt.plot(bin_centers_singola, percentuale_completezza_singola,
                     color='green',
                     marker='o',
                     markersize=4,
                     linestyle='-',
                     linewidth=2,
                     label='Percentuale di rilevamento')

            # traccio le linee di riferimento
            plt.axhline(100, color='gray', linestyle='--', alpha=0.5, label='100% Completezza')
            plt.axhline(50, color='red', linestyle=':', alpha=0.5, label='50% Completezza')

            plt.xlabel('Magnitudine (Centri dei Bin)', fontsize=12)
            plt.ylabel('Percentuale stelle correlate / totali (%)', fontsize=12)

            # definisco il titolo dinamicamente
            titolo_singolo = f'Funzione di Completezza ({nome_base})\n'
            titolo_singolo += f'{len(mags_correlate_singola)} match su {len(mags_catalogo_singola)} catalogate'
            if fwhm_usato and size_usato:
                titolo_singolo += f' (FWHM = {fwhm_usato}, size = {size_usato})'
            plt.title(titolo_singolo, fontsize=14)

            # inverto l'asse x
            plt.gca().invert_xaxis()

            # imposto il limite y
            plt.ylim(0, 110)

            plt.grid(True, which="both", linestyle='--', alpha=0.6)

            # formatto i tick sull'asse x per leggibilità
            tick_labels_singola = [f'{c:.2f}' for c in bin_centers_singola]
            step_singola = max(1, len(bin_centers_singola) // 10)
            subset_ticks_singola = bin_centers_singola[::step_singola]
            subset_labels_singola = tick_labels_singola[::step_singola]
            plt.gca().set_xticks(subset_ticks_singola)
            plt.gca().set_xticklabels(subset_labels_singola, rotation=45, ha='right')

            plt.legend(fontsize=11)
            plt.tight_layout()

            # salvo il grafico nella cartella specifica
            plt.savefig(out_run_dir / f"{nome_base}.png", dpi=300)

            # chiudo la figura per evitare di saturare la RAM
            plt.close()