import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings('ignore')


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    # risalgo l'albero delle directory per trovare la root del progetto
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
# 1. FUNZIONI DI SUPPORTO STATISTICO
# =============================================================================

def freedman_diaconis_bins(data, num_images=1, max_bins=60):
    # pulisco i dati e calcolo il numero di bin ottimali
    data_clean = data[~np.isnan(data)]
    n_effettivo = len(data_clean) / max(num_images, 1)
    if n_effettivo < 2: return 1

    iqr = np.percentile(data_clean, 75) - np.percentile(data_clean, 25)
    if iqr == 0: return 1

    bin_width = 2 * iqr / (n_effettivo ** (1 / 3))
    data_range = np.max(data_clean) - np.min(data_clean)
    bins = int(np.ceil(data_range / bin_width))

    # impongo un limite per non creare grafici illeggibili
    return min(max(bins, 1), max_bins)


# =============================================================================
# 2. RICERCA E RACCOLTA DATI (TUTTI I GIORNI E LE RUN DEL BLAZAR)
# =============================================================================

print("--- INIZIO CALCOLO DISTRIBUZIONE MEDIA (DATASET BLAZAR) ---")

# definisco le cartelle bersaglio
dir_unite = BASE_DIR / "tabelle_blazar_e_non_blazar" / "tabelle_unite"
dir_cataloghi = BASE_DIR / "tabelle_blazar_e_non_blazar" / "tabelle_cataloghi"

if not dir_unite.exists() or not dir_cataloghi.exists():
    print("ERRORE: Impossibile trovare le cartelle 'tabelle_unite' o 'tabelle_cataloghi' in blazar/tabelle/")
    exit()

# inizializzo le liste per accumulare i dati
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

        # recupero tutte le tabelle
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

            # prelevo i parametri di segmentazione dal primo header valido
            if fwhm_usato is None:
                header_dal_csv = leggi_header_da_csv(csv_unite_list[i])
                fwhm_usato = header_dal_csv.get('fwhm', header_dal_csv.get('FWHM'))
                size_usato = header_dal_csv.get('size', header_dal_csv.get('SIZE'))

            # filtro nativamente in pandas gli oggetti correlati
            mask_si = df_unite['Corrispondenza'].astype(str).str.startswith('SI', na=False)
            df_si = df_unite[mask_si]

            ids_trovati_e_correlati = set(df_si['ID'])

            # conto le stelle del catalogo che non compaiono nella lista dei match
            for star_id in df_cat['ID']:
                if star_id not in ids_trovati_e_correlati:
                    totale_perse += 1

            # rimuovo i match duplicati se una stella di catalogo si è agganciata a due sorgenti sporche
            df_uniche = df_si.drop_duplicates(subset=['ID'])

            totale_catalogate += len(df_cat)
            totale_correlate += len(df_uniche)

            # estraggo i vettori puri
            mags_correlate = df_uniche['Mag'].dropna().values
            mags_catalogo = df_cat['Mag'].dropna().values

            # li accodo ai dataset globali
            tutti_mag_data.extend(mags_correlate)
            tutti_mag_cat_data.extend(mags_catalogo)
            immagini_totali += 1

# prevengo errori in caso di cartelle vuote o dati corrotti
if len(tutti_mag_cat_data) == 0:
    print("ERRORE: Nessun dato valido caricato.")
    exit()

print(f"\n--- RIEPILOGO GLOBALE ---")
print(f"Totale Immagini Elaborate: {immagini_totali}")
print(f"Stelle totali di catalogo (tutte le run): {totale_catalogate}")
print(f"Stelle correlate uniche (tutte le run): {totale_correlate}")
print(f"Stelle di catalogo NON correlate/perse: {totale_perse}")

# =============================================================================
# 3. ELABORAZIONE STATISTICA E PLOTTING GLOBALE
# =============================================================================

# 1. calcolo i bin comuni GLOBALI
dati_totali = np.concatenate((tutti_mag_data, tutti_mag_cat_data))
n_bin = freedman_diaconis_bins(dati_totali, num_images=immagini_totali)
hist_range = (np.min(dati_totali), np.max(dati_totali))
bins = np.histogram_bin_edges(dati_totali, bins=n_bin, range=hist_range)

# 2. calcolo i conteggi GLOBALI
counts_cat, bin_edges = np.histogram(tutti_mag_cat_data, bins=bins)
counts_corr, _ = np.histogram(tutti_mag_data, bins=bins)

# 3. normalizzo le frequenze dividendole per il numero di immagini analizzate
media_counts_cat = counts_cat / immagini_totali
media_counts_corr = counts_corr / immagini_totali

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 4. genero il grafico
plt.figure(figsize=(14, 8))

# traccio la curva delle sorgenti catalogate (teoriche)
plt.plot(bin_centers, media_counts_cat,
         color='purple',
         linestyle='-',
         linewidth=1.5,
         label='Sorgenti Catalogate (Media)')

# traccio la curva delle sorgenti effettivamente identificate e correlate
plt.plot(bin_centers, media_counts_corr,
         color='red',
         linestyle='-',
         linewidth=1.5,
         label='Sorgenti Correlate (Media)')

# imposto la scala logaritmica perché la distribuzione delle stelle cresce esponenzialmente
plt.yscale('log')
plt.xlabel('Magnitudine (Centri dei Bin)', fontsize=12)
plt.ylabel('Frequenza Media (Conteggi / Immagine)', fontsize=12)

# definisco il titolo dinamicamente
titolo = f'Distribuzione Media delle Magnitudini: Catalogate vs Correlate (Dataset Blazar - {len(nomi_run_processate)} Run)\n'
titolo += f'Media di {totale_correlate / immagini_totali:.1f} match su {totale_catalogate / immagini_totali:.1f} catalogate per immagine'
if fwhm_usato and size_usato:
    titolo += f' (FWHM = {fwhm_usato}, size = {size_usato})'
plt.title(titolo, fontsize=14)

# inverto l'asse x per far crescere la luminosità verso sinistra come da convenzione astronomica
plt.gca().invert_xaxis()

plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

# salvo l'immagine generata
output_dir_grafici = BASE_DIR / "pmc_photometry" / "blazar" / "analisi" / "sensibilità_magnitudini"
output_dir_grafici.mkdir(parents=True, exist_ok=True)
output_path = output_dir_grafici / "distribuzione_media_magnitudini_blazar.png"

plt.savefig('distribuzione_media_magnitudini_blazar.png', dpi=300)
print(f"Grafico salvato in: {output_path.relative_to(BASE_DIR)}")

plt.close()

# =============================================================================
# 4. GENERAZIONE GRAFICI PER SINGOLE IMMAGINI DIVISE PER TARGET
# =============================================================================

print("\n--- INIZIO CALCOLO E PLOTTING SINGOLE IMMAGINI ---")

# definisco la directory base per i nuovi grafici
dir_output_singole = output_dir_grafici / "cumulativa_tutte_le_run"

# imposto le coordinate di riferimento dei due target
coord_mrk = SkyCoord(ra=166.1138 * u.deg, dec=38.2088 * u.deg, frame='icrs')
coord_crab = SkyCoord(ra=83.6331 * u.deg, dec=22.0145 * u.deg, frame='icrs')

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

        # analizzo la prima tabella della run per determinare a quale target puntano effettivamente le stelle
        primo_file = csv_unite_list[0]
        df_primo = pd.read_csv(primo_file, comment='#')

        target_nome = "sconosciuto"

        if not df_primo.empty and 'RA_centroid' in df_primo.columns and 'DEC_centroid' in df_primo.columns:
            ra_val = df_primo['RA_centroid'].median()
            dec_val = df_primo['DEC_centroid'].median()

            try:
                coord_centro = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame='icrs')

                # calcolo le distanze e decido il nome della cartella target
                sep_mrk = coord_centro.separation(coord_mrk)
                sep_crab = coord_centro.separation(coord_crab)

                if sep_mrk < sep_crab:
                    target_nome = "markarian"
                else:
                    target_nome = "crab"
            except Exception:
                pass
        else:
            # fallback all'header se la tabella è vuota
            header_primo_file = leggi_header_da_csv(primo_file)
            ra_val = header_primo_file.get('RA') or header_primo_file.get('RAJ2000') or header_primo_file.get('OBJ-RA')
            dec_val = header_primo_file.get('DEC') or header_primo_file.get('DEJ2000') or header_primo_file.get(
                'OBJ-DEC')

            if ra_val is not None and dec_val is not None:
                try:
                    if isinstance(ra_val, (int, float)):
                        coord_centro = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame='icrs')
                    else:
                        coord_centro = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg), frame='icrs')

                    sep_mrk = coord_centro.separation(coord_mrk)
                    sep_crab = coord_centro.separation(coord_crab)

                    if sep_mrk < sep_crab:
                        target_nome = "markarian"
                    else:
                        target_nome = "crab"
                except Exception:
                    pass

        # creo la sottocartella inserendo il livello del target tra giorno e run
        out_run_dir = dir_output_singole / giorno_nome / target_nome / run_nome
        out_run_dir.mkdir(parents=True, exist_ok=True)

        # analizzo l'immagine per generare il suo grafico
        for i in tqdm(range(num_file), desc=f"Plotting {giorno_nome} - {run_nome} ({target_nome})"):

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

            # calcolo il conteggio
            counts_cat_singola, bin_edges_singola = np.histogram(mags_catalogo_singola, bins=bins_singola)
            counts_corr_singola, _ = np.histogram(mags_correlate_singola, bins=bins_singola)
            bin_centers_singola = (bin_edges_singola[:-1] + bin_edges_singola[1:]) / 2

            # genero il grafico identico al precedente
            plt.figure(figsize=(14, 8))

            # traccio la curva della sorgente catalogata
            plt.plot(bin_centers_singola, counts_cat_singola,
                     color='purple',
                     linestyle='-',
                     linewidth=1.5,
                     label='Sorgenti Catalogate')

            # traccio la curva della sorgente correlata
            plt.plot(bin_centers_singola, counts_corr_singola,
                     color='red',
                     linestyle='-',
                     linewidth=1.5,
                     label='Sorgenti Correlate')

            plt.yscale('log')
            plt.xlabel('Magnitudine (Centri dei Bin)', fontsize=12)
            plt.ylabel('Frequenza (Conteggi)', fontsize=12)

            # definisco il titolo dinamicamente
            titolo_singolo = f'Distribuzione Magnitudini: Catalogate vs Correlate ({nome_base})\n'
            titolo_singolo += f'{len(mags_correlate_singola)} match su {len(mags_catalogo_singola)} catalogate'
            if fwhm_usato and size_usato:
                titolo_singolo += f' (FWHM = {fwhm_usato}, size = {size_usato})'
            plt.title(titolo_singolo, fontsize=14)

            # inverto l'asse x
            plt.gca().invert_xaxis()

            plt.grid(True, which="both", linestyle='--', alpha=0.6)
            plt.legend(fontsize=11)
            plt.tight_layout()

            # salvo il grafico nella cartella specifica
            plt.savefig(out_run_dir / f"{nome_base}.png", dpi=300)

            # chiudo la figura per evitare di saturare la RAM
            plt.close()

# =============================================================================
# 5. ANDAMENTO TEMPORALE DISTANZE DAI TARGET
# =============================================================================

print("\n--- GENERAZIONE GRAFICO DISTANZE TEMPORALI ---")

from astropy.time import Time

tempi_plot = []
distanze_mrk_plot = []
distanze_crab_plot = []

# scorro di nuovo le cartelle per raccogliere le informazioni cronologiche
for giorno_dir in sorted([d for d in dir_unite.iterdir() if d.is_dir()]):
    for run_dir in sorted([d for d in giorno_dir.iterdir() if d.is_dir()]):
        for file_csv in sorted(list(run_dir.glob("*.csv"))):
            df_temp = pd.read_csv(file_csv, comment='#')
            header_temp = leggi_header_da_csv(file_csv)

            if not df_temp.empty and 'RA_centroid' in df_temp.columns and 'DEC_centroid' in df_temp.columns:
                ra_val = df_temp['RA_centroid'].median()
                dec_val = df_temp['DEC_centroid'].median()
            else:
                ra_val = header_temp.get('RA') or header_temp.get('RAJ2000') or header_temp.get('OBJ-RA')
                dec_val = header_temp.get('DEC') or header_temp.get('DEJ2000') or header_temp.get('OBJ-DEC')

            if ra_val is not None and dec_val is not None:
                try:
                    if isinstance(ra_val, (int, float)):
                        coord_centro = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame='icrs')
                    else:
                        coord_centro = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg), frame='icrs')

                    # estraggo il tempo
                    tempo_obs_str = header_temp.get('DATE-OBS')
                    if tempo_obs_str:
                        tempo = Time(tempo_obs_str).unix
                    else:
                        tempo = file_csv.stat().st_mtime

                    tempi_plot.append(tempo)
                    distanze_mrk_plot.append(coord_centro.separation(coord_mrk).deg)
                    distanze_crab_plot.append(coord_centro.separation(coord_crab).deg)
                except Exception:
                    pass

if tempi_plot:
    # ordino i dati cronologicamente
    dati_ordinati = sorted(zip(tempi_plot, distanze_mrk_plot, distanze_crab_plot), key=lambda x: x[0])
    tempi_ord = np.array([x[0] for x in dati_ordinati])
    dist_mrk_ord = np.array([x[1] for x in dati_ordinati])
    dist_crab_ord = np.array([x[2] for x in dati_ordinati])

    # calcolo i secondi trascorsi dalla prima immagine
    tempi_sec = tempi_ord - tempi_ord[0]

    plt.figure(figsize=(14, 8), dpi=300)

    # traccio la distanza da markarian
    plt.plot(tempi_sec, dist_mrk_ord, color='teal', marker='.', linestyle='-', linewidth=1.5,
             label='Distanza da Markarian 421')

    # traccio la distanza dalla crab
    plt.plot(tempi_sec, dist_crab_ord, color='crimson', marker='.', linestyle='-', linewidth=1.5,
             label='Distanza da Crab Nebula')

    plt.xlabel('Tempo trascorso dalla prima immagine (secondi)', fontsize=12)
    plt.ylabel('Distanza (Gradi)', fontsize=12)
    plt.title('Andamento temporale della distanza dai target', fontsize=14)
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_dist_path = output_dir_grafici / "andamento_distanze_target.png"
    plt.savefig(out_dist_path)
    plt.show()
    plt.close()
    print(f"Grafico distanze salvato in: {out_dist_path.relative_to(BASE_DIR)}")