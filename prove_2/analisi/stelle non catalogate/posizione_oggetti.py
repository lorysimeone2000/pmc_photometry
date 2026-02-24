import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from tqdm import tqdm
from pathlib import Path
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI DINAMICA (PORTABILITÀ TOTALE)
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # risalgo l'albero delle directory per trovare la cartella base
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # cerco una specifica cartella in tutto il progetto
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


def converti_valore(valore):
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
    if valore.upper() in ['T', 'TRUE']: return True
    if valore.upper() in ['F', 'FALSE']: return False
    return valore


def leggi_header_da_csv(filename):
    # estraggo l'header FITS commentato in cima al CSV
    header_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                clean_line = line.strip()[1:].strip()
                if clean_line and ': ' in clean_line:
                    key, value = clean_line.split(': ', 1)
                    header_dict[key] = converti_valore(value)
            else:
                break
    return header_dict


def genera_wcs_da_dict(header_dict):
    # ricostruisco un oggetto WCS partendo dal dizionario dell'header
    header = fits.Header()
    for k, v in header_dict.items():
        if len(k) <= 8 and k not in ['COMMENT', 'HISTORY']:
            if v is not None and str(v).strip() != '':
                header[k] = v
    # sopprimo i warning per keyword mancanti non essenziali
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return WCS(header)


# definisco la BASE_DIR dinamicamente
BASE_DIR = trova_cartella_base("pmc_photometry")
print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")
print(f"------------------------------")

RUN = [1, 2, 3]
files_per_run = {}

print("\n--- RICERCA FILE CSV ---")
for r in RUN:
    nome_cartella_target = f"tabelle_unite_run_{r}"
    path_cartella_run = cerca_cartella_nel_progetto(BASE_DIR, nome_cartella_target)
    if path_cartella_run is None:
        print(f"[ATTENZIONE] Cartella '{nome_cartella_target}' non trovata. Salto la run {r}.")
        files_per_run[r] = []
        continue

    print(f"Run {r}: Cartella trovata in -> {path_cartella_run.relative_to(BASE_DIR)}")
    lista_csv = sorted(list(path_cartella_run.glob("*.csv")))
    files_per_run[r] = lista_csv

# =============================================================================
# 1. RACCOLTA DATI COORDINATE E DISTANZE
# =============================================================================

lista_totale_dfs = []
lista_distanze_stelle = []

for r in RUN:
    files_correnti = files_per_run.get(r, [])
    if not files_correnti: continue

    # cerco la cartella dei cataloghi originali per questa run
    cartella_cataloghi = cerca_cartella_nel_progetto(BASE_DIR, f"sorgenti_catalogate_run_{r}")

    descrizione_bar = f"Analisi Run {r} (Estrazione coords e distanze)"
    for nome_csv in tqdm(files_correnti, desc=descrizione_bar):
        try:
            df_temp = pd.read_csv(nome_csv, comment='#')

            if 'Corrispondenza' not in df_temp.columns or 'xcentroid' not in df_temp.columns:
                continue

            # filtro solo gli oggetti NON catalogati
            df_no = df_temp[df_temp['Corrispondenza'] == 'NO'].copy()

            if not df_no.empty:
                col_id = 'run_unique_id' if 'run_unique_id' in df_no.columns else 'ID'
                lista_totale_dfs.append(df_no[[col_id, 'xcentroid', 'ycentroid']])

                # --- CALCOLO DISTANZA DALLA STELLA PIÙ VICINA (IN GRADI) ---
                if cartella_cataloghi:
                    # estraggo il numero dell'immagine dal nome del file
                    match = re.search(r'immagine_(\d+)', nome_csv.name)
                    if match:
                        img_idx = match.group(1)
                        nome_file_cat = f"run_{r}_stelle_catalogate_immagine_{img_idx}.csv"
                        path_cat = cartella_cataloghi / nome_file_cat

                        if path_cat.exists():
                            df_cat = pd.read_csv(path_cat, comment='#')
                            df_cat = df_cat[df_cat['Mag'] <= 12]
                            header_dict = leggi_header_da_csv(path_cat)
                            wcs_img = genera_wcs_da_dict(header_dict)

                            # carico le coordinate celesti dal catalogo
                            ra_cat = df_cat['RAJ2000'].values
                            dec_cat = df_cat['DEJ2000'].values
                            coords_cat = SkyCoord(ra=ra_cat * u.deg, dec=dec_cat * u.deg)

                            # converto i centroidi (X, Y) degli oggetti NO in coordinate celesti (RA, DEC)
                            coords_no = wcs_img.pixel_to_world(df_no['xcentroid'].values, df_no['ycentroid'].values)

                            # calcolo la distanza sferica esatta in gradi tra gli oggetti e le stelle del catalogo
                            idx_cat, d2d, _ = coords_no.match_to_catalog_sky(coords_cat)

                            # salvo la distanza minima in gradi per ogni oggetto
                            for i, _id in enumerate(df_no[col_id].values):
                                lista_distanze_stelle.append({
                                    col_id: _id,
                                    'distanza_stella_deg': d2d[i].deg
                                })

        except Exception:
            continue

print("\n--- ELABORAZIONE DATI SPAZIALI E DISTANZE ---")

if lista_totale_dfs:
    # concateno tutto
    df_global = pd.concat(lista_totale_dfs, ignore_index=True)
    col_id = 'run_unique_id' if 'run_unique_id' in df_global.columns else 'ID'

    # siccome un oggetto compare in più frame, calcolo la sua posizione media per avere un dato univoco
    df_unique = df_global.groupby(col_id)[['xcentroid', 'ycentroid']].mean().reset_index()

    print(f"Totale oggetti 'NO' UNICI identificati: {len(df_unique)}")

    # =============================================================================
    # 2. CALCOLO DISTANZA DAL BORDO (GRAFICO 1)
    # =============================================================================
    W, H = 3072, 2048

    # calcolo la distanza dal bordo X più vicino (sinistro o destro)
    dist_x = np.minimum(df_unique['xcentroid'], W - df_unique['xcentroid'])
    # calcolo la distanza dal bordo Y più vicino (superiore o inferiore)
    dist_y = np.minimum(df_unique['ycentroid'], H - df_unique['ycentroid'])

    # la distanza assoluta dal bordo del sensore è il minimo tra le due
    df_unique['dist_edge'] = np.minimum(dist_x, dist_y)
    distanze_bordo = df_unique['dist_edge'].values

    # calcolo i bin equi-areali
    NUM_BINS = 15
    AREA_TOTALE = W * H
    area_per_bin = AREA_TOTALE / NUM_BINS

    bin_edges = [0.0]
    for k in range(1, NUM_BINS + 1):
        area_cumulata = k * area_per_bin
        x_k = ((W + H) - np.sqrt((W + H) ** 2 - 4 * area_cumulata)) / 4
        bin_edges.append(x_k)

    counts_bordo, _ = np.histogram(distanze_bordo, bins=bin_edges)

    # PLOTTING GRAFICO 1
    plt.figure(figsize=(15, 8))
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i + 1])} px" for i in range(NUM_BINS)]
    x_pos = np.arange(NUM_BINS)

    bars = plt.bar(x_pos, counts_bordo, color='crimson', edgecolor='black', alpha=0.8)

    conteggio_medio_atteso = len(distanze_bordo) / NUM_BINS
    plt.axhline(y=conteggio_medio_atteso, color='green', linestyle='--', linewidth=2,
                label=f'Valore atteso se uniformi ({conteggio_medio_atteso:.1f})')

    plt.xticks(x_pos, bin_labels, rotation=45, ha='right')
    plt.xlabel(f"Distanza dal bordo più vicino (Fasce di Aree Uguali, {int(area_per_bin)} px² ciascuna)", fontsize=12)
    plt.ylabel("Numero di Oggetti 'NO' Unici", fontsize=12)
    plt.title(
        f"Distribuzione Spaziale degli Oggetti Non Catalogati dal Bordo al Centro\n(Le barre dovrebbero essere piatte se la distribuzione fosse uniforme)",
        fontsize=14)

    plt.bar_label(bars, padding=3)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()

    output_plot_bordo = "distribuzione_bordo_equiareale.png"
    plt.savefig(output_plot_bordo, dpi=300)
    print(f"Grafico 1 (Bordo) salvato in: {output_plot_bordo}")
    plt.show()

    # =============================================================================
    # 3. DISTANZA DALLA STELLA CATALOGATA PIÙ VICINA (GRAFICO 2)
    # =============================================================================
    if lista_distanze_stelle:
        df_distanze = pd.DataFrame(lista_distanze_stelle)

        # raggruppo per ID univoco e calcolo la distanza media in GRADI tra tutti i frame in cui appare
        df_distanze_uniche = df_distanze.groupby(col_id)['distanza_stella_deg'].mean().reset_index()
        distanze_stella_deg = df_distanze_uniche['distanza_stella_deg'].values

        # converto da gradi in arcosecondi per una lettura più immediata
        distanze_stella_arcsec = distanze_stella_deg * 3600

        plt.figure(figsize=(15, 8))

        # genero i bin la cui larghezza cresce in modo esponenziale
        min_dist = max(distanze_stella_arcsec.min(), 1e-4)  # prevengo lo 0 assoluto per geomspace
        max_dist = distanze_stella_arcsec.max()
        bins_esponenziali = np.geomspace(min_dist, max_dist, 51)

        # creo l'istogramma standard per le distanze passando i bin esponenziali
        counts_stella, bins_stella, patches = plt.hist(distanze_stella_arcsec, bins=bins_esponenziali, color='indigo',
                                                       edgecolor='black', alpha=0.8)

        # imposto la scala logaritmica prima dei tick per evitare che vengano sovrascritti dai valori di default
        plt.xscale('log')

        # calcolo il centro di ogni bin in scala logaritmica usando la media geometrica
        centri_bins = np.sqrt(bins_esponenziali[:-1] * bins_esponenziali[1:])

        # imposto esplicitamente i valori calcolati posizionandoli al centro dei bin
        etichette_x = [f"{c:.3f}" for c in centri_bins]
        plt.xticks(centri_bins, etichette_x, rotation=60, ha='right', fontsize=7)

        # rimuovo i tick minori per evitare sovrapposizioni e mantenere puliti i numeri dei bin
        plt.gca().xaxis.set_minor_locator(plt.NullLocator())

        # aggiorno l'unità di misura in arcsec (coerente con * 3600)
        plt.xlabel("Distanza dalla stella catalogata più vicina (arcsec)", fontsize=12)
        plt.ylabel("Numero di Oggetti 'NO' Unici", fontsize=12)
        plt.title("Distribuzione della Distanza tra Oggetti 'NO' e la Stella Catalogata più vicina \n (mag <= 12))", fontsize=14)

        # aggiungo la linea fissa per la soglia di correlazione
        soglia_corr = 0.0033 * 3600
        plt.axvline(soglia_corr, color='orange', linestyle='dashed', linewidth=2,
                    label=f'Soglia correlazione: {soglia_corr:.2f} arcsec', zorder=10)

        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        output_plot_stella = "distribuzione_distanza_stella_vicina.png"
        plt.savefig(output_plot_stella, dpi=300)
        print(f"Grafico 2 (Distanza Stella) salvato in: {output_plot_stella}")
        plt.show()
    else:
        print(
            "Non è stato possibile calcolare le distanze dalle stelle catalogate (cataloghi mancanti o WCS non validi).")

else:
    print("Nessun dato da graficare.")