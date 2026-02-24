import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

            # Prendo tutti gli oggetti, con corrispondenza e non
            df_tot = df_temp.groupby(['xcentroid', 'ycentroid'], as_index=False).first()

            if not df_tot.empty:
                col_id = 'run_unique_id' if 'run_unique_id' in df_tot.columns else 'ID'
                lista_totale_dfs.append(df_tot[[col_id, 'xcentroid', 'ycentroid']])

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
                            coords_no = wcs_img.pixel_to_world(df_tot['xcentroid'].values, df_tot['ycentroid'].values)

                            # calcolo la distanza sferica esatta in gradi tra gli oggetti e le stelle del catalogo
                            idx_cat, d2d, _ = coords_no.match_to_catalog_sky(coords_cat)

                            # salvo la distanza minima in gradi insieme alle coordinate spaziali X, Y per ogni oggetto
                            for i, _id in enumerate(df_tot[col_id].values):
                                lista_distanze_stelle.append({
                                    col_id: _id,
                                    'xcentroid': df_tot['xcentroid'].values[i],
                                    'ycentroid': df_tot['ycentroid'].values[i],
                                    'distanza_stella_deg': d2d[i].deg
                                })

        except Exception:
            continue

print("\n--- ELABORAZIONE DATI SPAZIALI E DISTANZE ---")

# Dimensioni sensore
W, H = 3072, 2048

if lista_totale_dfs:
    # concateno tutto
    df_global = pd.concat(lista_totale_dfs, ignore_index=True)

    print(f"Totale rilevamenti analizzati: {len(df_global)}")

    # =============================================================================
    # 2. CALCOLO DISTANZA DAL BORDO (GRAFICO 1 - MANTENUTO A 15 BIN)
    # =============================================================================

    # calcolo la distanza dal bordo X più vicino (sinistro o destro)
    dist_x = np.minimum(df_global['xcentroid'], W - df_global['xcentroid'])
    # calcolo la distanza dal bordo Y più vicino (superiore o inferiore)
    dist_y = np.minimum(df_global['ycentroid'], H - df_global['ycentroid'])

    # la distanza assoluta dal bordo del sensore è il minimo tra le due
    df_global['dist_edge'] = np.minimum(dist_x, dist_y)
    distanze_bordo = df_global['dist_edge'].values

    # calcolo i bin equi-areali (QUI USO 15 BIN COME RICHIESTO)
    NUM_BINS_G1 = 15
    AREA_TOTALE = W * H
    area_per_bin_g1 = AREA_TOTALE / NUM_BINS_G1

    bin_edges_g1 = [0.0]
    for k in range(1, NUM_BINS_G1 + 1):
        area_cumulata = k * area_per_bin_g1
        x_k = ((W + H) - np.sqrt((W + H) ** 2 - 4 * area_cumulata)) / 4
        bin_edges_g1.append(x_k)

    counts_bordo, _ = np.histogram(distanze_bordo, bins=bin_edges_g1)

    # PLOTTING GRAFICO 1
    plt.figure(figsize=(15, 8))
    bin_labels_g1 = [f"{int(bin_edges_g1[i])}-{int(bin_edges_g1[i + 1])} px" for i in range(NUM_BINS_G1)]
    x_pos_g1 = np.arange(NUM_BINS_G1)

    bars = plt.bar(x_pos_g1, counts_bordo, color='crimson', edgecolor='black', alpha=0.8)

    conteggio_medio_atteso = len(distanze_bordo) / NUM_BINS_G1
    plt.axhline(y=conteggio_medio_atteso, color='green', linestyle='--', linewidth=2,
                label=f'Valore atteso se uniformi ({conteggio_medio_atteso:.1f})')

    plt.xticks(x_pos_g1, bin_labels_g1, rotation=45, ha='right')
    plt.xlabel(f"Distanza dal bordo più vicino (Fasce di Aree Uguali, {int(area_per_bin_g1)} px² ciascuna)",
               fontsize=12)
    plt.ylabel("Numero di Rilevamenti", fontsize=12)
    plt.title(
        f"Distribuzione Spaziale dei Rilevamenti dal Bordo al Centro\n(Le barre dovrebbero essere piatte se la distribuzione fosse uniforme)",
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
    # 3. GRAFICO 2: DISTANZA STELLA SOVRAPPOSTA + MAPPA SENSORE (4 FASCE)
    # =============================================================================
    if lista_distanze_stelle:
        df_distanze = pd.DataFrame(lista_distanze_stelle)

        # --- RICALCOLO FASCE EQUI-AREALI SOLO PER QUESTO GRAFICO (4 BIN) ---
        NUM_BINS_MAP = 4
        area_per_bin_map = AREA_TOTALE / NUM_BINS_MAP
        bin_edges_map = [0.0]
        for k in range(1, NUM_BINS_MAP + 1):
            area_cumulata = k * area_per_bin_map
            x_k = ((W + H) - np.sqrt((W + H) ** 2 - 4 * area_cumulata)) / 4
            bin_edges_map.append(x_k)

        bin_labels_map = [f"Fascia {i + 1} (Bordo {int(bin_edges_map[i])}-{int(bin_edges_map[i + 1])}px)" for i in
                          range(NUM_BINS_MAP)]

        # calcolo la distanza dal bordo per i dati delle stelle
        dist_x_stella = np.minimum(df_distanze['xcentroid'], W - df_distanze['xcentroid'])
        dist_y_stella = np.minimum(df_distanze['ycentroid'], H - df_distanze['ycentroid'])
        df_distanze['dist_edge'] = np.minimum(dist_x_stella, dist_y_stella)

        # categorizzo usando le 4 nuove fasce
        df_distanze['fascia_bordo'] = pd.cut(df_distanze['dist_edge'], bins=bin_edges_map, labels=bin_labels_map,
                                             include_lowest=True)

        # --- PREPARAZIONE LAYOUT AFFIANCATO ---
        # Creo una figura con due subplot affiancati (ax1: istogramma, ax2: mappa)
        # width_ratios dà leggermente più spazio all'istogramma
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.5, 1]})

        # --- SUBPLOT 1 (SINISTRA): ISTOGRAMMA SOVRAPPOSTO ---
        dati_per_fascia = []
        etichette_valide = []

        for label in bin_labels_map:
            subset = df_distanze[df_distanze['fascia_bordo'] == label]['distanza_stella_deg'].values * 3600
            if len(subset) > 0:
                dati_per_fascia.append(subset)
                etichette_valide.append(label)

        # genero i colori usando 'turbo' per le 4 fasce
        cmap = plt.get_cmap('turbo')
        colori = [cmap(i) for i in np.linspace(0, 1, NUM_BINS_MAP)]
        # Se alcune fasce sono vuote, filtro i colori per mantenere la corrispondenza
        colori_validi = [colori[i] for i, label in enumerate(bin_labels_map) if label in etichette_valide]

        # MODIFICA: ho rinominato 'patches' in '_patches' per evitare conflitti con il modulo matplotlib.patches
        counts_stella, bins_stella, _patches = ax1.hist(
            dati_per_fascia, bins=1000, color=colori_validi, histtype='stepfilled',
            linewidth=1.5, alpha=0.5, stacked=False, label=etichette_valide
        )

        centri_bins = (bins_stella[:-1] + bins_stella[1:]) / 2
        etichette_x = [f"{c:.3f}" for c in centri_bins]
        ax1.set_xticks(centri_bins)
        ax1.set_xticklabels(etichette_x, rotation=60, ha='right', fontsize=7)
        ax1.xaxis.set_minor_locator(plt.NullLocator())

        ax1.set_xlabel("Distanza dalla stella catalogata più vicina (arcsec)", fontsize=12)
        ax1.set_ylabel("Numero di Rilevamenti", fontsize=12)
        ax1.set_title("Distribuzione Distanza vs Stella Catalogata (mag <= 12)\nDiviso per 4 Fasce di Bordo",
                      fontsize=14)

        soglia_corr = 0.0033 * 3600
        ax1.axvline(soglia_corr, color='orange', linestyle='dashed', linewidth=2,
                    label=f'Soglia correlazione: {soglia_corr:.2f} arcsec', zorder=10)
        ax1.set_xlim(0, 22)
        ax1.grid(axis='y', linestyle=':', alpha=0.7)
        ax1.legend(title='Fasce (dal Bordo al Centro)', loc='upper right')

        # --- SUBPLOT 2 (DESTRA): MAPPA SENSORE COLORATA ---
        ax2.set_title(f"Mappa Regioni Sensore ({W}x{H})\nColori corrispondenti all'istogramma", fontsize=14)
        ax2.set_xlim(0, W)
        ax2.set_ylim(0, H)
        ax2.set_aspect('equal')  # mantiene le proporzioni corrette del sensore
        ax2.invert_yaxis()  # inverte l'asse Y per far corrispondere (0,0) all'angolo in alto a sinistra tipico delle immagini

        # Disegno rettangoli concentrici dal più esterno al più interno
        # Il rettangolo i-esimo copre l'area dove la distanza dal bordo è > bin_edges_map[i]
        # Usiamo il colore i-esimo per riempirlo. Sovrapponendoli si crea l'effetto fascia.
        for i in range(NUM_BINS_MAP):
            edge_dist = bin_edges_map[i]
            # coordinate angolo in basso a sinistra (o alto a sx dopo inversione)
            x0, y0 = edge_dist, edge_dist
            # larghezza e altezza del rettangolo interno
            width = W - 2 * edge_dist
            height = H - 2 * edge_dist

            # creo il rettangolo con il colore corrispondente alla fascia i
            rect = patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor='none', facecolor=colori[i],
                                     alpha=0.6)
            ax2.add_patch(rect)

            # aggiungo un'etichetta di testo al centro della fascia (approssimativo)
            if i < NUM_BINS_MAP - 1:
                next_edge = bin_edges_map[i + 1]
                mid_edge = (edge_dist + next_edge) / 2
                ax2.text(mid_edge, H / 2, f"F{i + 1}", color='white', ha='center', va='center', fontweight='bold')
            else:
                # centro assoluto per l'ultima fascia
                ax2.text(W / 2, H / 2, f"F{NUM_BINS_MAP}\n(Centro)", color='white', ha='center', va='center',
                         fontweight='bold')

        ax2.set_xlabel("Pixel X")
        ax2.set_ylabel("Pixel Y")
        ax2.grid(False)  # la griglia confonde sulla mappa

        plt.tight_layout()
        output_plot_stella_map = "distribuzione_distanza_stella_con_mappa.png"
        plt.savefig(output_plot_stella_map, dpi=300)
        print(f"Grafico 2 (Distanza Stella + Mappa) salvato in: {output_plot_stella_map}")
        plt.show()
    else:
        print(
            "Non è stato possibile calcolare le distanze dalle stelle catalogate (cataloghi mancanti o WCS non validi).")

else:
    print("Nessun dato da graficare.")