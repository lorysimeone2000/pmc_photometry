import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import simple_norm
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from pathlib import Path
import astropy.units as u
from astropy.table import Table
from tqdm import tqdm

# importo i moduli per l'image segmentation
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel, SourceFinder
from astropy.stats import sigma_clipped_stats
import warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)


# =============================================================================
# FUNZIONI DI UTILITÀ E GESTIONE PERCORSI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # risalgo l'albero delle directory per trovare la cartella principale del progetto
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent


BASE_DIR = trova_cartella_base("pmc_photometry")

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # cerco un file specifico in tutte le sottocartelle
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # cerco una cartella specifica in tutte le sottocartelle
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


def estrai_header_fits_da_csv(percorso_csv):
    # leggo le prime righe commentate del CSV per trovare il riferimento al FITS originale
    info = {}
    with open(percorso_csv, 'r') as f:
        for riga in f:
            if riga.startswith('#'):
                if ':' in riga:
                    parti = riga.strip('# \n').split(':', 1)
                    info[parti[0].strip()] = parti[1].strip()
            else:
                break
    return info


def trova_flusso_kron(riga_df):
    # cerco la colonna del flusso di Kron
    colonne_possibili = ['flusso_fisso_max_run', 'kron_manuale_aper', 'kron_flux', 'kron_manuale_seg']
    for col in colonne_possibili:
        if col in riga_df.index and pd.notna(riga_df[col]):
            try:
                return float(riga_df[col])
            except ValueError:
                continue
    return 0.0


def leggi_file_parametri(percorso):
    # leggo il file dei parametri per la segmentazione
    parametri = {}
    if not percorso or not os.path.exists(percorso): return {}
    with open(percorso, 'r') as file:
        next(file, None)
        for riga in file:
            riga = riga.split('#')[0].strip()
            if riga:
                parts = riga.split()
                if len(parts) >= 2:
                    try:
                        valore = float(parts[1]) if '.' in parts[1] else int(parts[1])
                        parametri[parts[0]] = valore
                    except ValueError:
                        pass
    return parametri


# =============================================================================
# LOGICA GRAFICA AVANZATA (DAL TUO SALVA_CUTOUT)
# =============================================================================

def genera_cutout_avanzato(x_centr, y_centr, data_img, wcs, img_idx, run_id, star_id, flusso, cartella_out,
                           parametri_seg, base_dir_progetto, df_img):
    # 1. Sottrazione del background dall'intera immagine prima del ritaglio
    _, median_bg, _ = sigma_clipped_stats(data_img, sigma=3.0)
    data_sub = data_img - median_bg

    # 2. Calcolo dimensioni ritaglio (box basato sui 35 arcsec)
    pixel_scale_deg = np.mean(proj_plane_pixel_scales(wcs))
    r_corr_px = 35 / 3600 / pixel_scale_deg

    side = 2 * r_corr_px
    half_side = side * 1.25 / 2.0

    ny, nx = data_sub.shape
    x_min = max(0, int(np.floor(x_centr - half_side)))
    x_max = min(nx, int(np.ceil(x_centr + half_side)))
    y_min = max(0, int(np.floor(y_centr - half_side)))
    y_max = min(ny, int(np.ceil(y_centr + half_side)))

    cutout = data_sub[y_min:y_max, x_min:x_max]
    if cutout.size == 0: return

    # 3. Setup Grafico
    fig, ax = plt.subplots(figsize=(7, 7))
    legend_elements = []

    norm = simple_norm(cutout, 'log', percent=99.9)
    img_plot = ax.imshow(cutout, cmap='gray_r', origin='lower', extent=[x_min, x_max, y_min, y_max], norm=norm)

    # Colorbar Intensità Pixel
    cbar_img = fig.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
    cbar_img.set_label('Intensità Pixel (ADU)')

    # 4. Applico Image Segmentation per tracciare i contorni
    fwhm = parametri_seg.get('fwhm', 3.0)
    size_kernel = int(parametri_seg.get('size', 5))
    threshold = parametri_seg.get('threshold_assoluta', 3.61)
    pixel_n = int(parametri_seg.get('pixel', 3))

    try:
        kernel = make_2dgaussian_kernel(fwhm, size=size_kernel)
        convolved_cutout = convolve(cutout, kernel)
        finder = SourceFinder(npixels=pixel_n, progress_bar=False)
        segment_map = finder(convolved_cutout, threshold)

        if segment_map is not None:
            ax.contour(segment_map.data > 0, levels=[0.5], colors='#00ff00', alpha=0.5, linewidths=1.5,
                       extent=[x_min, x_max, y_min, y_max], origin='lower', zorder=8)
            legend_elements.append(Patch(facecolor='none', edgecolor='#00ff00', alpha=0.5,
                                         label='Regione della segmentazione', linewidth=1.5))
    except Exception:
        pass

    # 5. Recupero il catalogo per questa specifica immagine
    nome_cartella_cat = f"prove_2/tabelle/sorgenti_catalogate_run/sorgenti_catalogate_run_{run_id}"
    cartella_cat = cerca_cartella_nel_progetto(base_dir_progetto, nome_cartella_cat)

    if cartella_cat:
        # Costruisco il nome considerando il formato tipico delle tue tabelle
        nome_file_cat = f"run_{run_id}_stelle_catalogate_immagine_{int(img_idx):03d}.csv"
        path_file_cat = cerca_file_nel_progetto(cartella_cat, nome_file_cat)

        if path_file_cat:
            try:
                df_cat_full = pd.read_csv(path_file_cat, comment='#')
                tbl_cat_full = Table.from_pandas(df_cat_full)

                if 'xcentroid' not in tbl_cat_full.colnames or 'ycentroid' not in tbl_cat_full.colnames:
                    coords_cat_sky = u.Quantity([tbl_cat_full['RAJ2000'], tbl_cat_full['DEJ2000']], unit=u.deg)
                    x_pix, y_pix = wcs.world_to_pixel_values(coords_cat_sky[0], coords_cat_sky[1])
                    tbl_cat_full['xcentroid'] = x_pix
                    tbl_cat_full['ycentroid'] = y_pix

                mask_in_box = (tbl_cat_full['xcentroid'] >= x_min) & (tbl_cat_full['xcentroid'] <= x_max) & \
                              (tbl_cat_full['ycentroid'] >= y_min) & (tbl_cat_full['ycentroid'] <= y_max)

                tbl_cat_box = tbl_cat_full[mask_in_box]

                if len(tbl_cat_box) > 0:
                    min_mag = min(np.nanmin(tbl_cat_box['Mag']), 5)
                    max_mag = 15

                    ax.scatter(tbl_cat_box['xcentroid'], tbl_cat_box['ycentroid'], c=tbl_cat_box['Mag'],
                               cmap='viridis_r', vmin=min_mag, vmax=max_mag, s=4, zorder=5)

                    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Stelle catalogate',
                                                  markerfacecolor='gray', markersize=5))

                    sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(vmin=min_mag, vmax=max_mag))
                    sm.set_array([])
                    cbar_mag = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.08)
                    cbar_mag.set_label('Mag Catalogo')
            except Exception:
                pass

    # 6. Disegno il centro e il cerchio di correlazione per l'oggetto target principale
    circle = Circle((x_centr, y_centr), r_corr_px, edgecolor='#ffff00', facecolor='none', linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    legend_elements.append(Patch(facecolor='none', edgecolor='#ffff00', label='Raggio tolleranza (35")', linewidth=1.5))

    # Disegno tutti i centroidi con Corrispondenza == 'NO' presenti in questa immagine
    # OTTIMIZZAZIONE: Filtro a monte solo quelli che cadono fisicamente dentro il riquadro
    mask_no = df_img['Corrispondenza'].astype(str).str.contains('NO')
    mask_spaziale = (
            (df_img['xcentroid'] >= x_min) & (df_img['xcentroid'] <= x_max) &
            (df_img['ycentroid'] >= y_min) & (df_img['ycentroid'] <= y_max)
    )

    df_no_filtrato = df_img[mask_no & mask_spaziale]

    for _, row in df_no_filtrato.iterrows():
        cx_no = row['xcentroid']
        cy_no = row['ycentroid']
        ax.plot(cx_no, cy_no, marker='+', color='red', markersize=15, markeredgewidth=2, zorder=15)

    legend_elements.append(Line2D([0], [0], marker='+', color='red', label='Centroidi (NO match)',
                                  markersize=10, linestyle='None', markeredgewidth=2))

    # 7. Formattazione titolo (Con Immagine, Run e ID)
    titolo = f"Run {run_id} - Immagine {img_idx}\nOggetto: {star_id} | Kron: {flusso:.2f}"
    ax.set_title(titolo, fontsize=11, fontweight='bold')

    # Reimposto i limiti stretti per tagliare fuori i centroidi rossi disegnati lontano
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.8)

    # 8. Salvataggio
    nome_png = f"{star_id}_run_{run_id}_img_{int(img_idx):03d}.png"
    percorso_salvataggio = cartella_out / nome_png
    plt.tight_layout()
    plt.savefig(percorso_salvataggio, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":

    RUNS = [1, 2, 3]
    BOX_SIZE_ARCSEC = 35.0

    # creo la cartella di output
    cartella_output = Path("./riquadri_di_tutti_gli_INT")
    cartella_output.mkdir(exist_ok=True)

    # carico i parametri di segmentazione
    file_parametri = cerca_file_nel_progetto(BASE_DIR, 'parametri_image_segmentation.txt')
    parametri_seg = leggi_file_parametri(str(file_parametri)) if file_parametri else {}

    print("--- FASE 1: Raccolta dati di tutti gli oggetti non catalogati ---")
    dati_tutti_oggetti = []

    for run_target in RUNS:
        cartella_csv = BASE_DIR / "prove_2" / "tabelle" / "tabelle_unite" / f"tabelle_unite_run_{run_target}"
        if not cartella_csv.exists():
            continue

        file_csv_list = sorted(list(cartella_csv.glob('*.csv')))

        # itero su tutti i file CSV per mappare le posizioni di tutti i NO
        for p_csv in tqdm(file_csv_list, desc=f"Scansione CSV Run {run_target}"):
            try:
                df = pd.read_csv(p_csv, comment='#')
            except Exception:
                continue

            if 'ID' not in df.columns or 'Corrispondenza' not in df.columns:
                continue

            # Isolo solo le righe con Corrispondenza "NO"
            df_no = df[df['Corrispondenza'].astype(str).str.contains('NO')].copy()
            if df_no.empty:
                continue

            # Leggo l'header solo se ci sono oggetti anomali per risparmiare tempo
            header_info = estrai_header_fits_da_csv(p_csv)
            nome_fits = header_info.get('NOME_FILE_FITS')
            percorso_fits_letto = header_info.get('PERCORSO_FILE')

            fallback_idx = 0
            if 'immagine_' in p_csv.name:
                try:
                    fallback_idx = int(p_csv.name.split('immagine_')[1].split('.csv')[0])
                except:
                    pass

            # Salvo ogni apparizione di ogni oggetto
            for _, row in df_no.iterrows():
                img_index = row.get('img_index', fallback_idx)
                if pd.isna(img_index) or img_index == 0:
                    img_index = fallback_idx

                dati_tutti_oggetti.append({
                    'ID': row['ID'],
                    'run_target': run_target,
                    'img_index': img_index,
                    'xcentroid': row['xcentroid'],
                    'ycentroid': row['ycentroid'],
                    'flusso_kron': trova_flusso_kron(row),
                    'p_csv': p_csv,
                    'nome_fits': nome_fits,
                    'percorso_fits_letto': percorso_fits_letto
                })

    df_totale = pd.DataFrame(dati_tutti_oggetti)
    if df_totale.empty:
        print("Nessun oggetto non catalogato trovato nelle run specificate.")
        sys.exit()

    print(f"\nTrovati {df_totale['ID'].nunique()} oggetti non catalogati unici in totale.")
    print("Calcolo dell'immagine di riferimento (punto medio) per ogni oggetto...")

    oggetti_da_processare = []
    gruppi = df_totale.groupby('ID')

    for star_id, group in gruppi:
        # Ordino temporalmente (per run e poi per frame)
        group = group.sort_values(by=['run_target', 'img_index'])
        # Prendo il record esattamente a metà del ciclo di vita dell'oggetto
        mid_idx = len(group) // 2
        oggetti_da_processare.append(group.iloc[mid_idx])

    df_da_processare = pd.DataFrame(oggetti_da_processare)

    print(f"\n--- FASE 2: Generazione di {len(df_da_processare)} riquadri ---")

    # Ottimizzazione cache: Salvo in memoria i percorsi FITS già trovati per non rispazzolare l'hard disk
    cache_fits_paths = {}

    for _, riga in tqdm(df_da_processare.iterrows(), total=len(df_da_processare), desc="Creazione Ritagli"):
        star_id = riga['ID']
        run_target = riga['run_target']
        img_index = riga['img_index']
        x_centr = riga['xcentroid']
        y_centr = riga['ycentroid']
        flusso_kron = riga['flusso_kron']
        p_csv = riga['p_csv']
        nome_fits = riga['nome_fits']
        percorso_fits_letto = riga['percorso_fits_letto']

        percorso_fits_reale = None

        if nome_fits in cache_fits_paths:
            percorso_fits_reale = cache_fits_paths[nome_fits]
        else:
            if percorso_fits_letto and os.path.exists(percorso_fits_letto):
                percorso_fits_reale = Path(percorso_fits_letto)
            elif nome_fits:
                percorso_fits_reale = cerca_file_nel_progetto(BASE_DIR, nome_fits)

            cache_fits_paths[nome_fits] = percorso_fits_reale

        if not percorso_fits_reale:
            continue

        # Ricostruisco il DataFrame completo dell'immagine per poter disegnare tutte le crocette rosse
        try:
            df_img = pd.read_csv(p_csv, comment='#')
        except Exception:
            continue

        try:
            with fits.open(percorso_fits_reale, memmap=False) as hdu:
                dati_img = hdu[0].data
                wcs = WCS(hdu[0].header)

                genera_cutout_avanzato(
                    x_centr=x_centr,
                    y_centr=y_centr,
                    data_img=dati_img,
                    wcs=wcs,
                    img_idx=img_index,
                    run_id=run_target,
                    star_id=star_id,
                    flusso=flusso_kron,
                    cartella_out=cartella_output,
                    parametri_seg=parametri_seg,
                    base_dir_progetto=BASE_DIR,
                    df_img=df_img
                )
        except Exception as e:
            # Ignora silenziosamente errori per singole immagini corrotte e passa alla successiva
            pass

    print(f"\nElaborazione totale completata. Controlla la cartella '{cartella_output}'.")