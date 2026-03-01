import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import simple_norm
from pathlib import Path


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


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # cerco un file specifico in tutte le sottocartelle
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


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
    # cerco la colonna del flusso di Kron tra quelle comunemente generate dallo script principale
    colonne_possibili = ['kron_manuale_aper', 'kron_flux', 'kron_manuale_seg']
    for col in colonne_possibili:
        if col in riga_df.index and pd.notna(riga_df[col]):
            return float(riga_df[col])
    return 0.0


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":

    TARGET_ID = "INT_81"
    RUN_TARGET = 1
    BOX_SIZE_ARCSEC = 35.0

    # creo la cartella di output nella stessa cartella in cui eseguo lo script
    cartella_output = Path("./riquadri_INT_81")
    cartella_output.mkdir(exist_ok=True)

    # imposto la cartella base del progetto
    BASE_DIR = trova_cartella_base("pmc_photometry")

    # individuo la cartella contenente i CSV uniti della run 1
    cartella_csv = BASE_DIR / "tabelle" / "tabelle_unite" / f"tabelle_unite_run_{RUN_TARGET}"
    if not cartella_csv.exists():
        print(f"ERRORE: La cartella {cartella_csv} non esiste.")
        exit()

    file_csv_list = sorted(list(cartella_csv.glob('*.csv')))
    print(f"Trovati {len(file_csv_list)} file CSV per la Run {RUN_TARGET}. Inizio l'estrazione...")

    oggetti_trovati = 0

    # itero su tutti i file CSV
    for p_csv in file_csv_list:
        try:
            df = pd.read_csv(p_csv, comment='#')
        except Exception as e:
            print(f"Errore nella lettura di {p_csv.name}: {e}")
            continue

        # verifico se l'oggetto target è presente nel dataframe corrente
        if 'ID' not in df.columns:
            continue

        riga_target = df[df['ID'] == TARGET_ID]
        if riga_target.empty:
            continue

        oggetti_trovati += 1
        riga = riga_target.iloc[0]

        # estraggo i metadati richiesti
        img_index = riga.get('img_index', 'N/A')
        ra_centr = riga.get('RA_centroid', 0.0)
        dec_centr = riga.get('DEC_centroid', 0.0)
        x_centr = riga.get('xcentroid', 0.0)
        y_centr = riga.get('ycentroid', 0.0)
        flusso_kron = trova_flusso_kron(riga)

        # recupero il file FITS originale leggendo l'header del CSV
        header_info = estrai_header_fits_da_csv(p_csv)
        nome_fits = header_info.get('NOME_FILE_FITS')
        percorso_fits_letto = header_info.get('PERCORSO_FILE')

        percorso_fits_reale = None

        # provo a usare il percorso assoluto se esiste
        if percorso_fits_letto and os.path.exists(percorso_fits_letto):
            percorso_fits_reale = Path(percorso_fits_letto)
        # altrimenti cerco il file per nome all'interno del progetto
        elif nome_fits:
            percorso_fits_reale = cerca_file_nel_progetto(BASE_DIR, nome_fits)

        if not percorso_fits_reale:
            print(f"ATTENZIONE: File FITS non trovato per l'immagine {img_index} (CSV: {p_csv.name}). Salto.")
            continue

        # apro il file FITS per estrarre l'immagine e calcolare il ritaglio
        try:
            with fits.open(percorso_fits_reale, memmap=False) as hdu:
                dati_img = hdu[0].data
                header_fits = hdu[0].header
                wcs = WCS(header_fits)

                # calcolo la scala in gradi per pixel
                pixel_scale_deg = np.mean(proj_plane_pixel_scales(wcs))

                # converto i 35 arcosecondi in pixel
                box_deg = BOX_SIZE_ARCSEC / 3600.0
                box_px = int(np.ceil(box_deg / pixel_scale_deg))
                half_box = box_px // 2

                # definisco i confini del ritaglio, assicurandomi di non uscire dall'immagine
                ny, nx = dati_img.shape
                x_min = max(0, int(np.floor(x_centr - half_box)))
                x_max = min(nx, int(np.ceil(x_centr + half_box)))
                y_min = max(0, int(np.floor(y_centr - half_box)))
                y_max = min(ny, int(np.ceil(y_centr + half_box)))

                ritaglio = dati_img[y_min:y_max, x_min:x_max]

                # creo e configuro il grafico
                fig, ax = plt.subplots(figsize=(6, 6))
                norma = simple_norm(ritaglio, 'log', percent=99.5)

                ax.imshow(ritaglio, cmap='gray', origin='lower', extent=[x_min, x_max, y_min, y_max], norm=norma)

                # disegno una croce rossa al centro esatto delle coordinate in pixel
                ax.plot(x_centr, y_centr, marker='+', color='red', markersize=15, markeredgewidth=1.5)

                # formatto il titolo con tutti i parametri richiesti
                titolo = (f"Img: {img_index} | ID: {TARGET_ID}\n"
                          f"RA: {ra_centr:.5f} | DEC: {dec_centr:.5f}\n"
                          f"Flusso Kron: {flusso_kron:.2f}")
                ax.set_title(titolo, fontsize=10)
                ax.set_xlabel("Pixel X")
                ax.set_ylabel("Pixel Y")

                # genero il nome del file png e lo salvo
                nome_png = f"img_{img_index}_RA_{ra_centr:.4f}_DEC_{dec_centr:.4f}_Kron_{flusso_kron:.1f}.png"
                percorso_salvataggio = cartella_output / nome_png

                plt.tight_layout()
                plt.savefig(percorso_salvataggio, dpi=150)
                plt.close(fig)

        except Exception as e:
            print(f"Errore nell'elaborazione del FITS {percorso_fits_reale.name}: {e}")

    print(f"\nElaborazione completata. Generati {oggetti_trovati} riquadri nella cartella '{cartella_output}'.")