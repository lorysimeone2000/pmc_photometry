import pandas as pd
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture
import numpy as np
import os
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
import warnings
from astropy.wcs import FITSFixedWarning

# Ignora i warning specifici sui fix automatici degli header FITS
warnings.simplefilter('ignore', category=FITSFixedWarning)


# --- 1. FUNZIONI HELPER ---

def leggi_file_parametri(percorso):
    """Legge il file dei parametri in un dizionario."""
    parametri = {}
    if not os.path.exists(percorso):
        return {}
    with open(percorso, 'r') as file:
        next(file, None)
        for riga in file:
            riga = riga.split('#')[0].strip()
            if riga:
                parts = riga.split()
                if len(parts) >= 2:
                    parametro = parts[0]
                    valore_str = parts[1]
                    try:
                        valore = float(valore_str) if '.' in valore_str else int(valore_str)
                        parametri[parametro] = valore
                    except ValueError:
                        pass
    return parametri


def elabora_file_fits(percorso_file_):
    """Carica il FITS e sottrae il fondo."""
    # memmap=False previene errori con BZERO/BSCALE
    with fits.open(percorso_file_, memmap=False) as hdu_list_:
        image_data_ = hdu_list_[0].data
        w_ = WCS(hdu_list_[0].header)
        mean_, median_, std_ = sigma_clipped_stats(image_data_, sigma=3.0)
        image_data_ = image_data_ - median_
        return image_data_, median_, w_


def calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pixel, k=2.5, r_min=3.5):
    """
    Esegue l'INTERA pipeline Kron per una singola stella:
    1. Calcola il primo momento dai pixel forniti.
    2. Determina il raggio di Kron.
    3. Esegue la fotometria di apertura sull'immagine originale.

    Returns:
    - float: Il flusso finale integrato (Kron Flux).
    """
    # 1. Calcolo del Raggio
    somma_intensita = np.sum(valori_pixel)

    if somma_intensita <= 0:
        return np.nan  # Impossibile calcolare raggio su flusso nullo/negativo

    somma_momenti = np.sum(valori_pixel * distanze_pixel)
    r_1 = somma_momenti / somma_intensita

    # Raggio finale con soglia minima
    r_kron_finale = max(k * r_1, r_min)

    # 2. Misura Fotometrica (Integrazione)
    # Creiamo l'apertura circolare con il raggio calcolato
    aper = CircularAperture((xc, yc), r=r_kron_finale)

    # Eseguiamo la fotometria sull'immagine completa 'data'
    phot = aperture_photometry(data, aper)

    return phot['aperture_sum'][0], r_kron_finale  # Ritorna (flusso, raggio)


def esegui_fotometria_variabile(data, positions, raggi):
    """
    Helper generico per altri tipi di flusso (es. Raggio Max)
    che richiedono raggi variabili.
    """
    flussi = []
    for (xc, yc), r in zip(positions, raggi):
        if r > 0 and not np.isnan(r):
            aper = CircularAperture((xc, yc), r=r)
            phot = aperture_photometry(data, aper)
            flussi.append(phot['aperture_sum'][0])
        else:
            flussi.append(np.nan)
    return flussi


def salva_csv_con_header_fits(dataframe, header_fits, filename, nome_file_fits, parametri_seg=None):
    """Salva CSV con header FITS come commenti."""
    with open(filename, 'w') as f:
        f.write("# Header FITS:\n")
        for key, value in header_fits.items():
            clean_val = str(value).replace('\n', ' ')
            f.write(f"# {key}: {clean_val}\n")
        f.write(f"# PERCORSO_FILE: {nome_file_fits}\n")
        f.write("#\n# PARAMETRI SEGMENTAZIONE:\n")
        if parametri_seg:
            for key, value in parametri_seg.items():
                f.write(f"# {key}: {value}\n")
        f.write("#\n")
        dataframe.to_csv(f, index=False)


# --- 2. FUNZIONE DI ANALISI PRINCIPALE ---

def analisi_image_segmentation(percorso_file_, parametri_globali):
    """
    Esegue image segmentation su un'immagine FITS e restituisce la tabella filtrata.
    """
    # A. Setup
    data, fondo_iniziale, w = elabora_file_fits(percorso_file_)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # B. Parametri
    fwhm = parametri_globali.get('fwhm', 3.0)
    size = parametri_globali.get('size', 5)
    threshold = parametri_globali.get('threshold_assoluta', 3.0)
    pixel_n = parametri_globali.get('pixel', 5)

    # C. Segmentazione
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    finder = SourceFinder(npixels=pixel_n, progress_bar=False)
    segment_map = finder(convolved_data, threshold)

    # D. Catalogo Base
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    if len(tbl) == 0:
        return tbl, parametri_globali

    for col in ['xcentroid', 'ycentroid', 'kron_flux']:
        tbl[col].info.format = '.2f'

    livello_saturazione = 255 - fondo_iniziale - median
    tbl['saturazione'] = np.where(tbl['max_value'] >= livello_saturazione, 'SI', 'NO')

    # E. CALCOLO E FILTRAGGIO (Ciclo Unico)
    K_KRON = 2.5
    R_MIN_KRON = 3.5
    soglia_assoluta = 2.5
    soglia_relativa = 0.05
    bordo = 7
    ny, nx = data.shape

    lista_raggi_max = []  # Serve ancora per l'altro flusso
    kron_manuale_seg = []  # qui salverò i kron delle segmentazioni
    kron_manuale_aper = [] # qui salverò i kron delle aperture
    raggi_kron_aper = []
    mask_keep = []

    # Iteriamo sulle proprietà
    for prop in cat:
        xc, yc = prop.xcentroid, prop.ycentroid # coordinate pixel del centroide

        # 1. Check Bordo
        dentro_riquadro = (xc >= bordo) and (xc < nx - bordo) and \
                          (yc >= bordo) and (yc < ny - bordo)

        if not dentro_riquadro:
            lista_raggi_max.append(0.5)
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan)
            raggi_kron_aper.append(np.nan)
            mask_keep.append(False)
            continue

        # 2. Recupero Pixel (Ottimizzato con Slice)
        slices = prop.slices # prendo il rettangolo minimo per velocizzare
        cutout_seg = segment_map.data[slices]
        y_loc, x_loc = np.where(cutout_seg == prop.label) # seleziono SOLO i pixel che appartengono alla segmentazione

        ypix = y_loc + slices[0].start
        xpix = x_loc + slices[1].start

        valori_pixel = data[ypix, xpix]

        if len(valori_pixel) == 0:
            lista_raggi_max.append(0.5)
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan)
            raggi_kron_aper.append(np.nan)
            mask_keep.append(False)
            continue

        # 3. Calcoli Geometrici di Base
        distanze_pix = np.hypot(xpix - xc, ypix - yc)

        # Raggio Massimo (per "somma apertura ultimo pixel")
        r_max_pix = np.max(distanze_pix) if len(distanze_pix) > 0 else 0.5
        r_max_pix = max(r_max_pix, 0.5)
        lista_raggi_max.append(r_max_pix)

        r_int = int(np.ceil(r_max_pix))  # Arrotondiamo per eccesso per il ritaglio array
        y_min_box = int(max(0, yc - r_int))
        y_max_box = int(min(data.shape[0], yc + r_int + 1))
        x_min_box = int(max(0, xc - r_int))
        x_max_box = int(min(data.shape[1], xc + r_int + 1))
        cutout_box = data[y_min_box:y_max_box, x_min_box:x_max_box]

        # Creiamo una griglia di coordinate per il ritaglio
        y_grid, x_grid = np.ogrid[y_min_box:y_max_box, x_min_box:x_max_box]

        # Calcoliamo la distanza di ogni pixel del ritaglio dal centroide reale
        distanze_box = np.hypot(x_grid - xc, y_grid - yc)

        # Selezioniamo solo i pixel dentro il cerchio
        mask_circle = distanze_box <= r_max_pix

        # Questo array contiene i valori di TUTTI i pixel dentro il cerchio (non solo la segmentazione)
        pixels_apertura_completa = cutout_box[mask_circle]
        distanze_apertura_completa = distanze_box[mask_circle]  # Le distanze corrispondenti (distanze_pixel)

        # calcolo flusso kron sull'apertura massima
        flusso_kron_apertura, raggio_usato = calcola_flusso_kron_completo(
            data=data,
            xc=xc,
            yc=yc,
            valori_pixel=pixels_apertura_completa,
            distanze_pixel=distanze_apertura_completa,
            k=K_KRON,
            r_min=R_MIN_KRON
        )
        kron_manuale_aper.append(flusso_kron_apertura)
        raggi_kron_aper.append(raggio_usato)

        # --- FLUSSO KRON COMPLETO (Helper Function) ---
        # Qui calcoliamo TUTTO: raggio e flusso finale
        flusso_kron_seg, raggio_valore = calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pix, k=K_KRON, r_min=R_MIN_KRON)
        kron_manuale_seg.append(flusso_kron_seg)
        # ----------------------------------------------

        # 4. Check Soglie (Filtraggio Qualità)
        pixel_sopra_soglia_assoluta = np.sum(valori_pixel > soglia_assoluta)
        pixel_sopra_soglia_relativa = np.sum(valori_pixel > soglia_relativa * prop.max_value)

        is_good = (pixel_sopra_soglia_assoluta >= 3) and (pixel_sopra_soglia_relativa >= 2)
        mask_keep.append(is_good)

    # F. ASSEGNAZIONE E CALCOLO RIMANENTE

    # 1. Assegnazione Kron (Già calcolato nel ciclo!)
    tbl['kron_manuale_seg'] = kron_manuale_seg
    tbl['kron_manuale_aper'] = kron_manuale_aper
    tbl['raggio_kron_aper'] = raggi_kron_aper

    # 2. Calcolo "Somma Apertura Ultimo Pixel" (Ancora da fare massivamente)
    positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))
    tbl['somma_apertura_ultimo_pixel'] = esegui_fotometria_variabile(data, positions, lista_raggi_max)

    # Formattazione
    tbl['somma_apertura_ultimo_pixel'].info.format = '%.2f'
    tbl['kron_manuale_seg'].info.format = '%.2f'
    tbl['kron_manuale_aper'].info.format = '%.2f'
    tbl['raggio_kron_aper'].info.format = '%.2f'

    # G. FILTRAGGIO FINALE
    tbl_filtrato = tbl[mask_keep]

    if len(tbl_filtrato) > 0:
        tbl_filtrato['label'] = np.arange(1, len(tbl_filtrato) + 1)

    return tbl_filtrato, parametri_globali


# --- 3. BLOCCO DI ESECUZIONE (MAIN) ---

if __name__ == "__main__":
    file_parametri = '/home/lorysimeone/tesi_magistrale/prove_2/parametri_image_segmentation.txt'
    parametri_caricati = leggi_file_parametri(file_parametri)

    try:
        run = int(input("Quale run vuoi elaborare: "))
    except ValueError:
        print("Input non valido.")
        exit()

    with open(f'/home/lorysimeone/tesi_magistrale/prove_2/liste_percorsi_run/lista_immagini_run_{run}.txt',
              'r') as file:
        file_list = file.read().splitlines()

    output_dir = f"/home/lorysimeone/tesi_magistrale/prove_2/tabelle/sorgenti_trovate_run/sorgenti_trovate_run_{run}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    numero_stelle = []
    n = 0

    for percorso_file in tqdm(file_list, desc="Elaborazione Files"):
        n += 1

        # Analisi
        tbl, _ = analisi_image_segmentation(percorso_file, parametri_caricati)
        numero_stelle.append(len(tbl))

        # Header e Salvataggio
        header_fits = fits.getheader(percorso_file)

        # Selezione colonne dinamica
        all_cols = tbl.colnames
        cols_base = ['label', 'xcentroid', 'ycentroid', 'area', 'max_value']
        cols_finali = []
        # 1. Aggiungi le colonne base (se presenti)
        cols_finali.extend([c for c in cols_base if c in all_cols])
        # 2. Aggiungi 'saturazione'
        if 'saturazione' in all_cols:
            cols_finali.append('saturazione')
        # 3. Aggiungi 'kron_flux' (SUBITO DOPO SATURAZIONE)
        if 'kron_flux' in all_cols:
            cols_finali.append('kron_flux')
        # 4. Aggiungi tutte le altre colonne dinamiche (per es. altri flussi calcolati)
        # Cerchiamo di prendere tutto ciò che segue 'saturazione' nel file originale
        try:
            if 'saturazione' in all_cols:
                idx_start = all_cols.index('saturazione') + 1
                cols_extra = all_cols[idx_start:]
            else:
                # Se manca saturazione, controlliamo tutto tranne le base
                cols_extra = [c for c in all_cols if c not in cols_base]
            for c in cols_extra:
                # Aggiungiamo solo se non è già stata inserita (evita duplicati di kron_flux)
                if c not in cols_finali:
                    cols_finali.append(c)
        except ValueError:
            pass
        tbl_save = tbl[cols_finali]

        dataframe = tbl_save.to_pandas()
        filename_out = os.path.join(output_dir, f'run_{run}_stelle_trovate_immagine_{n:03d}.csv')
        salva_csv_con_header_fits(dataframe, header_fits, filename_out, percorso_file, parametri_seg=parametri_caricati)

    # Plot
    print(f"\nTotale stelle per immagine: {numero_stelle}")
    array_finale = np.array(numero_stelle)

    plt.plot(array_finale, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel('Indice')
    plt.ylabel('Numero stelle')
    plt.title('Numero di stelle trovate in funzione della run')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    #plt.show()