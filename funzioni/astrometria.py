import requests
from datetime import timedelta
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder, SourceCatalog, make_2dgaussian_kernel
from photutils.aperture import aperture_photometry, CircularAperture
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, vstack

# importo la funzione di ricerca interna al mio nuovo modulo
from .utilita import cerca_file_nel_progetto

def ottieni_coordinate_telescopio(nome_telescopio, base_dir):
    # passo base_dir come argomento altrimenti la funzione non sa dove cercare
    file_posizioni = cerca_file_nel_progetto(base_dir, "posizione_telescopi.txt")
    if file_posizioni:
        with open(file_posizioni, 'r') as f:
            linee = [line.strip() for line in f.readlines() if line.strip()]
        for i, linea in enumerate(linee):
            if linea == nome_telescopio:
                try:
                    lon = float(linee[i + 1].split(':')[1].strip())
                    lat = float(linee[i + 2].split(':')[1].strip())
                    alt = float(linee[i + 3].split(':')[1].strip())
                    print(f"Coordinate caricate per {nome_telescopio}: Lat {lat}, Lon {lon}, Alt {alt}m")
                    return lat, lon, alt
                except Exception as e:
                    print(f"Errore durante l'estrazione delle coordinate per {nome_telescopio}: {e}")
                    break
    print(f"ATTENZIONE: Uso valori default per {nome_telescopio}.")
    return 28.3000, -16.505830555555555, 2370

def scarica_tle_storici(tempo_astropy, username, password, cartella_output):
    data_osservazione = tempo_astropy.datetime
    data_inizio = (data_osservazione - timedelta(days=0.5)).strftime('%Y-%m-%d')
    data_fine = (data_osservazione + timedelta(days=0.5)).strftime('%Y-%m-%d')
    nome_file = f"tle_storico_payload_{data_inizio}_to_{data_fine}.txt"
    percorso_output = cartella_output / nome_file

    if percorso_output.exists():
        print(f"TLE storici già presenti: {nome_file}")
        return str(percorso_output)
        
    print(f"Scaricando i TLE storici da Space-Track per le date {data_inizio} -> {data_fine}...")
    login_url = "https://www.space-track.org/ajaxauth/login"
    query_url = f"https://www.space-track.org/basicspacedata/query/class/gp_history/EPOCH/{data_inizio}--{data_fine}/OBJECT_TYPE/PAYLOAD/format/tle"

    with requests.Session() as session:
        risposta_login = session.post(login_url, data={'identity': username, 'password': password})
        if risposta_login.status_code != 200:
            print("ERRORE: Login su Space-Track fallito.")
            return None
        
        risposta_tle = session.get(query_url, stream=True)
        if risposta_tle.status_code == 200:
            testo_risposta = risposta_tle.text
            if "<html" in testo_risposta.lower()[:50]:
                print("ERRORE: Space-Track ha bloccato il download.")
                return None
            with open(percorso_output, 'w') as f:
                f.write(testo_risposta)
            print("Download TLE storici completato con successo!")
            return str(percorso_output)
        else:
            print(f"ERRORE: Download TLE fallito con codice HTTP {risposta_tle.status_code}")
            return None

def elabora_file_fits(percorso_file_):
    with fits.open(percorso_file_, memmap=False) as hdu_list_:
        image_data_ = hdu_list_[0].data
        w_ = WCS(hdu_list_[0].header)
        mean_, median_, std_ = sigma_clipped_stats(image_data_, sigma=3.0)
        image_data_ = image_data_ - median_
        return image_data_, median_, w_

def calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pixel, k=2.5, r_min=3.5):
    somma_intensita = np.sum(valori_pixel)
    if somma_intensita <= 0: return np.nan, np.nan
    somma_momenti = np.sum(valori_pixel * distanze_pixel)
    r_1 = somma_momenti / somma_intensita
    r_kron_finale = max(k * r_1, r_min)
    aper = CircularAperture((xc, yc), r=r_kron_finale)
    phot = aperture_photometry(data, aper)
    return phot['aperture_sum'][0], r_kron_finale

def tabella_catalogo(image_file_, tbl_vizier_cut, tbl_hipparco_run_clean):
    hdu_list_ = fits.open(image_file_)
    wcs = WCS(hdu_list_[0].header)
    data_ = hdu_list_[0].data
    h, w = data_.shape
    bordo = 7

    nome_catalogo_vizier = np.array(["II/389/ps1_dr2"] * len(tbl_vizier_cut), dtype=object)
    colonne_vizier = {
        'Catalogo': nome_catalogo_vizier,
        'ID': tbl_vizier_cut['objID'],
        'RAJ2000': tbl_vizier_cut['RAJ2000'],
        'DEJ2000': tbl_vizier_cut['DEJ2000'],
        'Mag': tbl_vizier_cut['gmag'],
    }

    nome_catalogo_hipparco = np.array(["I/239/hip_main"] * len(tbl_hipparco_run_clean), dtype=object)
    colonne_hipparco = {
        'Catalogo': nome_catalogo_hipparco,
        'ID': tbl_hipparco_run_clean['HIP'],
        'RAJ2000': tbl_hipparco_run_clean['_RAJ2000'],
        'DEJ2000': tbl_hipparco_run_clean['_DEJ2000'],
        'Mag': tbl_hipparco_run_clean['Vmag'],
    }

    t1 = Table(colonne_vizier)
    t2 = Table(colonne_hipparco)
    tbl_unita = vstack([t1, t2])

    coords = SkyCoord(ra=tbl_unita['RAJ2000'], dec=tbl_unita['DEJ2000'], unit=u.deg)
    x_pix, y_pix = wcs.world_to_pixel(coords)

    mask_bordo = ((x_pix >= bordo) & (x_pix < (w - bordo)) & (y_pix >= bordo) & (y_pix < (h - bordo)))
    hdu_list_.close()
    return tbl_unita[mask_bordo]

def esegui_fotometria_variabile(data, positions, raggi):
    flussi = []
    for (xc, yc), r in zip(positions, raggi):
        if r > 0 and not np.isnan(r):
            aper = CircularAperture((xc, yc), r=r)
            phot = aperture_photometry(data, aper)
            flussi.append(phot['aperture_sum'][0])
        else:
            flussi.append(np.nan)
    return flussi

def analisi_image_segmentation(percorso_file_, parametri_globali):
    data, fondo_iniziale, w = elabora_file_fits(percorso_file_)
    fwhm = parametri_globali.get('fwhm', 3.0)
    size = parametri_globali.get('size', 5)
    threshold = parametri_globali.get('threshold_assoluta', 3.0)
    pixel_n = parametri_globali.get('pixel', 5)

    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    finder = SourceFinder(npixels=pixel_n, progress_bar=False)
    segment_map = finder(convolved_data, threshold)
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    if len(tbl) == 0: return tbl, parametri_globali

    for col in ['xcentroid', 'ycentroid', 'kron_flux']:
        tbl[col].info.format = '.2f'

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    livello_saturazione = 255 - fondo_iniziale - median
    tbl['saturazione'] = np.where(tbl['max_value'] >= livello_saturazione, 'SI', 'NO')

    K_KRON, R_MIN_KRON = 2.5, 3.5
    soglia_assoluta, soglia_relativa = 2.5, 0.05
    bordo, ny, nx = 7, data.shape[0], data.shape[1]

    lista_raggi_max, kron_manuale_seg, kron_manuale_aper, raggi_kron_aper, mask_keep = [], [], [], [], []

    for prop in cat:
        xc, yc = prop.xcentroid, prop.ycentroid
        if not ((xc >= bordo) and (xc < nx - bordo) and (yc >= bordo) and (yc < ny - bordo)):
            lista_raggi_max.append(0.5);
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan);
            raggi_kron_aper.append(np.nan);
            mask_keep.append(False)
            continue

        slices = prop.slices
        valori_pixel = data[prop.slices][segment_map.data[prop.slices] == prop.label]

        if len(valori_pixel) == 0:
            lista_raggi_max.append(0.5);
            kron_manuale_seg.append(np.nan)
            kron_manuale_aper.append(np.nan);
            raggi_kron_aper.append(np.nan);
            mask_keep.append(False)
            continue

        y_idx, x_idx = np.indices(segment_map.data[prop.slices].shape)
        ypix = y_idx[segment_map.data[prop.slices] == prop.label] + slices[0].start
        xpix = x_idx[segment_map.data[prop.slices] == prop.label] + slices[1].start
        distanze_pix = np.hypot(xpix - xc, ypix - yc)

        r_max_pix = max(np.max(distanze_pix) if len(distanze_pix) > 0 else 0.5, 0.5)
        lista_raggi_max.append(r_max_pix)

        r_int = int(np.ceil(r_max_pix))
        y_min, y_max = max(0, int(yc - r_int)), min(data.shape[0], int(yc + r_int + 1))
        x_min, x_max = max(0, int(xc - r_int)), min(data.shape[1], int(xc + r_int + 1))
        cutout = data[y_min:y_max, x_min:x_max]
        y_g, x_g = np.ogrid[y_min:y_max, x_min:x_max]
        dist_box = np.hypot(x_g - xc, y_g - yc)
        mask_circle = dist_box <= r_max_pix

        fl_aper, r_used = calcola_flusso_kron_completo(data, xc, yc, cutout[mask_circle], dist_box[mask_circle], K_KRON, R_MIN_KRON)
        kron_manuale_aper.append(fl_aper)
        raggi_kron_aper.append(r_used)

        fl_seg, _ = calcola_flusso_kron_completo(data, xc, yc, valori_pixel, distanze_pix, K_KRON, R_MIN_KRON)
        kron_manuale_seg.append(fl_seg)

        is_good = (np.sum(valori_pixel > soglia_assoluta) >= 3) and (np.sum(valori_pixel > soglia_relativa * prop.max_value) >= 2)
        mask_keep.append(is_good)

    tbl['kron_manuale_seg'] = kron_manuale_seg
    tbl['kron_manuale_aper'] = kron_manuale_aper
    tbl['raggio_kron_aper'] = raggi_kron_aper
    tbl['somma_apertura_ultimo_pixel'] = esegui_fotometria_variabile(data, np.transpose((tbl['xcentroid'], tbl['ycentroid'])), lista_raggi_max)

    for col in ['somma_apertura_ultimo_pixel', 'kron_manuale_seg', 'kron_manuale_aper', 'raggio_kron_aper']:
        tbl[col].info.format = '%.2f'

    tbl_filtrato = tbl[mask_keep]
    if len(tbl_filtrato) > 0: tbl_filtrato['label'] = np.arange(1, len(tbl_filtrato) + 1)

    return tbl_filtrato, parametri_globali