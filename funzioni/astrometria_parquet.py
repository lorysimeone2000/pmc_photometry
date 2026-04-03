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


# estraggo solamente la logica per calcolare il raggio
def calcola_raggio_kron(valori_pixel, distanze_pixel, k=2.5, r_min=3.5):
    somma_intensita = np.sum(valori_pixel)
    if somma_intensita <= 0: return np.nan
    somma_momenti = np.sum(valori_pixel * distanze_pixel)
    r_1 = somma_momenti / somma_intensita
    return max(k * r_1, r_min)


def tabella_catalogo(image_file_, tbl_vizier_cut, tbl_hipparco_run_clean):
    hdu_list_ = fits.open(image_file_)
    wcs = WCS(hdu_list_[0].header)
    data_ = hdu_list_[0].data
    h, w = data_.shape
    bordo = 7

    nome_catalogo_vizier = np.array(["P"] * len(tbl_vizier_cut), dtype=object)
    colonne_vizier = {
        'Catalogo': nome_catalogo_vizier,
        'ID': tbl_vizier_cut['objID'],
        'RAJ2000': tbl_vizier_cut['RAJ2000'],
        'DEJ2000': tbl_vizier_cut['DEJ2000'],
        'Mag': tbl_vizier_cut['Mag'],
    }

    nome_catalogo_hipparco = np.array(["H"] * len(tbl_hipparco_run_clean), dtype=object)
    colonne_hipparco = {
        'Catalogo': nome_catalogo_hipparco,
        'ID': tbl_hipparco_run_clean['HIP'],
        'RAJ2000': tbl_hipparco_run_clean['_RAJ2000'],
        'DEJ2000': tbl_hipparco_run_clean['_DEJ2000'],
        'Mag': tbl_hipparco_run_clean['Mag'],
    }

    t1 = Table(colonne_vizier)
    t2 = Table(colonne_hipparco)
    tbl_unita = vstack([t1, t2], metadata_conflicts='silent')

    coords = SkyCoord(ra=tbl_unita['RAJ2000'], dec=tbl_unita['DEJ2000'], unit=u.deg)
    x_pix, y_pix = wcs.world_to_pixel(coords)

    mask_bordo = ((x_pix >= bordo) & (x_pix < (w - bordo)) & (y_pix >= bordo) & (y_pix < (h - bordo)))
    hdu_list_.close()
    return tbl_unita[mask_bordo]


def analisi_image_segmentation(percorso_file_, parametri_globali):
    data, fondo_iniziale, w = elabora_file_fits(percorso_file_)
    fwhm = parametri_globali.get('fwhm', 3.0)
    size = parametri_globali.get('size', 5)
    threshold = parametri_globali.get('threshold_assoluta', 3.0)
    pixel_n = parametri_globali.get('pixel', 3)

    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    finder = SourceFinder(npixels=pixel_n, progress_bar=False)
    segment_map = finder(convolved_data, threshold)
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()

    if len(tbl) == 0:
        # calcolo il fondo sull'intera immagine qualora non ci siano sorgenti
        somma_totale = np.sum(data)
        fondo_medio = somma_totale / data.size if data.size > 0 else 0
        return tbl, parametri_globali, somma_totale, fondo_medio

    # mantengo il format solo per i centroidi
    for col in ['xcentroid', 'ycentroid']:
        tbl[col].info.format = '.2f'

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    livello_saturazione = 255 - fondo_iniziale - median
    tbl['saturazione'] = tbl['max_value'] >= livello_saturazione

    K_KRON, R_MIN_KRON = 2.5, 3.5
    soglia_assoluta, soglia_relativa = 2.5, 0.05
    bordo, ny, nx = 7, data.shape[0], data.shape[1]

    # svuoto la memoria dai vettori dei flussi inutilizzati
    lista_raggi_max, raggi_kron_aper, mask_keep = [], [], []

    # imposto tutta la maschera a True all'inizio
    mask_sfondo = np.ones(data.shape, dtype=bool)

    for prop in cat:
        xc, yc = prop.xcentroid, prop.ycentroid
        slices = prop.slices
        valori_pixel = data[slices][segment_map.data[slices] == prop.label]

        # estraggo il raggio massimo per mascherare anche le sorgenti sui bordi
        if len(valori_pixel) == 0:
            r_max_pix = 0.5
        else:
            y_idx, x_idx = np.indices(segment_map.data[slices].shape)
            ypix = y_idx[segment_map.data[slices] == prop.label] + slices[0].start
            xpix = x_idx[segment_map.data[slices] == prop.label] + slices[1].start
            distanze_pix = np.hypot(xpix - xc, ypix - yc)
            r_max_pix = max(np.max(distanze_pix) if len(distanze_pix) > 0 else 0.5, 0.5)

        # applico il mascheramento del fondo raddoppiando l'apertura
        r_apertura = 2.0 * r_max_pix
        r_int_bg = int(np.ceil(r_apertura))
        y_min_bg = max(0, int(yc - r_int_bg))
        y_max_bg = min(data.shape[0], int(yc + r_int_bg + 1))
        x_min_bg = max(0, int(xc - r_int_bg))
        x_max_bg = min(data.shape[1], int(xc + r_int_bg + 1))
        y_g_bg, x_g_bg = np.ogrid[y_min_bg:y_max_bg, x_min_bg:x_max_bg]
        dist_box_bg = np.hypot(x_g_bg - xc, y_g_bg - yc)

        # escludo i pixel stellari dalla mia maschera
        mask_sfondo[y_min_bg:y_max_bg, x_min_bg:x_max_bg][dist_box_bg <= r_apertura] = False

        # salto l'analisi dettagliata per le sorgenti vicine ai bordi o invalide
        if len(valori_pixel) == 0 or not ((xc >= bordo) and (xc < nx - bordo) and (yc >= bordo) and (yc < ny - bordo)):
            lista_raggi_max.append(0.5)
            raggi_kron_aper.append(np.nan)
            mask_keep.append(False)
            continue

        lista_raggi_max.append(r_max_pix)

        r_int = int(np.ceil(r_max_pix))
        y_min, y_max = max(0, int(yc - r_int)), min(data.shape[0], int(yc + r_int + 1))
        x_min, x_max = max(0, int(xc - r_int)), min(data.shape[1], int(xc + r_int + 1))
        cutout = data[y_min:y_max, x_min:x_max]
        y_g, x_g = np.ogrid[y_min:y_max, x_min:x_max]
        dist_box = np.hypot(x_g - xc, y_g - yc)
        mask_circle = dist_box <= r_max_pix

        # estraggo solamente il raggio che mi serve
        r_used = calcola_raggio_kron(cutout[mask_circle], dist_box[mask_circle], K_KRON, R_MIN_KRON)
        raggi_kron_aper.append(r_used)

        is_good = (np.sum(valori_pixel > soglia_assoluta) >= 3) and (
                np.sum(valori_pixel > soglia_relativa * prop.max_value) >= 2)
        mask_keep.append(is_good)

    tbl['raggio_kron_aper'] = raggi_kron_aper
    tbl['raggio_kron_aper'].info.format = '%.2f'

    tbl_filtrato = tbl[mask_keep]
    if len(tbl_filtrato) > 0: tbl_filtrato['label'] = np.arange(1, len(tbl_filtrato) + 1)

    # sommo i pixel rimasti fuori dalle aperture per isolare il fondo
    somma_parziale = np.sum(data[mask_sfondo])
    pixel_contati = np.sum(mask_sfondo)
    pixel_totali = data.size

    # calcolo il rapporto finale e le medie relative
    rapporto_pixel = pixel_contati / pixel_totali if pixel_totali > 0 else 1
    somma_totale = somma_parziale / rapporto_pixel if rapporto_pixel > 0 else 0
    fondo_medio = somma_parziale / pixel_contati if pixel_contati > 0 else 0

    return tbl_filtrato, parametri_globali, somma_totale, fondo_medio

def modello_lineare(mag, m, q):
    return m * mag + q


def stampa_descrizioni_colonne_ps1():
    """
    Stampa tutte le informazioni disponibili per le colonne di magnitudine
    del catalogo Pan-STARRS DR2.
    """
    from astroquery.vizier import Vizier
    import pandas as pd

    print("\n" + "=" * 80)
    print("ANALISI COMPLETA COLONNE MAGNITUDINE - PAN-STARRS DR2 (II/389/ps1_dr2)")
    print("=" * 80)

    # Recupera il catalogo
    print("\n1. Scaricamento metadati del catalogo...")
    catalogo = Vizier.get_catalogs("II/389/ps1_dr2")[0]

    # Colori di magnitudine da analizzare
    bande = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']
    bande_err = ['e_gmag', 'e_rmag', 'e_imag', 'e_zmag', 'e_ymag']
    bande_std = ['gmagStd', 'rmagStd', 'imagStd', 'zmagStd', 'ymagStd']

    tutte_colonne = bande + bande_err + bande_std

    print("\n2. Analisi dettagliata per colonna:")
    print("-" * 80)

    risultati = {}

    for col_name in tutte_colonne:
        if col_name in catalogo.columns:
            print(f"\n>>> COLONNA: {col_name}")
            print("-" * 50)

            # Ottieni la colonna
            col = catalogo[col_name]

            # Stampa tutte le informazioni disponibili
            print(f"  Descrizione: {col.description if hasattr(col, 'description') else 'N/A'}")
            print(f"  Unità: {col.unit if hasattr(col, 'unit') else 'N/A'}")
            print(f"  Formato: {col.format if hasattr(col, 'format') else 'N/A'}")

            # Meta dati
            if hasattr(col, 'meta'):
                print(f"  Meta dati:")
                for key, value in col.meta.items():
                    print(f"    {key}: {value}")

            # UCD
            if hasattr(col, 'meta') and 'ucd' in col.meta:
                print(f"  UCD: {col.meta['ucd']}")

            # Cerca informazioni sulla lunghezza d'onda
            desc = col.description if hasattr(col, 'description') else ""
            if desc:
                # Cerca pattern di lunghezza d'onda
                import re
                # Cerca pattern come "4866Å", "4866 A", "4866A", "4866 nm"
                patterns = [
                    r'(\d+)\s*[ÅA]',  # 4866Å o 4866A
                    r'(\d+)\s*nm',  # 4866 nm
                    r'(\d+)\s*microns',  # 0.486 microns
                    r'(\d+\.?\d*)\s*μm',  # 0.486 μm
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, desc)
                    if matches:
                        print(f"  Lunghezza d'onda trovata: {matches[0]} Å")
                        break

            risultati[col_name] = {
                'description': desc,
                'ucd': col.meta.get('ucd', 'N/A') if hasattr(col, 'meta') else 'N/A'
            }
        else:
            print(f"\n>>> COLONNA: {col_name} - NON TROVATA nel catalogo")

    print("\n" + "=" * 80)
    print("RIASSUNTO UCD TROVATI:")
    print("=" * 80)
    for col_name, info in risultati.items():
        print(f"{col_name:10s} -> UCD: {info['ucd']}")

    print("\n" + "=" * 80)

    return risultati


def scarica_intervalli_bande_ps1_da_descrizioni():
    """
    Scarica gli intervalli delle bande Pan-STARRS DR2 estraendo
    le lunghezze d'onda centrali dalle descrizioni delle colonne.
    """
    from astroquery.vizier import Vizier
    import re

    print("\n" + "=" * 70)
    print("SCARICAMENTO INTERVALLI BANDE PAN-STARRS DR2")
    print("=" * 70)

    # Recupera il catalogo
    catalogo = Vizier.get_catalogs("II/389/ps1_dr2")[0]

    # FWHM delle bande Pan-STARRS da Tonry+ 2012
    fwhm_nm = {
        'gmag': 137,
        'rmag': 140,
        'imag': 130,
        'zmag': 104,
        'ymag': 83
    }

    # Valori di fallback validi (in nm)
    fallback_validi = {
        'gmag': 486.6,
        'rmag': 621.5,
        'imag': 754.5,
        'zmag': 867.9,
        'ymag': 963.3
    }

    # Pattern per estrarre la lunghezza d'onda centrale - ORDINE IMPORTANTE!
    # Prima cerco il pattern con {AA} che è specifico per gmag
    patterns = [
        r'\((\d+)\s*\{AA\}\)',  # (4866{AA}) - specifico per gmag
        r'\((\d+)\s*A\)',  # (6215A)
        r'(\d+)\s*[ÅA]',  # 4866Å o 4866A
        r'(\d+)\s*nm',  # 4866 nm
    ]

    bande = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']
    limiti_bande = {}

    print("\nEstrazione lunghezze d'onda dalle descrizioni:")
    print("-" * 70)

    for banda in bande:
        if banda in catalogo.columns:
            col = catalogo[banda]
            descrizione = col.description if hasattr(col, 'description') else ""

            print(f"\n{banda}:")
            print(f"  Descrizione: {descrizione}")

            # Estrai la lunghezza d'onda centrale
            lambda_centro = None

            # prima prova a trovare il pattern specifico per questa banda
            if banda == 'gmag':
                # cerco specificamente il pattern con {AA}
                match = re.search(r'\((\d+)\s*\{AA\}\)', descrizione)
                if match:
                    valore = float(match.group(1))
                    lambda_centro = valore / 10.0
                    # raddoppio le parentesi graffe per non farle interpretare come variabile
                    print(f"  -> Lunghezza d'onda estratta (pattern {{AA}}): {valore:.0f} Å = {lambda_centro:.1f} nm")

            # se non trovato, provo tutti i pattern
            if lambda_centro is None:
                for pattern in patterns:
                    match = re.search(pattern, descrizione)
                    if match:
                        valore = float(match.group(1))
                        # verifico che il valore sia in un range plausibile (300-2000 nm o 3000-20000 Å)
                        # aggiungo la 'r' per rendere la stringa raw ed evitare il SyntaxWarning
                        if '{AA}' in pattern or 'Å' in pattern or pattern.endswith(r'A\)') or pattern.endswith('[ÅA]'):
                            # è in Å, converto in nm
                            lambda_centro = valore / 10.0
                            print(f"  -> Lunghezza d'onda estratta: {valore:.0f} Å = {lambda_centro:.1f} nm")
                        else:
                            lambda_centro = valore
                            print(f"  -> Lunghezza d'onda estratta: {lambda_centro:.1f} nm")
                        break

            # Verifica che il valore sia plausibile (tra 300 e 2000 nm)
            if lambda_centro is not None:
                if lambda_centro < 300 or lambda_centro > 2000:
                    print(f"  -> ATTENZIONE: Valore {lambda_centro:.1f} nm non plausibile! Uso fallback.")
                    lambda_centro = fallback_validi.get(banda, 500.0)
            else:
                print(f"  -> ATTENZIONE: Nessuna lunghezza d'onda trovata! Uso fallback.")
                lambda_centro = fallback_validi.get(banda, 500.0)
                print(f"  -> Valore di fallback: {lambda_centro:.1f} nm")

            # Calcola l'intervallo usando 1.5×FWHM (copia ~93% della risposta)
            fwhm = fwhm_nm.get(banda, 100)
            fattore = 0.75  # 1.5×FWHM / 2 = 0.75×FWHM per lato
            w_min = int(round(lambda_centro - fwhm * fattore))
            w_max = int(round(lambda_centro + fwhm * fattore))

            # Limita al range del sensore (300-1100 nm)
            w_min = max(w_min, 300)
            w_max = min(w_max, 1100)

            # Verifica finale che w_min < w_max
            if w_min >= w_max:
                print(f"  -> ERRORE: Intervallo non valido ({w_min}-{w_max})! Uso fallback.")
                # Usa un intervallo di fallback
                fallback_intervalli = {
                    'gmag': (418, 555),
                    'rmag': (552, 692),
                    'imag': (690, 820),
                    'zmag': (816, 920),
                    'ymag': (922, 1005)
                }
                w_min, w_max = fallback_intervalli.get(banda, (400, 550))

            limiti_bande[banda] = (w_min, w_max)
            print(f"  -> FWHM: {fwhm} nm")
            print(f"  -> Intervallo finale: {w_min} - {w_max} nm")

    print("\n" + "=" * 70)
    print("DIZIONARIO FINALE:")
    print("=" * 70)
    for banda, (w_min, w_max) in limiti_bande.items():
        print(f"    '{banda}': ({w_min}, {w_max}),")
    print("=" * 70 + "\n")

    return limiti_bande