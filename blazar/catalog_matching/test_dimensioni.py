import numpy as np
import os
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
from astropy.wcs import FITSFixedWarning

# Questo disattiva specificamente gli avvisi di correzione automatica delle date
warnings.filterwarnings('ignore', category=FITSFixedWarning)


# =============================================================================
# FUNZIONI DI SUPPORTO
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


def confronta_immagini(path_foto1, path_foto2):
    """
    Confronto le caratteristiche tecniche di due immagini FITS
    per identificare differenze di scala e binning.
    """
    percorsi = [str(path_foto1), str(path_foto2)]
    risultati = []

    for p in percorsi:
        with fits.open(p) as hdu:
            header = hdu[0].header
            data = hdu[0].data
            w = WCS(header)

            # Calcolo la scala di piastra
            scale_deg = proj_plane_pixel_scales(w)
            scale_arcsec = np.mean(scale_deg) * 3600.0
            altezza, larghezza = data.shape

            # --- CALCOLO MEZZA DIAGONALE ---
            # Trovo le coordinate del centro (in pixel) e di un angolo
            centro_pix = [larghezza / 2, altezza / 2]
            angolo_pix = [larghezza, altezza]

            # Converto in coordinate celesti
            centro_coord = w.pixel_to_world(centro_pix[0], centro_pix[1])
            angolo_coord = w.pixel_to_world(angolo_pix[0], angolo_pix[1])

            # Calcolo la separazione in minuti d'arco (arcmin)
            mezza_diag_arcmin = centro_coord.separation(angolo_coord).arcmin
            # -------------------------------

            bin_x = header.get('XBINNING') or header.get('CCDBIN1') or '?'
            bin_y = header.get('YBINNING') or header.get('CCDBIN2') or '?'

            risultati.append({
                'file': os.path.basename(p),
                'dim': f"{larghezza}x{altezza}",
                'scale': f"{scale_arcsec:.3f} arcsec/px",
                'bin': f"{bin_x}x{bin_y}",
                'diag': f"{mezza_diag_arcmin:.2f}'"
            })

    print(f"\n{'PARAMETRO':<20} | {'FOTO 1 (GRAB)':<25} | {'FOTO 2 (BLAZAR)':<25}")
    print("-" * 75)
    print(f"{'Nome File':<20} | {risultati[0]['file']:<25} | {risultati[1]['file']:<25}")
    print(f"{'Risoluzione (px)':<20} | {risultati[0]['dim']:<25} | {risultati[1]['dim']:<25}")
    print(f"{'Scala di Piastra':<20} | {risultati[0]['scale']:<25} | {risultati[1]['scale']:<25}")
    print(f"{'Mezza Diagonale':<20} | {risultati[0]['diag']:<25} | {risultati[1]['diag']:<25}")
    print(f"{'Binning Header':<20} | {risultati[0]['bin']:<25} | {risultati[1]['bin']:<25}\n")


# =============================================================================
# ESECUZIONE
# =============================================================================

BASE_DIR = trova_cartella_base("Lorenzo")
run = 1

nome_file_csv = f"run_{run}_stelle_trovate_e_catalogate_immagine_035.csv"
percorso_grab = cerca_file_nel_progetto(BASE_DIR, nome_file_csv)

percorso_foto1 = None
if percorso_grab:
    with open(percorso_grab, 'r') as f:
        for line in f:
            if "NOME_FILE_FITS" in line or "PERCORSO_FILE" in line:
                nome_fits = line.split(':')[-1].strip()
                nome_fits = os.path.basename(nome_fits)
                percorso_foto1 = cerca_file_nel_progetto(BASE_DIR, nome_fits)
                break

cartella_dati_blazar = BASE_DIR / "PMC_DATA_BLAZAR"
lista_completa_fits = sorted([f for f in cartella_dati_blazar.rglob("*.fits") if f.is_file()])

if percorso_foto1 and lista_completa_fits:
    percorso_blazar = lista_completa_fits[0]
    confronta_immagini(percorso_foto1, percorso_blazar)
else:
    print("Errore: impossibile trovare uno dei due file FITS per il confronto.")