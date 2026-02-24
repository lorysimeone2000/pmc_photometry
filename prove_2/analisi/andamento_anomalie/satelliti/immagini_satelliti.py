import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import FITSFixedWarning, WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import simple_norm
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
import astropy.coordinates as coord
from astropy.table import Table, vstack
from matplotlib.lines import Line2D
from tqdm import tqdm
from pathlib import Path
from astropy.io.fits.verify import VerifyWarning

# importo le librerie per il calcolo orbitale e per il catalogo
from skyfield.api import load, wgs84
from astropy.time import Time
from astroquery.vizier import Vizier

# sopprimo i warning
warnings.filterwarnings('ignore', category=FITSFixedWarning)
# sopprimo i warning di validazione dei file FITS
warnings.filterwarnings('ignore', category=VerifyWarning)

# =============================================================================
# --- FUNZIONI DI UTILITÀ ---
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


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


def crea_tabella_catalogo_unita(tbl_viz, tbl_hip):
    # assemblo la tabella unita esattamente come nello script originale
    nome_catalogo_vizier = np.array(["II/389/ps1_dr2"] * len(tbl_viz), dtype=object)
    colonne_vizier = {
        'Catalogo': nome_catalogo_vizier,
        'ID': tbl_viz['objID'],
        'RAJ2000': tbl_viz['RAJ2000'],
        'DEJ2000': tbl_viz['DEJ2000'],
        'Mag': tbl_viz['gmag'],
    }

    nome_catalogo_hipparco = np.array(["I/239/hip_main"] * len(tbl_hip), dtype=object)
    colonne_hipparco = {
        'Catalogo': nome_catalogo_hipparco,
        'ID': tbl_hip['HIP'],
        'RAJ2000': tbl_hip['_RAJ2000'],
        'DEJ2000': tbl_hip['_DEJ2000'],
        'Mag': tbl_hip['Vmag'],
    }

    t1 = Table(colonne_vizier)
    t2 = Table(colonne_hipparco)

    return vstack([t1, t2])


def salva_cutout_satellite(xc, yc, x_sat, y_sat, data_sub, wcs_ref, img_idx, star_id, cartella_out, soglia_deg,
                           tbl_catalogo_unito):
    # calcolo la scala in pixel per convertire la soglia
    pixel_scale = np.mean(proj_plane_pixel_scales(wcs_ref))
    soglia_px = soglia_deg / pixel_scale

    # imposto il lato del riquadro grande il doppio della soglia di correlazione usata
    half_side = soglia_px * 2

    # imposto i limiti spaziali del cutout centrato sull'oggetto rilevato
    x_min = int(np.floor(xc - half_side))
    x_max = int(np.ceil(xc + half_side))
    y_min = int(np.floor(yc - half_side))
    y_max = int(np.ceil(yc + half_side))

    # prevengo sforamenti rispetto ai bordi dell'immagine reale
    ny, nx = data_sub.shape
    x_min = max(0, x_min)
    x_max = min(nx, x_max)
    y_min = max(0, y_min)
    y_max = min(ny, y_max)

    cutout = data_sub[y_min:y_max, x_min:x_max]

    fig, ax = plt.subplots(figsize=(8, 8))
    legend_elements = []

    if cutout.size > 0:
        norm = simple_norm(cutout, 'log', percent=99.9)
        ax.imshow(cutout, cmap='gray_r', origin='lower', extent=[x_min, x_max, y_min, y_max], norm=norm)

    # proietto le stelle del catalogo sulle coordinate pixel dell'immagine
    coords_cat_sky = SkyCoord(ra=tbl_catalogo_unito['RAJ2000'], dec=tbl_catalogo_unito['DEJ2000'], unit=u.deg)
    x_pix_cat, y_pix_cat = wcs_ref.world_to_pixel(coords_cat_sky)
    tbl_catalogo_unito['xcentroid'] = x_pix_cat
    tbl_catalogo_unito['ycentroid'] = y_pix_cat

    # filtro le stelle catalogate che cadono nel riquadro attuale
    mask_in_box = (tbl_catalogo_unito['xcentroid'] >= x_min) & (tbl_catalogo_unito['xcentroid'] <= x_max) & \
                  (tbl_catalogo_unito['ycentroid'] >= y_min) & (tbl_catalogo_unito['ycentroid'] <= y_max)

    tbl_cat_box = tbl_catalogo_unito[mask_in_box]

    if len(tbl_cat_box) > 0:
        # imposto i limiti della colorbar per la magnitudine
        min_mag = min(np.nanmin(tbl_cat_box['Mag']), 5)
        max_mag = 15

        # disegno le stelle catalogate imponendo la dimensione a 1
        ax.scatter(tbl_cat_box['xcentroid'], tbl_cat_box['ycentroid'], c=tbl_cat_box['Mag'],
                   cmap='viridis_r', vmin=min_mag, vmax=max_mag, s=1, zorder=5)

        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Stelle catalogate',
                                      markerfacecolor='gray', markersize=5))

        # aggiungo la colorbar laterale
        sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(vmin=min_mag, vmax=max_mag))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mag')

    # disegno il primo anello sottile per l'oggetto rilevato (non catalogato)
    ax.plot(xc, yc, marker='o', markerfacecolor='none', markeredgecolor='red', markersize=14, markeredgewidth=1.2, linestyle='None', zorder=10)
    legend_elements.append(
        Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor='red', label='Oggetto rilevato (NO match)', markersize=10, linestyle='None',
               markeredgewidth=1.2))

    # disegno il secondo anello sottile per la posizione calcolata del satellite
    # lo faccio leggermente più largo per poterli distinguere facilmente se i centri sono quasi sovrapposti
    ax.plot(x_sat, y_sat, marker='o', markerfacecolor='none', markeredgecolor='#ff00ff', markersize=22, markeredgewidth=1.2, linestyle='None', zorder=15)
    legend_elements.append(
        Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor='#ff00ff', label='Posizione calcolata Satellite', markersize=12,
               linestyle='None', markeredgewidth=1.2))

    ax.set_title(f"Correlazione Satellite - Immagine {img_idx:03d}\nID Oggetto: {star_id}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.8)

    nome_figura = f"correlazione_sat_img_{img_idx:03d}_obj_{star_id}.png"
    plt.savefig(cartella_out / nome_figura, dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# --- MAIN SCRIPT ---
# =============================================================================

BASE_DIR = trova_cartella_base("pmc_photometry")
print(f"--- CONFIGURAZIONE SISTEMA ---")
print(f"Cartella Base rilevata: {BASE_DIR}")

RUN_TARGET = 1
SOGLIA_CORRELAZIONE_DEG = 50 / 60
magnitudine_massima = 15

# preparo l'interrogazione a vizier
vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag'],
    row_limit=-1
)

# cerco le cartelle necessarie
cartella_unite = cerca_cartella_nel_progetto(BASE_DIR, "tabelle_unite")
if cartella_unite is None:
    print("ERRORE: Cartella 'tabelle_unite' non trovata.")
    exit()

cartella_tabelle_run = Path(cartella_unite) / f"tabelle_unite_run_{RUN_TARGET}"
if not cartella_tabelle_run.exists():
    print(f"ERRORE: Cartella {cartella_tabelle_run} non trovata.")
    exit()

cartella_output = BASE_DIR / "plot_satelliti_run_1"
cartella_output.mkdir(exist_ok=True)

# configuro i parametri di osservazione per skyfield
lat_oss, lon_oss, alt_oss = 28.3000, -16.50583, 2370
osservatorio = wgs84.latlon(lat_oss, lon_oss, elevation_m=alt_oss)
ts = load.timescale()

# carico il file TLE
file_tle = list(BASE_DIR.rglob("tle_storico_payload_*.txt"))
if not file_tle:
    print("ERRORE: Nessun file TLE trovato nel progetto.")
    exit()
# converto il percorso PosixPath in una stringa testuale
satelliti_attivi = load.tle_file(str(file_tle[0]))

# pre-carico e propago Hipparcos
file_hipparco = cerca_file_nel_progetto(BASE_DIR, "hipparco.fit")
hdu_list_hipparco = fits.open(file_hipparco)
tbl_catalogo_hipparco = Table(hdu_list_hipparco[1].data)
hdu_list_hipparco.close()

dt = 2000.0 - 1991.25
sigma_ra_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco['e_RAICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco['e_pmRA'])) ** 2) / 3600000.0
sigma_dec_deg = np.sqrt(np.nan_to_num(tbl_catalogo_hipparco['e_DEICRS']) ** 2 + (
            dt * np.nan_to_num(tbl_catalogo_hipparco['e_pmDE'])) ** 2) / 3600000.0
sigma_hip_deg = np.sqrt(sigma_ra_deg ** 2 + sigma_dec_deg ** 2)
sigma_vizier_deg = 0.1 / 3600.0
sigma_totale_deg = np.sqrt(sigma_hip_deg ** 2 + sigma_vizier_deg ** 2)
exclusion_radii_deg_ = 3.0 * sigma_totale_deg
exclusion_radii_deg = np.full(len(exclusion_radii_deg_), 2.5 / 3600.0)
coords_hipparco_global = SkyCoord(ra=tbl_catalogo_hipparco['_RAJ2000'], dec=tbl_catalogo_hipparco['_DEJ2000'],
                                  unit=u.deg)

file_csv_list = sorted(list(cartella_tabelle_run.glob('*.csv')))
print(f"Inizio analisi di {len(file_csv_list)} file per la Run {RUN_TARGET}...")

tbl_unita_globale = None

for n, p_csv in enumerate(tqdm(file_csv_list, desc="Ricerca e Plot Satelliti")):
    df = pd.read_csv(p_csv, comment='#')

    header_lines = []
    with open(p_csv, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line)
            else:
                break

    nome_fits = None
    for line in header_lines:
        if 'NOME_FILE_FITS:' in line:
            nome_fits = line.split('NOME_FILE_FITS:')[1].strip()
            break

    if not nome_fits: continue

    percorso_fits = cerca_file_nel_progetto(BASE_DIR, nome_fits)
    if not percorso_fits: continue

    df_no = df[df['Corrispondenza'] == 'NO'].copy()

    with fits.open(str(percorso_fits), memmap=False) as hdu:
        data = hdu[0].data
        wcs_ref = WCS(hdu[0].header)
        header_date_obs = hdu[0].header['DATE-OBS']
        ra_c, dec_c = hdu[0].header["RA"], hdu[0].header["DEC"]
        _, median_bg, _ = sigma_clipped_stats(data[::10, ::10], sigma=3.0)
        data_sub = data - median_bg

    # calcolo il catalogo filtrato per l'intero campo visivo alla primissima immagine
    if n == 0 or tbl_unita_globale is None:
        alto_destra = wcs_ref.pixel_to_world(3071, 2047)
        centro = SkyCoord(ra_c, dec_c, unit=u.deg)
        raggio_ricerca = Angle(centro.separation(alto_destra) * 1.5, "deg")

        riquadro_esterno_vizier = vizier.query_region(
            coord.SkyCoord(ra=ra_c, dec=dec_c, unit=(u.deg, u.deg), frame='icrs'),
            radius=raggio_ricerca,
            column_filters={'gmag': f'<{15}'}
        )
        tbl_riquadro_esterno_vizier = riquadro_esterno_vizier[0]

        distanze_hip = centro.separation(coords_hipparco_global)
        mask_hip_fov = distanze_hip < raggio_ricerca
        tbl_hipparco_run_subset = tbl_catalogo_hipparco[mask_hip_fov]
        coords_hipparco_run_subset = coords_hipparco_global[mask_hip_fov]
        exclusion_radii_run_subset = exclusion_radii_deg[mask_hip_fov]

        coords_vizier = SkyCoord(ra=tbl_riquadro_esterno_vizier['RAJ2000'], dec=tbl_riquadro_esterno_vizier['DEJ2000'],
                                 unit=u.deg)
        max_threshold_deg = np.max(exclusion_radii_run_subset)
        seplimit = max_threshold_deg * u.deg

        idx_A, idx_B, d2d_1, _ = coords_hipparco_run_subset.search_around_sky(coords_vizier, seplimit)

        if len(idx_A) > 0 and np.max(idx_A) >= len(coords_hipparco_run_subset):
            idx_viz_1, idx_hip_1 = idx_A, idx_B
        else:
            idx_hip_1, idx_viz_1 = idx_A, idx_B

        mask_threshold = d2d_1.deg <= exclusion_radii_run_subset[idx_hip_1]
        idx_hip_valid = idx_hip_1[mask_threshold]
        idx_viz_valid = idx_viz_1[mask_threshold]

        mask_keep_hipparco = np.ones(len(tbl_hipparco_run_subset), dtype=bool)
        mask_keep_vizier = np.ones(len(tbl_riquadro_esterno_vizier), dtype=bool)
        unique_hip_idx = np.unique(idx_hip_valid)

        for i_hip in unique_hip_idx:
            viz_matches = idx_viz_valid[idx_hip_valid == i_hip]
            if len(viz_matches) > 0:
                mag_viz_matches = np.nan_to_num(tbl_riquadro_esterno_vizier['gmag'][viz_matches], nan=99.0)
                idx_min_mag = np.argmin(mag_viz_matches)
                best_viz_idx = viz_matches[idx_min_mag]
                best_viz_mag = mag_viz_matches[idx_min_mag]
                hip_mag = np.nan_to_num(tbl_hipparco_run_subset['Vmag'][i_hip], nan=99.0)

                if best_viz_mag <= hip_mag:
                    mask_keep_hipparco[i_hip] = False
                else:
                    mask_keep_vizier[best_viz_idx] = False

        mask_keep_hipparco[tbl_hipparco_run_subset['Vmag'] >= magnitudine_massima] = False
        tbl_hipparco_run_clean = tbl_hipparco_run_subset[mask_keep_hipparco]
        tbl_riquadro_esterno_vizier_CLEAN = tbl_riquadro_esterno_vizier[mask_keep_vizier]
        tbl_vizier_cut = tbl_riquadro_esterno_vizier_CLEAN[
            tbl_riquadro_esterno_vizier_CLEAN['gmag'] < magnitudine_massima]

        # creo la tabella finale unita valida per tutto il run
        tbl_unita_globale = crea_tabella_catalogo_unita(tbl_vizier_cut, tbl_hipparco_run_clean)

    if df_no.empty: continue

    try:
        img_idx = int(str(p_csv.stem).split('_')[-1])
    except:
        img_idx = 0

    tempo_scatto = Time(header_date_obs, format='isot', scale='utc')
    tempo_skyfield = ts.from_astropy(tempo_scatto)

    ra_sat_list, dec_sat_list = [], []
    for sat in satelliti_attivi:
        topocentrica = (sat - osservatorio).at(tempo_skyfield)
        ra_sat, dec_sat, _ = topocentrica.radec()
        if not np.isnan(ra_sat.hours) and not np.isnan(dec_sat.degrees):
            ra_sat_list.append(ra_sat.hours * 15)
            dec_sat_list.append(dec_sat.degrees)

    if not ra_sat_list: continue

    catalogo_satelliti = SkyCoord(ra=ra_sat_list * u.deg, dec=dec_sat_list * u.deg)
    coords_oggetti_no = SkyCoord(ra=df_no['RA_centroid'].values * u.deg, dec=df_no['DEC_centroid'].values * u.deg)

    idx_sat, d2d_sat, _ = coords_oggetti_no.match_to_catalog_sky(catalogo_satelliti)
    mask_match = d2d_sat.deg < SOGLIA_CORRELAZIONE_DEG

    for i, is_match in enumerate(mask_match):
        if is_match:
            riga_oggetto = df_no.iloc[i]
            xc_obj = riga_oggetto['xcentroid']
            yc_obj = riga_oggetto['ycentroid']
            id_obj = riga_oggetto['ID']

            sat_match_idx = idx_sat[i]
            ra_s = ra_sat_list[sat_match_idx]
            dec_s = dec_sat_list[sat_match_idx]

            coord_sat_sky = SkyCoord(ra=ra_s * u.deg, dec=dec_s * u.deg)
            x_sat, y_sat = wcs_ref.world_to_pixel(coord_sat_sky)

            salva_cutout_satellite(xc_obj, yc_obj, float(x_sat), float(y_sat), data_sub, wcs_ref, img_idx, id_obj,
                                   cartella_output, SOGLIA_CORRELAZIONE_DEG, tbl_unita_globale.copy())

print(f"\nElaborazione completata. I plot sono stati salvati nella cartella: {cartella_output}")