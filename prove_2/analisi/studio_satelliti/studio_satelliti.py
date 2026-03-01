import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
from skyfield.api import load, wgs84
from datetime import timedelta


# =============================================================================
# CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # Risalgo l'albero delle directory per trovare la cartella base del progetto
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata. Uso la directory dello script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("pmc_photometry")

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Importo i moduli personalizzati
from funzioni.utilita import *
from funzioni.astrometria import *

# Inizializzo Skyfield
ts = load.timescale()

# Imposto le coordinate del telescopio
lat_oss, lon_oss, alt_oss = ottieni_coordinate_telescopio('ASTRI 1', BASE_DIR)
osservatorio = wgs84.latlon(lat_oss, lon_oss, elevation_m=alt_oss)

RUNS = [1, 2, 3]


def estrai_header_manuale(percorso_csv):
    # Leggo manualmente le prime righe commentate del CSV per estrarre con sicurezza DATE-OBS e DATE-END
    header_data = {}
    with open(percorso_csv, 'r') as f:
        for riga in f:
            if riga.startswith('#'):
                parti = riga.strip('# \n').split(': ', 1)
                if len(parti) == 2:
                    chiave = parti[0].strip()
                    valore = parti[1].strip()
                    header_data[chiave] = valore
            else:
                break  # Fine dei commenti header
    return header_data


if __name__ == "__main__":
    tuo_user = "lorenzo.simeone@studenti.unipg.it"
    tua_password = "Cazzata_2002348"

    # Cerco un FITS qualsiasi per scaricare i TLE del giorno corretto
    file_fits_riferimento = None
    for r in RUNS:
        cartella_run_temp = list(BASE_DIR.rglob(f"20250120_run{r}"))
        if cartella_run_temp:
            f_list = list(cartella_run_temp[0].glob('*.fit')) + list(cartella_run_temp[0].glob('*.fits'))
            if f_list:
                file_fits_riferimento = f_list[0]
                break

    if file_fits_riferimento:
        import astropy.io.fits as fits

        hdu_ref = fits.open(file_fits_riferimento)
        tempo_ref_astropy = Time(hdu_ref[0].header['DATE-OBS'], format='isot', scale='utc')
        hdu_ref.close()

        cartella_tabelle = BASE_DIR / "tabelle"
        cartella_tabelle.mkdir(exist_ok=True)

        print("Scaricamento/Caricamento TLE in corso...")
        percorso_tle = scarica_tle_storici(tempo_ref_astropy, tuo_user, tua_password, cartella_tabelle)
        if percorso_tle:
            satelliti_attivi = load.tle_file(percorso_tle)
            print(f"Caricati {len(satelliti_attivi)} satelliti storici.")
        else:
            print("Errore nel caricamento dei TLE. Esco.")
            exit()
    else:
        print("Nessun file FITS trovato per calcolare la data. Esco.")
        exit()

    for run in RUNS:
        cartella_run_csv = BASE_DIR / "prove_2" / "tabelle" / "tabelle_unite" / f"tabelle_unite_run_{run}"
        if not cartella_run_csv.exists():
            continue

        file_csv_list = sorted(list(cartella_run_csv.glob('*.csv')))
        if not file_csv_list:
            continue

        print(f"\n==================== ANALISI SATELLITI RUN {run} ====================")

        for percorso_file in tqdm(file_csv_list, desc=f"Verifica satelliti Run {run}"):
            # Estraggo l'indice dell'immagine dal nome del file
            nome_file = percorso_file.name

            # Leggo l'header commentato
            header_dict = estrai_header_manuale(percorso_file)

            if 'DATE-OBS' not in header_dict or 'DATE-END' not in header_dict:
                continue

            # Calcolo il tempo medio di esposizione
            t_inizio = Time(header_dict['DATE-OBS'], format='isot', scale='utc')
            t_fine = Time(header_dict['DATE-END'], format='isot', scale='utc')
            t_medio = t_inizio + (t_fine - t_inizio) / 2.0
            tempo_skyfield = ts.from_astropy(t_medio)

            # Leggo il dataframe
            df = pd.read_csv(percorso_file, comment='#')

            # Filtro solo gli oggetti senza corrispondenza
            if 'Corrispondenza' not in df.columns:
                continue

            df_no = df[df['Corrispondenza'] == 'NO'].copy()
            if df_no.empty:
                continue

            # Propago le posizioni dei satelliti
            ra_sat_list, dec_sat_list = [], []
            nomi_satelliti = []
            navstar_presente = False

            for sat in satelliti_attivi:
                # Traccio specificamente il NAVSTAR 74
                if sat.model.satnum == 40105:
                    navstar_presente = True
                    topo_nav = (sat - osservatorio).at(tempo_skyfield)
                    ra_n, dec_n, _ = topo_nav.radec()
                    pos_navstar_ra = ra_n.hours * 15
                    pos_navstar_dec = dec_n.degrees

                topocentrica = (sat - osservatorio).at(tempo_skyfield)
                ra_sat, dec_sat, _ = topocentrica.radec()

                if np.isnan(ra_sat.hours) or np.isnan(dec_sat.degrees):
                    continue

                ra_sat_list.append(ra_sat.hours * 15)
                dec_sat_list.append(dec_sat.degrees)
                nomi_satelliti.append(sat.name)

            # Eseguo il cross-match con SkyCoord
            catalogo_satelliti = SkyCoord(ra=ra_sat_list * u.deg, dec=dec_sat_list * u.deg)
            coords_oggetti_no = SkyCoord(ra=df_no['RA_centroid'].values * u.deg,
                                         dec=df_no['DEC_centroid'].values * u.deg)

            idx_sat, d2d_sat, _ = coords_oggetti_no.match_to_catalog_sky(catalogo_satelliti)

            # Stampo i risultati per questa immagine
            print(f"\n--- File: {nome_file} ---")
            print(f"Tempo calcolato (centro esposizione): {t_medio.isot}")
            if navstar_presente:
                print(f"Posizione calcolata NAVSTAR 74 -> RA: {pos_navstar_ra:.5f}, DEC: {pos_navstar_dec:.5f}")
            else:
                print("ATTENZIONE: NAVSTAR 74 (40105) NON TROVATO NEL CATALOGO TLE!")

            for k in range(len(df_no)):
                obj_id = df_no.iloc[k].get('ID', f'Idx_{k}')
                label = df_no.iloc[k].get('label', 'N/A')
                ra_obj = coords_oggetti_no[k].ra.deg
                dec_obj = coords_oggetti_no[k].dec.deg
                dist_arcmin = d2d_sat[k].arcmin
                sat_vicino = nomi_satelliti[idx_sat[k]]

                print(f"  -> Oggetto {obj_id} (Label: {label}) alle coordinate RA: {ra_obj:.5f}, DEC: {dec_obj:.5f}")
                print(f"     Il satellite più vicino è {sat_vicino} a una distanza di {dist_arcmin:.2f} arcminuti.")

                if navstar_presente:
                    # Calcolo manualmente la distanza specifica dal NAVSTAR 74
                    coord_nav = SkyCoord(ra=pos_navstar_ra * u.deg, dec=pos_navstar_dec * u.deg)
                    dist_da_navstar = coords_oggetti_no[k].separation(coord_nav).arcmin
                    print(f"     Distanza specifica dal NAVSTAR 74: {dist_da_navstar:.2f} arcminuti.")