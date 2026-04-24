import numpy as np
import os
import sys
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import warnings
from astropy.wcs import FITSFixedWarning
from tqdm import tqdm
import pandas as pd

# importo i moduli necessari per la statistica e la visualizzazione
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats

# Importo i moduli necessari per il ritaglio spaziale
from astropy.nddata import Cutout2D

warnings.filterwarnings('ignore', category=FITSFixedWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    # cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


# trovo la cartella base del mio progetto
BASE_DIR = trova_cartella_base("Lorenzo")

PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")
if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

# =============================================================================
# 1. RICERCA E RACCOLTA DATI (TUTTI I GIORNI E LE RUN DEL BLAZAR)
# =============================================================================
print("--- INIZIO RICERCA IMMAGINI BLAZAR ---")

# imposto la cartella radice dove si trovano i FITS grezzi
dir_dati = BASE_DIR / "PMC_DATA_BLAZAR"

if not dir_dati.exists():
    print(f"ERRORE: Impossibile trovare la cartella dati {dir_dati}")
    exit()

tutti_file_fits = []
nomi_run_processate = []

# esploro la cartella dei giorni
for giorno_dir in sorted([d for d in dir_dati.iterdir() if d.is_dir()]):
    giorno_nome = giorno_dir.name

    # esploro le run all'interno del giorno
    for run_dir in sorted([d for d in giorno_dir.iterdir() if d.is_dir()]):
        run_nome = run_dir.name

        # cerco tutti i file FITS all'interno della cartella della run e li ordino alfabeticamente
        estensioni_valide = ['.fit', '.fits']
        file_run = sorted([str(f) for f in run_dir.rglob('*') if f.suffix.lower() in estensioni_valide and f.is_file()])

        # salto la prima e le ultime due immagini della singola run per evitare scarti
        if len(file_run) > 3:
            file_run_validi = file_run[1:-2]
            tutti_file_fits.extend(file_run_validi)
            nomi_run_processate.append(f"{giorno_nome}/{run_nome}")

if not tutti_file_fits:
    print("ERRORE: Nessun file FITS valido trovato per il blazar!")
    exit()

print(f"Trovate {len(tutti_file_fits)} immagini valide in {len(nomi_run_processate)} run totali.")

# =============================================================================
# 2. DEFINIZIONE DEL SISTEMA DI RIFERIMENTO E RITAGLIO (MRK 421)
# =============================================================================
# prendo un'immagine centrale (o la prima disponibile) come riferimento
indice_ref = min(12, len(tutti_file_fits) - 1)
print(f"Caricamento riferimento: {Path(tutti_file_fits[indice_ref]).name}")

hdu_ref = fits.open(tutti_file_fits[indice_ref])[0]
target_header_full = hdu_ref.header.copy()

# aggiungo relax=True per leggere i polinomi di distorsione senza warning
target_wcs_full = WCS(target_header_full, relax=True)

# imposto le coordinate esatte del blazar Markarian 421
coord_mrk421 = SkyCoord('11h04m27.31s', '+38d12m31.8s', frame='icrs')

# imposto la dimensione del riquadro richiesta (1.6x1.6 minuti d'arco)
dimensione_riquadro = u.Quantity((1.6, 1.6), u.arcmin)

# eseguo il ritaglio sull'immagine di riferimento per ottenere le coordinate target ristrette
print("Calcolo il riquadro di 1.6x1.6 arcmin centrato su Markarian 421...")
try:
    # uso mode='partial' nel caso il blazar sia vicino ai bordi dell'immagine
    ritaglio_ref = Cutout2D(hdu_ref.data, coord_mrk421, dimensione_riquadro, wcs=target_wcs_full, mode='partial')
except Exception as e:
    raise ValueError(f"Errore durante il ritaglio: {e}")

# estraggo il WCS e la forma limitati al solo riquadro
target_wcs = ritaglio_ref.wcs
target_shape = ritaglio_ref.shape

# aggiorno l'header con le nuove informazioni astrometriche ridotte
target_header = target_wcs.to_header()
target_header['DATE-OBS'] = target_header_full.get('DATE-OBS', 'UNKNOWN')

# creo la matrice finale (accumulatore) piena di zeri
final_image_sum = np.zeros(target_shape)
coverage_map = np.zeros(target_shape)

print(f"Inizio stacking globale...")
print(f"Dimensioni target (ritagliate): {target_shape} pixel")

# =============================================================================
# 3. LOOP DI RIPROIEZIONE E STACKING GLOBALE
# =============================================================================
for percorso_file_fits in tqdm(tutti_file_fits, desc="Stacking", unit="img"):
    try:
        with fits.open(percorso_file_fits) as hdu_list:
            data = hdu_list[0].data
            header = hdu_list[0].header

            # aggiungo relax=True anche qui per gestire le distorsioni dell'immagine corrente
            wcs_input = WCS(header, relax=True)

            mean, median, std = sigma_clipped_stats(data, sigma=3.0)

            # Sottraggo il fondo
            data_sub = data - median

            # riproietto direttamente sul target_wcs ristretto
            array_reprojected, footprint = reproject_interp(
                (data_sub, wcs_input),
                target_wcs,
                shape_out=target_shape
            )

            # converto i NaN in zeri per permettere la somma matriciale
            array_reprojected = np.nan_to_num(array_reprojected, nan=0.0)
            final_image_sum += array_reprojected
            coverage_map += np.nan_to_num(footprint, nan=0.0)

    except Exception as e:
        tqdm.write(f"Errore nel file {Path(percorso_file_fits).name}: {e}")

# =============================================================================
# 4. SCALATURA E SALVATAGGIO
# =============================================================================
max_coverage = np.max(coverage_map)

# calcolo e applico il mio fattore di scala per ottenere il flusso medio (come una singola immagine)
scale_factor_map = np.zeros_like(coverage_map, dtype=float)
np.divide(1.0, coverage_map,
          out=scale_factor_map,
          where=coverage_map > 0)

final_image_sum = final_image_sum * scale_factor_map

# imposto la cartella di output
output_dir = BASE_DIR / 'pmc_photometry' / 'blazar' / 'stacking' / 'stacking_mrk421'
output_dir.mkdir(parents=True, exist_ok=True)

output_filename = output_dir / 'stacked_sum_mrk421_globale.fits'
header_finale = target_header.copy()
header_finale[
    'HISTORY'] = f'Stacking 1.6x1.6 arcmin su Mrk 421 ({len(tutti_file_fits)} immagini da {len(nomi_run_processate)} run)'

fits.writeto(str(output_filename), final_image_sum, header_finale, overwrite=True)
print(f"\nFatto! Immagine salvata come: {output_filename.name}")

coverage_filename = output_dir / 'coverage_map_mrk421_globale.fits'
header_coverage = target_header.copy()
header_coverage['HISTORY'] = 'Mappa di copertura globale (numero di immagini per pixel)'

fits.writeto(str(coverage_filename), coverage_map, header_coverage, overwrite=True)
print(f"Fatto! Coverage map salvata come: {coverage_filename.name}")

# =============================================================================
# 5. VISUALIZZAZIONE VELOCE
# =============================================================================
norm = simple_norm(final_image_sum, 'sqrt')
plt.figure(figsize=(8, 8))
ax = plt.subplot(projection=target_wcs)

# genero l'immagine impostando origin='lower' per rispettare gli standard astronomici
ax.imshow(final_image_sum, origin='lower', norm=norm, cmap='viridis')

# cerco la prima tabella csv in tabelle_blazar
dir_tabelle = BASE_DIR / "tabelle_blazar"
if dir_tabelle.exists():
    tabelle_csv = sorted(list(dir_tabelle.rglob("*.csv")))
    if tabelle_csv:
        prima_tabella = tabelle_csv[0]
        print(f"Sovrappongo lo scatter della tabella: {prima_tabella.name}")

        # leggo i dati del catalogo
        df_cat = pd.read_csv(prima_tabella, comment='#')

        # identifico le colonne delle coordinate astronomiche (uso un fallback standard)
        col_ra = 'RAJ2000' if 'RAJ2000' in df_cat.columns else (
            'RA_centroid' if 'RA_centroid' in df_cat.columns else 'RA')
        col_dec = 'DEJ2000' if 'DEJ2000' in df_cat.columns else (
            'DEC_centroid' if 'DEC_centroid' in df_cat.columns else 'DEC')

        if col_ra in df_cat.columns and col_dec in df_cat.columns:
            # applico il transform='world' per allineare direttamente RA/DEC all'immagine
            ax.scatter(df_cat[col_ra], df_cat[col_dec], transform=ax.get_transform('world'),
                       s=4, color='red', label='Catalogo', zorder=10)
            plt.legend()

plt.xlabel('RA')
plt.ylabel('Dec')
plt.title(f'Stacking Globale Mrk 421 (1.6x1.6 arcmin)\nCopertura max: {int(max_coverage)} immagini')

output_png = output_dir / 'stacking_mrk421_globale.png'
plt.savefig(str(output_png))

# =============================================================================
# 6. STACKING SEPARATO PER ANNO (2025 E 2026)
# =============================================================================
print("\n--- INIZIO STACKING PER ANNI SEPARATI ---")

# filtro le liste dei file basandomi sul nome della cartella del giorno
file_fits_2025 = [f for f in tutti_file_fits if Path(f).parents[1].name.startswith('2025')]
file_fits_2026 = [f for f in tutti_file_fits if Path(f).parents[1].name.startswith('2026')]

# creo un dizionario per ciclare comodamente sui due anni
dizionari_anni = {
    '2025': file_fits_2025,
    '2026': file_fits_2026
}

for anno, lista_file_anno in dizionari_anni.items():
    if not lista_file_anno:
        print(f"Nessun file trovato per l'anno {anno}, salto lo stacking.")
        continue

    print(f"\nElaborazione anno {anno} ({len(lista_file_anno)} immagini)...")

    # inizializzo le matrici vuote specifiche per l'anno corrente
    final_image_sum_anno = np.zeros(target_shape)
    coverage_map_anno = np.zeros(target_shape)

    for percorso_file_fits in tqdm(lista_file_anno, desc=f"Stacking {anno}", unit="img"):
        try:
            with fits.open(percorso_file_fits) as hdu_list:
                data = hdu_list[0].data
                header = hdu_list[0].header

                # leggo il wcs con relax=True
                wcs_input = WCS(header, relax=True)

                mean, median, std = sigma_clipped_stats(data, sigma=3.0)

                # Sottraggo il fondo
                data_sub = data - median

                # riproietto
                array_reprojected, footprint = reproject_interp(
                    (data_sub, wcs_input),
                    target_wcs,
                    shape_out=target_shape
                )

                # converto e sommo
                array_reprojected = np.nan_to_num(array_reprojected, nan=0.0)
                final_image_sum_anno += array_reprojected
                coverage_map_anno += np.nan_to_num(footprint, nan=0.0)

        except Exception as e:
            tqdm.write(f"Errore nel file {Path(percorso_file_fits).name}: {e}")

    # scalo la mia immagine basandomi sulla copertura per ottenere il flusso medio
    max_coverage_anno = np.max(coverage_map_anno)
    scale_factor_map_anno = np.zeros_like(coverage_map_anno, dtype=float)
    np.divide(1.0, coverage_map_anno,
              out=scale_factor_map_anno,
              where=coverage_map_anno > 0)

    final_image_sum_anno = final_image_sum_anno * scale_factor_map_anno

    # salvo l'immagine FITS
    output_filename_anno = output_dir / f'stacked_sum_mrk421_{anno}.fits'
    header_finale_anno = target_header.copy()
    header_finale_anno['HISTORY'] = f'Stacking 1.6x1.6 arcmin su Mrk 421 ({len(lista_file_anno)} immagini dell\'anno {anno})'

    fits.writeto(str(output_filename_anno), final_image_sum_anno, header_finale_anno, overwrite=True)
    print(f"Fatto! Immagine {anno} salvata come: {output_filename_anno.name}")

    # salvo la mappa di copertura FITS
    coverage_filename_anno = output_dir / f'coverage_map_mrk421_{anno}.fits'
    header_coverage_anno = target_header.copy()
    header_coverage_anno['HISTORY'] = f'Mappa di copertura anno {anno}'

    fits.writeto(str(coverage_filename_anno), coverage_map_anno, header_coverage_anno, overwrite=True)
    print(f"Fatto! Coverage map {anno} salvata come: {coverage_filename_anno.name}")

    # preparo la visualizzazione veloce
    norm_anno = simple_norm(final_image_sum_anno, 'sqrt')
    plt.figure(figsize=(8, 8))
    ax_anno = plt.subplot(projection=target_wcs)

    # genero l'immagine
    ax_anno.imshow(final_image_sum_anno, origin='lower', norm=norm_anno, cmap='viridis')

    # aggiungo lo scatter rosso del catalogo se precedentemente calcolato e disponibile
    if dir_tabelle.exists() and tabelle_csv:
        if col_ra in df_cat.columns and col_dec in df_cat.columns:
            ax_anno.scatter(df_cat[col_ra], df_cat[col_dec], transform=ax_anno.get_transform('world'),
                       s=4, color='red', label='Catalogo', zorder=10)
            plt.legend()

    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title(f'Stacking Mrk 421 - Anno {anno} (1.6x1.6 arcmin)\nCopertura max: {int(max_coverage_anno)} immagini')

    output_png_anno = output_dir / f'stacking_mrk421_{anno}.png'
    plt.savefig(str(output_png_anno))