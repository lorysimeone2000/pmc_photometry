import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent

BASE_DIR = trova_cartella_base("pmc_photometry")

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Importo solo le utilità necessarie
from funzioni.utilita import leggi_header_da_csv, cerca_file_nel_progetto

print(f"--- CONFIGURAZIONE PLOT BLAZAR ---")
print(f"Cartella Base rilevata: {BASE_DIR}")

# =============================================================================
# 1. RICERCA ESATTA DEL 34° FILE (1° GIORNO, 1° RUN)
# =============================================================================

# Definisco le cartelle radice per unite e cataloghi
dir_unite = BASE_DIR / "blazar" / "tabelle" / "tabelle_unite"
dir_cataloghi = BASE_DIR / "blazar" / "tabelle" / "tabelle_cataloghi"

if not dir_unite.exists() or not dir_cataloghi.exists():
    print("ERRORE: Le cartelle delle tabelle (unite o cataloghi) non esistono.")
    exit()

# Trovo il primo giorno
giorni_unite = sorted([d for d in dir_unite.iterdir() if d.is_dir()])
giorni_cat = sorted([d for d in dir_cataloghi.iterdir() if d.is_dir()])
giorno_1_unite = giorni_unite[0]
giorno_1_cat = giorni_cat[0]

# Trovo la prima run del primo giorno
run_unite = sorted([d for d in giorno_1_unite.iterdir() if d.is_dir()])
run_cat = sorted([d for d in giorno_1_cat.iterdir() if d.is_dir()])
run_1_unite = run_unite[0]
run_1_cat = run_cat[0]

# Trovo tutti i CSV della run e seleziono il 34° (indice 33 in Python)
csv_unite_list = sorted(list(run_1_unite.glob("*.csv")))
csv_cat_list = sorted(list(run_1_cat.glob("*.csv")))

if len(csv_unite_list) < 34 or len(csv_cat_list) < 34:
    print("ERRORE: Non ci sono abbastanza file CSV (minimo 34) nella cartella selezionata.")
    exit()

csv_unite = csv_unite_list[33]
csv_cat = csv_cat_list[33]

print(f"\nSelezionati i seguenti file per il plot:")
print(f"- Unite: {csv_unite.relative_to(BASE_DIR)}")
print(f"- Cataloghi: {csv_cat.relative_to(BASE_DIR)}")

# =============================================================================
# 2. LETTURA DATI E RICERCA FITS ORIGINALE
# =============================================================================

# Leggo i DataFrame
df_unite = pd.read_csv(csv_unite, comment='#')
df_cat = pd.read_csv(csv_cat, comment='#')
header_info = leggi_header_da_csv(csv_unite)

# Trovo il file FITS basandomi sull'header usando la stessa logica robusta
path_fits = header_info.get('PERCORSO_FILE', '')
nome_fits = header_info.get('NOME_FILE_FITS', '')

if not path_fits or not os.path.exists(path_fits):
    if path_fits:
        p_obj = Path(path_fits)
        try:
            if "pmc_photometry" in p_obj.parts:
                idx_part = p_obj.parts.index("pmc_photometry")
                new_path = BASE_DIR.joinpath(*p_obj.parts[idx_part + 1:])
                if new_path.exists(): path_fits = str(new_path)
        except: pass

    if (not path_fits or not os.path.exists(path_fits)) and nome_fits:
        found = cerca_file_nel_progetto(BASE_DIR, str(nome_fits).strip())
        if found: path_fits = str(found)

if not path_fits or not os.path.exists(path_fits):
    print(f"ERRORE: File FITS '{nome_fits}' non trovato. Impossibile procedere.")
    exit()

# Apro il FITS, calcolo il fondo e lo sottraggo
with fits.open(path_fits, memmap=False) as hdu:
    w = WCS(hdu[0].header)
    image_data = hdu[0].data.astype(float)
    _, median_bg, _ = sigma_clipped_stats(image_data[::10, ::10], sigma=3.0)
    print("Mediana: ", median_bg)
    data_pmc = image_data - median_bg

# =============================================================================
# 3. PREPARAZIONE DATI E LIMITI CAMPO VISIVO
# =============================================================================

# Trovo gli estremi (WCS)
alto_destra = w.pixel_to_world(3072, 2048)
alto_sinistra = w.pixel_to_world(3072, 0)
basso_sinistra = w.pixel_to_world(0, 0)
basso_destra = w.pixel_to_world(0, 2048)

ra_max = np.max([alto_destra.ra.deg, basso_sinistra.ra.deg, basso_destra.ra.deg, alto_sinistra.ra.deg])
ra_min = np.min([alto_destra.ra.deg, basso_sinistra.ra.deg, basso_destra.ra.deg, alto_sinistra.ra.deg])
dec_max = np.max([alto_destra.dec.deg, basso_sinistra.dec.deg, basso_destra.dec.deg, alto_sinistra.dec.deg])
dec_min = np.min([alto_destra.dec.deg, basso_sinistra.dec.deg, basso_destra.dec.deg, alto_sinistra.dec.deg])

# =============================================================================
# 4. PRIMO GRAFICO: MAPPA CELESTE DEL CATALOGO
# =============================================================================

magnitudini_cat = df_cat['Mag']
sizes = 15 * (8 - magnitudini_cat)
sizes = np.clip(sizes, 10, 200)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_cat['RAJ2000'], df_cat['DEJ2000'],
                      c=magnitudini_cat, s=sizes, alpha=0.7, cmap='viridis_r')
plt.colorbar(scatter, label='Magnitudine (Catalogo)')

plt.gca().invert_xaxis()  # RA aumenta verso est
plt.grid(True, alpha=0.3)
plt.xlim(ra_max, ra_min)
plt.ylim(dec_min, dec_max)
plt.xlabel('Ascensione Retta (RA J2000) [gradi]')
plt.ylabel('Declinazione (DEC J2000) [gradi]')
plt.title(f'Mappa del catalogo combinato ({len(df_cat)} stelle), FOV {nome_fits}')
plt.tight_layout()
plt.show()

# =============================================================================
# 5. SECONDO GRAFICO: OVERLAY SU IMMAGINE FITS
# =============================================================================

# Trasformo in coordinate pixel gli oggetti di catalogo e ne dimensiono i cerchi
coords_cat = SkyCoord(ra=df_cat['RAJ2000'].values * u.deg, dec=df_cat['DEJ2000'].values * u.deg)
pos_cat_x, pos_cat_y = w.world_to_pixel(coords_cat)
posizioni_vere_pixel = np.column_stack((pos_cat_x, pos_cat_y))

raggio_min = 4.0
raggio_max = 20.0

if len(magnitudini_cat) > 1 and magnitudini_cat.max() != magnitudini_cat.min():
    raggi = raggio_max - (magnitudini_cat - magnitudini_cat.min()) * (raggio_max - raggio_min) / (magnitudini_cat.max() - magnitudini_cat.min())
else:
    raggi = np.full(len(magnitudini_cat), 10.0)

cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=magnitudini_cat.min(), vmax=magnitudini_cat.max())

fig = plt.figure(figsize=(12, 8))
ax = plt.subplot()

# Plot immagine FITS: applico un clip ai valori negativi (dovuti alla sottrazione del fondo) per evitare problemi con LogNorm
img_pmc = ax.imshow(np.clip(data_pmc, a_min=1e-3, a_max=None), cmap='gray_r', origin='lower', norm=LogNorm(), interpolation='nearest')

# Aggiungo la colorbar per l'intensità dei pixel (legata all'immagine)
cbar_img = plt.colorbar(img_pmc, ax=ax, fraction=0.046, pad=0.04, label='Intensità Pixel (ADU)')

# Overlay cerchi catalogo
for i, (position, radius) in enumerate(zip(posizioni_vere_pixel, raggi)):
    color = cmap(norm(magnitudini_cat.iloc[i]))
    aperture = CircularAperture(position, r=radius)
    aperture.plot(color=color, lw=1.0, alpha=0.6, fill=True)

# Aggiungo la colorbar per le magnitudini (legata ai cerchi)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_mag = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.08, label='Magnitudine Catalogo')

# Plot dei centroidi estratti (le sorgenti effettivamente identificate nello step precedente)
posizioni_trovate = np.column_stack((df_unite['xcentroid'].values, df_unite['ycentroid'].values))
apertures_trovate = CircularAperture(posizioni_trovate, r=5.0)
apertures_trovate.plot(color='red', lw=1.)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Matching {nome_fits}\nStelle catalogo (dimensionate per Mag) vs Sorgenti Rilevate')

# Legenda customizzata
legend_elements = [
    Circle((0.5, 0.5), 0.4, facecolor='blue', alpha=0.7, edgecolor='black', linewidth=1,
          label=f'Stelle catalogo ({len(df_cat)} oggetti)'),
    Line2D([0], [0], marker='o', color='red', linestyle='None',
           markersize=8, markerfacecolor='none', markeredgewidth=1,
           label=f'Sorgenti rilevate ({len(df_unite)} oggetti)')
]

ax.legend(handles=legend_elements, loc='upper right', framealpha=0.85, fancybox=True, shadow=True)

plt.tight_layout()
plt.show()