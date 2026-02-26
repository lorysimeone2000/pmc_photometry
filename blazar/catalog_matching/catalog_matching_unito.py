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

# Disabilito i warning per pulizia nell'output
warnings.filterwarnings('ignore')

# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E IMPORTAZIONE MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent

BASE_DIR = trova_cartella_base("pmc_photometry")

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from funzioni.utilita import leggi_header_da_csv, cerca_file_nel_progetto

# =============================================================================
# 1. RICERCA FILE E LETTURA DATI
# =============================================================================

dir_unite = BASE_DIR / "blazar" / "tabelle" / "tabelle_unite"
dir_cataloghi = BASE_DIR / "blazar" / "tabelle" / "tabelle_cataloghi"

giorno_1_unite = sorted([d for d in dir_unite.iterdir() if d.is_dir()])[0]
run_1_unite = sorted([d for d in giorno_1_unite.iterdir() if d.is_dir()])[0]
csv_unite = sorted(list(run_1_unite.glob("*.csv")))[33]

giorno_1_cat = sorted([d for d in dir_cataloghi.iterdir() if d.is_dir()])[0]
run_1_cat = sorted([d for d in giorno_1_cat.iterdir() if d.is_dir()])[0]
csv_cat = sorted(list(run_1_cat.glob("*.csv")))[33]

df_unite = pd.read_csv(csv_unite, comment='#')
df_cat = pd.read_csv(csv_cat, comment='#')
df_cat = df_cat[df_cat["Mag"] <= 12]
header_info = leggi_header_da_csv(csv_unite)

# Ricerca del file FITS
path_fits = header_info.get('PERCORSO_FILE', '')
nome_fits = header_info.get('NOME_FILE_FITS', '')

if not path_fits or not os.path.exists(path_fits):
    found = cerca_file_nel_progetto(BASE_DIR, str(nome_fits).strip())
    if found: path_fits = str(found)

# Apertura FITS e WCS
with fits.open(path_fits) as hdu:
    header = hdu[0].header
    w = WCS(header)
    image_data = hdu[0].data.astype(float)
    _, median_bg, _ = sigma_clipped_stats(image_data[::10, ::10], sigma=3.0)
    data_pmc = image_data - median_bg

# =============================================================================
# 2. CALCOLO RAGGIO APERTURA IN PIXEL (35 arcsec)
# =============================================================================

# Estraggo la scala dei pixel (gradi per pixel) dalla matrice WCS
# Uso np.mean per gestire eventuali asimmetrie tra gli assi
pixel_scale_deg = np.mean(np.abs(w.pixel_scale_matrix.diagonal()))
pixel_scale_arcsec = pixel_scale_deg * 3600.0

# Calcolo il raggio corrispondente a 35 arcosecondi
raggio_arcsec = 35.0
raggio_pixel = raggio_arcsec / pixel_scale_arcsec

print(f"Scala pixel: {pixel_scale_arcsec:.3f} arcsec/px")
print(f"Raggio apertura (35 arcsec) in pixel: {raggio_pixel:.2f} px")

# =============================================================================
# 3. GENERAZIONE PLOT
# =============================================================================

coords_cat = SkyCoord(ra=df_cat['RAJ2000'].values * u.deg, dec=df_cat['DEJ2000'].values * u.deg)
pos_cat_x, pos_cat_y = w.world_to_pixel(coords_cat)

magnitudini_cat = df_cat['Mag']
cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=magnitudini_cat.min(), vmax=magnitudini_cat.max())

fig, ax = plt.subplots(figsize=(12, 8))

# Immagine di fondo
img_pmc = ax.imshow(np.clip(data_pmc, a_min=1e-3, a_max=None), cmap='gray_r',
                   origin='lower', norm=LogNorm(), interpolation='nearest')

# Colorbar per ADU
plt.colorbar(img_pmc, ax=ax, fraction=0.046, pad=0.04, label='Intensità Pixel (ADU)')

# Overlay stelle catalogo
scatter_cat = ax.scatter(pos_cat_x, pos_cat_y, c=magnitudini_cat, s=10,
                        cmap=cmap, norm=norm, alpha=0.6, zorder=5)

# Colorbar per Magnitudine
plt.colorbar(scatter_cat, ax=ax, fraction=0.046, pad=0.08, label='Magnitudine Catalogo')

# Plot dei centroidi con raggio impostato a 35 arcsec (in pixel)
posizioni_trovate = np.column_stack((df_unite['xcentroid'].values, df_unite['ycentroid'].values))
apertures_trovate = CircularAperture(posizioni_trovate, r=raggio_pixel)
apertures_trovate.plot(color='red', lw=1.2, zorder=6)

ax.set_xlabel('x [pixel]')
ax.set_ylabel('y [pixel]')
ax.set_title(f'Matching {nome_fits}\nCerchi rossi: aperture da {raggio_arcsec}" ({raggio_pixel:.2f} px)')

# Legenda
legend_elements = [
    Line2D([0], [0], marker='o', color='yellow', linestyle='None', markersize=6, label='Stelle Catalogo'),
    Line2D([0], [0], marker='o', color='none', markeredgecolor='red', linestyle='None',
           markersize=10, markeredgewidth=1.2, label=f'Oggetti rilevati')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.savefig("catalog_matching_unito.png", bbox_inches='tight')

plt.tight_layout()
plt.show()