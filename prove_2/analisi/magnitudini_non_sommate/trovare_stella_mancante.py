import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.colors import LogNorm
from astropy.wcs.utils import proj_plane_pixel_scales
from pathlib import Path
import warnings
from astropy.wcs import FITSFixedWarning
from astroquery.vizier import Vizier
from astropy.table import Table

# ignoro i warning FITS noti che non intralciano l'analisi
warnings.filterwarnings('ignore', category=FITSFixedWarning)

# --- FUNZIONI E CONFIGURAZIONE ---

# risalgo l'albero delle directory per trovare la cartella base del progetto
def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent

# definisco la funzione per ottenere le magnitudini da VizieR
def ottieni_magnitudini_vizier_con_coordinate(ra, dec, id_stella, raggio_arcsec=2.0):
    # preparo il dizionario con i valori nulli di default
    dati_stella = {
        'ID': id_stella,
        'gmag': np.nan,
        'rmag': np.nan,
        'imag': np.nan,
        'zmag': np.nan,
        'ymag': np.nan
    }

    try:
        # configuro Vizier per il catalogo Pan-STARRS
        vizier_ps1 = Vizier(
            catalog="II/389/ps1_dr2",
            columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
            row_limit=-1
        )

        # ricerco nella regione attorno alle coordinate
        result = vizier_ps1.query_region(
            SkyCoord(ra=ra, dec=dec, unit=u.deg),
            radius=raggio_arcsec * u.arcsec
        )

        if len(result) > 0:
            tabella_res = result[0]

            # converto l'ID in intero per il confronto
            try:
                id_intero = int(float(id_stella))
            except (ValueError, TypeError):
                id_intero = None

            # cerco la riga con l'ID corrispondente
            if id_intero is not None and 'objID' in tabella_res.colnames:
                for riga in tabella_res:
                    try:
                        if int(riga['objID']) == id_intero:
                            for banda in ['gmag', 'rmag', 'imag', 'zmag', 'ymag']:
                                if banda in riga.colnames and not pd.isna(riga[banda]):
                                    dati_stella[banda] = riga[banda]
                            return dati_stella
                    except (ValueError, TypeError):
                        continue

            # se non trovo corrispondenza per ID, prendo la stella più vicina
            riga = tabella_res[0]
            for banda in ['gmag', 'rmag', 'imag', 'zmag', 'ymag']:
                if banda in riga.colnames and not pd.isna(riga[banda]):
                    dati_stella[banda] = riga[banda]

    except Exception as e:
        print(f"  Attenzione: Errore per stella {id_stella} a coordinate ({ra:.6f}, {dec:.6f}): {e}")

    return dati_stella


# imposto la base directory
BASE_DIR = trova_cartella_base("Lorenzo")

# imposto la run che voglio analizzare
run = 1

# definisco i percorsi delle due cartelle contenenti i csv che mi interessano
cartella_cataloghi = BASE_DIR / "tabelle" / "sorgenti_catalogate_run" / f"sorgenti_catalogate_run_{run}"
cartella_unite = BASE_DIR / "tabelle" / "tabelle_unite" / f"tabelle_unite_run_{run}"

# --- CARICAMENTO DATI ---

# controllo che le cartelle esistano
if not cartella_cataloghi.exists() or not cartella_unite.exists():
    print(f"Errore: Cartelle dati non trovate in {BASE_DIR}")
    exit()

# prendo il primo file CSV del catalogo disponibile
file_catalogo_list = sorted(list(cartella_cataloghi.glob("*.csv")))
if not file_catalogo_list:
    print(f"Errore: Nessun file CSV trovato in {cartella_cataloghi}")
    exit()
file_catalogo = file_catalogo_list[0]

# ricavo dinamicamente il nome del file corrispondente in tabelle_unite
nome_file_unito = file_catalogo.name.replace("stelle_catalogate", "stelle_trovate_e_catalogate")
file_unito = cartella_unite / nome_file_unito

# leggo i due dataframe ignorando l'header FITS e forzando low_memory=False
df_cat = pd.read_csv(file_catalogo, comment='#', low_memory=False)
df_trovate = pd.read_csv(file_unito, comment='#', low_memory=False)

# cerco il file FITS corrispondente
cartella_run_list = list(BASE_DIR.rglob(f"20250120_run{run}"))
if not cartella_run_list:
    print(f"Errore: Cartella run FITS non trovata.")
    exit()
cartella_run = cartella_run_list[0]

# estraggo il numero progressivo dell'immagine dal nome del file CSV
try:
    num_img = int(file_catalogo.stem.split('_')[-1])
except (ValueError, IndexError):
    num_img = 1

file_fits_list = sorted([f for ext in ['*.fit', '*.fits', '*.FIT', '*.FITS'] for f in cartella_run.glob(ext)])
if not file_fits_list or num_img > len(file_fits_list):
    print(f"Errore: File FITS numero {num_img} non trovato in {cartella_run}")
    exit()
percorso_fits = str(file_fits_list[num_img - 1])

# --- ELABORAZIONE DATI FITS E COORDINATE ---

# apro il file FITS e carico i dati e il WCS
hdu = fits.open(percorso_fits)
data = hdu[0].data
header = hdu[0].header
w = WCS(header)

# estraggo gli ID delle stelle del catalogo che hanno trovato una corrispondenza
mask_si = df_trovate['Corrispondenza'].str.startswith('SI', na=False)
id_trovati = df_trovate.loc[mask_si, 'ID'].values

# filtro il catalogo originario per isolare le stelle che NON ho trovato
df_mancanti = df_cat[~df_cat['ID'].isin(id_trovati)]

if df_mancanti.empty:
    print("Nessuna stella del catalogo risulta mancante in questa immagine.")
    hdu.close()
    exit()

# trovo la stella mancante più luminosa prendendo l'indice con il valore Mag più basso
df_mancanti['Mag_num'] = pd.to_numeric(df_mancanti['Mag'], errors='coerce')
df_mancanti_pulite = df_mancanti.dropna(subset=['Mag_num', 'RAJ2000', 'DEJ2000'])

if df_mancanti_pulite.empty:
    print("Nessuna stella mancante ha coordinate e magnitudine valide.")
    hdu.close()
    exit()

stella_piu_luminosa = df_mancanti_pulite.loc[df_mancanti_pulite['Mag_num'].idxmin()]
ra_stella = float(stella_piu_luminosa['RAJ2000'])
dec_stella = float(stella_piu_luminosa['DEJ2000'])
mag_stella = float(stella_piu_luminosa['Mag_num'])
id_stella = stella_piu_luminosa['ID']

print(f"Analisi Immagine: {Path(percorso_fits).name}")
print(f"Stella mancante più luminosa: ID={id_stella}, Mag={mag_stella:.2f}, RA={ra_stella:.5f}, DEC={dec_stella:.5f}")

# --- RICERCA MAGNITUDINI SU VIZIER E GENERAZIONE TABELLA ---
print("\nRicerca delle magnitudini su VizieR (Pan-STARRS DR2)...")
dati_mag = ottieni_magnitudini_vizier_con_coordinate(ra_stella, dec_stella, id_stella)

# creo il dataframe per organizzare la visualizzazione
df_risultati_mag = pd.DataFrame([dati_mag])

# aggiungo coordinate e magnitudine sintetica di riferimento
df_risultati_mag.insert(1, 'Mag_Sintetica', mag_stella)
df_risultati_mag.insert(2, 'RAJ2000', ra_stella)
df_risultati_mag.insert(3, 'DEJ2000', dec_stella)

# converto in tabella Astropy e stampo a video
tabella_astropy_mag = Table.from_pandas(df_risultati_mag)
print("\n" + "=" * 100)
print("TABELLA MAGNITUDINI DELLA STELLA MANCANTE")
print("=" * 100)
print(tabella_astropy_mag)
print("=" * 100 + "\n")

# --- CONTINUO PREPARAZIONE PER IL PLOTTING ---

# converto le coordinate celesti della stella mancante in coordinate pixel per centrare il ritaglio
coord_stella = SkyCoord(ra=ra_stella*u.deg, dec=dec_stella*u.deg, frame='icrs')
x_pix_targ, y_pix_targ = w.world_to_pixel(coord_stella)
x_pix_targ, y_pix_targ = int(round(float(x_pix_targ))), int(round(float(y_pix_targ)))

# definisco il raggio in pixel del mio ritaglio
raggio_ritaglio = 45

# calcolo i confini del riquadro
y_min, y_max = max(0, y_pix_targ - raggio_ritaglio), min(data.shape[0], y_pix_targ + raggio_ritaglio)
x_min, x_max = max(0, x_pix_targ - raggio_ritaglio), min(data.shape[1], x_pix_targ + raggio_ritaglio)

# eseguo materialmente il ritaglio dell'array dati
ritaglio = data[y_min:y_max, x_min:x_max]

# --- PREPARAZIONE COORDINATE PER SOVRAPPOSIZIONE ---

# 1. Coordinate TUTTO IL CATALOGO
ra_cat = pd.to_numeric(df_cat['RAJ2000'], errors='coerce').values
dec_cat = pd.to_numeric(df_cat['DEJ2000'], errors='coerce').values
mag_cat = pd.to_numeric(df_cat['Mag'], errors='coerce').values
mask_valid_cat = ~np.isnan(ra_cat) & ~np.isnan(dec_cat) & ~np.isnan(mag_cat)
coords_cat_all = SkyCoord(ra=ra_cat[mask_valid_cat]*u.deg, dec=dec_cat[mask_valid_cat]*u.deg, frame='icrs')
mag_cat_all = mag_cat[mask_valid_cat]

x_cat_all, y_cat_all = w.world_to_pixel(coords_cat_all)
# traslo per adattarle al ritaglio
x_cat_cut_all = x_cat_all - x_min
y_cat_cut_all = y_cat_all - y_min

# 2. Coordinate CENTROIDI TROVATI
df_trovate_uniche = df_trovate.drop_duplicates(subset=['RA_centroid', 'DEC_centroid'])
ra_trovate = pd.to_numeric(df_trovate_uniche['RA_centroid'], errors='coerce').dropna().values
dec_trovate = pd.to_numeric(df_trovate_uniche['DEC_centroid'], errors='coerce').dropna().values

coords_trovate = SkyCoord(ra=ra_trovate*u.deg, dec=dec_trovate*u.deg, frame='icrs')
x_trovate, y_trovate = w.world_to_pixel(coords_trovate)
# traslo per adattarle al ritaglio
x_trovate_cut = x_trovate - x_min
y_trovate_cut = y_trovate - y_min

# --- PLOTTING ---

fig, ax = plt.subplots(figsize=(9, 8))

# 1. Sfondo: mostro il ritaglio FITS in scala logaritmica
ax.imshow(ritaglio, cmap='gray', origin='lower', norm=LogNorm(), zorder=1)

# 2. Sovrappongo lo scatterplot di TUTTE le stelle del catalogo
sc = ax.scatter(x_cat_cut_all, y_cat_cut_all, c=mag_cat_all, cmap='viridis_r', s=20, alpha=0.8, zorder=2, label='Stelle Catalogo')

# 3. Disegno croci e cerchi di correlazione sui centroidi trovati
ax.scatter(x_trovate_cut, y_trovate_cut, marker='+', color='red', s=60, linewidth=1.2, zorder=3, label='Centroidi Segmentation')

# calcolo la scala dell'immagine in gradi per pixel
scala_pixel = proj_plane_pixel_scales(w)

# converto il raggio di 35 arcosecondi in pixel
raggio_35_arcsec_gradi = 35.0 / 3600.0
raggio_correlazione_pixel = raggio_35_arcsec_gradi / np.mean(scala_pixel)

# disegno i cerchietti rossi vuoti sui centroidi
for xt, yt in zip(x_trovate_cut, y_trovate_cut):
    cerchio = plt.Circle((xt, yt), raggio_correlazione_pixel, edgecolor='red', facecolor='none', linewidth=1.5, zorder=4)
    ax.add_patch(cerchio)

# aggiungo un marker invisibile per mantenere la voce del cerchio nella legenda
ax.plot([], [], 'o', markeredgecolor='red', markerfacecolor='none', markersize=10, markeredgewidth=1.5, label='Area Correlazione (35")')

# --- CONFIGURAZIONE ASSI E TITOLI ---

# blocco gli assi esattamente sui bordi del ritaglio
ax.set_xlim(0, ritaglio.shape[1] - 1)
ax.set_ylim(0, ritaglio.shape[0] - 1)

# titoli e legenda
ax.set_title(f"Ritaglio (r={raggio_ritaglio}px) - Img {num_img:03d}\nTarget Mancante: ID {id_stella}, Mag={mag_stella:.2f}")

# raccolgo i label per la legenda
handles, labels = ax.get_legend_handles_labels()
ax.legend(loc='upper right', fontsize='small', framealpha=0.8)

# barra del colore
fig.colorbar(sc, ax=ax, label='Magnitudine Catalogo (Mag)', fraction=0.046, pad=0.04)

# salvo e mostro il plot a schermo
plt.tight_layout()
nome_output = BASE_DIR / f'ritaglio_croci_cerchi_run{run}_img{num_img:03d}.png'
plt.savefig(nome_output, dpi=300, bbox_inches='tight')
print(f"Immagine diagnostica salvata in: {nome_output}")
plt.show()

# chiudo correttamente il file FITS
hdu.close()