import pandas as pd
# pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # mi permette di avere la scala logaritmica
import matplotlib.cm as cm
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
import numpy as np
import os
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture
from astropy.wcs import WCS
from pathlib import Path
from shapely.geometry import Point, Polygon


# =============================================================================
# FUNZIONI DI GESTIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # cerco una cartella specifica ricorsivamente
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # cerco un file specifico ricorsivamente
    files_trovati = list(base_dir.rglob(nome_file_esatto))
    if not files_trovati: return None
    if len(files_trovati) > 1:
        files_trovati.sort(key=lambda p: len(str(p)))
    return files_trovati[0]


def leggi_file_parametri(percorso):
    parametri = {}
    if not os.path.exists(percorso): return {}
    with open(percorso, 'r') as file:
        next(file, None)
        for riga in file:
            riga = riga.split('#')[0].strip()
            if riga:
                parts = riga.split()
                if len(parts) >= 2:
                    try:
                        valore = float(parts[1]) if '.' in parts[1] else int(parts[1])
                        parametri[parts[0]] = valore
                    except ValueError:
                        pass
    return parametri


# --- INIZIO CODICE ---
# imposto la cartella base in modo dinamico
BASE_DIR = trova_cartella_base("Lorenzo")

# cerco e leggo il file dei parametri
file_parametri = cerca_file_nel_progetto(BASE_DIR, 'parametri_image_segmentation.txt')
if file_parametri:
    parametri = leggi_file_parametri(file_parametri)
else:
    print("ERRORE: File parametri non trovato.")
    parametri = {}

RUN_REF = 1

fwhm = parametri.get('fwhm', 2.8)
size = parametri.get('size', 5)
t = parametri.get('threshold_sigma', 3.0)
threshold = parametri.get('threshold_assoluta', 3.0)
n = parametri.get('pixel', 5)

# cerco dinamicamente la cartella tabelle_unite_run_{RUN_REF}
cartella_csv_path = cerca_cartella_nel_progetto(BASE_DIR,
                                                f"tabelle_unite_senza_taglio/tabelle_unite_senza_taglio_run_{RUN_REF}")
if cartella_csv_path is None:
    print(f"ERRORE CRITICO: Cartella 'tabelle_unite_run_{RUN_REF}' non trovata.")
    exit()
cartella_csv = str(cartella_csv_path)

# preparo la lista di tutti i file CSV
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
print("lista:")
print(f"Trovati {len(file_csv)} file CSV:")

i = 0
j = 0
posizioni_lista = []  # lista che dovrà essere riempita con tutte le posizioni di tutte le tabelle
colori_lista = []  # lista per i colori di ogni punto
numeri_immagine_lista = []  # memorizzo l'indice dell'immagine per poterlo stampare accanto a ogni pallino
id_lista = []  # memorizzo l'ID assegnato all'oggetto

# creo una colormap per la sfumatura
colormap = cm.viridis

# itero su tutti i file CSV
for nome_file in file_csv:
    i += 1
    percorso_completo = os.path.join(cartella_csv, nome_file)

    if i <= 2:
        print(f"\n{'=' * 50}")
        print(f"Elaborazione: {nome_file}")
        print(f"{'=' * 50}")

    # leggo il file CSV
    try:
        df = pd.read_csv(percorso_completo, comment="#")
        tbl = Table.from_pandas(df)
        j = j + len(tbl)

        mask_si = np.char.startswith(tbl['Corrispondenza'].astype(str), 'SI')
        tbl_no = tbl[~mask_si]

        posizioni_file = np.transpose(
            (tbl_no['RA_centroid'], tbl_no['DEC_centroid']))  # creo l'array di posizioni per questo file
        posizioni_lista.append(posizioni_file)  # lo aggiungo alla lista totale

        # salvo il numero dell'immagine e l'ID per ciascuna delle posizioni appena trovate
        numeri_immagine_lista.extend([i] * len(posizioni_file))
        id_lista.extend(tbl_no['label'])

        # calcolo il colore per questo file
        if len(file_csv) > 1:
            colore_valore = i / (len(file_csv))
        else:
            colore_valore = 0.5

        colore_rgb = colormap(colore_valore)

        if i < 3:  # Debug
            print(f"Punti: {len(posizioni_file)}")
            print(f"Valore colore: {colore_valore:.3f}")
            print(f"Colore RGB: {colore_rgb[:3]}")

        # aggiungo lo stesso colore per tutti i punti di questo file
        for _ in range(len(posizioni_file)):
            colori_lista.append(colore_rgb)

    except Exception as e:
        print(f"Errore nella lettura di {nome_file}: {e}")

posizioni_array = np.vstack(posizioni_lista)
colori_array = np.array(colori_lista)
numeri_immagine_array = np.array(numeri_immagine_lista)  # converto in array la lista dei numeri immagine
id_array = np.array(id_lista)  # converto in array gli id

print(f"\n{'=' * 60}")
print(f"ARRAY FINALE CREATO")
print(f"{'=' * 60}")
print(f"Dimensioni array posizioni: {posizioni_array.shape}")
print(f"Massimo RA: {np.max(posizioni_array[:, 0])}")
print(f"Massimo DEC: {np.max(posizioni_array[:, 1])}")
print("Ho verificato che l'asse x e l'asse y sono corrispondenti a quelli dell'immagine")

x = []
fov_ra = []
fov_dec = []

# cerco dinamicamente la lista delle immagini
file_lista_immagini = cerca_file_nel_progetto(BASE_DIR, f'lista_immagini_run_{RUN_REF}.txt')
if not file_lista_immagini:
    print("ERRORE: File 'lista_immagini_run_1.txt' non trovato.")
    exit()

# leggo la lista
with open(file_lista_immagini, 'r') as file:
    file_list_raw = file.read().splitlines()  # creo una lista di stringhe

file_list = []
# risolvo dinamicamente i percorsi dei file FITS
for p in file_list_raw:
    p_obj = Path(p)
    if not os.path.exists(p):
        try:
            if "pmc_photometry" in p_obj.parts:
                idx = p_obj.parts.index("pmc_photometry")
                new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                if new_path.exists():
                    file_list.append(str(new_path))
                else:
                    file_list.append(p)
        except:
            file_list.append(p)
    else:
        file_list.append(p)

n = 0
# elaboro tutti i file
for percorso_file in file_list:
    n = n + 1
    try:
        with fits.open(percorso_file) as hdu_list:
            image_header = hdu_list[0].header
            if n == 1:
                x.append(0)
                t1 = image_header["TSTART"]

                # estraggo i confini della fotocamera dalla prima immagine
                w_first = WCS(image_header)
                nx = image_header.get('NAXIS1', 3072)
                ny = image_header.get('NAXIS2', 2048)

                # creo i 4 angoli del sensore per passarlo a Shapely
                corners_pix = np.array([
                    [0, 0],
                    [nx, 0],
                    [nx, ny],
                    [0, ny]
                ])
                corners_world = w_first.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])

                # creo il poligono matematico della camera
                polygon_coords = np.column_stack((corners_world.ra.deg, corners_world.dec.deg))
                poly_fov = Polygon(polygon_coords)

                # estraggo il perimetro esterno chiuso
                fov_ra, fov_dec = poly_fov.exterior.xy

            else:
                x.append((image_header["TSTART"] - t1) / np.float64(1e3))
    except Exception as e:
        print(f"Errore caricamento FITS {percorso_file}: {e}")

x = np.array(x)

# creo la figura impostando la dimensione ottimizzata per 0.45\textwidth in un A4
plt.figure(figsize=(4.5, 4))

plt.scatter(posizioni_array[:, 0], posizioni_array[:, 1],
            s=4,
            alpha=1,
            color=colori_array,
            linewidth=0)

# aggiungo il testo con il numero dell'immagine riducendo notevolmente il font per la figura piccola
for x_val, y_val, num_img in zip(posizioni_array[:, 0], posizioni_array[:, 1], numeri_immagine_array):
    plt.text(x_val, y_val, f"{num_img}", fontsize=4, alpha=0.7, ha='left', va='bottom')

# aggiungo il bordo sottile traducendo l'etichetta
if len(fov_ra) > 0 and len(fov_dec) > 0:
    plt.plot(fov_ra, fov_dec, color='black', linewidth=0.4, linestyle='-', zorder=1, label='Camera Field of View')
    plt.legend(loc='upper right', fontsize=8)

# traduco e dimensiono le etichette degli assi
plt.xlabel('RA (Deg)', fontsize=10)
plt.ylabel('DEC (Deg)', fontsize=10)

# dimensiono i tick per la figura piccola
plt.tick_params(axis='both', which='major', labelsize=8)

plt.grid(True, alpha=0.4, linestyle='--')

# =========================================================
# MODIFICHE ASTRONOMICHE PER RA / DEC
# =========================================================
# 1. Inverto l'asse X. In astronomia, l'Est (RA maggiore) sta a sinistra!
plt.gca().invert_xaxis()

# 2. Correggo l'aspect ratio
dec_media = np.mean(posizioni_array[:, 1])
aspect_ratio = 1.0 / np.cos(np.radians(dec_media))
plt.gca().set_aspect(aspect_ratio)
# =========================================================

# aggiungo la colorbar traducendo l'etichetta e scalando i font
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=np.min(x), vmax=np.max(x)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Seconds', fontsize=10)
cbar.ax.tick_params(labelsize=8)

# Imposta i ticks per evitare che si estendano nella parte bianca
cbar.set_ticks(np.linspace(np.min(x), np.max(x), 10))

plt.tight_layout()

# salvo il file con un'alta risoluzione per il LaTeX
plt.savefig(f'path_non_correlate_tutte_run{RUN_REF}.png', dpi=300, bbox_inches='tight')