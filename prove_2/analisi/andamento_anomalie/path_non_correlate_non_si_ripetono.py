import pandas as pd
# pd.set_option('display.show_dimensions', False)
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # permette di avere la scala logaritmica
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
from pathlib import Path


# =============================================================================
# FUNZIONI DI GESTIONE PERCORSI E UTILITÀ
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    # Cerco la cartella base risalendo l'albero delle directory
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    # Cerco una cartella specifica ricorsivamente
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    return cartelle_trovate[0]


def cerca_file_nel_progetto(base_dir, nome_file_esatto):
    # Cerco un file specifico ricorsivamente
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
# Imposto la cartella base in modo dinamico
BASE_DIR = trova_cartella_base("pmc_photometry")

# Cerco e leggo il file dei parametri
file_parametri = cerca_file_nel_progetto(BASE_DIR, 'parametri_image_segmentation.txt')
if file_parametri:
    parametri = leggi_file_parametri(file_parametri)
else:
    print("ERRORE: File parametri non trovato.")
    parametri = {}

RUN_REF = 2

fwhm = parametri.get('fwhm', 2.8)
size = parametri.get('size', 5)
t = parametri.get('threshold_sigma', 3.0)
# threshold = t * std # per adesso lascio stare questo metodo
threshold = parametri.get('threshold_assoluta', 3.0)
n = parametri.get('pixel', 5)

# Cerco dinamicamente la cartella tabelle_unite_run_{RUN_REF}
cartella_csv_path = cerca_cartella_nel_progetto(BASE_DIR, f"tabelle_unite_run_{RUN_REF}")
if cartella_csv_path is None:
    print(f"ERRORE CRITICO: Cartella 'tabelle_unite_run_{RUN_REF}' non trovata.")
    exit()
cartella_csv = str(cartella_csv_path)

# Lista tutti i file CSV e ordinali
file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])
print("lista:")

print(f"Trovati {len(file_csv)} file CSV:")
'''for file in file_csv:
    print(f"  - {file}")'''

i = 0
j = 0
posizioni_lista = []  # lista che dovrà essere riempita con tutte le poszioni di tutte le tabelle
colori_lista = []  # Lista per i colori di ogni punto

# creo una colormap per la sfumatura
colormap = cm.viridis  # posso cambiare con: 'plasma', 'inferno', 'magma', 'cool', 'spring', etc.

# Itera su tutti i file CSV
for nome_file in file_csv:
    i += 1
    percorso_completo = os.path.join(cartella_csv, nome_file)
    # print(f"Nome file csv: {nome_file}")

    if i <= 2:
        print(f"\n{'=' * 50}")
        print(f"Elaborazione: {nome_file}")
        print(f"{'=' * 50}")
    # Leggi il file CSV
    try:
        df = pd.read_csv(percorso_completo, comment="#")
        tbl = Table.from_pandas(df)
        j = j + len(tbl)

        mask_si = np.char.startswith(tbl['Corrispondenza'].astype(str), 'SI')
        tbl_no = tbl[~mask_si]
        mask_no_ripetizioni = tbl_no['ripetizioni'] == 1
        tbl_no_no_ripetizioni = tbl_no[mask_no_ripetizioni]
        # print(f"Trovati {len(tbl_no_no_ripetizioni)} oggetti")

        posizioni_file = np.transpose((tbl_no_no_ripetizioni['RA_centroid'], tbl_no_no_ripetizioni[
            'DEC_centroid']))  # creo l'array di posizioni per questo file
        posizioni_lista.append(posizioni_file)  # lo aggiungo alla lista totale

        # Calcola il colore per questo file
        if len(file_csv) > 1:
            colore_valore = i / (len(file_csv))
        else:
            colore_valore = 0.5  # Se c'è solo un file

        colore_rgb = colormap(colore_valore)

        if i < 3:  # Debug
            print(f"Punti: {len(posizioni_file)}")
            print(f"Valore colore: {colore_valore:.3f}")
            print(f"Colore RGB: {colore_rgb[:3]}")

        # Aggiungo lo stesso colore per tutti i punti di questo file
        for _ in range(len(posizioni_file)):
            colori_lista.append(colore_rgb)

    except Exception as e:
        print(f"Errore nella lettura di {nome_file}: {e}")

posizioni_array = np.vstack(posizioni_lista)
print(f"Nella run {RUN_REF} ci sono {len(posizioni_array)} oggetti che compaiono una sola volta nella stessa posizione")
colori_array = np.array(colori_lista)
print(f"\n{'=' * 60}")
print(f"ARRAY FINALE CREATO")
print(f"{'=' * 60}")
print(f"Dimensioni array posizioni: {posizioni_array.shape}")

print(f"Massimo RA: {np.max(posizioni_array[:, 0])}")
print(f"Massimo DEC: {np.max(posizioni_array[:, 1])}")
print("Ho verificato che l'asse x e l'asse y sono corrispondenti a quelli dell'immagine")

# secondi
x = []

# Cerco dinamicamente la lista delle immagini
file_lista_immagini = cerca_file_nel_progetto(BASE_DIR, f'lista_immagini_run_{RUN_REF}.txt')
if not file_lista_immagini:
    print("ERRORE: File 'lista_immagini_run_1.txt' non trovato.")
    exit()

# Leggo la lista
with open(file_lista_immagini, 'r') as file:
    file_list_raw = file.read().splitlines()  # creo una lista di stringhe che sono i percorsi

file_list = []
# Risolvo dinamicamente i percorsi dei file FITS contenuti nel TXT per garantirne la portabilità
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
# Elaboro tutti i file
for percorso_file in file_list:
    n = n + 1
    try:
        with fits.open(percorso_file) as hdu_list:
            image_header = hdu_list[0].header
            if n == 1:
                x.append(0)
                t1 = image_header["TSTART"]
                # print(0 , "secondi")
            else:
                x.append((image_header["TSTART"] - t1) / np.float64(1e3))
                # print((image_header["TSTART"]-t1)/np.float64(1e3) , "secondi")
    except Exception as e:
        print(f"Errore caricamento FITS {percorso_file}: {e}")

x = np.array(x)

# creo la figura impostando una dimensione di base
plt.figure(figsize=(10, 8))

plt.scatter(posizioni_array[:, 0], posizioni_array[:, 1],
            s=4,
            alpha=1,
            color=colori_array,
            linewidth=0)  # ⬅ Nessuna linea di bordo

plt.xlabel('RA (Gradi)')
plt.ylabel('DEC (Gradi)')
plt.title(
    f'Posizioni delle sorgenti non catalogate che non si ripetono nella stessa posizione della run {RUN_REF}\nTotale: {len(posizioni_array)} sorgenti da {len(file_csv)} file della run')
plt.grid(True, alpha=0.4, linestyle='--')

# =========================================================
# MODIFICHE ASTRONOMICHE PER RA / DEC
# =========================================================
# 1. Inverto l'asse X. In astronomia, l'Est (RA maggiore) sta a sinistra!
plt.gca().invert_xaxis()

# 2. Correggo l'aspect ratio. Un grado di RA non è "largo" quanto un grado di DEC.
# Calcolo il coseno della Declinazione media (convertita in radianti)
dec_media = np.mean(posizioni_array[:, 1])
aspect_ratio = 1.0 / np.cos(np.radians(dec_media))
plt.gca().set_aspect(aspect_ratio)
# =========================================================

# aggiungo la colorbar per mostrare la progressione
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=np.min(x), vmax=np.max(x)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label='Secondi')

# Imposta i ticks per evitare che si estendano nella parte bianca
cbar.set_ticks(np.linspace(np.min(x), np.max(x), 10))  # 10 ticks equidistanti

plt.tight_layout()
plt.savefig(f'path_non_correlate_no_ripetizioni_run_{RUN_REF}.png')
plt.show()

# =============================================================================
# NUOVO GRAFICO: Numero totale di oggetti 'NO' cumulativo nel tempo (Run 1, 2, 3)
# =============================================================================

plt.figure(figsize=(12, 6))

colori_runs = {1: 'blue', 2: 'green', 3: 'red'}

# Inizializzo le liste globali per l'asse temporale cumulativo e i conteggi
tutti_i_tempi = []
tutti_i_conteggi_no = []
tutti_i_colori = []  # Per assegnare a ogni punto il colore della sua run

t0_global = None  # Tempo zero assoluto (inizio Run 1)
run_boundaries = []  # Per memorizzare il tempo di fine di ogni run e tracciare la linea verticale

for r in [1, 2, 3]:
    cartella_csv_r_path = cerca_cartella_nel_progetto(BASE_DIR, f"tabelle_unite_run_{r}")
    if not cartella_csv_r_path:
        continue
    cartella_csv_r = str(cartella_csv_r_path)

    file_csv_r = sorted([f for f in os.listdir(cartella_csv_r) if f.endswith('.csv')])

    file_lista_immagini_r = cerca_file_nel_progetto(BASE_DIR, f'lista_immagini_run_{r}.txt')
    if not file_lista_immagini_r:
        continue

    with open(file_lista_immagini_r, 'r') as file:
        file_list_raw_r = file.read().splitlines()

    file_list_r = []
    for p in file_list_raw_r:
        p_obj = Path(p)
        if not os.path.exists(p):
            try:
                if "pmc_photometry" in p_obj.parts:
                    idx = p_obj.parts.index("pmc_photometry")
                    new_path = BASE_DIR.joinpath(*p_obj.parts[idx + 1:])
                    if new_path.exists():
                        file_list_r.append(str(new_path))
                    else:
                        file_list_r.append(p)
            except:
                file_list_r.append(p)
        else:
            file_list_r.append(p)

    # Variabile per memorizzare l'ultimo tempo calcolato di questa run
    ultimo_tempo_run = 0

    for n_img, (percorso_file, nome_file_csv) in enumerate(zip(file_list_r, file_csv_r)):
        try:
            # 1. Estraggo il tempo
            with fits.open(percorso_file) as hdu_list:
                header = hdu_list[0].header
                t_curr = header["TSTART"]

                # Imposto il t0 globale solo alla prima immagine della primissima run valida
                if t0_global is None:
                    t0_global = t_curr

                # Calcolo il tempo cumulativo in secondi
                tempo_relativo = (t_curr - t0_global) / 1000.0
                ultimo_tempo_run = tempo_relativo

            # 2. Conto gli oggetti "NO"
            percorso_completo_csv = os.path.join(cartella_csv_r, nome_file_csv)
            df = pd.read_csv(percorso_completo_csv, comment="#")
            tbl = Table.from_pandas(df)

            mask_no = ~np.char.startswith(tbl['Corrispondenza'].astype(str), 'SI')
            mask_no_ripetizioni = tbl['ripetizioni'] == 1
            mask_combinata = mask_no_ripetizioni & mask_no&mask_no_ripetizioni
            conteggio_no = np.sum(mask_combinata)

            # 3. Salvo nelle liste globali
            tutti_i_tempi.append(tempo_relativo)
            tutti_i_conteggi_no.append(conteggio_no)
            tutti_i_colori.append(colori_runs[r])

        except Exception as e:
            # Ignoro i file corrotti ma stampo l'errore per sicurezza
            print(f"Errore {percorso_file}: {e}")
            pass

    # Alla fine di ogni run, salvo il punto di demarcazione temporale
    run_boundaries.append((r, ultimo_tempo_run))

# Converto in array per comodità di plotting
tempi_arr = np.array(tutti_i_tempi)
conteggi_arr = np.array(tutti_i_conteggi_no)

# Disegno un'unica linea grigia di base per collegare tutti i punti in modo continuo
plt.plot(tempi_arr, conteggi_arr, linestyle='-', color='lightblue', alpha=0.5, zorder=1)

# Disegno i punti colorati in base alla run a cui appartengono
for r in [1, 2, 3]:
    mask_run = np.array(tutti_i_colori) == colori_runs[r]
    plt.scatter(tempi_arr[mask_run], conteggi_arr[mask_run],
                color=colori_runs[r], s=5, label=f'Run {r}', zorder=2)

# Disegno le linee verticali di separazione e le etichette per ogni run
for r_idx, (run_num, t_end) in enumerate(run_boundaries):
    plt.axvline(x=t_end, color='black', linestyle='--', alpha=0.5)

    # Calcolo il centro dell'asse Y per posizionare la scritta verticalmente al centro
    y_min, y_max = plt.ylim()
    y_mid = (0 + y_max) / 2

    # Scritta "Fine Run X"
    plt.text(t_end, y_mid, f"Fine Run {run_num}",
             rotation=90,
             horizontalalignment='right',  # Leggermente a sinistra della linea
             verticalalignment='center',
             color='#333333',
             fontsize=10,
             fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.xlabel('Tempo dall\'inizio della Run 1 (secondi)')
plt.ylabel('Numero totale di oggetti non catalogati (NO)')
plt.title('Andamento degli oggetti non catalogati cumulativo (Run 1, 2, 3) \n solo quelli che non si ripetono')
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig('andamento_non_catalogati_no_ripetizioni_temporale_continuo.png')
plt.show()