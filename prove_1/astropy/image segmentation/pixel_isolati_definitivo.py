from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # permette di avere la scala logaritmica
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
import numpy as np
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.segmentation import deblend_sources
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceFinder
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture


# DEFINISCI PRIMA LA FUNZIONE PER IL GOMITO
def trova_gomito_dettagliato(image_data, bins=1000):
    # Crea l'istogramma
    counts, bins = np.histogram(image_data.flatten(), bins=bins)

    # Applica log per accentuare le variazioni (evitando log(0))
    counts_log = np.log10(counts + 1)

    # Calcola la derivata prima (tasso di cambio)
    derivata = np.diff(counts_log)

    # Trova il punto di massima curvatura (minimo della derivata)
    indice_gomito = np.argmin(derivata) + 1  # +1 per compensare np.diff
    valore_gomito = bins[indice_gomito]

    return valore_gomito, counts, bins, derivata


parametri = {}

with open('/home/lorysimeone/tesi_magistrale/prove/parametri_image_segmentation.txt', 'r') as file:
    # Salta la prima riga (intestazione)
    next(file)

    # Legge le righe vuota e successive
    for riga in file:
        riga = riga.strip()
        if riga:  # Ignora righe vuote
            parametro, valore = riga.split()
            print(f"{parametro} = {valore}")
            # AGGIUNGI al dizionario
            parametri[parametro] = float(valore) if '.' in valore else int(valore)

print('---------------------------------------------------------------')

# Accedo ai parametri dal dizionario
print(f"FWHM = ", parametri['fwhm'])
print(f"Size = ", parametri['size'])
print(f"Threshold (n. sigma) = ", parametri['threshold_sigma'])
print(f"threshold assoluta = ", parametri['threshold_assoluta'])

# Analisi di tutti i pixel

# prendo la prima immagine della run
image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_212815.fits"

hdu_list = fits.open(image_file)
hdu_list.info()  # dà le informazioni del file

image_data = hdu_list[0].data  # creo la matrice dei valori dei pixel

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_data = image_data - median
data = image_data

print(image_data.shape)  # dà le dimensioni della matrice

# Creazione istogramma

print(type(image_data.flatten()))  # verifico di aver creato un array 1D
print(image_data.flatten().shape)  # dà le dimensioni dell'array

histogram = plt.hist(image_data.flatten(), bins=254, range=(-0.5, 255.5))  # genero l'istogramma dell'array con i valori
plt.yscale("log")
plt.show()

# metodo 1

# Calcola statistiche robustes del fondo
from astropy.stats import sigma_clipped_stats

mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)

# Soglia classica: n-sigma sopra il fondo
n_sigma = 5  # Puoi regolare questo parametro
soglia_sigma = median + n_sigma * std

print(f"Mediana del fondo: {median:.2f}")
print(f"Deviazione standard del fondo: {std:.2f}")
print(f"Soglia a {n_sigma}σ: {soglia_sigma:.2f}")

bkg_estimator = MedianBackground()
bkg = Background2D(image_data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
threshold = bkg.background + n_sigma * bkg.background_rms
print(f'soglia a {n_sigma} σ = {threshold}')

# metodo 2

# Crea l'istogramma in scala log
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(image_data.flatten(), bins=254, log=True)
plt.xlabel('Valore del pixel')
plt.ylabel('Frequenza (log)')
plt.title('Distribuzione dei valori dei pixel')
plt.yscale('log')
plt.show()

# Trova il "gomito" nell'istogramma
# Dove la distribuzione cambia da fondo a sorgenti
differenze = np.diff(np.log10(counts + 1))  # +1 per evitare log(0)
punto_gomito = bins[np.argmin(differenze) + 1]

print(f"Punto di gomito stimato: {punto_gomito:.2f}")

# metodo 3

# Filtra solo i pixel di alto valore
pixel_alti = image_data[image_data > np.percentile(image_data, 95)]

plt.figure(figsize=(10, 6))
plt.hist(np.log10(pixel_alti + 1), bins=50, alpha=0.7)
plt.xlabel('log10(Valore pixel)')
plt.ylabel('Frequenza')
plt.title('Distribuzione della coda alta (sopra 95° percentile)')
plt.axvline(x=np.log10(soglia_sigma), color='red', linestyle='--', label=f'Soglia {n_sigma}σ')
plt.legend()
plt.show()


# metodo 4

def trova_soglia_ottimale(image_data):
    # Metodo 1: Sigma clipping
    mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
    soglia_sigma = median + 5 * std

    # Metodo 3: Analisi istogramma
    counts, bins = np.histogram(image_data.flatten(), bins=254)
    counts_log = np.log10(counts + 1)
    derivata = np.diff(counts_log)
    punto_transizione = bins[np.argmin(derivata) + 1]

    # Prendi il massimo tra i metodi per essere conservativi
    soglia_finale = max(soglia_sigma, punto_transizione)

    print("=== ANALISI SOGLIA ===")
    print(f"Soglia sigma (5σ): {soglia_sigma:.2f}")
    print(f"Punto transizione: {punto_transizione:.2f}")
    print(f"SOGLIA CONSIGLIATA: {soglia_finale:.2f}")

    return soglia_finale


soglia = trova_soglia_ottimale(image_data)

# tecnica gomito - ORA LA FUNZIONE È DEFINITA PRIMA
print('Tecnica gomito')

# Calcola il gomito
gomito, counts, bins, derivata = trova_gomito_dettagliato(image_data)

print(f"Punto di gomito calcolato: {gomito:.2f}")

# Visualizza l'istogramma con il punto di gomito
plt.figure(figsize=(15, 5))

# Subplot 1: Istogramma normale
plt.subplot(1, 2, 1)
plt.hist(image_data.flatten(), bins=1000, log=True, alpha=0.7)
plt.axvline(gomito, color='red', linestyle='--', linewidth=2,
            label=f'Gomito: {gomito:.2f}')
plt.xlabel('Valore del pixel')
plt.ylabel('Frequenza (log)')
plt.title('Distribuzione pixel con punto di gomito')
plt.legend()
plt.yscale('log')

# Subplot 2: Derivata per identificare il gomito
plt.subplot(1, 2, 2)
plt.plot(bins[:-1], derivata, 'b-', alpha=0.7)  # invece di bins[1:]
plt.axvline(gomito, color='red', linestyle='--', linewidth=2,
            label=f'Gomito: {gomito:.2f}')
plt.xlabel('Valore del pixel')
plt.ylabel('Derivata (tasso di cambio)')
plt.title('Derivata - Punto di minimo = Gomito')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Confronto finale tra tutti i metodi
print("\n" + "=" * 50)
print("CONFRONTO FINALE SOGLIE")
print("=" * 50)
print(f"Metodo Sigma (5σ): {soglia_sigma:.2f}")
print(f"Metodo Gomito: {gomito:.2f}")
print(f"Metodo Combinato: {soglia:.2f}")

# Applica la soglia e conta i pixel sopra soglia
mask_sopra_soglia = image_data > soglia
pixel_sopra_soglia = image_data[mask_sopra_soglia]

print(f"\nPixel totali: {image_data.size}")
print(f"Pixel sopra soglia: {np.sum(mask_sopra_soglia)}")
print(f"Percentuale sopra soglia: {np.sum(mask_sopra_soglia) / image_data.size * 100:.4f}%")