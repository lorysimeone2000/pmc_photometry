from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

# === 1. Apri il file FITS ===
# Sostituisci 'immagine.fits' con il nome del tuo file
image_file = "/home/lorysimeone/tesi_magistrale/prove/20250120_run1/20250120_215217.fits"
hdul = fits.open(image_file)
data = hdul[0].data
header = hdul[0].header
hdul.close()

# Crea un oggetto WCS
wcs = WCS(header)

# Coordinate del centro in pixel
ny, nx = data.shape
xc, yc = nx / 2, ny / 2

# Converti in coordinate celesti (RA, Dec)
ra_center, dec_center = wcs.wcs_pix2world(xc, yc, 0)
print(f"Centro celeste ricavato dai pixel: RA = {ra_center:.6f}°, Dec = {dec_center:.6f}°")
print(f"Centro celeste ricavato dall'header: RA = {header["RA"]}° , DEC = {header["DEC"]}° ")

ny, nx = data.shape

# Definisci i quattro angoli dell'immagine in pixel
pixels = np.array([
    [0, 0],
    [0, ny - 1],
    [nx - 1, 0],
    [nx - 1, ny - 1]
])

print(f"Quattro angoli in pixel: {pixels}")
print(f"Quattro angoli in gradi: {wcs.pixel_to_world(pixels)}")

# Converte i pixel in coordinate celesti (RA, Dec)
world = wcs.wcs_pix2world(pixels, 0)

ra_vals = world[:, 0]
dec_vals = world[:, 1]

# === 4. Calcola minimi e massimi ===
ra_min, ra_max = np.min(ra_vals), np.max(ra_vals)
dec_min, dec_max = np.min(dec_vals), np.max(dec_vals)

print(f"RA min = {ra_min:.6f}°, RA max = {ra_max:.6f}°")
print(f"Dec min = {dec_min:.6f}°, Dec max = {dec_max:.6f}°")

# Stampa informazioni sui vari HDU (Header Data Units)
hdul.info()

# === 2. Estrai i dati scientifici (solitamente HDU[0] o HDU[1]) ===
data = hdul[0].data  # se l’immagine è nel primo HDU
hdul.close()

# === 3. Gestisci eventuali NaN o valori negativi ===
data = np.nan_to_num(data, nan=0.0)

'''# === 4. Visualizza in scala di grigi ===
plt.figure(figsize=(8, 8))
plt.imshow(np.log10(data + 1), cmap='gray_r', origin='lower')
plt.colorbar(label='Count rate')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.show()'''
