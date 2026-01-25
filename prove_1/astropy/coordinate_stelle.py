#In questo codice costruisco la matrice dei valori di un file FITS, ne ricavo l'immagine in scala logaritmica e un istogramma dei valori dei pixel
#Guida: https://learn.astropy.org/tutorials/FITS-images.html#

import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm #permette di avere la scala logaritmica

#%matplotlib inline

from astropy.io import fits
from astropy.utils.data import download_file

#coordinate
from photutils import DAOStarFinder
from astropy.stats import mad_std



image_file = "20250106_231255.fits"

hdu_list = fits.open(image_file)
hdu_list.info() #dà le informazioni del file

image_data = hdu_list[0].data #creo la matrice dei valori dei pixel
#print(hdu_list[0].header) #mette tutti i dati dell'header
image_header = hdu_list[0].header
print("RA",image_header["RA"]) #dà la RA
print("DEC",image_header["DEC"]) #dà il DEC
# Ottenere le coordinate dal WCS (World Coordinate System)
from astropy.wcs import WCS

wcs = WCS(header)

# Se l'immagine contiene già informazioni sulle stelle (come in cataloghi)
if 'RA' in header and 'DEC' in header:
    ra = header['RA']
    dec = header['DEC']
    print(f"Coordinate centro: RA={ra}, DEC={dec}")

print(image_data) #dà la matrice dei valori dei pixel
print(image_data[0,0]) #dà il valore dell'pixel di coordinate [0,0]
print(image_data.shape) #dà le dimensioni della matrice



plt.imshow(image_data, cmap="gray", norm=LogNorm()) #genero l'immagine con scala di colori bianco e nero
plt.colorbar()

plt.show()


print(type(image_data.flatten())) #verifico di aver creato un array 1D
print(image_data.flatten().shape) #dà le dimensioni dell'array

histogram = plt.hist(image_data.flatten(), bins=256,range=(-0.5,255.5)) #genero l'istogramma dell'array con i valori
plt.yscale("log")


# Rilevamento automatico delle stelle
def detect_stars(data, threshold=5.0):
    # Calcolo del rumore di fondo
    bkg_sigma = mad_std(data)

    # Trova le stelle
    daofind = DAOStarFinder(fwhm=3.0, threshold=threshold * bkg_sigma)
    sources = daofind(data)

    return sources


# Rileva le stelle
sources = detect_stars(image_data)

if sources is not None:
    print(f"Trovate {len(sources)} stelle")

    # Converti coordinate pixel in coordinate celesti
    coords = wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])

    # Estrai RA e DEC
    ra_values = coords.ra.deg
    dec_values = coords.dec.deg

    # Crea tabella con i risultati
    from astropy.table import Table

    results = Table()
    results['ID'] = np.arange(len(sources)) + 1
    results['RA'] = ra_values
    results['DEC'] = dec_values
    results['FLUX'] = sources['flux']

    print(results)

    # Salva i risultati
    results.write('coordinate_stelle.csv', format='csv', overwrite=True)

plt.show()