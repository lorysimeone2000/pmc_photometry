import astropy.units as u
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from photutils.aperture import CircularAperture
from photutils.aperture import SkyCircularAperture
from photutils.aperture import aperture_photometry

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm #permette di avere la scala logaritmica

# importo i dati del telescopio Spitzer
from photutils.datasets import load_spitzer_catalog, load_spitzer_image

hdu = load_spitzer_image() # restituisce un oggetto HDU (Header/Data Unit) del genere FITS con header e dati
data = u.Quantity(hdu.data, unit=hdu.header['BUNIT']) # creo un array Astropy Quantity
# che include sia i valori che le unità di misura.
# "data" è un array con unità di misura corretta, utile per calcoli astronomici

# rappresento giusto per vedere
plt.imshow(data.value, cmap="grey", norm=LogNorm()) # genero l'immagine con scala di colori bianco e nero
plt.colorbar()

plt.show()

wcs = WCS(hdu.header) # creo un oggetto WCS dall'header per localizzare oggetti nell'immagine usando coordinate celesti
catalog = load_spitzer_catalog() # carica un catalogo di sorgenti associate all'immagine Spitzer. È una tabella

# definisco le posizioni delle aperture in base alle posizioni del catalogo che ci sono

positions = SkyCoord(catalog['l'], catalog['b'], frame='galactic') # l e b sono le coordinate galattiche
aperture = SkyCircularAperture(positions, r=4.8 * u.arcsec) # creo le aperture in coordinate celesti

phot_table = aperture_photometry(data, aperture, wcs=wcs) # converto le aperture in coordinate pixel

# ora converto le misure di flusso in un'unità milliJansky per pixel
import astropy.units as u
factor = (1.2 * u.arcsec) ** 2 / u.pixel # fattore di conversione dal pixel all'arcosecondo quadrato
fluxes_catalog = catalog['f4_5']  # estraggo la colonna chiamata "f4_55" dal catalogo
converted_aperture_sum = (phot_table['aperture_sum'] * factor).to(u.mJy / u.pixel) # converto le misure di flusso in un'unità milliJansky per pixel

# infine rappresento il confronto della fotometria

import matplotlib.pyplot as plt
plt.scatter(fluxes_catalog, converted_aperture_sum.value)
plt.xlabel('Spitzer catalog PSF-fit fluxes ')
plt.ylabel('Aperture photometry fluxes')

plt.show()

import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from photutils.aperture import SkyCircularAperture, aperture_photometry
from photutils.datasets import load_spitzer_catalog, load_spitzer_image

# Load dataset
hdu = load_spitzer_image()
data = u.Quantity(hdu.data, unit=hdu.header['BUNIT'])
wcs = WCS(hdu.header)
catalog = load_spitzer_catalog()

# Set up apertures
positions = SkyCoord(catalog['l'], catalog['b'], frame='galactic')
aperture = SkyCircularAperture(positions, r=4.8 * u.arcsec)
phot_table = aperture_photometry(data, aperture, wcs=wcs)

# Convert to correct units
factor = (1.2 * u.arcsec) ** 2 / u.pixel
fluxes_catalog = catalog['f4_5']
converted_aperture_sum = (phot_table['aperture_sum'] * factor).to(u.mJy / u.pixel)

# Plot
plt.scatter(fluxes_catalog, converted_aperture_sum.value)
plt.xlabel('Spitzer catalog PSF-fit fluxes ')
plt.ylabel('Aperture photometry fluxes')
plt.plot([40, 100, 450], [40, 100, 450], color='black', lw=2) # disegno una bisettrice per far vedere la corrispondenza

plt.show()