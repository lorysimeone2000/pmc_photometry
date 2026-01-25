from photutils.aperture import CircularAperture

# Creare due aperture circolari di raggio r fissato

'''positions = [(30.0, 30.0), (40.0, 40.0)] # coordinate in pixel
aperture = CircularAperture(positions, r=3.0)'''


# se volessi usare le coordinate celesti
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture

'''positions = SkyCoord(l=[1.2, 2.3] * u.deg, b=[0.1, 0.2] * u.deg, frame='galactic')
aperture = SkyCircularAperture(positions, r=4.0 * u.arcsec)'''

# Convertire pixel in coordinate celesti

from photutils.datasets import make_wcs

wcs = make_wcs((100, 100)) # creo un oggetto wcs
aperture = CircularAperture((10, 20), r=4.0)
sky_aperture = aperture.to_sky(wcs) # conversione
print(sky_aperture) # rappresento le informazioni dell'apertura in coordinate in RA/DEC e in gradi

# Convertire coordinate celesti in pixel

'''position = SkyCoord(197.893, -1.366, unit='deg', frame='icrs')
aperture = SkyCircularAperture(position, r=0.4 * u.arcsec)
pix_aperture = aperture.to_pixel(wcs)
pix_aperture # rappresento le informazioni dell'apertura in coordinate in RA/DEC e in gradi'''