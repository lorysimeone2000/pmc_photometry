import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.visualization import simple_norm
from photutils.centroids import centroid_quadratic
from photutils.datasets import make_noise_image
from photutils.profiles import CurveOfGrowth
from photutils.profiles import RadialProfile

# create an artificial single source
gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
yy, xx = np.mgrid[0:100, 0:100]
data = gmodel(xx, yy)
bkg_sig = 2.4
noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=123)
data += noise
error = np.zeros_like(data) + bkg_sig

fig, ax = plt.subplots(figsize=(5, 5))
norm = simple_norm(data, 'sqrt')
ax.imshow(data, norm=norm)

plt.show()

# trovo il baricentro
xycen = centroid_quadratic(data, xpeak=47, ypeak=52)

# creo il profilo radiale
edge_radii = np.arange(26)
rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

# rappresento il profilo radiale
fig, ax = plt.subplots(figsize=(8, 6))
rp.plot(ax=ax, color='C0', label='Profilo radiale') # linea del profilo radiale
rp.plot_error(ax=ax) # fascia di errore del profilo radiale
ax.plot(rp.radius, rp.gaussian_profile, color= 'red' , label='Fit gaussiano') # fit gaussiano
ax.scatter(rp.data_radius, rp.data_profile, s=1, color='C1' , label='Valori profilo') # valori del profilo radiale
ax.legend(loc='best', fontsize=12)

print(rp.gaussian_fit) # fit gaussiano
print(rp.gaussian_fwhm) # fwhm

#plt.show()

# Curva di crescita

from photutils.profiles import CurveOfGrowth
radii = np.arange(1, 26)
cog = CurveOfGrowth(data, xycen, radii, error=error, mask=None)
print(cog.radius) # rappresento i vari raggi
print(cog.profile) # array della curva di crescita
print(cog.profile_error) # array degli errori della curva di crescita

# plot the radial profile
fig, ax = plt.subplots(figsize=(8, 6))
cog.plot(ax=ax, label='Curva di crescita')
cog.plot_error(ax=ax)
ax.legend()
norm = simple_norm(data, 'sqrt')
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(data, norm=norm)
cog.apertures[5].plot(ax=ax, color='C0', lw=2)
cog.apertures[10].plot(ax=ax, color='C1', lw=2)
cog.apertures[15].plot(ax=ax, color='C3', lw=2)

plt.show()
