# Set up astropy
from astropy.stats import sigma_clipped_stats
import astropy.visualization
from astropy.visualization import simple_norm

# Set up photutils
from photutils.aperture import ApertureStats, CircularAperture
from photutils.datasets import make_4gaussians_image
from photutils.datasets import make_100gaussians_image
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.aperture import aperture_photometry

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica



# Sottrazione sfondo globale

'''data = make_4gaussians_image() # creo un'immagine sintetica di test contenente 4 sorgenti gaussiane
_, median, _ = sigma_clipped_stats(data, sigma=3.0) # rimuovo iterativamente i pixel che deviano più di 3σ dalla media
data = data - median  # sottraggo lo sfondo dai dati. L'immagine ora ha fondo circa zero
aper = CircularAperture((150, 25), 8) # creo un'apertura circolare


# Questo se lo sfondo è una matrice esterna su cui stanno i dati

aperstats = ApertureStats(data, aper, local_bkg=bkg)'''

# Sottrazione sfondo locale

from photutils.datasets import make_100gaussians_image
data = make_100gaussians_image() # Questa immagine artificiale ha un livello di sfondo costante noto pari a 5

# definisco le aprture e gli anelli
from photutils.aperture import CircularAnnulus, CircularAperture
positions = [(145.1, 168.3), (84.5, 224.1), (48.3, 200.3)] # definisco le posizioni
aperture = CircularAperture(positions, r=5) # creo le aperture circolari
annulus_aperture = CircularAnnulus(positions, r_in=10, r_out=15) # creo gli anelli

norm = simple_norm(data, 'sqrt', percent=99)
plt.imshow(data, norm=norm, interpolation='nearest')

# prendo una sezione delle 100 gaussiane
plt.xlim(0, 170)
plt.ylim(130, 250)

ap_patches = aperture.plot(color='white', lw=2, label='Photometry aperture') # disegno le aperture circolari
ann_patches = annulus_aperture.plot(color='red', lw=2,label='Background annulus') # disegno gli anelli
handles = (ap_patches[0], ann_patches[0]) # tupla contenente i riferimenti agli elementi da mostrare in legenda
plt.legend(loc=(0.17, 0.05), facecolor='#458989', labelcolor='white', handles=handles, prop={'weight': 'bold', 'size': 11}) # creo la legenda
plt.colorbar() # barra dei colori

plt.show()

# Media semplice all'interno di un anello circolare

aperstats = ApertureStats(data, annulus_aperture)
bkg_mean = aperstats.mean
print(bkg_mean)

# stavolta utilizzo il metodo aperture_photometry

phot_table = aperture_photometry(data, aperture)
for col in phot_table.colnames:
    phot_table[col].info.format = '%.8g'  # for consistent table output

print(phot_table) # mostro le caratteristiche delle aperture

aperture_area = aperture.area_overlap(data) # calcolo l'area delle aperture

print('Area delle aperture: ' , aperture_area) # in questo caso vengono tutte uguali perché hanno tutte lo stesso raggio

# ora calcolo il fondo totale all'interno delle aperture

total_bkg = bkg_mean * aperture_area # sfondo calcolato dagli anelli per l'area delle aperture
print('Sfondi totali: ' , total_bkg)

phot_bkgsub = phot_table['aperture_sum'] - total_bkg # calcolo la fotometria con lo sfondo sottratto
print('Fotometria con sfondo sottratto: ' , phot_bkgsub)

# Aggiungo lo sfondo totale e la fotometria con lo sfondo sottratto alle info delle aperture

phot_table['total_bkg'] = total_bkg # sfondo totale
phot_table['aperture_sum_bkgsub'] = phot_bkgsub # fotometria con lo sfondo sottratto

for col in phot_table.colnames:
    phot_table[col].info.format = '%.8g'  # for consistent table output

print('Nuova tabella: ')
print(phot_table)

# Mediana sigma-clipped dentro un anello circolare

from astropy.stats import SigmaClip

sigclip = SigmaClip(sigma=3.0, maxiters=10) # scarta i valori che deviano più di 3σ dalla media e lo itera al massimo maxiters volte

# stavolta utilizzo il metodo ApertureStats

aper_stats = ApertureStats(data, aperture, sigma_clip=None) # non applica la pulizia dentro l'apertura circolare
bkg_stats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip) # applica la pulizia nell'anello

print('Sfondi semplici per pixel: ')
print(bkg_mean)
print('Sfondi sigma-clippati per pixel: ')
print(bkg_stats.median)

# ora faccio come prima

total_bkg_sigma = bkg_stats.median * aper_stats.sum_aper_area.value # sfondo calcolato dagli anelli per l'area delle aperture
print('Sfondi semplici totali: ')
print(total_bkg)
print('Sfondi sigma-clippati totali: ')
print(total_bkg_sigma)

apersum_bkgsub_sigma = aper_stats.sum - total_bkg_sigma # calcolo la fotometria con lo sfondo sottratto
print('Fotometria con sfondo semplice sottratto: ')
print(phot_bkgsub)
print('Fotometria con sfondo sigma-clippato sottratto: ')
print(apersum_bkgsub_sigma)

'''Si noti che se si desidera calcolare tutte le proprietà di origine 
(cioè, in aggiunta al solo `.ApertureStats.sum`) 
sul dati sottratti in background locale, è possibile inserire i valori di sfondo per pixel locali a 
`photutils.aperture.ApertureStats` via il locale_bkg parola chiave:'''

aper_stats_bkgsub = ApertureStats(data, aperture, local_bkg=bkg_stats.median)
print(aper_stats_bkgsub.sum)