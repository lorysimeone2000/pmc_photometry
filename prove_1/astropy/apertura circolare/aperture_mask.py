import matplotlib.pyplot as plt
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.datasets import make_100gaussians_image

data = make_100gaussians_image()

# creo aperture e anelli circolari
positions = [(145.1, 168.3), (84.5, 224.1), (48.3, 200.3)]
aperture = CircularAperture(positions, r=5) # aperture
annulus_aperture = CircularAnnulus(positions, r_in=10, r_out=15) #anelli
masks = annulus_aperture.to_mask(method='exact') # creo una lista di oggetti mask, uno per ogni anello
plt.imshow(masks[0]) # mostro la maschera del primo anello
# è utile per visualizzare bene la forma dell'anello e i pixel coinvolti

plt.show()

masks2 = aperture.to_mask(method='center') # creo una lista di oggetti mask, uno per ogni apertura
plt.imshow(masks2[0]) # mostro la maschera della prima apertura

plt.show()

'''posso anche creare un ritaglio ponderato 
della maschera di apertura dai dati per gestire correttamente 
i casi di sovrapposizione parziale o nulla della maschera di apertura con i dati. 
Tracciamo i pesi della maschera di apertura (usando la maschera generato sopra con il metodo "esatto"(exact) 
moltiplicato con i dati'''

data_weighted = masks[0].multiply(data)
plt.imshow(data_weighted)

plt.show()