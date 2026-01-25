import numpy as np
import os

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

from astropy.io import fits
from astropy.utils.data import download_file
from astropy.table import Table

from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clipped_stats

#image_file = "/home/lorysimeone/tesi_magistrale/prove/astropy/catalog matching/bright star catalogue/asu.fit"
image_file = "/home/lorysimeone/tesi_magistrale/prove/astropy/catalog matching/hipparco/hipparco.fit"

# Apri il file FITS
hdu_list = fits.open(image_file)
hdu_list.info()

# I dati sono nella seconda estensione (V_SO_catalog), non nella prima
table_data = Table(aprire_immagine.pyhdu_list[1].data)  # Uso l'indice 1 per la seconda estensione

# Crea la mappa RA/DEC delle stelle
plt.figure(figsize=(12, 8))

# Usa la magnitudine per la dimensione e colore dei punti
# Stelle più brillanti (magnitudine minore) = punti più grandi e gialli
#sizes = 50 * (8 - table_data['Vmag'])  # Scala le dimensioni
sizes = 15 * (8 - table_data['Vmag'])  # Scala le dimensioni
sizes = np.clip(sizes, 10, 200)  # Limita dimensioni min/max

# Colori basati sulla magnitudine
colors = table_data['Vmag']

#scatter = plt.scatter(table_data['_RAJ2000'], table_data['_DEJ2000'],
#                     c=colors, s=sizes, alpha=0.7, cmap='viridis_r')

scatter = plt.scatter(table_data['_RAJ2000'], table_data['_DEJ2000'],
                      c=colors, s = sizes, alpha=0.7, cmap='viridis_r')

plt.colorbar(scatter, label='Magnitudine Visuale (Vmag)')
plt.xlabel('Ascensione Retta (RA J2000) [gradi]')
plt.ylabel('Declinazione (DEC J2000) [gradi]')
plt.title(f'Mappa del Catalogo Stellare Bright Star ({len(table_data)} stelle)')
plt.gca().invert_xaxis()  # RA aumenta verso est (convenzione astronomica)
plt.grid(True, alpha=0.3)

plt.show()