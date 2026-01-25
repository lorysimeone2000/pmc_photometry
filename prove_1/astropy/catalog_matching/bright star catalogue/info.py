#In questo codice costruisco la matrice dei valori di un file FITS, ne ricavo l'immagine in scala logaritmica e un istogramma dei valori dei pixel
#Guida: https://learn.astropy.org/tutorials/FITS-images.html#

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

image_file = "/home/lorysimeone/tesi_magistrale/prove/astropy/catalog matching/bright star catalogue/asu.fit"
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250106_231255.fits"  # prima immagine
#image_file = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits" # seconda immagine

# Apri il file FITS
hdu_list = fits.open(image_file)
hdu_list.info()

# I dati sono nella seconda estensione (V_SO_catalog), non nella prima
table_data = Table(hdu_list[1].data)  # Uso l'indice 1 per la seconda estensione

# Chiudo il file FITS
hdu_list.close()

# Esplora il catalogo
print("\n=== INFORMAZIONI DEL CATALOGO ===")
print(f"Numero di stelle nel catalogo: {len(table_data)}")
print(f"Nomi delle colonne: {table_data.colnames}")

# Mostra le prime righe della tabella
print("\n=== PRIME 5 RIGHE DEL CATALOGO ===")
print(table_data[:5])

# Informazioni sulle colonne disponibili
print("\n=== STRUTTURA DELLA TABELLA ===")
table_data.info()

# Esempio: se vuoi accedere a coordinate RA e DEC (supponendo siano colonne comuni)
# Controlla quali colonne sono disponibili
print("\nColonne disponibili:")
for col in table_data.colnames:
    print(f"  - {col}")

# Se ci sono coordinate, puoi fare un plot semplice
if 'RA' in table_data.colnames and 'DEC' in table_data.colnames:
    plt.figure(figsize=(10, 8))
    plt.scatter(table_data['RA'], table_data['DEC'], s=1, alpha=0.5)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('Mappa delle stelle nel catalogo')
    plt.gca().invert_xaxis()  # Per convenzione RA aumenta verso est
    plt.show()

# Statistiche di base su alcune colonne numeriche
print("\n=== STATISTICHE DI BASE ===")
for col in table_data.colnames:
    if table_data[col].dtype.kind in 'if':  # Colonne numeriche
        print(f"\nColonna {col}:")
        print(f"  Min: {np.min(table_data[col])}")
        print(f"  Max: {np.max(table_data[col])}")
        print(f"  Media: {np.mean(table_data[col])}")