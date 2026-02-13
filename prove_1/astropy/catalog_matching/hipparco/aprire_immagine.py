import numpy as np
import os
import warnings
from astropy.utils.exceptions import AstropyWarning

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

from astropy.io import fits
from astropy.utils.data import download_file
from astropy.table import Table
from astropy.stats import sigma_clipped_stats

# Ignoro i warning specifici di verifica di astropy
warnings.simplefilter('ignore', category=AstropyWarning)

#image_file = "/home/lorysimeone/tesi_magistrale/prove_1/astropy/catalog_matching/bright star catalogue/asu.fit"
image_file = "/home/lorysimeone/tesi_magistrale/prove_1/astropy/catalog_matching/hipparco/hipparco.fit"

# Apro il file FITS
hdu_list = fits.open(image_file)
hdu_list.info()

# I dati sono nella seconda estensione (V_SO_catalog), non nella prima
table_data = Table(hdu_list[1].data)  # Uso l'indice 1 per la seconda estensione

colonne = hdu_list[1].columns
print(f"Colonne: {colonne}")

# Accedo all'header della tabella (estensione 1)
header = hdu_list[1].header

# Cerco le descrizioni specifiche per _RAJ2000 e _DEJ2000
# Le colonne sono numerate progressivamente nell'header come TTYPE1, TTYPE2, ecc.

for i in range(1, len(colonne) + 1):
    key_name = f"TTYPE{i}"  # Costruisco la chiave es. TTYPE1

    # Verifico se la chiave esiste nell'header (per sicurezza)
    if key_name in header:
        col_name = header[key_name]  # Il valore è il nome della colonna (es. _RAJ2000)

        # Se è una delle colonne che ci interessano
        if col_name in ['e_RAICRS', 'e_DEICRS']:
            description = header.comments[key_name]  # .comments estrae il testo dopo lo slash /
            print(f"Colonna: {col_name}")
            print(f"Descrizione: {description}")
            print("-" * 40)

# Fattore di conversione da mas a gradi
# 1 grado = 3600 arcsec = 3.600.000 mas
mas_to_deg = 1.0 / 3600000.0

# 1. Estraggo la Declinazione (necessaria per il coseno) e converto in radianti
dec_rad = np.radians(table_data['_DEJ2000'])

# 2. Gestione dell'errore su RA (e_RAICRS)
# Il catalogo fornisce: sigma_RA_sky = sigma_RA_coordinate * cos(dec)

# OPZIONE A: Errore "Visivo" (Distanza angolare nel cielo)
# Questo è quello che si usa per i plot di errore o cross-match spaziale
e_ra_sky_deg = table_data['e_RAICRS'] * mas_to_deg

# OPZIONE B: Errore sulla Coordinata (Valore numerico di RA)
# Devo "annullare" il coseno per trovare l'errore puro della coordinata
# Attenzione: cos(90°) = 0, quindi evito divisioni per zero o valori instabili ai poli
cos_dec = np.abs(np.cos(dec_rad))
cos_dec = np.where(cos_dec < 1e-10, 1e-10, cos_dec) # Evito divisione per zero ai poli

e_ra_coord_deg = (table_data['e_RAICRS'] * mas_to_deg) / cos_dec


# 3. Gestione dell'errore su DEC (e_DEICRS)
# Per la declinazione non c'è fattore coseno, è diretta
e_dec_deg = table_data['e_DEICRS'] * mas_to_deg


# --- ESEMPIO DI STAMPA DI VERIFICA ---
print(f"Esempio stella 0:")
print(f"Dec: {table_data['_DEJ2000'][0]:.4f} deg")
print(f"Errore RA (distanza cielo): {e_ra_sky_deg[0]:.8f} deg")
print(f"Errore RA (coordinata):     {e_ra_coord_deg[0]:.8f} deg")

# Creo la mappa RA/DEC delle stelle
plt.figure(figsize=(12, 8))

# Uso la magnitudine per la dimensione e colore dei punti
# Stelle più brillanti (magnitudine minore) = punti più grandi e gialli
#sizes = 50 * (8 - table_data['Vmag'])  # Scalo le dimensioni
sizes = 15 * (8 - table_data['Vmag'])  # Scalo le dimensioni
sizes = np.clip(sizes, 10, 200)  # Limito dimensioni min/max

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

# plt.show()