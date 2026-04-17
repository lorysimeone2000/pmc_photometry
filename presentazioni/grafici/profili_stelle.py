import numpy as np
from scipy.interpolate import interp1d

# Imposto matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # Importo LogNorm per permettermi di avere la scala logaritmica

#%matplotlib inline

from astropy.io import fits
from astropy.utils.data import download_file

from astropy.stats import sigma_clipped_stats

#image_file  = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits"
image_file = "/home/lorysimeone/tesi_magistrale/prove_1/20250106_231255.fits"

hdu_list = fits.open(image_file)
hdu_list.info() # Stampo le informazioni del file

data = hdu_list[0].data # Creo la matrice dei valori dei pixel

'''plt.imshow(data, cmap="grey_r", norm=LogNorm()) # Genero l'immagine con scala di colori bianco e nero
plt.colorbar()
plt.gca().invert_yaxis()
#plt.show()'''

# Inserisco le coordinate invertite rispetto a quelle che vedo nell'immagine
y_centro = 390
x_centro = 282
raggio = 6
stella = 4

print(data[x_centro,y_centro]) # Stampo il valore centrale

#plt.imshow(data, cmap="grey_r", norm=LogNorm()) # Genero l'immagine con scala di colori bianco e nero
#plt.gca().invert_yaxis()
#plt.colorbar()
##plt.show()

profilo = hdu_list[0].data[x_centro, y_centro-raggio:y_centro+raggio]
print(profilo)
porzione_stella = hdu_list[0].data[x_centro-raggio:x_centro+raggio , y_centro-raggio:y_centro+raggio]

print(len(profilo))

'''# Genero un istogramma semplice

plt.bar(range(len(profilo)), profilo, color='skyblue', edgecolor='navy', alpha=0.7, width=0.8) # Traccio l'istogramma della stella

plt.xlabel('Profilo')
plt.ylabel('valori pixel')
plt.grid(axis='y', alpha=0.3)

#plt.show()'''


# Calcolo la FWHM

def calculate_fwhm(profile):
    # Trovo il valore massimo e minimo
    max_val = np.max(profile)
    min_val = np.min(profile)

    # Calcolo la metà altezza
    half_max = min_val + (max_val - min_val) / 2

    # Trovo dove il mio profilo attraversa la metà altezza

    # Creo un array booleano che mi indica per ogni posizione se il valore del profilo è sopra (True) o sotto (False) la metà altezza
    above_half_max = profile > half_max

    # Trovo gli indici in cui il profilo supera la metà altezza
    indices = np.where(above_half_max)[0] # Prendo il primo indice True, ovvero sopra la metà altezza

    # Controllo se non ci sono punti sopra la metà altezza. Se vero, restituisco valori di default per evitare errori
    if len(indices) == 0:
        return 0, half_max, max_val, min_val

    # Segno il primo e ultimo indice sopra la metà altezza
    left_index = indices[0] # Segno il primo pixel sopra la metà altezza
    right_index = indices[-1] # Segno l'ultimo pixel sotto la metà altezza

    # Eseguo l'interpolazione per avere maggiore precisione
    x = np.arange(len(profile))

    # Faccio l'interpolazione a sinistra
    if left_index > 0:
        left_x = [x[left_index - 1], x[left_index]]
        left_y = [profile[left_index - 1], profile[left_index]]
        f_left = interp1d(left_y, left_x, kind='linear')
        try:
            left_interp = float(f_left(half_max))
        except:
            left_interp = left_index
    else:
        left_interp = left_index

    # Faccio l'interpolazione a destra
    if right_index < len(profile) - 1:
        right_x = [x[right_index], x[right_index + 1]]
        right_y = [profile[right_index], profile[right_index + 1]]
        f_right = interp1d(right_y, right_x, kind='linear')
        try:
            right_interp = float(f_right(half_max))
        except:
            right_interp = right_index
    else:
        right_interp = right_index

    fwhm = right_interp - left_interp

    return fwhm, half_max, max_val, min_val, left_interp, right_interp


# Calcolo la FWHM
# Chiamo la funzione con il profilo della stella e salvo tutti i valori che mi restituisce in variabili separate
fwhm, half_max, max_val, min_val, left_edge, right_edge = calculate_fwhm(profilo)

# Genero il plot del profilo con FWHM
# Imposto una dimensione più compatta che risalta i font scalati per 0.45\textwidth
plt.figure(figsize=(6, 4.5))

# Traccio il profilo
plt.bar(range(len(profilo)), profilo, color='skyblue', edgecolor='navy', alpha=0.7, width=0.8, label='Stellar profile')

# Aggiungo le linee per la FWHM
plt.axhline(y=half_max, color='red', linestyle='--', linewidth=2, label=f'Half maximum: {half_max:.2f}')
plt.axhline(y=max_val, color='green', linestyle=':', linewidth=1, label=f'Maximum: {max_val:.2f}')
#plt.axhline(y=min_val, color='orange', linestyle=':', linewidth=1, label=f'Minimo: {min_val:.2f}')

# Segno i bordi della FWHM
plt.axvline(x=left_edge, color='red', linestyle='--', linewidth=1, alpha=0.7)
plt.axvline(x=right_edge, color='red', linestyle='--', linewidth=1, alpha=0.7)

# Imposto etichette e titolo in inglese e con dimensioni maggiorate per la stampa
plt.xlabel('Pixels position in x axis', fontsize=14)
plt.ylabel('Pixel value (ADU)', fontsize=14)
plt.title(f'Stellar profile - FWHM = {fwhm:.2f} pixels', fontsize=14, fontweight='bold', pad=10)
plt.grid(axis='y', alpha=0.3)

# Ingrandisco i valori sugli assi
plt.tick_params(axis='both', which='major', labelsize=12)

# Inserisco la legenda dimensionata in modo leggibile e posizionata in alto a destra
plt.legend(loc='upper right', fontsize=9)

# Aggiungo il testo con i valori calcolati in inglese
plt.text(0.02, 0.98, f'FWHM: {fwhm:.2f} pixels\nMaximum: {max_val:.2f}\nHalf maximum: {half_max:.2f}',
         transform=plt.gca().transAxes, verticalalignment='top', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Ottimizzo il layout prima di salvare per non tagliare le etichette
plt.tight_layout()

# Salvo il primo grafico in un file dedicato
plt.savefig(f'stella_{stella}_profilo.png', dpi=300, bbox_inches='tight')
#plt.show()

print(f"\n=== RISULTATI FWHM ===")
print(f"Valore massimo: {max_val:.2f}")
print(f"Valore minimo: {min_val:.2f}")
print(f"Metà altezza: {half_max:.2f}")
print(f"Bordo sinistro: {left_edge:.2f} pixel")
print(f"Bordo destro: {right_edge:.2f} pixel")
print(f"FWHM: {fwhm:.2f} pixel")

# Sezione per l'istogramma 3D (che ho commentato perché lentissimo)
'''
x, y = np.meshgrid(range(porzione_stella.shape[1]), range(porzione_stella.shape[0]))
x = x.flatten()
y = y.flatten()
z = np.zeros_like(x)
dx = dy = 0.8  # Imposto la larghezza delle barre
dz = porzione_stella.flatten()  # Altezza delle barre = valori della matrice

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(raggio, projection='3d')

# Creo le barre 3D
colors = plt.cm.viridis(dz / dz.max())  # Assegno i colori basati sui valori
ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.7, shade=True)

plt.tight_layout()
#plt.show()'''

# Creo una nuova figura per assicurarmi che il secondo plot sia separato dal primo
plt.figure()

plt.imshow(porzione_stella, cmap="grey_r", norm=LogNorm()) # Genero la porzione di immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # Inverto l'asse y

plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.colorbar()

# Salvo il secondo grafico in un altro file
plt.savefig(f'stella_{stella}.png', dpi=300)
#plt.show()