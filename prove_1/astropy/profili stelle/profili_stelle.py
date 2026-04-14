import numpy as np
from scipy.interpolate import interp1d

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # permette di avere la scala logaritmica

#%matplotlib inline

from astropy.io import fits
from astropy.utils.data import download_file

from astropy.stats import sigma_clipped_stats

#image_file  = "/home/lorysimeone/tesi_magistrale/prove/20250107_060735.fits"
image_file = "/home/lorysimeone/tesi_magistrale/prove_1/20250106_231255.fits"

hdu_list = fits.open(image_file)
hdu_list.info() # dà le informazioni del file

data = hdu_list[0].data # creo la matrice dei valori dei pixel

'''plt.imshow(data, cmap="grey_r", norm=LogNorm()) #genero l'immagine con scala di colori bianco e nero
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()'''

# inserisco le coordinate invertite rispetto a quelle che vedo nell'immagine
y_centro = int(input("x centro: "))
x_centro = int(input("y centro: "))
raggio = int(input("raggio: "))

print(data[x_centro,y_centro]) # valore centrale

#plt.imshow(data, cmap="grey_r", norm=LogNorm()) #genero l'immagine con scala di colori bianco e nero
#plt.gca().invert_yaxis()
#plt.colorbar()
#plt.show()



profilo = hdu_list[0].data[x_centro, y_centro-raggio:y_centro+raggio]
print(profilo)
porzione_stella = hdu_list[0].data[x_centro-raggio:x_centro+raggio , y_centro-raggio:y_centro+raggio]



print(len(profilo))

'''# istogramma semplice

plt.bar(range(len(profilo)), profilo, color='skyblue', edgecolor='navy', alpha=0.7, width=0.8) # istogramma stella

plt.xlabel('Profilo')
plt.ylabel('valori pixel')
plt.grid(axis='y', alpha=0.3)

plt.show()'''


# CALCOLO FWHM

def calculate_fwhm(profile):
    # Trovo il valore massimo e minimo
    max_val = np.max(profile)
    min_val = np.min(profile)

    # Calcolo metà altezza
    half_max = min_val + (max_val - min_val) / 2

    # Trovo dove il profilo attraversa la metà altezza

    # creo un array booleano che indica per ogni posizione se il valore del profilo è sopra (True) o sotto (False) la metà altezza
    above_half_max = profile > half_max

    # trovo gli indici dove il profilo supera la metà altezza
    indices = np.where(above_half_max)[0] # prendo il primo indice True, ovvero sopra la metà altezza

    # Controllo se non ci sono punti sopra la metà altezza. Se vero, restituisce valori di default per evitare errori
    if len(indices) == 0:
        return 0, half_max, max_val, min_val

    # Primo e ultimo indice sopra la metà altezza
    left_index = indices[0] # primo pixel sopra la metà altezza
    right_index = indices[-1] # ultimo pixel sotto la metà altezza

    # Interpolazione per maggiore precisione
    x = np.arange(len(profile))

    # Interpolazione a sinistra
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

    # Interpolazione a destra
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


# Calcolo FWHM
# Chiamo la funzione con il profilo della stella e salvo tutti i valori restituiti in variabili separate

fwhm, half_max, max_val, min_val, left_edge, right_edge = calculate_fwhm(profilo)

# Genero il plot del profilo con FWHM
plt.figure(figsize=(12, 6))

# Profilo
plt.bar(range(len(profilo)), profilo, color='skyblue', edgecolor='navy', alpha=0.7, width=0.8, label='Profilo stellare')

# Linee per FWHM
plt.axhline(y=half_max, color='red', linestyle='--', linewidth=2, label=f'Metà altezza: {half_max:.2f}')
plt.axhline(y=max_val, color='green', linestyle=':', linewidth=1, label=f'Massimo: {max_val:.2f}')
#plt.axhline(y=min_val, color='orange', linestyle=':', linewidth=1, label=f'Minimo: {min_val:.2f}')

# Segno i bordi del FWHM
plt.axvline(x=left_edge, color='red', linestyle='--', linewidth=1, alpha=0.7)
plt.axvline(x=right_edge, color='red', linestyle='--', linewidth=1, alpha=0.7)

plt.xlabel('Posizione (pixel)')
plt.ylabel('Intensità pixel')
plt.title(f'Profilo stellare - FWHM = {fwhm:.2f} pixel')
plt.grid(axis='y', alpha=0.3)
plt.legend()

# Aggiungo testo con i valori
plt.text(0.02, 0.98, f'FWHM: {fwhm:.2f} pixel\nMassimo: {max_val:.2f}\nMetà altezza: {half_max:.2f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Salvo il primo grafico in un file dedicato
plt.savefig('grafico_fwhm.png', dpi=300)
plt.show()

print(f"\n=== RISULTATI FWHM ===")
print(f"Valore massimo: {max_val:.2f}")
print(f"Valore minimo: {min_val:.2f}")
print(f"Metà altezza: {half_max:.2f}")
print(f"Bordo sinistro: {left_edge:.2f} pixel")
print(f"Bordo destro: {right_edge:.2f} pixel")
print(f"FWHM: {fwhm:.2f} pixel")

# istogramma 3D (lentissimo)
'''

x, y = np.meshgrid(range(porzione_stella.shape[1]), range(porzione_stella.shape[0]))
x = x.flatten()
y = y.flatten()
z = np.zeros_like(x)
dx = dy = 0.8  # Larghezza delle barre
dz = porzione_stella.flatten()  # Altezza delle barre = valori della matrice

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(raggio, projection='3d')

# Crea le barre 3D
colors = plt.cm.viridis(dz / dz.max())  # Colori basati sui valori
ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.7, shade=True)

plt.tight_layout()
plt.show()'''

# Creo una nuova figura per assicurarmi che il secondo plot sia separato dal primo
plt.figure()

plt.imshow(porzione_stella, cmap="grey_r", norm=LogNorm()) #genero porzione immagine con scala di colori bianco e nero
plt.gca().invert_yaxis() # inverto asse y
plt.colorbar()

# Salvo il secondo grafico in un altro file
plt.savefig('immagine_porzione_stella.png', dpi=300)