import numpy as np
import matplotlib.pyplot as plt

# definisco i parametri principali del profilo stellare ricavati dall'immagine
picco = 255.0
meta_altezza = 128.0
fwhm = 9.53
centro = 25.0 # Posizione centrale stimata dal grafico

# calcolo la deviazione standard (sigma) a partire dalla FWHM
sigma = fwhm / 2.355

# genero i dati per l'asse X (i pixel) e per l'asse Y (l'intensità gaussiana)
x = np.arange(0, 51, 1)
y = picco * np.exp(-0.5 * ((x - centro) / sigma) ** 2)

# inizializzo la figura
plt.figure(figsize=(10, 6))

# traccio la curva principale con i punti marcati
plt.plot(x, y, marker='o', linestyle='-', color='#1f77b4', label='Stellar profile')

# calcolo i punti esatti in cui la curva incrocia la metà altezza per disegnare la linea della FWHM
x_sinistra = centro - (fwhm / 2)
x_destra = centro + (fwhm / 2)

# disegno la linea orizzontale tratteggiata per la FWHM
plt.plot([x_sinistra, x_destra], [meta_altezza, meta_altezza], color='green', linestyle='--', linewidth=2)

# disegno la linea verticale tratteggiata dal centro fino al picco
plt.plot([centro, centro], [0, picco], color='red', linestyle='--', linewidth=1.5)

# imposto il titolo e le etichette degli assi ingrandendo il testo per il formato A4
plt.title('Stellar profile - FWHM = 9.53 pixels', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Position (pixels)', fontsize=14)
plt.ylabel('Pixel intensity', fontsize=14)

# ingrandisco i valori sugli assi
plt.tick_params(axis='both', which='major', labelsize=12)

# aggiungo il riquadro di testo in alto a sinistra con i valori esatti
testo_statistiche = (
    f"FWHM: {fwhm:.2f} pixels\n"
    f"Maximum: {picco:.2f}\n"
    f"Half maximum: {meta_altezza:.2f}"
)
plt.text(0.05, 0.95, testo_statistiche, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# aggiungo la legenda in basso a sinistra
plt.legend(loc='lower left', fontsize=12)

# imposto i limiti degli assi per centrare il grafico in modo simile all'originale
plt.xlim(0, 50)
plt.ylim(0, 280)

# aggiungo la griglia di sfondo leggera
plt.grid(True, linestyle=':', alpha=0.7)

# ottimizzo i margini ed esporto l'immagine
plt.tight_layout()
plt.savefig('stellar_profile_english.png', dpi=300, bbox_inches='tight')

