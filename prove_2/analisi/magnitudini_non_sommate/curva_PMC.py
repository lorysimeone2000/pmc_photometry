import pandas as pd
import matplotlib.pyplot as plt

# Carico il dataset dal file CSV che ho appena creato
df = pd.read_csv('curva_PMC.csv')

# Inizializzo la figura e gli assi impostando dimensioni simili all'immagine di riferimento
fig, ax = plt.subplots(figsize=(10, 5))

# Definizione degli intervalli delle bande con i relativi colori
intervalli = {
    'gmag': (400, 550),    # banda g: da 400 a 550 nm
    'rmag': (555, 705),    # banda r: da 555 a 705 nm
    'imag': (688, 850),    # banda i: da 688 a 850 nm
    'zmag': (812, 920),    # banda z: da 812 a 920 nm
    'ymag': (922, 1005)    # banda y: da 922 a 1005 nm
}

# Mappa dei colori per le bande (sfumature tenui e riconoscibili)
colori_bande = {
    'gmag': '#33cc33',    # verde
    'rmag': '#ff3333',    # rosso
    'imag': '#ff66cc',    # magenta
    'zmag': '#8b4513',    # marrone
    'ymag': '#66ccff'     # blu chiaro
}

# Aggiungo le strisce verticali per ogni banda
for banda, (inizio, fine) in intervalli.items():
    ax.axvspan(inizio, fine, alpha=0.3, color=colori_bande[banda], label=banda)

# Traccio la curva principale usando il nero come colore
ax.plot(df['Wavelength'], df['QE'], color='#111111', linewidth=2)

# Inserisco la linea verticale tratteggiata in corrispondenza del valore 525
ax.axvline(x=525, color='#111111', linestyle=':', linewidth=1.5)

# Imposto i limiti degli assi per farli coincidere esattamente con il grafico originale
ax.set_xlim(300, 1100)
ax.set_ylim(0, 90)

# Aggiungo i testi descrittivi per gli assi X e Y
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Quantum Efficiency (%)')

# Posiziono il titolo in alto a sinistra, fuori dal perimetro del riquadro del grafico
fig.text(0.08, 0.95, 'BFS-PGE-63S4M', fontsize=16, fontweight='bold', fontfamily='sans-serif')

# Aggiungo la legenda per le bande
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

# Configuro la griglia per mostrare solo le linee orizzontali in tinta unita
ax.yaxis.grid(True, linestyle='-', color='#c0c0c0', alpha=0.8)
ax.xaxis.grid(False)

# Mi assicuro che la linea dati parta esattamente dai bordi senza spazi vuoti
ax.margins(x=0, y=0)

# Aggiusto i margini della figura per fare in modo che il titolo non venga tagliato
plt.subplots_adjust(top=0.85)

# Salvo l'immagine
plt.savefig('curva_PMC.png', dpi=300, bbox_inches='tight')

# Avvio il render a schermo del grafico finale
plt.show()