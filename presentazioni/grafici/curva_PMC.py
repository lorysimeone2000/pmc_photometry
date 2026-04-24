import pandas as pd
import matplotlib.pyplot as plt

# Carico il dataset dal file CSV che ho appena creato
df = pd.read_csv('curva_PMC.csv')

# Inizializzo la figura e gli assi impostando dimensioni simili all'immagine di riferimento
fig, ax = plt.subplots(figsize=(10, 5))

# Definizione degli intervalli delle bande con i relativi colori
'''intervalli = { # versione panstarr
    'm_g [414;551]': (414, 551),    # banda g: da 400 a 550 nm
    'm_r [551;689]': (551, 689),    # banda r: da 555 a 705 nm
    'm_i [690;819]': (690, 819),    # banda i: da 688 a 850 nm
    'm_z [819;922]': (819, 922),    # banda z: da 812 a 920 nm
    'm_y [922;1001]': (922, 1001)    # banda y: da 922 a 1005 nm
}'''


intervalli = { # versione classica
    'm_g [400;550]': (400, 550),    # banda g: da 400 a 550 nm
    'm_r [555;705]': (555, 705),    # banda r: da 555 a 705 nm
    'm_i [688;850]': (688, 850),    # banda i: da 688 a 850 nm
    'm_z [812;920]': (812, 920),    # banda z: da 812 a 920 nm
    'm_y [922;1005]': (922, 1005)    # banda y: da 922 a 1005 nm
}

# Mappa dei colori per le bande (sfumature tenui e riconoscibili)
colori_bande = {
    'm_g': '#33cc33',    # verde
    'm_r': '#ff3333',    # rosso
    'm_i': '#ff66cc',    # magenta
    'm_z': '#8b4513',    # marrone
    'm_y': '#66ccff'     # blu chiaro
}

# Aggiungo le strisce verticali per ogni banda
for banda, (inizio, fine) in intervalli.items():
    # Estraggo il nome base della banda separando la stringa allo spazio per trovare il colore corretto nel dizionario
    chiave_colore = banda.split()[0]
    ax.axvspan(inizio, fine, alpha=0.3, color=colori_bande[chiave_colore], label=banda)

# Traccio la curva principale usando il nero come colore
ax.plot(df['Wavelength'], df['QE'], color='#111111', linewidth=2)

# Inserisco la linea verticale tratteggiata in corrispondenza del valore 525
ax.axvline(x=525, color='#111111', linestyle=':', linewidth=1.5)

# Imposto i limiti degli assi per farli coincidere esattamente con il grafico originale
ax.set_xlim(300, 1100)
ax.set_ylim(0, 90)

# Aggiungo i testi descrittivi per gli assi X e Y, impostando font grandi per la stampa su A4
ax.set_xlabel('Wavelength (nm)', fontsize=16)
ax.set_ylabel('Quantum Efficiency (%)', fontsize=16)

# Ingrandisco le etichette numeriche sugli assi per renderle leggibili in scala
ax.tick_params(axis='both', which='major', labelsize=14)

# Posiziono il titolo in alto a sinistra, fuori dal perimetro del riquadro del grafico
fig.text(0.08, 0.95, 'BFS-PGE-63S4M', fontsize=16, fontweight='bold', fontfamily='sans-serif')

# Aggiungo la legenda per le bande, aumentando leggermente il font per la leggibilità
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

# Configuro la griglia per mostrare solo le linee orizzontali in tinta unita
ax.yaxis.grid(True, linestyle='-', color='#c0c0c0', alpha=0.8)
ax.xaxis.grid(False)

# Mi assicuro che la linea dati parta esattamente dai bordi senza spazi vuoti
#ax.margins(x=0, y=0)

# Disegno un intervallo per la banda Vmag sopra il riquadro per non intaccare i colori delle altre bande.
# Utilizzo ax.plot al posto di ax.annotate per fare in modo che le righe verticali siano solo sotto quella orizzontale.
#ax.plot([500, 500, 600, 600], [1.00, 1.02, 1.02, 1.00], color='black', lw=1., transform=ax.get_xaxis_transform(), clip_on=False)

# Aggiungo l'etichetta di testo "Vmag" centrata appena sopra la linea dell'intervallo
#ax.text(550, 1.03, 'm_V [500;600]', transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='black', clip_on=False)

# Aggiusto i margini della figura per fare in modo che il titolo non venga tagliato
plt.subplots_adjust(top=0.85)

# Salvo l'immagine
plt.savefig('curva_PMC_intervalli_ufficiali.png', dpi=300, bbox_inches='tight')

# Avvio il render a schermo del grafico finale
#plt.show()