import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =============================================================================
# ANALISI GRAFICA DEL FLUSSO TRAMITE MEDIA MOBILE
# =============================================================================

# individuo la cartella dello script e carico il file csv salvato prima
cartella_script = Path(__file__).resolve().parent
file_csv = cartella_script / 'curva_di_luce_mrk421_valori.csv'

# leggo i dati tramite pandas
df = pd.read_csv(file_csv)

# estraggo le colonne per comodità
tempi = df['Tempo_trascorso_minuti']
flussi = df['Flusso_netto']

# imposto la dimensione della finestra per la media mobile
# 30 significa che ogni punto della linea rossa sarà la media di 30 scatti consecutivi
finestra = 30

# calcolo la media mobile centrata
media_mobile = flussi.rolling(window=finestra, center=True).mean()

# preparo il grafico
plt.figure(figsize=(14, 7))

# traccio i dati grezzi in grigio e in trasparenza per non distrarre la vista
plt.plot(tempi, flussi, marker='.', linestyle='', color='gray', alpha=0.3, markersize=4, label='Dati grezzi (singole immagini)')

# traccio la media mobile in rosso acceso per evidenziare il trend del flusso
plt.plot(tempi, media_mobile, color='red', linewidth=1, label=f'Media Mobile (finestra={finestra} img)')

plt.title("Analisi del Flusso Medio nel Tempo (Markarian 421)", fontsize=16)
plt.xlabel("Tempo trascorso dalla prima osservazione (Minuti)", fontsize=14)
plt.ylabel("Somma apertura di 0.6 arcmin", fontsize=14)

# imposto il limite dell'asse y basandomi sul massimo valore dei flussi
plt.ylim(0, flussi.max() * 1.05)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# salvo il grafico
output_png = cartella_script / 'analisi_flusso_medio_mrk421.png'
plt.savefig(output_png, dpi=300)
plt.show()