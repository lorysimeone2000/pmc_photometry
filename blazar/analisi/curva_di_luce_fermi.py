import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

def trova_cartella_base(nome_target="Lorenzo"):
    # risalgo l'albero delle directory per trovare la radice del progetto
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")

PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

from funzioni.utilita import *
from funzioni.astrometria import *

percorso_csv = cerca_file_nel_progetto(BASE_DIR,"4FGL_J1104.4+3812_daily_04_03_2026.csv")

# leggo il file CSV scaricato dal telescopio Fermi-LAT
df = pd.read_csv(percorso_csv)

# identifico le colonne esatte di flusso ed errore
colonna_flusso = 'Photon Flux [0.1-100 GeV](photons cm-2 s-1)'
colonna_errore = 'Photon Flux Error(photons cm-2 s-1)'

# converto la colonna delle date in un formato datetime comprensibile per il grafico
df['Data_Formattata'] = pd.to_datetime(df['Date(UTC)'])

# controllo se nei dati del flusso ci sono dei limiti superiori (indicati con il simbolo '<')
df['limite_superiore'] = df[colonna_flusso].astype(str).str.contains('<')

# pulisco la colonna del flusso rimuovendo eventuali '<' e convertendo tutto in valori numerici decimali
df['Flusso_Numerico'] = df[colonna_flusso].astype(str).str.replace('<', '').astype(float)

# pulisco la colonna dell'errore convertendo i dati in numerici e gestendo eventuali trattini ('-') come assenza di dati (NaN)
df['Errore_Numerico'] = pd.to_numeric(df[colonna_errore], errors='coerce')

# imposto la figura in modo da renderla larga e chiara da leggere
fig, ax = plt.subplots(figsize=(14, 7))

# separo i dati rilevati effettivamente dai limiti superiori
rilevamenti = df[~df['limite_superiore']]
limiti = df[df['limite_superiore']]

# disegno i punti dei flussi misurati con le loro barre di errore in grigio
ax.errorbar(rilevamenti['Data_Formattata'], rilevamenti['Flusso_Numerico'],
            yerr=rilevamenti['Errore_Numerico'], fmt='o', color='black',
            ecolor='gray', elinewidth=1, markersize=3, label='Rilevamenti validi')

# disegno i limiti superiori usando dei triangolini rossi rivolti verso il basso per distinguerli
ax.scatter(limiti['Data_Formattata'], limiti['Flusso_Numerico'],
           marker='v', color='red', s=10, alpha=0.5, label='Limiti superiori')

# aggiungo i titoli, le etichette per gli assi e la griglia di sfondo
ax.set_title('Curva di Luce Fermi-LAT: Markarian 421', fontsize=16)
ax.set_xlabel('Data di Osservazione (UTC)', fontsize=14)
ax.set_ylabel('Flusso Fotoni [0.1-100 GeV] (ph $cm^{-2} s^{-1}$)', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)

# definisco i limiti temporali per l'asse X
data_inizio = pd.to_datetime('2025-12-18')
data_fine = pd.to_datetime('2026-01-25')

# applico il limite all'asse X
ax.set_xlim(data_inizio, data_fine)

# filtro i dati all'interno di questo intervallo per trovare il valore massimo del flusso
mask_intervallo = (df['Data_Formattata'] >= data_inizio) & (df['Data_Formattata'] <= data_fine)
flusso_massimo = df.loc[mask_intervallo, 'Flusso_Numerico'].max()

# imposto i limiti dell'asse Y da 0 al valore massimo (con un 5% di margine in alto per estetica)
ax.set_ylim(0, flusso_massimo * 1.25)

# inserisco la legenda
ax.legend(fontsize=12)

# organizzo lo spazio e salvo l'immagine
plt.tight_layout()
plt.savefig('curva_di_luce_fermi_mrk421.png', dpi=300)
plt.show()