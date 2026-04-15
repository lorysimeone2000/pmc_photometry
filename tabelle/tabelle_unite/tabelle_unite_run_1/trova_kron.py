import pandas as pd
import matplotlib.pyplot as plt

# Leggo il file CSV ignorando le righe iniziali di commento
df = pd.read_csv('run_1_stelle_trovate_e_catalogate_immagine_001.csv', comment='#')

# Estraggo i dati dalla colonna relativa al raggio di Kron
raggi_kron = df['raggio_kron_aper']

# Imposto la figura per il grafico
plt.figure(figsize=(10, 6))

# Genero l'istogramma definendo il numero di bin e i colori
plt.hist(raggi_kron, bins=100, color='skyblue', edgecolor='black')

# Aggiungo il titolo principale e le etichette per gli assi
plt.title('Istogramma dei Raggi di Kron')
plt.xlabel('Raggio di Kron')
plt.ylabel('Frequenza')

# Visualizzo a schermo il risultato finale
plt.show()