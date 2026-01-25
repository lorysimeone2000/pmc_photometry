from astropy.table import Table
import numpy as np

from astropy.table import Table, vstack

# Tabella 1 con colonne A, B
t1 = Table()
t1['ID'] = [1, 2]
t1['RA'] = [83.82, 84.12]
t1['FLUX'] = [150.5, 89.3]  # Solo in t1

# Tabella 2 con colonne A, C
t2 = Table()
t2['ID'] = [3, 4]
t2['RA'] = [84.25, 84.50]
t2['SNR'] = [15.2, 8.7]    # Solo in t2

# vstack riempie con valori mancanti
t_unita = vstack([t1, t2])
print(t_unita)

