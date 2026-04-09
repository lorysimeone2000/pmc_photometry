import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from pathlib import Path
import sys
import os
import re
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import warnings

# gestisco i warning ignorandoli per mantenere pulito il mio output
warnings.filterwarnings('ignore')


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E MODULI ESTERNI
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"WARNING: '{nome_target}' folder not found in the tree. Using script directory.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")
PERCORSO_FUNZIONI = os.path.join(str(BASE_DIR), "pmc_photometry")

if PERCORSO_FUNZIONI not in sys.path:
    sys.path.append(PERCORSO_FUNZIONI)

# importo l'utilità per estrarre i metadati
from funzioni.utilita_parquet import leggi_header_da_parquet

# nome della colonna del flusso che mi hai richiesto
COLONNA_FLUSSO = 'flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE'

# =============================================================================
# 1. IDENTIFICAZIONE DI UNA STELLA CASUALE VICINO A M1
# =============================================================================

print("Caricamento del catalogo delle stelle persistenti...")
percorso_catalogo = BASE_DIR / "catalogo_stelle_persistente_COLOSSALE.parquet"

if not percorso_catalogo.exists():
    print(f"ERRORE: Catalogo non trovato in {percorso_catalogo}")
    sys.exit()

# leggo solo la colonna label per non appesantire la memoria
df_cat = pd.read_parquet(percorso_catalogo, columns=['label'])

# estraggo RA e DEC da tutte le label usando un'espressione regolare vettorializzata (molto più veloce)
pattern = r'RA_([\d\.]+)DEC([\-]?\d+\.?\d*)'
estratti = df_cat['label'].str.extract(pattern).astype(float)

# filtro le label formattate male
maschera_validi = estratti[0].notna() & estratti[1].notna()
df_cat_valido = df_cat[maschera_validi].copy()
estratti_validi = estratti[maschera_validi]

print("Calcolo delle distanze angolari dalla Nebulosa del Granchio (M1)...")
# definisco le coordinate esatte di M1
coord_m1 = SkyCoord(ra=83.63308 * u.deg, dec=22.0145 * u.deg)

# creo un array di coordinate per tutte le stelle del catalogo
coord_stelle = SkyCoord(ra=estratti_validi[0].values * u.deg, dec=estratti_validi[1].values * u.deg)

# misuro le separazioni e applico il filtro dei 2 gradi
separazioni = coord_stelle.separation(coord_m1)
maschera_vicini = separazioni < 2.0 * u.deg

labels_vicini = df_cat_valido['label'][maschera_vicini].values

if len(labels_vicini) == 0:
    print("ERRORE: Nessuna stella trovata entro 2 gradi da M1 nel catalogo fornito.")
    sys.exit()

# scelgo una stella bersaglio in maniera completamente casuale
label_bersaglio = np.random.choice(labels_vicini)
print(f"Trovate {len(labels_vicini)} stelle. Selezionata casualmente: {label_bersaglio}")

# =============================================================================
# 2. RICERCA DEL FLUSSO E DEI TEMPI NEI FILE PARQUET
# =============================================================================

cartella_tabelle = BASE_DIR / "tabelle_COLOSSALE_alleggerito"
file_parquet = list(cartella_tabelle.rglob("*run_*_run_*_immagine_*.parquet"))

if not file_parquet:
    print("ERRORE: Nessun file trovato nella cartella tabelle_COLOSSALE_alleggerito.")
    sys.exit()

tempi_tstart = []
flussi = []

for file_p in tqdm(file_parquet, desc=f"Ricerca del flusso per {label_bersaglio}"):
    try:
        # sfrutto pyarrow per caricare SOLTANTO la riga della stella che mi interessa e la sola colonna del flusso
        tabella = pq.read_table(file_p, columns=['label', COLONNA_FLUSSO], filters=[('label', '=', label_bersaglio)])

        # se ho trovato l'oggetto in questo frame
        if tabella.num_rows > 0:
            header_dict = leggi_header_da_parquet(file_p)
            tstart = header_dict.get('TSTART')

            if tstart is not None:
                # estraggo il valore del flusso convertendolo da array PyArrow a float nativo
                flusso_estratto = tabella.column(COLONNA_FLUSSO)[0].as_py()

                tempi_tstart.append(float(tstart))
                flussi.append(flusso_estratto)
    except Exception:
        # ignoro i file corrotti o che non contengono la colonna specificata
        continue

if not tempi_tstart:
    print(f"Nessun dato di flusso trovato nei file per la stella {label_bersaglio}.")
    sys.exit()

# =============================================================================
# 3. ORDINAMENTO E CREAZIONE DELLA CURVA DI LUCE
# =============================================================================

print("\nOrdinamento cronologico e generazione del grafico in corso...")
# creo un dataframe temporaneo per allineare temporalmente TSTART e Flusso
df_curva = pd.DataFrame({'TSTART': tempi_tstart, 'Flusso': flussi})
df_curva = df_curva.sort_values(by='TSTART').reset_index(drop=True)

# genero il grafico
plt.figure(figsize=(12, 6))
plt.plot(df_curva['TSTART'], df_curva['Flusso'], marker='o', linestyle='-', color='indigo', alpha=0.8, markersize=5)

# calcolo e aggiungo statistiche rapide sul grafico
media_flusso = df_curva['Flusso'].mean()
std_flusso = df_curva['Flusso'].std()
plt.axhline(media_flusso, color='red', linestyle='--', alpha=0.7, label=f'Mean Flux: {media_flusso:.2f}')

# personalizzo le etichette
plt.title(f"Light Curve for Star: {label_bersaglio}\n(Within 2° of Crab Nebula)", fontsize=14)
plt.xlabel("Observation Time (TSTART)", fontsize=12)
plt.ylabel("Corrected Aperture Flux", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# salvo il risultato
cartella_output = BASE_DIR / "studio_colossale"
cartella_output.mkdir(parents=True, exist_ok=True)
nome_plot = cartella_output / f"light_curve_{label_bersaglio}.png"
plt.tight_layout()
plt.savefig(nome_plot, dpi=300)

print(f"Operazione completata! Curva di luce salvata in: {nome_plot}")
plt.show()