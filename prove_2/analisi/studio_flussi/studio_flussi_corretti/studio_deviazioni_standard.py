import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning

# gestisco i warning ignorandoli
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=VerifyWarning)


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI E CREAZIONE CARTELLA OUTPUT
# =============================================================================

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")
base_path = BASE_DIR / "tabelle/tabelle_unite"
run_list = [1, 2, 3]

# creo la cartella per il salvataggio degli istogrammi nella directory di esecuzione corrente
cartella_output = Path(os.getcwd()) / "istogrammi_dev"
cartella_output.mkdir(parents=True, exist_ok=True)
print(f"Cartella di output creata/verificata: {cartella_output}")

# --- STRUTTURE DATI GLOBALI ---

# definisco le categorie di stringhe per comporre tutti i nomi delle colonne
flussi_base = [
    'kron_manuale_aper',
    'kron_manuale_seg',
    'somma_apertura_ultimo_pixel',
    'flusso_fisso_max_run',
    'flusso_raggio_fisso_doppio',
    'flusso_intera_segmentazione',
    'flusso_kron_intera_segmentazione'
]

varianti_correzione = [
    '',
    '_CORRETTO_Correzione_Additiva_dell_Apertura',
    '_FONDO_SOTTRATTO'
]

varianti_decorrelazione = [
    '',
    '_DECORRELAZIONE_LINEARE',
    '_DECORRELAZIONE_STELLE',
    '_DECORRELAZIONE_LINEARE_DECORRELAZIONE_STELLE'
]

# genero dinamicamente tutti i nomi delle colonne di flusso possibili
colonne_flusso = []
for base in flussi_base:
    for var in varianti_correzione:
        for dec in varianti_decorrelazione:
            colonne_flusso.append(base + var + dec)

# =============================================================================
# 1. RACCOLTA DATI
# =============================================================================

print("--- FASE 1: Lettura dei dati dalle Run ---")
lista_df = []
totale_immagini = 0

for run in run_list:
    cartella_csv = os.path.join(base_path, f"tabelle_unite_run_{run}")

    if not os.path.exists(cartella_csv):
        continue

    file_csv = sorted([f for f in os.listdir(cartella_csv) if f.endswith('.csv')])

    for f in tqdm(file_csv):
        percorso_file = os.path.join(cartella_csv, f)
        try:
            # carico solo le righe relative a stelle catalogate per avere ID stabili
            df_temp = pd.read_csv(percorso_file, comment='#')
            if 'Corrispondenza' in df_temp.columns:
                mask_si = df_temp['Corrispondenza'].astype(str).str.startswith('SI')
                lista_df.append(df_temp[mask_si])
            totale_immagini += 1
        except Exception:
            pass

print(f"Elaborate {totale_immagini} immagini totali.")

# unisco tutto in un unico grande DataFrame
big_df = pd.concat(lista_df, ignore_index=True)

# filtro per mantenere solo le stelle con almeno 30 rilevazioni
conteggi_stelle = big_df['ID'].value_counts()
stelle_valide = conteggi_stelle[conteggi_stelle >= 30].index
big_df_valido = big_df[big_df['ID'].isin(stelle_valide)].copy()

print(f"Trovate {len(stelle_valide)} stelle con campionamento sufficiente.")

# =============================================================================
# 2. CALCOLO, GENERAZIONE ISTOGRAMMI E RACCOLTA STATISTICHE
# =============================================================================

print("\n--- FASE 2: Calcolo Deviazioni Standard e Creazione Grafici ---")

# inizializzo una lista in cui memorizzare le statistiche finali per la stampa
risultati_statistici = []

for col in tqdm(colonne_flusso):
    if col not in big_df_valido.columns:
        continue

    # mi assicuro che i dati siano numerici
    big_df_valido[col] = pd.to_numeric(big_df_valido[col], errors='coerce')

    # raggruppo per stella e calcolo la deviazione standard assoluta
    std_per_stella = big_df_valido.groupby('ID')[col].std().dropna()

    if len(std_per_stella) == 0:
        continue

    # estraggo la mediana globale delle deviazioni standard
    mediana_std = std_per_stella.median()
    # estraggo la deviazione standard di questa distribuzione di deviazioni standard
    std_delle_std = std_per_stella.std()

    # memorizzo i risultati per la stampa finale
    risultati_statistici.append({
        'flusso': col,
        'mediana_std': mediana_std,
        'std_della_std': std_delle_std
    })

    # imposto un limite superiore (98° percentile) per ignorare gli outlier estremi
    limite_superiore = std_per_stella.quantile(0.98)
    std_filtrate = std_per_stella[std_per_stella <= limite_superiore]

    # preparo il grafico
    plt.figure(figsize=(10, 6))

    # disegno l'istogramma
    plt.hist(std_filtrate, bins=60, color='purple', edgecolor='black', alpha=0.75)

    # aggiungo la linea della mediana
    plt.axvline(mediana_std, color='red', linestyle='dashed', linewidth=2,
                label=f'Mediana della Dev. Std: {mediana_std:.2f} ADU')

    # formatto il grafico
    plt.title(f"Distribuzione Deviazione Standard Assoluta tra le stelle\nFlusso: [{col}]", fontsize=12,
              fontweight='bold')
    plt.xlabel("Deviazione Standard nel tempo (ADU)", fontsize=11)
    plt.ylabel("Numero di Stelle", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # salvo e chiudo il grafico
    nome_file = f"istogramma_std_{col}.png"
    percorso_salvataggio = cartella_output / nome_file
    plt.savefig(percorso_salvataggio, dpi=300)
    plt.close()

# ordino i risultati in base alla mediana in ordine decrescente
risultati_statistici.sort(key=lambda x: x['mediana_std'], reverse=True)

# stampo a terminale la classifica
print("\n=================================================================================================")
print("CLASSIFICA DEVIAZIONI STANDARD (Mediana delle dev. std. per stella)")
print("=================================================================================================")
for ris in risultati_statistici:
    print(f"-> {ris['flusso']}:")
    print(f"   Mediana STD: {ris['mediana_std']:.2f} ADU  |  Std delle STD: {ris['std_della_std']:.2f} ADU")
print("=================================================================================================")

print(f"\nGenerazione completata. Tutti i grafici sono stati salvati in: {cartella_output}")