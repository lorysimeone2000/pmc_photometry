import pandas as pd
import pyarrow.parquet as pq

file_parquet = 'run_1_stelle_trovate_e_catalogate_immagine_001.parquet'

# Leggo esclusivamente lo schema del file Parquet per accedere ai metadati
schema = pq.read_schema(file_parquet)

# Recupero il dizionario dei metadati salvati nello schema
metadati = schema.metadata

# Verifico se i metadati esistono e se contengono la mia chiave personalizzata
if metadati and b'metadati_intestazione_csv' in metadati:
    # Decodifico la stringa da byte a testo normale usando la codifica utf-8
    metadati_testo = metadati[b'metadati_intestazione_csv'].decode('utf-8')
    print("--- Metadati estratti ---")
    print(metadati_testo)
    print("-------------------------\n")
else:
    print("Nessun metadato personalizzato trovato.")

# Carico l'intero contenuto del file Parquet estraendo la tabella in un DataFrame
df = pd.read_parquet(file_parquet)

# Stampo a schermo le prime righe del DataFrame per verificare i dati estratti
print("--- Dati della tabella ---")
print(df[df['media_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE']>1000]['media_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE'])