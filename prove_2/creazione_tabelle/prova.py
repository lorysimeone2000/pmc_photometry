import pandas as pd
import numpy as np
from pathlib import Path

def trova_cartella_base(nome_target="pmc_photometry"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"ATTENZIONE: Cartella '{nome_target}' non trovata nell'albero. Uso la directory dello script.")
    return path_corrente.parent

def cerca_cartella_nel_progetto(base_dir, nome_cartella_esatto):
    cartelle_trovate = [p for p in base_dir.rglob(nome_cartella_esatto) if p.is_dir()]
    if not cartelle_trovate: return None
    cartelle_trovate.sort(key=lambda p: len(str(p)))
    if len(cartelle_trovate) > 1:
        print(
            f"INFO: Trovate {len(cartelle_trovate)} cartelle '{nome_cartella_esatto}'. Uso la prima: {cartelle_trovate[0].relative_to(base_dir)}")
    return cartelle_trovate[0]

pmc_photometry = trova_cartella_base("pmc_photometry")

colonne = ['albero', 'gino']

cc1 = np.arange(10, 20)
cc2 = np.linspace(-2,2, 10)

df = pd.DataFrame(columns=colonne)

df['albero'] = cc1
df['gino'] = cc2

output_dir = cerca_cartella_nel_progetto(pmc_photometry,"tabelle_unite_run_1")

print(output_dir)

file_path = output_dir / 'prova.csv'
print(file_path)




df.to_csv(file_path, index=False)