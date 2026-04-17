import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import re
from pathlib import Path
import sys

# gestisco i warning ignorandoli per mantenere pulito il mio output
warnings.filterwarnings('ignore')


# =============================================================================
# 0. CONFIGURAZIONE PERCORSI
# =============================================================================

def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    print(f"WARNING: '{nome_target}' folder not found in the tree. Using script directory.")
    return path_corrente.parent


BASE_DIR = trova_cartella_base("Lorenzo")

# =============================================================================
# 1. CARICAMENTO DATI DAL CSV
# =============================================================================

# individuo il file CSV con gli oggetti consecutivi
percorso_csv = None
for file in BASE_DIR.rglob("oggetti_presenza_multirun.csv"):
    percorso_csv = file
    break

if percorso_csv is None:
    print("ERROR: Could not find 'oggetti_presenza_multirun.csv'.")
    sys.exit()

# carico i dati
df = pd.read_csv(percorso_csv)
print(f"Loaded {len(df)} objects from CSV")

# estraggo label e occorrenze
labels = df['label'].values
occorrenze = df['occorrenze'].values


# =============================================================================
# 2. ESTRAZIONE COORDINATE RA/DEC
# =============================================================================

def estrai_coordinate(label):
    """
    Estrae RA e DEC da un label nel formato RA_123.45DEC-67.89
    """
    pattern = r'RA_([\d\.]+)DEC([\-]?\d+\.?\d*)'
    match = re.match(pattern, label)

    if match:
        ra = float(match.group(1))
        dec = float(match.group(2))
        return ra, dec
    return None, None


# estraggo coordinate per tutti i label
ra_list = []
dec_list = []

for label in labels:
    ra, dec = estrai_coordinate(label)
    if ra is not None and dec is not None:
        ra_list.append(ra)
        dec_list.append(dec)
    else:
        print(f"Warning: Could not parse label: {label}")

print(f"Successfully extracted coordinates for {len(ra_list)} objects")

# =============================================================================
# 3. STATISTICHE DI BASE
# =============================================================================

print("\n=== Basic statistics ===")
print(f"Average run: {np.mean(occorrenze):.2f}")
print(f"Median run: {np.median(occorrenze):.2f}")
print(f"Min run: {np.min(occorrenze)}")
print(f"Max run: {np.max(occorrenze)}")
print(f"Standard deviation: {np.std(occorrenze):.2f}")

# =============================================================================
# 4. CREAZIONE GRAFICI
# =============================================================================

# individuo la cartella di output
cartella_output = None
for cartella in BASE_DIR.rglob("presenza_consecutiva_multirun"):
    cartella_output = cartella
    break

if cartella_output is None:
    cartella_output = Path.cwd()

# 4.1 ISTOGRAMMA DELLE occorrenze
plt.figure(figsize=(12, 7))
plt.hist(occorrenze, bins=range(min(occorrenze), max(occorrenze) + 2),
         color='skyblue', edgecolor='black', align='left', alpha=0.8)

plt.title("Distribution of object appearances", fontsize=14)
plt.xlabel("Occorrences", fontsize=12)
plt.ylabel("Number of objects", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

nome_istogramma = cartella_output / "label_occorrences_histogram.png"
plt.savefig(nome_istogramma, dpi=300)
print(f"\nHistogram saved at: {nome_istogramma}")
plt.show()

# 4.2 SCATTER PLOT SPAZIALE (colori = occorrenze)
plt.figure(figsize=(14, 10))

scatter = plt.scatter(ra_list, dec_list, c=occorrenze,
                      cmap='viridis', s=30, alpha=0.7,
                      edgecolors='black', linewidth=0.5)

cbar = plt.colorbar(scatter)
cbar.set_label('Occorrences', fontsize=12)

plt.title('Spatial distribution of objects (coloured by occurrence count)', fontsize=14)
plt.xlabel('Right Ascension (RA)', fontsize=12)
plt.ylabel('Declination (DEC)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')

# inverto l'asse RA (convenzione astronomica)
plt.gca().invert_xaxis()

# statistiche nel box
stats_text = f'Total objects: {len(ra_list)}\n'
stats_text += f'Min run: {np.min(occorrenze)}\n'
stats_text += f'Max run: {np.max(occorrenze)}\n'
stats_text += f'Mean run: {np.mean(occorrenze):.1f}'

plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

nome_scatter = cartella_output / "label_spatial_distribution_occorrences.png"
plt.savefig(nome_scatter, dpi=300, bbox_inches='tight')
print(f"Spatial distribution plot saved at: {nome_scatter}")
plt.show()

# 4.3 GRAFICO COMBINATO (opzionale)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# istogramma
ax1.hist(occorrenze, bins=range(min(occorrenze), max(occorrenze) + 2),
         color='skyblue', edgecolor='black', align='left', alpha=0.8)
ax1.set_title("Distribution of appearances", fontsize=12)
ax1.set_xlabel("Occorrences", fontsize=10)
ax1.set_ylabel("Number of objects", fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# scatter plot
scatter2 = ax2.scatter(ra_list, dec_list, c=occorrenze,
                       cmap='viridis', s=25, alpha=0.7,
                       edgecolors='black', linewidth=0.3)
ax2.set_title("Spatial distribution (colour = run)", fontsize=12)
ax2.set_xlabel("Right Ascension (RA)", fontsize=10)
ax2.set_ylabel("Declination (DEC)", fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.invert_xaxis()

cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('run', fontsize=10)

plt.tight_layout()
nome_combinato = cartella_output / "combined_analysis_occorrences.png"
plt.savefig(nome_combinato, dpi=300, bbox_inches='tight')
print(f"Combined plot saved at: {nome_combinato}")
plt.show()

# =============================================================================
# 5. STATISTICHE SPAZIALI
# =============================================================================

print("\n=== Spatial statistics ===")
print(f"RA range: {min(ra_list):.2f} - {max(ra_list):.2f}")
print(f"DEC range: {min(dec_list):.2f} - {max(dec_list):.2f}")

# identifico la regione con maggiore densità di oggetti
ra_bins = np.linspace(min(ra_list), max(ra_list), 20)
dec_bins = np.linspace(min(dec_list), max(dec_list), 20)
H, _, _ = np.histogram2d(ra_list, dec_list, bins=[ra_bins, dec_bins])

max_density_idx = np.unravel_index(np.argmax(H), H.shape)
ra_peak = (ra_bins[max_density_idx[0]] + ra_bins[max_density_idx[0] + 1]) / 2
dec_peak = (dec_bins[max_density_idx[1]] + dec_bins[max_density_idx[1] + 1]) / 2
print(f"Highest density region centred at: RA ≈ {ra_peak:.2f}, DEC ≈ {dec_peak:.2f}")

# oggetto con più occorrenze
idx_max = np.argmax(occorrenze)
print(f"\nObject with highest run ({occorrenze[idx_max]}): {labels[idx_max]}")
print(f"  -> RA: {ra_list[idx_max]:.2f}, DEC: {dec_list[idx_max]:.2f}")