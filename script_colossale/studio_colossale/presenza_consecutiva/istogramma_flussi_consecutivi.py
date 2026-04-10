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
for file in BASE_DIR.rglob("oggetti_con_presenza_consecutiva.csv"):
    percorso_csv = file
    break

if percorso_csv is None:
    print("ERROR: Could not find 'oggetti_con_presenza_consecutiva.csv'.")
    sys.exit()

# carico i dati
df = pd.read_csv(percorso_csv)
print(f"Loaded {len(df)} objects from CSV")

# estraggo label e il flusso rinominando la variabile in kron_flux_mean
labels = df['label'].values
kron_flux_mean = df['media_flusso_fisso_max_run_CORRETTO_Correzione_Additiva_dell_Apertura_DECORRELAZIONE_STELLE_GLOBALE'].values


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
print(f"Average Kron flux mean: {np.mean(kron_flux_mean):.2f}")
print(f"Median Kron flux mean: {np.median(kron_flux_mean):.2f}")
print(f"Min Kron flux mean: {np.min(kron_flux_mean)}")
print(f"Max Kron flux mean: {np.max(kron_flux_mean)}")
print(f"Standard deviation: {np.std(kron_flux_mean):.2f}")

# =============================================================================
# 4. CREAZIONE GRAFICI
# =============================================================================

# individuo la cartella di output
cartella_output = None
for cartella in BASE_DIR.rglob("presenza_consecutiva"):
    cartella_output = cartella
    break

if cartella_output is None:
    cartella_output = Path.cwd()

# 4.1 ISTOGRAMMA DEL KRON FLUX MEAN
plt.figure(figsize=(12, 7))
plt.hist(kron_flux_mean, bins='auto',
         color='skyblue', edgecolor='black', alpha=0.8)

plt.title("Distribution of Kron flux mean", fontsize=14)
plt.xlabel("Kron flux mean", fontsize=12)
plt.ylabel("Number of objects", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

nome_istogramma = cartella_output / "label_flux_histogram.png"
plt.savefig(nome_istogramma, dpi=300)
print(f"\nHistogram saved at: {nome_istogramma}")
plt.show()

# 4.2 SCATTER PLOT SPAZIALE (colori = kron_flux_mean)
plt.figure(figsize=(14, 10))

scatter = plt.scatter(ra_list, dec_list, c=kron_flux_mean,
                      cmap='viridis', s=30, alpha=0.7,
                      edgecolors='black', linewidth=0.5)

cbar = plt.colorbar(scatter)
cbar.set_label('Kron flux mean', fontsize=12)

plt.title('Spatial distribution of objects (coloured by Kron flux mean)', fontsize=14)
plt.xlabel('Right Ascension (RA)', fontsize=12)
plt.ylabel('Declination (DEC)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')

# inverto l'asse RA (convenzione astronomica)
plt.gca().invert_xaxis()

# statistiche nel box
stats_text = f'Total objects: {len(ra_list)}\n'
stats_text += f'Min Kron flux mean: {np.min(kron_flux_mean):.2f}\n'
stats_text += f'Max Kron flux mean: {np.max(kron_flux_mean):.2f}\n'
stats_text += f'Mean Kron flux mean: {np.mean(kron_flux_mean):.2f}'

plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

nome_scatter = cartella_output / "label_spatial_distribution.png"
plt.savefig(nome_scatter, dpi=300, bbox_inches='tight')
print(f"Spatial distribution plot saved at: {nome_scatter}")
plt.show()

# 4.3 GRAFICO COMBINATO (opzionale)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# istogramma
ax1.hist(kron_flux_mean, bins='auto',
         color='skyblue', edgecolor='black', alpha=0.8)
ax1.set_title("Distribution of Kron flux mean", fontsize=12)
ax1.set_xlabel("Kron flux mean", fontsize=10)
ax1.set_ylabel("Number of objects", fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# scatter plot
scatter2 = ax2.scatter(ra_list, dec_list, c=kron_flux_mean,
                       cmap='viridis', s=25, alpha=0.7,
                       edgecolors='black', linewidth=0.3)
ax2.set_title("Spatial distribution (colour = Kron flux mean)", fontsize=12)
ax2.set_xlabel("Right Ascension (RA)", fontsize=10)
ax2.set_ylabel("Declination (DEC)", fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.invert_xaxis()

cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Kron flux mean', fontsize=10)

plt.tight_layout()
nome_combinato = cartella_output / "combined_analysis.png"
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

# oggetto con più flusso
idx_max = np.argmax(kron_flux_mean)
print(f"\nObject with highest Kron flux mean ({kron_flux_mean[idx_max]:.2f}): {labels[idx_max]}")
print(f"  -> RA: {ra_list[idx_max]:.2f}, DEC: {dec_list[idx_max]:.2f}")

# =============================================================================
# 6. HYPER-IN-DEPTH ANALYSIS: POSITION, OCCURRENCES AND FLUX
# =============================================================================

print("\n=== Hyper-In-Depth Analysis: Position, Occurrences, and Flux ===")

# estraggo la colonna delle occorrenze originaria dal dataset
occorrenze_totali = df['occorrenze'].values

# ricostruisco i miei array per assicurarmi che tutti i dati siano perfettamente allineati
# nel caso in cui un label non venisse convertito
occorrenze_list = []
flux_list = []
ra_valid_list = []
dec_valid_list = []

for idx, label in enumerate(labels):
    ra, dec = estrai_coordinate(label)
    if ra is not None and dec is not None:
        ra_valid_list.append(ra)
        dec_valid_list.append(dec)
        occorrenze_list.append(occorrenze_totali[idx])
        flux_list.append(kron_flux_mean[idx])

# creo un DataFrame dedicato alla mia analisi correlazionale
df_corr = pd.DataFrame({
    'RA': ra_valid_list,
    'DEC': dec_valid_list,
    'Occurrences': occorrenze_list,
    'Kron_flux_mean': flux_list
})

# 6.1 Matrice di Correlazione (Spearman per dipendenze non lineari)
correlazioni = df_corr.corr(method='spearman')
print("\nSpearman Correlation Matrix:")
print(correlazioni.to_string())

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(correlazioni, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)

# imposto i miei tick grafici per la matrice
ax.set_xticks(range(len(correlazioni.columns)))
ax.set_yticks(range(len(correlazioni.columns)))
ax.set_xticklabels(['RA', 'DEC', 'Occurrences', 'Kron flux mean'], rotation=45, ha='left', fontsize=10)
ax.set_yticklabels(['RA', 'DEC', 'Occurrences', 'Kron flux mean'], fontsize=10)

plt.title("Spearman Correlation Matrix", pad=20, fontsize=14)
plt.tight_layout()

nome_corr = cartella_output / "correlation_matrix.png"
plt.savefig(nome_corr, dpi=300)
print(f"Correlation matrix plot saved at: {nome_corr}")
plt.show()

# 6.2 Relazione diretta: Occorrenze vs Kron Flux Mean
plt.figure(figsize=(10, 6))
scatter_rel = plt.scatter(df_corr['Occurrences'], df_corr['Kron_flux_mean'],
                          c=df_corr['DEC'], cmap='plasma', alpha=0.7, edgecolors='black')
plt.xlabel("Occurrences", fontsize=12)
plt.ylabel("Kron flux mean", fontsize=12)
plt.title("Occurrences vs Kron flux mean (coloured by DEC)", fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')

cbar_rel = plt.colorbar(scatter_rel)
cbar_rel.set_label('Declination (DEC)', fontsize=12)

nome_rel = cartella_output / "occurrences_vs_flux.png"
plt.savefig(nome_rel, dpi=300)
print(f"Occurrences vs Flux plot saved at: {nome_rel}")
plt.show()

# 6.3 Analisi Spaziale 3D: RA, DEC e Occorrenze
# imposto il mio grafico 3D interattivo
fig = plt.figure(figsize=(12, 8))
ax3d = fig.add_subplot(111, projection='3d')
scatter_3d = ax3d.scatter(df_corr['RA'], df_corr['DEC'], df_corr['Occurrences'],
                          c=df_corr['Kron_flux_mean'], cmap='viridis', s=40, depthshade=True, edgecolors='k', linewidth=0.2)

ax3d.set_xlabel('Right Ascension (RA)')
ax3d.set_ylabel('Declination (DEC)')
ax3d.set_zlabel('Occurrences')
ax3d.set_title('3D Spatial-Temporal Distribution (coloured by Kron flux mean)', fontsize=14)
# inverto l'asse RA per convenzione astronomica
ax3d.invert_xaxis()

cbar_3d = fig.colorbar(scatter_3d, ax=ax3d, pad=0.1)
cbar_3d.set_label('Kron flux mean', fontsize=12)

nome_3d = cartella_output / "3d_spatial_analysis.png"
plt.savefig(nome_3d, dpi=300)
print(f"3D Analysis plot saved at: {nome_3d}")
plt.show()

# 6.4 Analisi delle distribuzioni e densità spaziali (Hexbin)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# valuto la mappa delle occorrenze medie spaziali
hb1 = ax1.hexbin(df_corr['RA'], df_corr['DEC'], C=df_corr['Occurrences'],
                 gridsize=20, cmap='inferno', reduce_C_function=np.mean)
ax1.set_title("Spatial map weighted by average occurrences", fontsize=12)
ax1.set_xlabel("Right Ascension (RA)")
ax1.set_ylabel("Declination (DEC)")
ax1.invert_xaxis()
cb1 = plt.colorbar(hb1, ax=ax1)
cb1.set_label('Average occurrences')

# valuto la mappa del Kron flux mean medio
hb2 = ax2.hexbin(df_corr['RA'], df_corr['DEC'], C=df_corr['Kron_flux_mean'],
                 gridsize=20, cmap='magma', reduce_C_function=np.mean)
ax2.set_title("Spatial map weighted by average Kron flux mean", fontsize=12)
ax2.set_xlabel("Right Ascension (RA)")
ax2.set_ylabel("Declination (DEC)")
ax2.invert_xaxis()
cb2 = plt.colorbar(hb2, ax=ax2)
cb2.set_label('Average Kron flux mean')

plt.tight_layout()
nome_hexbin = cartella_output / "hexbin_spatial_analysis.png"
plt.savefig(nome_hexbin, dpi=300)
print(f"Hexbin spatial analysis saved at: {nome_hexbin}")
plt.show()