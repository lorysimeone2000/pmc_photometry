import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import re
from pathlib import Path
import sys
from matplotlib.ticker import MaxNLocator

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

# estraggo label e il flusso rinominando la variabile in mag_max
labels = df['label'].values
mag_max = df['Mag_estratta_max'].values


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
print(f"Average  mag max: {np.max(mag_max):.2f}")
print(f"Median  mag max: {np.median(mag_max):.2f}")
print(f"Min  mag max: {np.min(mag_max)}")
print(f"Max  mag max: {np.max(mag_max)}")
print(f"Standard deviation: {np.std(mag_max):.2f}")

# =============================================================================
# 4. CREAZIONE GRAFICI
# =============================================================================

# individuo la mia cartella di output di base
cartella_output_base = None
for cartella in BASE_DIR.rglob("presenza_consecutiva_multirun"):
    cartella_output_base = cartella
    break

if cartella_output_base is None:
    cartella_output_base = Path.cwd()

# creo la mia sottocartella a parte per salvare i png
cartella_output = cartella_output_base / "studio_flusso_multirun"
cartella_output.mkdir(parents=True, exist_ok=True)

# 4.1 ISTOGRAMMA DEL  mag max
plt.figure(figsize=(12, 7))
plt.hist(mag_max, bins='auto',
         color='skyblue', edgecolor='black', alpha=0.8)

plt.title("Distribution of  mag max", fontsize=14)
plt.xlabel(" mag max", fontsize=12)
plt.ylabel("Number of objects", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

nome_istogramma = cartella_output / "label_mag_histogram.png"
plt.savefig(nome_istogramma, dpi=300)
print(f"\nHistogram saved at: {nome_istogramma}")
#plt.show()

# 4.2 SCATTER PLOT SPAZIALE (colori = mag_max)
plt.figure(figsize=(14, 10))

scatter = plt.scatter(ra_list, dec_list, c=mag_max,
                      cmap='viridis_r', s=30, alpha=0.7,
                      edgecolors='black', linewidth=0.5)

cbar = plt.colorbar(scatter)
cbar.set_label(' mag max', fontsize=12)

plt.title('Spatial distribution of objects (coloured by mag max)', fontsize=14)
plt.xlabel('Right Ascension (RA)', fontsize=12)
plt.ylabel('Declination (DEC)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')

# inverto l'asse RA (convenzione astronomica)
plt.gca().invert_xaxis()

# statistiche nel box
stats_text = f'Total objects: {len(ra_list)}\n'
stats_text += f'Min  mag max: {np.min(mag_max):.2f}\n'
stats_text += f'Max  mag max: {np.max(mag_max):.2f}\n'
stats_text += f'max  mag max: {np.max(mag_max):.2f}'

plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

nome_scatter = cartella_output / "label_spatial_distribution.png"
plt.savefig(nome_scatter, dpi=300, bbox_inches='tight')
print(f"Spatial distribution plot saved at: {nome_scatter}")
#plt.show()

# 4.3 GRAFICO COMBINATO (opzionale)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# istogramma
ax1.hist(mag_max, bins='auto',
         color='skyblue', edgecolor='black', alpha=0.8)
ax1.set_title("Distribution of  mag max", fontsize=12)
ax1.set_xlabel(" mag max", fontsize=10)
ax1.set_ylabel("Number of objects", fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# scatter plot
scatter2 = ax2.scatter(ra_list, dec_list, c=mag_max,
                       cmap='viridis_r', s=25, alpha=0.7,
                       edgecolors='black', linewidth=0.3)
ax2.set_title("Spatial distribution (colour =  mag max)", fontsize=12)
ax2.set_xlabel("Right Ascension (RA)", fontsize=10)
ax2.set_ylabel("Declination (DEC)", fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.invert_xaxis()

cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label(' mag max', fontsize=10)

plt.tight_layout()
nome_combinato = cartella_output / "combined_analysis_mag.png"
plt.savefig(nome_combinato, dpi=300, bbox_inches='tight')
print(f"Combined plot saved at: {nome_combinato}")
#plt.show()

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
idx_max = np.argmax(mag_max)
print(f"\nObject with highest  mag max ({mag_max[idx_max]:.2f}): {labels[idx_max]}")
print(f"  -> RA: {ra_list[idx_max]:.2f}, DEC: {dec_list[idx_max]:.2f}")

# =============================================================================
# 6. HYPER-IN-DEPTH ANALYSIS: POSITION, OCCURRENCES, RUNS AND mag
# =============================================================================

print("\n=== Hyper-In-Depth Analysis: Position, Occurrences, Runs, and mag ===")

# estraggo le colonne dal dataset originale
occorrenze_totali = df['occorrenze'].values
# Estraggo anche il numero di run, che hai aggiunto precedentemente al CSV
numero_run_totali = df['numero_di_run'].values

# ricostruisco i miei array per assicurarmi che tutti i dati siano perfettamente allineati
occorrenze_list = []
numero_run_list = []
mag_list = []
ra_valid_list = []
dec_valid_list = []

for idx, label in enumerate(labels):
    ra, dec = estrai_coordinate(label)
    if ra is not None and dec is not None:
        ra_valid_list.append(ra)
        dec_valid_list.append(dec)
        occorrenze_list.append(occorrenze_totali[idx])
        numero_run_list.append(numero_run_totali[idx])
        mag_list.append(mag_max[idx])

# creo un DataFrame dedicato alla mia analisi correlazionale includendo Numero_Run
df_corr = pd.DataFrame({
    'RA': ra_valid_list,
    'DEC': dec_valid_list,
    'Occurrences': occorrenze_list,
    'Numero_Run': numero_run_list,
    'mag_max': mag_list
})

# 6.1 Matrice di Correlazione (Spearman per dipendenze non lineari)
correlazioni = df_corr.corr(method='spearman')
print("\nSpearman Correlation Matrix:")
print(correlazioni.to_string())

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(correlazioni, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)

# imposto i miei tick grafici per la matrice (ora ci sono 5 variabili)
ax.set_xticks(range(len(correlazioni.columns)))
ax.set_yticks(range(len(correlazioni.columns)))
etichette_matrice = ['RA', 'DEC', 'Occurrences', 'Numero Run', ' mag max']
ax.set_xticklabels(etichette_matrice, rotation=45, ha='left', fontsize=10)
ax.set_yticklabels(etichette_matrice, fontsize=10)

plt.title("Spearman Correlation Matrix", pad=20, fontsize=14)
plt.tight_layout()

nome_corr = cartella_output / "correlation_matrix_mag.png"
plt.savefig(nome_corr, dpi=300)
print(f"Correlation matrix plot saved at: {nome_corr}")
#plt.show()

# 6.2 Relazione diretta: Occorrenze vs mag max
plt.figure(figsize=(10, 6))
scatter_rel = plt.scatter(df_corr['Occurrences'], df_corr['mag_max'],
                        color='black', alpha=1, s=5)
plt.xlabel("Occurrences", fontsize=12)
plt.ylabel("Mag max", fontsize=12)
plt.title("Occurrences vs mag max", fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.gca().invert_yaxis()
plt.xscale('log', base=2)

nome_rel = cartella_output / "occurrences_vs_mag_versione_logaritmica_estrema.png"
plt.savefig(nome_rel, dpi=300)
print(f"Occurrences vs mag plot saved at: {nome_rel}")
#plt.show()

# 6.2 Relazione diretta: Occorrenze vs mag max
plt.figure(figsize=(10, 6))
scatter_rel = plt.scatter(df_corr['Numero_Run'], df_corr['mag_max'],
                        color='black', alpha=1, s=5)
plt.xlabel("Number of Runs", fontsize=12)
plt.ylabel("Mag max", fontsize=12)
plt.title("N run vs mag max", fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.gca().invert_yaxis()
plt.xscale('log')

nome_rel = cartella_output / "Numero_Run_vs_mag_versione_logaritmica.png"
plt.savefig(nome_rel, dpi=300)
print(f"Numero_Run vs mag plot saved at: {nome_rel}")
#plt.show()

# 6.3 Analisi Spaziale 3D: RA, DEC e Occorrenze
fig = plt.figure(figsize=(12, 8))
ax3d = fig.add_subplot(111, projection='3d')
scatter_3d = ax3d.scatter(df_corr['RA'], df_corr['DEC'], df_corr['Occurrences'],
                          c=df_corr['mag_max'], cmap='viridis_r', s=40, depthshade=True, edgecolors='k', linewidth=0.2)

ax3d.set_xlabel('Right Ascension (RA)')
ax3d.set_ylabel('Declination (DEC)')
ax3d.set_zlabel('Occurrences')
ax3d.set_title('3D Spatial-Temporal Distribution (coloured by  mag max)', fontsize=14)
ax3d.invert_xaxis()

cbar_3d = fig.colorbar(scatter_3d, ax=ax3d, pad=0.1)
cbar_3d.set_label(' mag max', fontsize=12)

nome_3d = cartella_output / "3d_spatial_analysis_mag.png"
plt.savefig(nome_3d, dpi=300)
print(f"3D Analysis plot saved at: {nome_3d}")
#plt.show()

# 6.4 Analisi delle distribuzioni e densità spaziali (Hexbin)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

hb1 = ax1.hexbin(df_corr['RA'], df_corr['DEC'], C=df_corr['Occurrences'],
                 gridsize=20, cmap='inferno', reduce_C_function=np.max)
ax1.set_title("Spatial map weighted by average occurrences", fontsize=12)
ax1.set_xlabel("Right Ascension (RA)")
ax1.set_ylabel("Declination (DEC)")
ax1.invert_xaxis()
cb1 = plt.colorbar(hb1, ax=ax1)
cb1.set_label('Average occurrences')

hb2 = ax2.hexbin(df_corr['RA'], df_corr['DEC'], C=df_corr['mag_max'],
                 gridsize=20, cmap='magma', reduce_C_function=np.max)
ax2.set_title("Spatial map weighted by average  mag max", fontsize=12)
ax2.set_xlabel("Right Ascension (RA)")
ax2.set_ylabel("Declination (DEC)")
ax2.invert_xaxis()
cb2 = plt.colorbar(hb2, ax=ax2)
cb2.set_label('Average  mag max')

plt.tight_layout()
nome_hexbin = cartella_output / "hexbin_spatial_analysis_mag.png"
plt.savefig(nome_hexbin, dpi=300)
print(f"Hexbin spatial analysis saved at: {nome_hexbin}")
#plt.show()

# 6.5 NUOVA Analisi 3D: Numero di Run, Occorrenze e Flusso
fig_run = plt.figure(figsize=(12, 8))
ax3d_run = fig_run.add_subplot(111, projection='3d')

scatter_3d_run = ax3d_run.scatter(df_corr['Numero_Run'], df_corr['Occurrences'], df_corr['mag_max'],
                                  c=df_corr['mag_max'], cmap='plasma', s=40, depthshade=True, edgecolors='k', linewidth=0.2)

ax3d_run.set_xlabel('Numero di Run')
ax3d_run.set_ylabel('Occorrenze')
ax3d_run.set_zlabel(' mag max')
ax3d_run.set_title('3D Analysis: Numero di Run, Occorrenze e Flusso', fontsize=14)

# Forza l'asse X a visualizzare solo numeri interi (es. 1, 2, 3...)
ax3d_run.xaxis.set_major_locator(MaxNLocator(integer=True))

cbar_3d_run = fig_run.colorbar(scatter_3d_run, ax=ax3d_run, pad=0.1)
cbar_3d_run.set_label(' mag max', fontsize=12)

nome_3d_run_plot = cartella_output / "3d_run_occurrences_mag.png"
plt.savefig(nome_3d_run_plot, dpi=300)
print(f"3D Analysis (Run/Occurrences/mag) plot saved at: {nome_3d_run_plot}")
#plt.show()