import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import re
from scipy.stats import skew, kurtosis


# Definisco la mia funzione per localizzare la cartella di base del progetto
def trova_cartella_base(nome_target="Lorenzo"):
    path_corrente = Path(__file__).resolve()
    for parent in [path_corrente] + list(path_corrente.parents):
        if parent.name == nome_target:
            return parent
    return path_corrente.parent


# Inizializzo i miei percorsi di lavoro
BASE_DIR = trova_cartella_base("Lorenzo")
percorso_csv = BASE_DIR / "pmc_photometry" / "script_colossale"/"studio_colossale"/"analisi_costanza_dinamica_ULTRA_PRECISA.csv"
output_dir = BASE_DIR / "pmc_photometry"/"script_colossale"/"studio_colossale"/"risultati_analisi_dettagliata"
output_dir.mkdir(parents=True, exist_ok=True)


# Definisco la funzione per salvare i plot con nome basato sul titolo
def salva_plot_con_titolo(titolo, formato='png', dpi=150):
    """
    Salva il plot corrente con nome file derivato dal titolo.
    Converte il titolo in un nome file valido (sostituisce spazi, punteggiatura, ecc.)
    """
    # Pulisco il titolo per creare un nome file valido
    nome_file = titolo.lower()
    nome_file = re.sub(r'[^\w\s-]', '', nome_file)  # Rimuove punteggiatura
    nome_file = re.sub(r'[-\s]+', '_', nome_file)  # Sostituisce spazi e trattini con underscore
    nome_file = nome_file.strip('_')

    percorso_completo = output_dir / f"{nome_file}.{formato}"
    plt.savefig(percorso_completo, dpi=dpi, bbox_inches='tight')
    print(f"✅ Salvato: {percorso_completo}")
    return percorso_completo


# Carico il mio dataset e verifico la consistenza dei dati
if not percorso_csv.exists():
    print(f"Errore: Non trovo il file {percorso_csv}")
    exit()

df = pd.read_csv(percorso_csv)

# Pulisco i nomi delle colonne
df.columns = df.columns.str.strip()

# Mi assicuro che le colonne numeriche siano del tipo corretto
numeric_cols = ['RA_centroid', 'DEC_centroid', 'Copertura_Immagini', 'Presenza_Effettiva', 'Costanza_Percentuale']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Rimuovo le righe con NaN nelle colonne essenziali
df_clean = df.dropna(subset=['Costanza_Percentuale', 'Copertura_Immagini', 'Presenza_Effettiva'])

print("📊 DATASET ORIGINALE")
print(f"Righe totali: {len(df)}")
print(f"Righe valide (dopo pulizia): {len(df_clean)}")
print(f"Output salvati in: {output_dir}")
print("\n" + "=" * 80 + "\n")

# ------------------------------
# 1. STATISTICHE DESCRITTIVE
# ------------------------------
print("📈 STATISTICHE DESCRITTIVE")
stats = df_clean['Costanza_Percentuale'].describe()
print(stats)
print(f"\nSkewness (asimmetria): {skew(df_clean['Costanza_Percentuale'].dropna()):.3f}")
print(f"Kurtosis (curtosi): {kurtosis(df_clean['Costanza_Percentuale'].dropna()):.3f}")
print("\n" + "=" * 80 + "\n")

# ------------------------------
# 2. DISTRIBUZIONE COSTANZA
# ------------------------------
print("🔹 DISTRIBUZIONE COSTANZA")
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
df_clean['Costanza_Range'] = pd.cut(df_clean['Costanza_Percentuale'], bins=bins, labels=labels, right=False)
freq = df_clean['Costanza_Range'].value_counts().sort_index()
print(freq)
print("\n" + "=" * 80 + "\n")


# ------------------------------
# 3. CLASSIFICAZIONE STABILITÀ
# ------------------------------
def classifica_stabilita(val):
    if val >= 90:
        return 'Very stable'
    elif val >= 70:
        return 'Stable'
    elif val >= 50:
        return 'Moderately stable'
    elif val >= 25:
        return 'Unstable'
    else:
        return 'Very unstable'


df_clean['Stabilita'] = df_clean['Costanza_Percentuale'].apply(classifica_stabilita)
stabilita_counts = df_clean['Stabilita'].value_counts()
print("📉 CLASSI DI STABILITÀ")
print(stabilita_counts)
print("\n" + "=" * 80 + "\n")

# ------------------------------
# 4. GRAFICO 1: Istogramma della Costanza
# ------------------------------
plt.figure(figsize=(12, 8))
sns.histplot(df_clean['Costanza_Percentuale'], bins=30, kde=True, color='teal', edgecolor='black', alpha=0.7)
plt.title('Percentage Constancy Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Constancy (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(df_clean['Costanza_Percentuale'].median(), color='red', linestyle='--',
            linewidth=2, label=f'Median: {df_clean["Costanza_Percentuale"].median():.1f}%')
plt.axvline(df_clean['Costanza_Percentuale'].mean(), color='orange', linestyle='--',
            linewidth=2, label=f'Mean: {df_clean["Costanza_Percentuale"].mean():.1f}%')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
salva_plot_con_titolo('Distribuzione della Costanza Percentuale')
plt.close()

# ------------------------------
# 5. GRAFICO 2: Boxplot classi di stabilità
# ------------------------------
plt.figure(figsize=(12, 8))
order = ['Very unstable', 'Unstable', 'Moderately stable', 'Stable', 'Very stable']
colors_box = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850']
sns.boxplot(x='Stabilita', y='Costanza_Percentuale', data=df_clean,
            order=order, palette=colors_box, flierprops={'markersize': 3, 'alpha': 0.5})
plt.title('Constancy Distribution by Stability Class', fontsize=14, fontweight='bold')
plt.xlabel('Stability Class', fontsize=12)
plt.ylabel('Constancy (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
salva_plot_con_titolo('Distribuzione della Costanza per Classe di Stabilita')
plt.close()

# ------------------------------
# 6. GRAFICO 3: Bar plot classi di stabilità
# ------------------------------
plt.figure(figsize=(12, 8))

# Uso .get(cat, 0) per avere sempre 5 elementi e prevenire errori di lunghezza se una classe è assente
counts_ordered = [stabilita_counts.get(cat, 0) for cat in order]
bars = plt.bar(order, counts_ordered, color=colors_box, edgecolor='black', linewidth=1.5)
plt.title('Number of Objects by Stability Class', fontsize=14, fontweight='bold')
plt.xlabel('Stability Class', fontsize=12)
plt.ylabel('Number of Objects', fontsize=12)

# Applica la scala logaritmica solo se ci sono valori per evitare errori grafici
if max(counts_ordered) > 0:
    plt.yscale('log')
plt.grid(True, alpha=0.3, axis='y')

# Aggiungo le percentuali sulle barre ignorando quelle a zero
total = len(df_clean)
for bar, count in zip(bars, counts_ordered):
    if count > 0:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{count}\n({count / total * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.xticks(rotation=45, ha='right')
salva_plot_con_titolo('Numero di Oggetti per Classe di Stabilita')
plt.close()

# ------------------------------
# 7. GRAFICO 4: Mappa spaziale RA vs DEC
# ------------------------------
plt.figure(figsize=(14, 10))
scatter = plt.scatter(df_clean['RA_centroid'], df_clean['DEC_centroid'],
                      c=df_clean['Costanza_Percentuale'], cmap='RdYlGn_r',
                      s=15, alpha=0.6, edgecolors='black', linewidth=0.3)
plt.colorbar(scatter, label='Constancy (%)', fraction=0.046, pad=0.04)
plt.xlabel('RA (degrees)', fontsize=12)
plt.ylabel('DEC (degrees)', fontsize=12)
plt.title('Constancy Map in the RA-DEC Plane', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
salva_plot_con_titolo('Mappa della Costanza nel Piano RA-DEC')
plt.close()

# ------------------------------
# 8. GRAFICO 5: Presenza vs Copertura (con evidenziazione persistenti)
# ------------------------------
plt.figure(figsize=(12, 10))

persistenti = df_clean[df_clean['Costanza_Percentuale'] == 100]
non_persistenti = df_clean[df_clean['Costanza_Percentuale'] < 100]

plt.scatter(non_persistenti['Copertura_Immagini'],
            non_persistenti['Presenza_Effettiva'],
            c=non_persistenti['Costanza_Percentuale'],
            cmap='viridis', s=10, alpha=0.5, label='Constancy < 100%')

plt.scatter(persistenti['Copertura_Immagini'],
            persistenti['Presenza_Effettiva'],
            c='red', s=80, marker='*', edgecolors='black', linewidth=1,
            label=f'100% Persistent (n={len(persistenti)})')

# Traccio la linea y=x
max_val = max(df_clean['Copertura_Immagini'].max(), df_clean['Presenza_Effettiva'].max())
plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=1.5, label='y=x (perfect persistence)')

plt.colorbar(label='Constancy (%)')
plt.xlabel('Image Coverage (total number of observations)', fontsize=12)
plt.ylabel('Effective Presence (number of detections)', fontsize=12)
plt.title('Relationship between Coverage and Effective Presence', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3)
salva_plot_con_titolo('Relazione tra Copertura e Presenza Effettiva')
plt.close()

# ------------------------------
# 9. GRAFICO 6: Heatmap 2D (hexbin) della costanza
# ------------------------------
plt.figure(figsize=(14, 10))
hb = plt.hexbin(df_clean['RA_centroid'], df_clean['DEC_centroid'],
                C=df_clean['Costanza_Percentuale'], gridsize=80,
                cmap='plasma', mincnt=1, edgecolors='none')
plt.colorbar(hb, label='Constancy (%)', fraction=0.046, pad=0.04)
plt.xlabel('RA (degrees)', fontsize=12)
plt.ylabel('DEC (degrees)', fontsize=12)
plt.title('2D Constancy Heatmap (hexagonal binning)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.2)
salva_plot_con_titolo('Heatmap 2D della Costanza')
plt.close()

# ------------------------------
# 10. GRAFICO 7: Pie chart classi di stabilità (escludendo la categoria dominante)
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Definisco una mappa per i colori e i valori di explode per adattarmi dinamicamente alle classi presenti
mappa_colori = {
    'Very unstable': '#d73027',
    'Unstable': '#fc8d59',
    'Moderately stable': '#fee08b',
    'Stable': '#d9ef8b',
    'Very stable': '#1a9850'
}
mappa_explode = {
    'Very unstable': 0.05,
    'Unstable': 0.02,
    'Moderately stable': 0.02,
    'Stable': 0.02,
    'Very stable': 0.05
}

# Plotto tutte le categorie
colori_effettivi1 = [mappa_colori[cat] for cat in stabilita_counts.index]
explode_effettivo1 = [mappa_explode[cat] for cat in stabilita_counts.index]

wedges1, texts1, autotexts1 = ax1.pie(stabilita_counts.values,
                                      labels=stabilita_counts.index,
                                      autopct='%1.1f%%', colors=colori_effettivi1,
                                      textprops={'fontsize': 11},
                                      explode=explode_effettivo1)
ax1.set_title('Complete Distribution of Stability Classes', fontsize=14, fontweight='bold')

# Plotto solo le categorie non dominanti (escludo "Very unstable" per vedere meglio)
non_dominanti = stabilita_counts[stabilita_counts.index != 'Very unstable']

# Se ci sono ancora dati da plottare creo il secondo grafico, altrimenti lascio uno spazio informativo
if len(non_dominanti) > 0:
    colori_effettivi2 = [mappa_colori[cat] for cat in non_dominanti.index]
    wedges2, texts2, autotexts2 = ax2.pie(non_dominanti.values,
                                          labels=non_dominanti.index,
                                          autopct='%1.1f%%', colors=colori_effettivi2,
                                          textprops={'fontsize': 11})
    ax2.set_title('Distribution (excluding "Very unstable" category)', fontsize=14, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Nessun dato per questa selezione', ha='center', va='center', fontsize=12)
    ax2.axis('off')

plt.tight_layout()
salva_plot_con_titolo('Pie Chart Classi di Stabilita')
plt.close()

# ------------------------------
# 11. GRAFICO 8: Violin plot della costanza per range di copertura
# ------------------------------
df_clean['Copertura_Classe'] = pd.cut(df_clean['Copertura_Immagini'],
                                      bins=[0, 50, 100, 500, 1000, 5000, 10000],
                                      labels=['<50', '50-100', '100-500', '500-1000', '1000-5000', '>5000'])

plt.figure(figsize=(14, 8))
sns.violinplot(x='Copertura_Classe', y='Costanza_Percentuale', data=df_clean,
               palette='Set2', cut=0, inner='quartile')
plt.title('Constancy Distribution by Coverage Classes', fontsize=14, fontweight='bold')
plt.xlabel('Image Coverage (classes)', fontsize=12)
plt.ylabel('Constancy (%)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
salva_plot_con_titolo('Distribuzione della Costanza per Classi di Copertura')
plt.close()

# ------------------------------
# 12. GRAFICO 9: Top e Bottom regioni
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Seleziono le top 20 regioni
top20 = df_clean.nlargest(20, 'Costanza_Percentuale')
bars1 = ax1.barh(range(len(top20)), top20['Costanza_Percentuale'].values, color='green', alpha=0.7)
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels([f"{row['RA_centroid']:.2f}, {row['DEC_centroid']:.2f}" for _, row in top20.iterrows()], fontsize=8)
ax1.set_xlabel('Constancy (%)', fontsize=12)
ax1.set_title('Top 20 Regions by Constancy', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# Seleziono le bottom 20 regioni
bottom20 = df_clean.nsmallest(20, 'Costanza_Percentuale')
bars2 = ax2.barh(range(len(bottom20)), bottom20['Costanza_Percentuale'].values, color='red', alpha=0.7)
ax2.set_yticks(range(len(bottom20)))
ax2.set_yticklabels([f"{row['RA_centroid']:.2f}, {row['DEC_centroid']:.2f}" for _, row in bottom20.iterrows()],
                    fontsize=8)
ax2.set_xlabel('Constancy (%)', fontsize=12)
ax2.set_title('Bottom 20 Regions by Constancy', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
salva_plot_con_titolo('Top e Bottom Regioni per Costanza')
plt.close()

# ------------------------------
# 13. GRAFICO 10: Distribuzione della copertura immagini
# ------------------------------
plt.figure(figsize=(12, 8))
sns.histplot(df_clean['Copertura_Immagini'], bins=50, color='purple', alpha=0.7, edgecolor='black')
plt.title('Image Coverage Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Image Coverage (total number of observations)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(df_clean['Copertura_Immagini'].median(), color='red', linestyle='--',
            linewidth=2, label=f'Median: {df_clean["Copertura_Immagini"].median():.0f}')
plt.axvline(df_clean['Copertura_Immagini'].mean(), color='orange', linestyle='--',
            linewidth=2, label=f'Mean: {df_clean["Copertura_Immagini"].mean():.0f}')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xscale('log')
salva_plot_con_titolo('Distribuzione della Copertura Immagini')
plt.close()

print("\n" + "=" * 80)
print(f"✅ Analisi completata! Tutti i grafici sono stati salvati in: {output_dir}")
print("=" * 80)