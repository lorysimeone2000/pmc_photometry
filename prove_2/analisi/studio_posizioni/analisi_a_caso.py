import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

# --- CONFIGURAZIONE ---
base_path = "/home/lorysimeone/tesi_magistrale/prove_2/tabelle/tabelle_unite"
path_stats_run1 = "/home/lorysimeone/tesi_magistrale/prove_2/analisi/studio_flussi/dispersione_flussi/risultati_analisi_run_1.csv"
run_to_analyze = 1
# Percentuale dove avviene lo scalino (es. 0.75 = 75% del tempo)
STEP_THRESHOLD = 0.75

# Dimensioni sensore
W, H = 3072, 2048
CENTER_X, CENTER_Y = W / 2, H / 2
MAX_RADIUS = np.hypot(CENTER_X, CENTER_Y)

# --- 1. CARICAMENTO DATI ---
print(">>> Caricamento dati...")

# A. Carichiamo le statistiche generali (Std, Media)
df_stats = pd.read_csv(path_stats_run1, comment='#')

# B. Recuperiamo le coordinate (X, Y) da un file di riferimento
# Usiamo il primo file disponibile della Run 1 per associare ID -> (X, Y)
cartella_run = os.path.join(base_path, f"tabelle_unite_run_{run_to_analyze}")
files_run = sorted([f for f in os.listdir(cartella_run) if f.endswith('.csv')])

if not files_run:
    print("Errore: Nessun file trovato nella cartella della run.")
    exit()

# Leggiamo il file a metà run per avere coordinate stabili, oppure il primo
path_ref = os.path.join(cartella_run, files_run[0])
df_coords = pd.read_csv(path_ref, comment='#', usecols=['ID', 'xcentroid', 'ycentroid'])

# C. Uniamo tutto
# Teniamo solo le stelle che sono sia nel file stats che nel file coords
df_main = pd.merge(df_stats, df_coords, on='ID', how='inner')

# Calcolo distanza dal centro
df_main['dist_center'] = np.hypot(df_main['xcentroid'] - CENTER_X, df_main['ycentroid'] - CENTER_Y)

# Calcolo Deviazione Standard Relativa (Sigma / Flux) per normalizzare la luminosità
df_main['rel_std'] = df_main['std_flusso_fisso_max_run'] / df_main['media_flusso_fisso_max_run']

print(f"Stelle analizzate: {len(df_main)}")

# --- 2. CALCOLO DELLO 'SCALINO' (STEP ANALYSIS) ---
print(">>> Analisi dello 'Scalino' temporale (richiede lettura file run)...")

# Dividiamo i file in "Prima dello scalino" e "Dopo lo scalino"
split_index = int(len(files_run) * STEP_THRESHOLD)
files_pre = files_run[:split_index]
files_post = files_run[split_index:]

print(f"File Pre-Scalino: {len(files_pre)} | File Post-Scalino: {len(files_post)}")

# Dizionari per accumulare i flussi: {ID: [lista_flussi]}
fluxes_pre = {id_: [] for id_ in df_main['ID']}
fluxes_post = {id_: [] for id_ in df_main['ID']}


# Funzione helper per caricare flussi velocemente
def load_partial_flux(file_list, target_dict):
    # Leggiamo un campione di file per non metterci troppo tempo (es. 1 ogni 5)
    # Se vuoi precisione massima togli lo step [::5]
    for f in file_list[::5]:
        p = os.path.join(cartella_run, f)
        try:
            d = pd.read_csv(p, comment='#', usecols=['ID', 'flusso_fisso_max_run'])
            # Filtra solo gli ID che ci interessano
            d = d[d['ID'].isin(target_dict.keys())]
            for _, row in d.iterrows():
                target_dict[row['ID']].append(row['flusso_fisso_max_run'])
        except:
            pass


load_partial_flux(files_pre, fluxes_pre)
load_partial_flux(files_post, fluxes_post)

# Calcolo entità dello scalino per ogni stella
step_metrics = []
for star_id in df_main['ID']:
    vals_pre = np.array(fluxes_pre[star_id])
    vals_post = np.array(fluxes_post[star_id])

    if len(vals_pre) > 0 and len(vals_post) > 0:
        mean_pre = np.mean(vals_pre)
        mean_post = np.mean(vals_post)
        # Calcolo variazione percentuale
        delta_perc = ((mean_post - mean_pre) / mean_pre) * 100
        step_metrics.append(delta_perc)
    else:
        step_metrics.append(np.nan)

df_main['step_perc'] = step_metrics

# Rimuoviamo NaN e outlier estremi per i plot
df_clean = df_main.dropna(subset=['step_perc', 'rel_std', 'dist_center'])
# Filtro per vedere meglio (rimuovo variazioni > 50% che sono probabilmente errori)
df_clean = df_clean[np.abs(df_clean['step_perc']) < 50]

# --- 3. VISUALIZZAZIONE ---

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

# --- RIGA 1: SCATTER PLOTS (Correlazione con Distanza) ---

# 1. Deviazione Standard Relativa vs Distanza
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(df_clean['dist_center'], df_clean['rel_std'] * 100, alpha=0.3, s=5, c='blue')
ax1.set_title("1. Stabilità (Std/Mean) vs Distanza")
ax1.set_xlabel("Distanza dal Centro (px)")
ax1.set_ylabel("Std Relativa (%)")
ax1.grid(True, alpha=0.3)
# Trendline
z = np.polyfit(df_clean['dist_center'], df_clean['rel_std'] * 100, 1)
p = np.poly1d(z)
ax1.plot(df_clean['dist_center'], p(df_clean['dist_center']), "r--", lw=1)

# 2. Entità Scalino vs Distanza
ax2 = plt.subplot(gs[0, 1])
ax2.scatter(df_clean['dist_center'], df_clean['step_perc'], alpha=0.3, s=5, c='green')
ax2.set_title(f"2. Entità Scalino (post {int(STEP_THRESHOLD * 100)}%) vs Distanza")
ax2.set_xlabel("Distanza dal Centro (px)")
ax2.set_ylabel("Variazione Flusso (%)")
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linestyle='--')
# Trendline
z = np.polyfit(df_clean['dist_center'], df_clean['step_perc'], 1)
p = np.poly1d(z)
ax2.plot(df_clean['dist_center'], p(df_clean['dist_center']), "r--", lw=1)

# 3. Media Flusso vs Distanza (Vignettatura check)
ax3 = plt.subplot(gs[0, 2])
ax3.scatter(df_clean['dist_center'], df_clean['media_flusso_fisso_max_run'], alpha=0.3, s=5, c='purple')
ax3.set_title("3. Flusso Medio vs Distanza")
ax3.set_xlabel("Distanza dal Centro (px)")
ax3.set_ylabel("Flusso Medio (ADU)")
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# --- RIGA 2: MAPPE 2D (Dove succede?) ---

# 4. Mappa 2D Std Relativa
ax4 = plt.subplot(gs[1, 0])
sc4 = ax4.scatter(df_clean['xcentroid'], df_clean['ycentroid'],
                  c=df_clean['rel_std'] * 100, s=15, cmap='plasma', alpha=0.8,
                  norm=Normalize(vmin=0, vmax=np.percentile(df_clean['rel_std'] * 100, 95)))
plt.colorbar(sc4, ax=ax4, label="Std Relativa (%)")
ax4.set_title("Mappa Stabilità (Più giallo = Più rumore)")
ax4.set_aspect('equal')

# 5. Mappa 2D Scalino
# Usiamo una mappa divergente (blu=cala, rosso=sale, bianco=stabile)
ax5 = plt.subplot(gs[1, 1])
limit = np.percentile(np.abs(df_clean['step_perc']), 95)
sc5 = ax5.scatter(df_clean['xcentroid'], df_clean['ycentroid'],
                  c=df_clean['step_perc'], s=15, cmap='seismic', alpha=0.8,
                  norm=Normalize(vmin=-limit, vmax=limit))
plt.colorbar(sc5, ax=ax5, label="Variazione (%)")
ax5.set_title("Mappa Scalino (Rosso=Salto in su, Blu=Giù)")
ax5.set_aspect('equal')

# 6. Mappa 2D Flusso Medio
ax6 = plt.subplot(gs[1, 2])
sc6 = ax6.scatter(df_clean['xcentroid'], df_clean['ycentroid'],
                  c=np.log10(df_clean['media_flusso_fisso_max_run']), s=15, cmap='viridis', alpha=0.8)
plt.colorbar(sc6, ax=ax6, label="Log10(Flusso)")
ax6.set_title("Mappa Luminosità (Vignettatura?)")
ax6.set_aspect('equal')

plt.tight_layout()
filename_out = "analisi_spaziale_scalino.png"
plt.savefig(filename_out, dpi=300)
print(f"Grafico salvato in: {filename_out}")
plt.show()