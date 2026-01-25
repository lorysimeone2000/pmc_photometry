import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
import numpy as np # Aggiunto per sicurezza se serve np

# Caricamento dati
filename = 'statistiche_distanze_ordinate.csv'
df = pd.read_csv(filename)

# --- FILTRAGGIO RICHIESTO ---
print(f"Totale stelle originali: {len(df)}")

# Filtra via tutte le stelle con magnitudine inferiore a 10 (tiene quelle >= 10)
df = df[df['Magnitudine'] <= 10].copy()

print(f"Totale stelle dopo il filtro (Mag <= 10): {len(df)}")

if len(df) == 0:
    print("ATTENZIONE: Nessuna stella rimasta dopo il filtro!")
    exit()
# ----------------------------

# Creazione del grafico 1: Distanze vs Magnitudine
plt.figure(figsize=(12, 7))

# Plotting delle tre serie
plt.plot(df['Magnitudine'], df['Distanza_max'],
         label='Distanza Max', color='red', alpha=0.5, linewidth=1, marker='o', markersize=2)

plt.plot(df['Magnitudine'], df['Distanza_media'],
         label='Distanza Media', color='blue', linewidth=1.5, marker='o', markersize=2)

plt.plot(df['Magnitudine'], df['Distanza_min'],
         label='Distanza Min', color='green', alpha=0.5, linewidth=1, marker='o', markersize=2)

# Configurazione assi e titoli
plt.title('Analisi Breathing: Spostamento vs Magnitudine (Mag <= 10)')
plt.xlabel('Magnitudine')
plt.ylabel('Distanza (pixel)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Inversione asse X per convenzione astronomica
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig('grafico_distanze_magnitudine_filtrato.png')
plt.show()

# Calcolo Std Percentuale
df['Std_Percentuale'] = (df['Distanza_std'] / df['Distanza_media']) * 100

# Creazione del grafico 2: Std % vs Magnitudine
plt.figure(figsize=(12, 7))

plt.plot(df['Magnitudine'], df['Std_Percentuale'],
         label='Deviazione Standard %', color='purple', alpha=0.7, linewidth=1, marker='o', markersize=3)

# Configurazione assi e titoli
plt.title('Stabilità Relativa: Std % vs Magnitudine (Mag <= 10)')
plt.xlabel('Magnitudine (più luminose a sinistra)')
plt.ylabel('Deviazione Standard (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Inversione asse X
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig('grafico_std_percentuale_filtrato.png')
plt.show()

# --- ANALISI PULITA (Outliers) ---
# Pulizia outliers per le statistiche robuste
clean_df = df[df['Std_Percentuale'] < 50].copy()

# Ordiniamo per Magnitudine per poter calcolare il trend grafico
clean_df = clean_df.sort_values(by='Magnitudine')

# 4. Calcolo Statistiche di Riepilogo
median_abs_std = clean_df['Distanza_std'].median()
median_perc_std = clean_df['Std_Percentuale'].median()
num_totali = len(df)
num_pulite = len(clean_df)

print(f"\n=== RISULTATI ANALISI STABILITÀ (Mag <= 10) ===")
print(f"Totale stelle analizzate: {num_totali}")
print(f"Stelle considerate 'stabili' (<50% var): {num_pulite}")
print(f"---------------------------------------")
print(f"Mediana della Deviazione Standard Assoluta: {median_abs_std:.4f} pixel")
print(f"Mediana della Deviazione Standard Percentuale: {median_perc_std:.2f}%")

# --- SIGMA CLIPPING ---
# Calcolo statistiche sigma-clipped sulla STD
mean, median, std = sigma_clipped_stats(df['Distanza_std'], sigma=3.0, maxiters=5)

print(f"\n[Distanza_std] Media sigma-clipped: {mean:.4f}")
print(f"[Distanza_std] Mediana sigma-clipped: {median:.4f}")
print(f"[Distanza_std] Dev.Std sigma-clipped: {std:.4f}")

# Calcolo statistiche sigma-clipped sulla MEDIA
mean_m, median_m, std_m = sigma_clipped_stats(df['Distanza_media'], sigma=3.0, maxiters=5)

print(f"\n[Distanza_media] Media sigma-clipped: {mean_m:.4f}")
print(f"[Distanza_media] Mediana sigma-clipped: {median_m:.4f}")
print(f"[Distanza_media] Dev.Std sigma-clipped: {std_m:.4f}")

# 5. Calcolo Trend per il Grafico (Rolling Median)
window_size = 20
# Rolling su clean_df che è già ordinato per magnitudine
clean_df['Rolling_Std_Abs'] = clean_df['Distanza_std'].rolling(window=window_size, center=True).median()
clean_df['Rolling_Std_Perc'] = clean_df['Std_Percentuale'].rolling(window=window_size, center=True).median()

# --- BLOCCO GRAFICO FINALE ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Grafico 1: Assoluto
ax1.scatter(clean_df['Magnitudine'], clean_df['Distanza_std'], color='gray', alpha=0.3, s=15, label='Dati singoli')
ax1.plot(clean_df['Magnitudine'], clean_df['Rolling_Std_Abs'], color='red', linewidth=2.5, label='Trend Mediano')
ax1.set_ylabel('Deviazione Standard (pixel)')
ax1.set_title(f'Stabilità Assoluta (Mag >= 10) - Mediana: {median_abs_std:.2f} px')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# Grafico 2: Percentuale
ax2.scatter(clean_df['Magnitudine'], clean_df['Std_Percentuale'], color='gray', alpha=0.3, s=15, label='Dati singoli')
ax2.plot(clean_df['Magnitudine'], clean_df['Rolling_Std_Perc'], color='blue', linewidth=2.5, label='Trend Mediano')
ax2.set_xlabel('Magnitudine')
ax2.set_ylabel('Errore Percentuale (%)')
ax2.set_title(f'Stabilità Percentuale (Mag <= 10) - Mediana: {median_perc_std:.1f}%')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

# Invertiamo l'asse X per convenzione astronomica
ax2.invert_xaxis()

plt.tight_layout()
plt.show()