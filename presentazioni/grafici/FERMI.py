import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Importo le librerie necessarie per l'elaborazione dei dati e il grafico
# Leggo il file CSV contenente i dati del satellite
df = pd.read_csv("4FGL_J1104.4+3812_daily_18_04_2026.csv")

# Converto le colonne di flusso e di errore in valori numerici. Se incontro stringhe, le forzo a essere valori numerici validi
df["Photon Flux [0.1-100 GeV](photons cm-2 s-1)"] = pd.to_numeric(df["Photon Flux [0.1-100 GeV](photons cm-2 s-1)"], errors='coerce')
df["Photon Flux Error(photons cm-2 s-1)"] = pd.to_numeric(df["Photon Flux Error(photons cm-2 s-1)"], errors='coerce')

# Converto i valori della Julian Date in date standard per poterli formattare
df["Date_Converted"] = pd.to_datetime(df["Julian Date"], origin='julian', unit='D')

# Imposto la dimensione della figura ottimizzata per 0.65\textwidth in un documento A4
plt.figure(figsize=(9, 4))

# Isolo le rilevazioni certe basandomi sulla Test Statistic
mask_detections = df["TS"] >= 9

# Traccio i punti con le barre di errore per le rilevazioni con TS elevato. Inserisco 'r' prima della label per interpretare il LaTeX
# Uso la nuova colonna delle date per l'asse x
plt.errorbar(df.loc[mask_detections, "Date_Converted"],
             df.loc[mask_detections, "Photon Flux [0.1-100 GeV](photons cm-2 s-1)"],
             yerr=df.loc[mask_detections, "Photon Flux Error(photons cm-2 s-1)"],
             fmt='o', color='black', markersize=4, capsize=3, elinewidth=1, label=r'Detections (TS $\geq$ 9)')

# Confino i dati tra il 19 dicembre 2025 e il 20 gennaio 2026
start_date = pd.to_datetime("2025-12-19")
end_date = pd.to_datetime("2026-01-20")
plt.xlim(start_date, end_date)

# Isolo i dati dell'intervallo temporale richiesto per calcolare dinamicamente i limiti dell'asse y
mask_time = (df["Date_Converted"] >= start_date) & (df["Date_Converted"] <= end_date)

# Calcolo il massimo e il minimo considerando i valori del flusso e i relativi errori
y_max = (df.loc[mask_time, "Photon Flux [0.1-100 GeV](photons cm-2 s-1)"] + df.loc[mask_time, "Photon Flux Error(photons cm-2 s-1)"].fillna(0)).max()
y_min = (df.loc[mask_time, "Photon Flux [0.1-100 GeV](photons cm-2 s-1)"] - df.loc[mask_time, "Photon Flux Error(photons cm-2 s-1)"].fillna(0)).min()

# Aggiungo un piccolo margine del 5% per evitare che i punti estremi tocchino la cornice del grafico e imposto i limiti
margin = (y_max - y_min) * 0.05
plt.ylim(y_min - margin, y_max + margin)

# Imposto le etichette dell'asse x ogni 5 giorni e nel formato giorno/mese/anno
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))

# Ruoto le etichette per facilitare la mia lettura ed evitare sovrapposizioni
plt.xticks(rotation=45)

# Dimensiono i tick degli assi scalandoli per renderli adatti alle dimensioni della figura in LaTeX
plt.tick_params(axis='both', which='major', labelsize=8)

# Configuro i nomi degli assi utilizzando la formattazione corretta e dimensiono i testi
#plt.xlabel("Date", fontsize=10)
plt.ylabel(r"Photon Flux [0.1-100 GeV] (photons cm$^{-2}$ s$^{-1}$)", fontsize=10)

# Attivo la griglia di sfondo per facilitare la mia lettura dei valori
plt.grid(True, which="both", linestyle="--", alpha=0.5)

# Inserisco la legenda dimensionandola per LaTeX
plt.legend(fontsize=8)

# Ottimizzo gli spazi e mostro il risultato a schermo
plt.tight_layout()
plt.savefig("grafico_markarian.png", dpi=300)
plt.show()