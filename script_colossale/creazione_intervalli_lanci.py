import datetime

# definisco la mia data di inizio e la mia data di fine
data_inizio = datetime.date(2024, 11, 1)
data_fine = datetime.date(2025, 12, 31)

# imposto la durata del mio intervallo aggiungendo 4 giorni alla data di inizio per avere blocchi di 5 giorni esatti
giorni_da_aggiungere = 5
giorni_da_aggiungere = giorni_da_aggiungere - 1

# apro il mio file di testo in modalità scrittura nella cartella corrente
with open("lanci.txt", "w") as file_comandi:
    data_corrente = data_inizio
    
    # inizio il mio ciclo per generare gli intervalli fino alla data di fine
    while data_corrente <= data_fine:
        # calcolo la data finale del mio intervallo corrente
        fine_intervallo = data_corrente + datetime.timedelta(days=giorni_da_aggiungere)
        
        # se la fine del mio intervallo supera la data di fine assoluta, la forzo all'ultimo giorno utile
        if fine_intervallo > data_fine:
            fine_intervallo = data_fine
            
        # formatto le mie date nel formato stringa richiesto (YYYYMMDD)
        str_inizio = data_corrente.strftime("%Y%m%d")
        str_fine = fine_intervallo.strftime("%Y%m%d")
        
        # assemblo la mia riga di comando e la scrivo nel file
        riga_comando = f"python3 creazione_tabelle_principale_colossale_alleggerito.py s {str_inizio} {str_fine}\n"
        file_comandi.write(riga_comando)
        
        # aggiorno la mia data corrente al giorno immediatamente successivo alla fine dell'intervallo appena scritto
        data_corrente = fine_intervallo + datetime.timedelta(days=1)

print("File 'comandi_batch_colossale.txt' generato con successo.")
