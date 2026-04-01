import datetime
import subprocess
import os

# definisco la mia data di inizio e la mia data di fine
data_inizio = datetime.date(2024, 1, 1)
data_fine = datetime.date(2025, 4, 1)

# imposto la durata del mio intervallo aggiungendo 4 giorni alla data di inizio per avere blocchi di 5 giorni esatti
giorni_da_aggiungere = 5
giorni_da_aggiungere = giorni_da_aggiungere - 1

# definisco il nome del mio file contenente i comandi
file_comandi = "lanci.txt"

# apro il mio file di testo in modalità scrittura nella cartella corrente
with open(file_comandi, "w") as file_out:
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
        file_out.write(riga_comando)
        
        # aggiorno la mia data corrente al giorno immediatamente successivo alla fine dell'intervallo appena scritto
        data_corrente = fine_intervallo + datetime.timedelta(days=1)

print(f"File '{file_comandi}' generato con successo.\n")

# controllo che il mio file esista prima di procedere
if not os.path.exists(file_comandi):
    print(f"Errore: Il file {file_comandi} non esiste nella cartella corrente.")
    exit()

# apro il file e memorizzo tutte le righe
with open(file_comandi, "r") as file_in:
    comandi = file_in.readlines()

# filtro le eventuali righe vuote per evitare problemi durante l'esecuzione
comandi_validi = [cmd.strip() for cmd in comandi if cmd.strip()]

totale_comandi = len(comandi_validi)
print(f"Trovati {totale_comandi} comandi da eseguire in sequenza.\n")

# avvio il mio ciclo per lanciare ogni comando uno dopo l'altro
for indice, comando in enumerate(comandi_validi, start=1):
    print(f"\n======================================================================")
    print(f"LANCIO BATCH {indice}/{totale_comandi}")
    print(f"Eseguo: {comando}")
    print(f"======================================================================\n")
    
    # eseguo il mio comando nel terminale e attendo che finisca
    # uso shell=True per passare l'intera riga esattamente come se la digitassi io
    risultato = subprocess.run(comando, shell=True)
    
    # verifico se l'esecuzione ha generato un errore (returncode diverso da 0)
    if risultato.returncode != 0:
        print(f"\nATTENZIONE: L'esecuzione si è interrotta con un errore (codice {risultato.returncode}).")
        print("Ignoro l'errore e passo al lancio successivo.")
        
print("\nElaborazione della coda terminata.")
