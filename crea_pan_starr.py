from astroquery.vizier import Vizier
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import vstack, unique
import time
import sys
import warnings

warnings.simplefilter('ignore')

# Configurazione di VizieR (nessun limite di righe)
v = Vizier(
    columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    column_filters={'gmag': '<=15', 'rmag': '<=15', 'imag': '<=15', 'zmag': '<=15', 'ymag': '<=15'},
    row_limit=-1,
    timeout=600
)

# Griglia: riquadri da 10x10 gradi
step = 10
ra_centers = list(range(5, 360, step))
dec_centers = list(range(-25, 90, step))

# Contatori
totale_riquadri = len(ra_centers) * len(dec_centers)
riquadri_elaborati = 0
lista_tabelle = []
oggetti_totali = 0

print("=== INIZIO DOWNLOAD VIA VIZIER API NATIVA (ASTROQUERY) ===")
print(f"Totale riquadri da analizzare: {totale_riquadri}")

# Avvio il cronometro globale
tempo_inizio = time.time()

for dec in dec_centers:
    for ra in ra_centers:
        riquadri_elaborati += 1
        print(f"\n[+] Riquadro {riquadri_elaborati}/{totale_riquadri} | Centro RA={ra}°, DEC={dec}°")

        centro = coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        tentativi = 0
        successo = False

        while tentativi < 3 and not successo:
            try:
                risultato = v.query_region(centro, width=step * u.deg, height=step * u.deg, catalog='II/389/ps1_dr2')

                if len(risultato) > 0:
                    tabella_tassello = risultato[0]
                    num_oggetti = len(tabella_tassello)
                    print(f"    -> Trovati: {num_oggetti} oggetti.")
                    lista_tabelle.append(tabella_tassello)
                    oggetti_totali += num_oggetti
                else:
                    print("    -> Nessun oggetto trovato in questo riquadro.")

                successo = True

            except Exception as e:
                tentativi += 1
                print(f"    -> [ERRORE di Rete] Tentativo {tentativi}/3 fallito: {e}")
                if tentativi < 3:
                    print("    -> Attendo 5 secondi e riprovo...")
                    time.sleep(5)
                else:
                    print("    -> Passo al riquadro successivo.")

        # === CALCOLO STIME IN TEMPO REALE ===
        tempo_trascorso = time.time() - tempo_inizio
        tempo_medio = tempo_trascorso / riquadri_elaborati
        riquadri_rimanenti = totale_riquadri - riquadri_elaborati
        tempo_stimato_sec = tempo_medio * riquadri_rimanenti

        minuti = int(tempo_stimato_sec // 60)
        secondi = int(tempo_stimato_sec % 60)

        # 44 byte a riga convertiti in Megabyte
        peso_stimato_mb = (oggetti_totali * 44) / (1024 * 1024)

        print(f"    -> ETA (Tempo rimasto) : ~{minuti} min {secondi} sec")
        print(f"    -> Accumulo totale     : {oggetti_totali} stelle (Peso stimato file: {peso_stimato_mb:.2f} MB)")

        time.sleep(1)

print("\n=== FASE DI DOWNLOAD COMPLETATA ===")

if lista_tabelle:
    print("\nUnione dei riquadri e pulizia in corso...")
    tabella_finale = vstack(lista_tabelle)

    tabella_pulita = unique(tabella_finale, keys='objID')

    oggetti_unici = len(tabella_pulita)
    peso_finale_mb = (oggetti_unici * 44) / (1024 * 1024)
    print(f"Stelle uniche finali: {oggetti_unici} (Rimosse {oggetti_totali - oggetti_unici} sovrapposizioni)")
    print(f"Peso esatto dei dati: {peso_finale_mb:.2f} MB")

    nome_file = "panstarrs_dr2_mag15_finale.fits"
    print(f"Salvataggio del file '{nome_file}'...")
    tabella_pulita.write(nome_file, format='fits', overwrite=True)
    print("FATTO! Dati scaricati e salvati in modo sicuro.")
else:
    print("Nessun dato recuperato.")