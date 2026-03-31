import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from shapely.geometry import Polygon
import pandas as pd
import warnings
from astropy.wcs import FITSFixedWarning
import glob
import os

# Ignoro i warning del WCS
warnings.filterwarnings('ignore', category=FITSFixedWarning)


def gradi_a_sessagesimale(gradi):
    """
    Converto gradi decimali in formato sessagesimale (Gradi° Minuti' Secondi")

    Parametri:
    gradi: valore in gradi decimali

    Restituisce:
    stringa nel formato "DD° MM' SS.SS\""
    """
    gradi_int = int(gradi)
    minuti_float = (gradi - gradi_int) * 60
    minuti_int = int(minuti_float)
    secondi = (minuti_float - minuti_int) * 60

    return f"{gradi_int}° {minuti_int}' {secondi:.2f}\""


def calcola_dimensioni_fov(file_fits):
    """
    Calcolo le dimensioni del campo visivo (FOV) di un'immagine FITS in modo rigoroso.

    Parametri:
    file_fits: percorso del file FITS

    Restituisce:
    dizionario con larghezza, altezza, area e scala pixel
    """
    # Apro il file FITS
    with fits.open(file_fits) as hdu:
        header = hdu[0].header

        # Ottengo le dimensioni dell'immagine in pixel
        naxis1 = header.get('NAXIS1', 0)  # larghezza in pixel
        naxis2 = header.get('NAXIS2', 0)  # altezza in pixel

        if naxis1 == 0 or naxis2 == 0:
            raise ValueError("Dimensioni dell'immagine non trovate nell'header")

        # Ottengo il WCS
        wcs = WCS(header)

        # Trovo il centro dell'immagine per creare un piano di proiezione
        centro_x = naxis1 / 2
        centro_y = naxis2 / 2
        centro_coord = wcs.pixel_to_world(centro_x, centro_y)

        # Definisco i vertici esatti dell'immagine in pixel
        vertici_pixel = [
            (0, 0),
            (naxis1, 0),
            (naxis1, naxis2),
            (0, naxis2)
        ]

        # Trasformo i pixel in coordinate celesti
        vertici_mondo = [wcs.pixel_to_world(x, y) for x, y in vertici_pixel]

        # Proietto i vertici su un piano cartesiano locale tangente al centro
        # In questo modo evito le distorsioni sferiche e posso usare shapely rigorosamente
        vertici_proiettati = []
        for coord in vertici_mondo:
            # Calcolo gli offset sferici ortogonali dal centro
            ra_offset, dec_offset = centro_coord.spherical_offsets_to(coord)
            # Salvo le coordinate proiettate in gradi
            vertici_proiettati.append((ra_offset.deg, dec_offset.deg))

        # Costruisco il poligono esatto con shapely sul piano locale
        poligono_fov = Polygon(vertici_proiettati)

        # Calcolo l'area rigorosa utilizzando il poligono
        area_deg2 = poligono_fov.area

        # Estraggo le coordinate del poligono per misurare i lati
        coords = list(poligono_fov.exterior.coords)

        # Calcolo la larghezza (distanza tra vertice 0 e 1) e altezza (tra 0 e 3) in modo rigoroso
        larghezza_deg = np.sqrt((coords[1][0] - coords[0][0])**2 + (coords[1][1] - coords[0][1])**2)
        altezza_deg = np.sqrt((coords[3][0] - coords[0][0])**2 + (coords[3][1] - coords[0][1])**2)

        # Calcolo la scala dei pixel
        pixel_scale_arcsec = (larghezza_deg / naxis1) * 3600

        # Converto in formato sessagesimale
        larghezza_sess = gradi_a_sessagesimale(larghezza_deg)
        altezza_sess = gradi_a_sessagesimale(altezza_deg)

        # Creo il dizionario con i risultati (senza nome file)
        risultati = {
            'larghezza_deg': larghezza_deg,
            'altezza_deg': altezza_deg,
            'larghezza_sessagesimale': larghezza_sess,
            'altezza_sessagesimale': altezza_sess,
            'area_deg2': area_deg2,
            'pixel_scale_arcsec': pixel_scale_arcsec,
            'n_pixel_x': naxis1,
            'n_pixel_y': naxis2
        }

        return risultati


def genera_csv_dimensioni(lista_file, output_csv="dimensioni_fov.csv"):
    """
    Genero un CSV con le dimensioni del FOV per tutti i file nella lista.
    Le dimensioni sono organizzate come righe invece che colonne.

    Parametri:
    lista_file: lista di percorsi dei file FITS
    output_csv: nome del file CSV di output
    """
    risultati = []

    print(f"Elaborazione di {len(lista_file)} file...")

    for i, file_fits in enumerate(lista_file, 1):
        try:
            dim = calcola_dimensioni_fov(file_fits)
            risultati.append(dim)
            nome_file = os.path.basename(file_fits)
            print(f"[{i}/{len(lista_file)}] Elaborato: {nome_file}")
        except Exception as e:
            print(f"[{i}/{len(lista_file)}] ERRORE per {file_fits}: {e}")

    if not risultati:
        print("Nessun file elaborato con successo!")
        return

    # Creo DataFrame con i risultati
    df_raw = pd.DataFrame(risultati)

    # Inverto righe e colonne (transpose)
    # Ogni riga rappresenta una metrica, ogni colonna rappresenta un file
    df_transposed = df_raw.T

    # Rinomino le colonne con indici numerici
    df_transposed.columns = [f"File_{i + 1}" for i in range(len(df_transposed.columns))]

    # Aggiungo una colonna con le metriche
    df_transposed.insert(0, 'Metrica', df_transposed.index)

    # Resetto l'indice
    df_transposed.reset_index(drop=True, inplace=True)

    # Riordino le metriche in modo più logico
    ordine_metriche = [
        'larghezza_sessagesimale',
        'altezza_sessagesimale',
        'larghezza_deg',
        'altezza_deg',
        'area_deg2',
        'pixel_scale_arcsec',
        'n_pixel_x',
        'n_pixel_y'
    ]

    # Filtro solo le metriche nell'ordine desiderato
    df_transposed = df_transposed[df_transposed['Metrica'].isin(ordine_metriche)]
    df_transposed = df_transposed.set_index('Metrica').loc[ordine_metriche].reset_index()

    # Imposto i nomi descrittivi per le metriche
    nomi_metriche = {
        'larghezza_sessagesimale': 'Larghezza (sessagesimale)',
        'altezza_sessagesimale': 'Altezza (sessagesimale)',
        'larghezza_deg': 'Larghezza (gradi)',
        'altezza_deg': 'Altezza (gradi)',
        'area_deg2': 'Area (gradi²)',
        'pixel_scale_arcsec': 'Scala pixel (arcsec/pixel)',
        'n_pixel_x': 'Pixel X (larghezza)',
        'n_pixel_y': 'Pixel Y (altezza)'
    }

    df_transposed['Metrica'] = df_transposed['Metrica'].map(nomi_metriche)

    # Salvo il CSV
    df_transposed.to_csv(output_csv, index=False, float_format='%.6f')

    print(f"\n✅ CSV generato con successo: {output_csv}")
    print(f"   Totale file elaborati: {len(risultati)}")
    print(f"   Struttura: {len(df_transposed)} righe (metriche) × {len(df_transposed.columns) - 1} colonne (file)")

    return df_transposed


def genera_csv_da_cartella(cartella, pattern="*.fits", output_csv="dimensioni_fov.csv"):
    """
    Genero un CSV con le dimensioni del FOV per tutti i file in una cartella.
    Le dimensioni sono organizzate come righe invece che colonne.

    Parametri:
    cartella: percorso della cartella contenente i file FITS
    pattern: pattern per cercare i file (default: "*.fits")
    output_csv: nome del file CSV di output
    """
    # Cerco tutti i file FITS nella cartella
    file_pattern = os.path.join(cartella, pattern)
    lista_file = glob.glob(file_pattern)

    if not lista_file:
        print(f"Nessun file trovato con pattern: {file_pattern}")
        return None

    print(f"Trovati {len(lista_file)} file nella cartella: {cartella}")

    return genera_csv_dimensioni(lista_file, output_csv)


# Esempio di utilizzo
if __name__ == "__main__":
    # Processo un singolo file
    singolo_file = "/home/lorysimeone/tesi_magistrale/prove_1/20250106_231255.fits"
    df_risultati = genera_csv_dimensioni([singolo_file], "dimensioni_singolo_file.csv")

    # Visualizzo i risultati se presenti
    if df_risultati is not None:
        print("\n" + "=" * 80)
        print("ANTEPRIMA DEI RISULTATI:")
        print("=" * 80)
        print(df_risultati.to_string())