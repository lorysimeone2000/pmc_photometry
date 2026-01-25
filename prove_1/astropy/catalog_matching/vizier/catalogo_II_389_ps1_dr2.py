from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import Angle
from astropy.table import Table
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import matplotlib.pyplot as plt

# questa funzione restituisce la tabella delle sorgenti trovate
def analisi_image_segmentation(data):
    """
    Esegue image segmentation su un'immagine FITS e restituisce la tabella filtrata delle sorgenti

    Returns:
    astropy.table.Table: Tabella delle sorgenti filtrate con label riordinati
    """

    mean, median, std = sigma_clipped_stats(data, sigma=3.0) # nel caso me lo chiedessi, la std prima o dopo aver sottratto il fondo è la stessa

    # Lettura parametri
    parametri = {}
    with open('/home/lorysimeone/tesi_magistrale/prove/analisi/parametri_image_segmentation.txt', 'r') as file:
        next(file)  # Salta intestazione
        for riga in file:
            riga = riga.strip()
            if riga and not riga.startswith('#'):
                parametro, valore = riga.split()
                parametri[parametro] = float(valore) if '.' in valore else int(valore)

    # Convoluzione
    fwhm = parametri['fwhm']
    size = parametri['size']
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    convolved_data = convolve(data, kernel)
    mean_c, median_c, std_c = sigma_clipped_stats(convolved_data, sigma=3.0)

    # Sourcefinder
    t = parametri['threshold_sigma']
    # threshold = t * std # per adesso lascio stare questo metodo
    threshold = parametri['threshold_assoluta']
    n = parametri['pixel']

    finder = SourceFinder(npixels=n, progress_bar=True)
    segment_map = finder(convolved_data, threshold)

    # Catalogo sorgenti
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'

    # filtraggio sorgenti
    soglia_assoluta = 2.5
    soglia_relativa = 0.4

    indici_validi = []
    indici_non_validi = []

    for i, sorgente in enumerate(tbl):
        label = sorgente['label']
        mask_sorgente = (segment_map.data == label)
        valori_originali = data[mask_sorgente]

        pixel_sopra_soglia_assoluta = np.sum(valori_originali > soglia_assoluta)
        pixel_sopra_soglia_relativa = np.sum(valori_originali > soglia_relativa * sorgente['max_value'])

        if pixel_sopra_soglia_assoluta >= 2 and pixel_sopra_soglia_relativa >= 2:
            indici_validi.append(i)

    # Creazione tabella filtrata
    tbl_filtrato = tbl[indici_validi]
    new_labels_validi = np.arange(1, len(tbl_filtrato) + 1)
    tbl_filtrato['label'] = new_labels_validi

    # print(f"Sorgenti dopo filtro: {len(tbl_filtrato)} / {len(tbl)}")

    return tbl_filtrato

agn = Vizier(catalog="VII/258/vv10",
             columns=['*', '_RAJ2000', '_DEJ2000'])

print(agn.query_constraints(Vmag="10.0..11.0")[0])

nome = "II/389/ps1_dr2"

# inizializzo Vizier con i suoi parametri di default

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1
)

catalogo = vizier
print(f"Questo è il catalogo: \n{catalogo}\n ------------")

catalogs = Vizier.get_catalogs(nome)

'''# Stampa i nomi delle colonne per ciascuna tabella trovata
for i, cat in enumerate(catalogs):

    title = cat.meta.get('title', 'Titolo non disponibile')
    print(f"\n--- Tabella {i + 1}: {title} ---")
    print(f"Numero colonne: {len(cat.colnames)}")

    # Stampa tutte le colonne
    for col in cat.colnames:
        print(col)'''

print(f"Descrizione gmag: {catalogs[0]["gmag"].description}")

print("Filtro gli oggetti all'interno di un quadratro di lato 'width'")

riquadro = vizier.query_region(coord.SkyCoord(ra=270, dec=35.201,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                        width=0.1 * u.deg,  # <-- Larghezza in RA
                        height=0.2 * u.deg) # <-- Altezza in Dec

print(riquadro) # dovrei ottenere una tabella sola

print(riquadro[0])

# Se vuoi solo la mappa spaziale principale
if riquadro and len(riquadro[0]) > 0:
    tabella = riquadro[0]

    plt.figure(figsize=(12, 8))

    ra = tabella['RAJ2000']
    dec = tabella['DEJ2000']

    scatter = plt.scatter(ra, dec, c=tabella['rmag'], cmap='viridis',
                          s=50, alpha=0.8, edgecolors='black', linewidth=0.5)

    plt.xlabel('Ascensione Retta (RA J2000) [gradi]', fontsize=12)
    plt.ylabel('Declinazione (Dec J2000) [gradi]', fontsize=12)
    plt.title(f'Catalogo Pan-STARRS - {len(tabella)} oggetti', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()  # Convenzione astronomica

    cbar = plt.colorbar(scatter)
    cbar.set_label('Magnitudine r', fontsize=12)

    plt.tight_layout()
    plt.show()

    print(f"Oggetti trovati: {len(tabella)}")
    print(f"Range RA: {np.min(ra):.3f}° - {np.max(ra):.3f}°")
    print(f"Range Dec: {np.min(dec):.3f}° - {np.max(dec):.3f}°")

print("Applico vincoli diretti su RAJ2000 e DEJ2000 con query_constraints")

# Definisci i vincoli su RA e Dec
ra_min = 290
ra_max = 299.8
dec_min = 30
dec_max = 35.4

# Usa query_constraints per applicare filtri diretti
riquadro = vizier.query_constraints(
    RAJ2000='50 .. 55',
    DEJ2000='40 .. 45'
)

print(riquadro)



