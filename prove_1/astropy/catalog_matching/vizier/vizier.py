from astroquery.vizier import Vizier
from astropy.coordinates import Angle
import astropy.units as u
import astropy.coordinates as coord

vizier = Vizier() # inizializzo Vizier con i suoi parametri di default
catalog_list = vizier.find_catalogs('hot jupiter exoplanet transit') # .find_catalogs()
# è il metodo che cerca cataloghi che corrispondono alle parole chiave
# le chiavi sono i codici identificativi dei cataloghi (es: "J/ApJ/885/46")
# i valori sono oggetti che contengono informazioni sui cataloghi
for k, v in catalog_list.items(): # restituisce tutte le coppie (chiave, valore) del dizionario
    # k contiene il codice identificativo del catalogo (es: "J/ApJ/885/46")
    # v contiene l'oggetto con le informazioni del catalogo
    print(k, ":", v.description)

vizier.ROW_LIMIT = 1 # fisso il numero di righe del catalogo, altrimenti è 50 di default
info = Vizier(catalog="VII/74A").get_catalog_metadata() # recupero una tabella con le informazioni del catalogo

# print(info.info)
print("-------------------------------------------------------------------")
print("Interrogare un catalogo")

# vizier = Vizier(row_limit=1)
# result = vizier.query_object("sirius")
# print(result)

'''vizier = Vizier()
result = vizier.query_region("3C 273", radius=Angle(0.1, "deg"), catalog='GSC') # interrogo una regione
print(result)'''

vizier = Vizier(columns=['_RAJ2000', '_DEJ2000','B-V', 'Vmag', 'Plx'],
           column_filters={"Vmag":">10"}, keywords=["optical"])
result = vizier.query_object("HD 226868", catalog=["NOMAD", "UCAC"])
print(result)
print(result['I/322A/out'])
# riordino la tabella in base alla distanza da HD 226868
vizier = Vizier(columns=["*", "+_r"], catalog="II/246")
result = vizier.query_region("HD 226868", radius="20s")
print(result[0])

print("--------------AGN--------------")

agn = Vizier(catalog="VII/258/vv10",
             columns=['*', '_RAJ2000', '_DEJ2000']).query_constraints(Vmag="10.0..11.0")[0] # AGN con Vmag compreso tra 10.0 e 11.0
print(agn)

guide = Vizier(catalog="II/246", column_filters={"Kmag":"<9.0"}).query_region(agn, radius="30s", inner_radius="2s")[0] # stelle distanti al massimo 30 secondi d'arco daagli agn di prima
guide.pprint()

print("Filtro gli oggetti all'interno di un quadratro di lato 'width'")

nome = "II/389/ps1_dr2"

vizier = Vizier(
    catalog="II/389/ps1_dr2",
    columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'],
    row_limit=-1
)


result = vizier.query_region(coord.SkyCoord(ra=299.590, dec=35.201,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                        width="30m")

result = vizier

print(result)
