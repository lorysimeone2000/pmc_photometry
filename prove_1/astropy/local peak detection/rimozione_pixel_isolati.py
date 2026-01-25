# Rimozione pixel isolati (std è 0.4)

def azzera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3):
    """
    Versione ottimizzata per azzerare pixel isolati
    """
    from scipy import ndimage

    data_pulita = data.copy()

    # Soglia per considerare un pixel significativo
    soglia_minima = 5 * std
    mask_significativi = data > soglia_minima

    # Crea una mappa dei vicini forti
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Esclude il pixel centrale

    # Per ogni pixel, conta quanti vicini sono sopra threshold_ratio del pixel centrale
    mask_vicini_forti = np.zeros_like(data, dtype=bool)

    y_coords, x_coords = np.where(mask_significativi)
    for y, x in zip(y_coords, x_coords):
        pixel_val = data[y, x]
        threshold_locale = threshold_ratio * pixel_val

        # Controlla regione 3x3
        y_min, y_max = max(0, y - 1), min(data.shape[0], y + 2)
        x_min, x_max = max(0, x - 1), min(data.shape[1], x + 2)

        region = data[y_min:y_max, x_min:x_max]
        mask_forti = region > threshold_locale

        # Conta vicini forti (escludendo il centro)
        center_y, center_x = 1 if y_min < y else 0, 1 if x_min < x else 0
        mask_forti[center_y, center_x] = False  # Esclude il centro

        vicini_forti = np.sum(mask_forti)

        if vicini_forti < min_neighbors:
            data_pulita[y, x] = 0  # Azzera il pixel isolato

    return data_pulita

data = image_data
data_pulita = azzera_pixel_isolati_veloce(data, min_neighbors=2, threshold_ratio=0.3)