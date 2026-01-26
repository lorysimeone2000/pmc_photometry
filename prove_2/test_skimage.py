import sys
print(f"Python esecutivo: {sys.executable}")

try:
    print("Tentativo importazione numpy...")
    import numpy
    print(f"OK Numpy: {numpy.__version__}")

    print("Tentativo importazione scipy...")
    import scipy
    print(f"OK Scipy: {scipy.__version__}")

    print("Tentativo importazione scikit-image...")
    import skimage
    from skimage import segmentation
    print(f"OK Scikit-image: {skimage.__version__}")
    print("TUTTO OK! Le librerie funzionano.")

except ImportError as e:
    print(f"\nERRORE CRITICO DI IMPORTAZIONE:\n{e}")
    print("\nSOLUZIONE: Devi reinstallare le librerie come spiegato nel PASSO 2.")
except RuntimeError as e:
    print(f"\nERRORE RUNTIME (conflitto versioni):\n{e}")