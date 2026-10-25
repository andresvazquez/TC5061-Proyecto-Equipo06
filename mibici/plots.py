import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay


def graficar_curva_pr(modelo, X, y, nombre, ruta):
    # Curva precision-recall del modelo guardada en ruta, devuelve su PR-AUC
    curva = PrecisionRecallDisplay.from_estimator(modelo, X, y, name=nombre)
    curva.figure_.savefig(ruta)
    plt.close(curva.figure_)
    return curva.average_precision


def graficar_matriz_confusion(modelo, X, y, ruta):
    # Matriz de confusión del modelo guardada en ruta
    matriz = ConfusionMatrixDisplay.from_estimator(modelo, X, y)
    matriz.figure_.savefig(ruta)
    plt.close(matriz.figure_)
