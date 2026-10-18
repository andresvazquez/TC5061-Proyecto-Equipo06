import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay


def graficar_curva_pr(modelo, X, y, nombre, ruta):
    # Curva precision-recall del modelo guardada en ruta, devuelve su PR-AUC
    curva = PrecisionRecallDisplay.from_estimator(modelo, X, y, name=nombre)
    curva.figure_.savefig(ruta)
    plt.close(curva.figure_)
    return curva.average_precision
