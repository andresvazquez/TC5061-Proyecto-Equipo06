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

def graficar_real_vs_predicho(y_real, y_pred, ruta):
    """
    Genera un gráfico de dispersión entre valores reales y predichos (P1)
    y lo guarda en la ruta especificada.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_real, y_pred, alpha=0.3, edgecolors="k", s=20)

    # Línea ideal y = x
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Predicción Perfecta",
    )

    ax.set_xlabel("Demanda Real")
    ax.set_ylabel("Demanda Predicha")
    ax.set_title("P1: Demanda Real vs. Predicha")
    ax.legend()
    fig.tight_layout()

    fig.savefig(ruta)
    plt.close(fig)