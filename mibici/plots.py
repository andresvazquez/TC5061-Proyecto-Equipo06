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

def graficar_perfiles_clusters(resumen, ruta):
    """
    Grafica el perfil promedio de uso de cada cluster P3.

    Muestra salidas, llegadas y volumen diario promedio por grupo.

    Args:
        resumen (pd.DataFrame): Resumen de estaciones con cluster
            y variables de uso.
        ruta (str | Path): Ruta donde se guarda la figura.
    """
    columnas = {
        "cluster",
        "salidas",
        "llegadas",
        "volumen_diario",
    }

    faltantes = columnas - set(resumen.columns)

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(sorted(faltantes))
        )

    perfiles = (
        resumen.groupby("cluster")[
            [
                "salidas",
                "llegadas",
                "volumen_diario",
            ]
        ]
        .mean()
        .sort_index()
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    variables = [
        ("salidas", "Salidas promedio"),
        ("llegadas", "Llegadas promedio"),
        ("volumen_diario", "Volumen diario promedio"),
    ]

    for ax, (variable, titulo) in zip(
        axes,
        variables,
        strict=True,
    ):
        ax.bar(
            perfiles.index.astype(str),
            perfiles[variable],
        )

        ax.set_title(titulo)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Promedio")

    fig.suptitle(
        "Perfiles promedio de los clusters P3"
    )

    fig.tight_layout()

    fig.savefig(
        ruta,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)
