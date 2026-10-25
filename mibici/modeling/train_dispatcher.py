import argparse
import subprocess
import sys

MODELOS_VALIDOS = {
    "p1": {
        "baseline",
        "hgb",
        "search",
        "features",
        "all",
    },
    "p2": {
        "baseline",
        "rf",
        "lr",
        "rf-search",
        "rf-features-search",
        "all",
    },
    "p3": {
        "kmeans",
        "agglomerative",
        "kmeans-search",
        "kmeans-full",
        "kmeans-stability",
    },
}


def parse_args():
    """
    Lee los parametros comunes del entry point train del MLflow Project.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Despacha el entrenamiento de P1, P2 o P3 "
            "desde el MLflow Project."
        )
    )

    parser.add_argument(
        "--question",
        required=True,
        choices=["p1", "p2", "p3"],
        help="Pregunta del proyecto que se desea entrenar.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Modelo o configuracion que se desea ejecutar.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="Numero de arboles para P2 Random Forest.",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Profundidad maxima para P2 Random Forest.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Numero maximo de iteraciones para P1 HGB y P2 regresion logistica.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Tasa de aprendizaje para P1 HGB.",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Numero de clusters para P3.",
    )

    return parser.parse_args()


def validar_modelo(question, model):
    """
    Valida que el modelo solicitado corresponda con la pregunta.
    """
    modelos = MODELOS_VALIDOS[question]

    if model not in modelos:
        raise ValueError(
            f"Modelo '{model}' no valido para {question}. "
            f"Opciones: {sorted(modelos)}"
        )

def construir_comando(args):
    """
    Construye el comando del trainer correspondiente.

    Las rutas, fechas y la semilla permanecen centralizadas
    en config.py y no se reciben como argumentos.
    """
    validar_modelo(
        args.question,
        args.model,
    )

    if args.question == "p1":
        return [
            sys.executable,
            "-m",
            "mibici.modeling.train_p1",
            "--model",
            args.model,
            "--max-iter",
            str(args.max_iter),
            "--learning-rate",
            str(args.learning_rate),
        ]

    if args.question == "p2":
        return [
            sys.executable,
            "-m",
            "mibici.modeling.train_p2",
            "--model",
            args.model,
            "--n-estimators",
            str(args.n_estimators),
            "--max-depth",
            str(args.max_depth),
            "--max-iter",
            str(args.max_iter),
        ]

    if args.model == "kmeans-search":
        return [
            sys.executable,
            "-m",
            "mibici.modeling.train_p3_search",
        ]

    if args.model == "kmeans-stability":
        return [
            sys.executable,
            "-m",
            "mibici.modeling.train_p3_stability",
        ]

    modelo_p3 = {
        "kmeans": "baseline",
        "agglomerative": "agglomerative",
        "kmeans-full": "full-profile",
    }[args.model]

    return [
        sys.executable,
        "-m",
        "mibici.modeling.train_p3",
        "--model",
        modelo_p3,
        "--n-clusters",
        str(args.n_clusters),
    ]

def main():
    """
    Ejecuta el trainer asociado a la pregunta solicitada.
    """
    args = parse_args()

    comando = construir_comando(args)

    print(
        "Ejecutando:",
        " ".join(comando),
    )

    subprocess.run(
        comando,
        check=True,
    )


if __name__ == "__main__":
    main()
