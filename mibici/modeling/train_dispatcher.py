import argparse
import subprocess
import sys

MODELOS_VALIDOS = {
    "p1": {
        "baseline",
        "hgb",
        "all",
    },
    "p2": {
        "baseline",
        "rf",
        "all",
    },
    "p3": {
        "kmeans",
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
        ]

    if args.question == "p2":
        return [
            sys.executable,
            "-m",
            "mibici.modeling.train_p2",
        ]

    return [
        sys.executable,
        "-m",
        "mibici.modeling.train_p3",
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
