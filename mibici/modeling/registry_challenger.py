import argparse

import mlflow
from mlflow.tracking import MlflowClient

from mibici import config
from mibici.modeling.registry import MODELOS

# Registra la versión alternativa de P2 con el alias challenger.
# El alias champion no se mueve: eso se decide con el A/B testing.
ALIAS = "challenger"


def registrar_challenger(run_id):
    client = MlflowClient()

    # La corrida debe traer el commit y la versión de datos (igual que registry.py)
    run = client.get_run(run_id)
    git_commit = run.data.tags.get("git_commit")
    version_datos = run.data.tags.get("version_datos")
    if not git_commit or not version_datos:
        raise ValueError("La corrida debe tener los tags git_commit y version_datos.")

    # Mismo nombre que el champion: se crea una versión nueva sin borrar las anteriores
    nombre = MODELOS["p2"]["registry_name"]
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=nombre,
    )

    client.update_model_version(
        name=nombre,
        version=model_version.version,
        description="Versión alternativa de P2, reentrenada con datos hasta 2026-06.",
    )

    # Mismas etiquetas de linaje que registry.py
    etiquetas = {
        "source_run_id": run_id,
        "git_commit": git_commit,
        "version_datos": version_datos,
    }
    for clave, valor in etiquetas.items():
        client.set_model_version_tag(
            name=nombre,
            version=model_version.version,
            key=clave,
            value=str(valor),
        )

    client.set_registered_model_alias(
        name=nombre,
        alias=ALIAS,
        version=model_version.version,
    )

    print(f"Registrado: {nombre}, versión {model_version.version}, alias {ALIAS}")
    return model_version


def parse_args():
    parser = argparse.ArgumentParser(
        description="Registra la versión alternativa de P2 con el alias challenger."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="ID de la corrida de la versión reentrenada.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mlflow.set_tracking_uri(config.TRACKING_URI)
    registrar_challenger(args.run_id)
