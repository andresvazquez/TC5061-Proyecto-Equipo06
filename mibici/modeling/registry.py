import argparse

import mlflow
from mlflow.tracking import MlflowClient

from mibici import config


def registrar_modelo_elegido(
    *,
    run_id: str,
    registry_name: str,
    description: str,
    artifact_path: str,
):
    """Registra un modelo existente en el Model Registry."""

    client = MlflowClient()

    # 1. Recuperar la corrida ganadora.
    run = client.get_run(run_id)

    # 2. Recuperar el experimento de origen.
    experiment = client.get_experiment(
        run.info.experiment_id
    )

    # 3. Recuperar el commit y la versión de datos.
    tags = run.data.tags
    git_commit = tags.get("git_commit")
    version_datos = tags.get("version_datos")

    if not git_commit or not version_datos:
        raise ValueError(
            "La corrida debe tener los tags git_commit y version_datos."
        )

    # 4. Registrar el modelo guardado en la corrida.
    model_uri = f"runs:/{run_id}/{artifact_path}"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registry_name,
    )

    # 5. Agregar la descripción.
    client.update_registered_model(
        name=registry_name,
        description=description,
    )

    client.update_model_version(
        name=registry_name,
        version=model_version.version,
        description=description,
    )

    # 6. Guardar metadatos de linaje en la versión registrada.
    metadatos = {
        "source_run_id": run_id,
        "source_experiment_id": run.info.experiment_id,
        "source_experiment_name": experiment.name,
        "git_commit": git_commit,
        "version_datos": version_datos,
    }

    # El umbral de las alertas (P2) viaja con la versión registrada
    umbral = run.data.params.get("umbral")
    if umbral is not None:
        metadatos["umbral"] = umbral

    for clave, valor in metadatos.items():
        client.set_model_version_tag(
            name=registry_name,
            version=model_version.version,
            key=clave,
            value=str(valor),
        )

    # 7. Marcar la versión registrada como champion.
    client.set_registered_model_alias(
        name=registry_name,
        alias="champion",
        version=model_version.version,
    )

    print(
        f"Registrado: {registry_name}, "
        f"versión {model_version.version}, alias champion"
    )

    return model_version


# ============================================================
# MODELOS GANADORES
# ============================================================

MODELOS = {
    "p1": {
        "registry_name": "mibici-equipo06-p1-demanda",
        "description": (
            "Modelo seleccionado para predecir la demanda de MiBici, "
            "elegido mediante comparación contra el baseline de P1."
        ),
    },
    "p2": {
        "registry_name": "mibici-equipo06-p2-riesgo",
        "description": (
            "Modelo seleccionado para clasificar el riesgo de las estaciones de MiBici, "
            "elegido mediante comparación contra el baseline de P2."
        ),
    },
    "p3": {
        "registry_name": "mibici-equipo06-p3-grupos",
        "description": (
            "Modelo seleccionado para agrupar las estaciones de MiBici por perfil de uso, "
            "elegido mediante comparación contra el baseline de P3."
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Registra en el Model Registry el modelo elegido de P1, P2 o P3."
    )

    parser.add_argument(
        "--question",
        required=True,
        choices=sorted(MODELOS),
        help="Pregunta del modelo que se registra.",
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="ID de la corrida ganadora de esa pregunta.",
    )

    return parser.parse_args()


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    mlflow.set_tracking_uri(config.TRACKING_URI)

    modelo = MODELOS[args.question]

    # Los modelos de P1, P2 y P3 se guardan en cada corrida con el nombre model
    registrar_modelo_elegido(
        run_id=args.run_id,
        registry_name=modelo["registry_name"],
        description=modelo["description"],
        artifact_path="model",
    )