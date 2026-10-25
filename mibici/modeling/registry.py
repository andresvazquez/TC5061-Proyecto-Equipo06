import mlflow
from mlflow.tracking import MlflowClient


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

    for clave, valor in metadatos.items():
        client.set_model_version_tag(
            name=registry_name,
            version=model_version.version,
            key=clave,
            value=str(valor),
        )

    print(
        f"Registrado: {registry_name}, "
        f"versión {model_version.version}"
    )

    return model_version


# ============================================================
# EJECUCION: MODELOS GANADORES
# ============================================================

if __name__ == "__main__":

    # TODO: sustituir por el ID REAL de la corrida ganadora de P1.
    best_run_id_p1 = "ID_REAL_CORRIDA_P1"

    registrar_modelo_elegido(
        run_id=best_run_id_p1,
        registry_name="mibici-equipo06-p1-demanda",
        description=(
            "Modelo seleccionado para predecir la demanda de MiBici, "
            "elegido mediante comparación contra el baseline de P1."
        ),
        artifact_path="modelo_hist_gb",
    )