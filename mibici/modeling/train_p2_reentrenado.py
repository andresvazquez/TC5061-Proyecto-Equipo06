import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mibici import config
from mibici.dataset import calcular_hash
from mibici.features import agregar_features_temporales_p2
from mibici.modeling.tracking import obtener_git_commit
from mibici.modeling.train_p2 import (
    VARIABLES_RF_FEATURES,
    cargar_datos,
    evaluar,
    parametros_bosque,
    partir_por_fecha,
    preparar_malla,
)

# Versión alternativa de P2 para el A/B testing: el mismo bosque del champion
# (100 árboles y profundidad 15), entrenado con los datos hasta junio de 2026
N_ESTIMADORES = 100
PROFUNDIDAD_MAXIMA = 15


def correr_run_reentrenado(nombre, modelo, params, datos, version_datos):
    # Igual que correr_run de train_p2.py, pero sin validación: abril a junio ya entra al entrenamiento
    X_train, y_train, X_test, y_test = datos

    with mlflow.start_run(run_name=nombre):
        modelo.fit(X_train, y_train)

        metricas, figuras = evaluar(modelo, X_test, y_test, nombre, "test")

        mlflow.log_params(
            {
                "fecha_inicio": config.FECHA_INICIO,
                "fecha_fin": config.FECHA_FIN,
                "inicio_validacion": config.INICIO_VALIDACION,
                "inicio_prueba": config.INICIO_PRUEBA,
                "version_datos": version_datos,
                "semilla": config.SEMILLA,
                "modelo": type(modelo).__name__,
                **params,
            }
        )
        mlflow.log_metrics(metricas)
        mlflow.set_tag("equipo", "06")
        mlflow.set_tag("pregunta", "P2")
        mlflow.set_tag("git_commit", obtener_git_commit())
        mlflow.set_tag("model_family", "random_forest")
        mlflow.set_tag("stage_of_work", "retrained")
        mlflow.set_tag("version_datos", version_datos)
        for figura in figuras:
            mlflow.log_artifact(figura, artifact_path="plots")

        # Versión de los datos y configuración de la corrida
        mlflow.log_artifact(config.MANIFEST_PATH, artifact_path="data")
        mlflow.log_artifact("mibici/config.py", artifact_path="config")

        # Modelo con firma y ejemplo de entrada
        signature = infer_signature(X_train, modelo.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        train_df = pd.concat([X_train, y_train], axis=1)
        dataset = mlflow.data.from_pandas(
            train_df, source=str(config.PROCESSED_DATA_DIR), name="p2-train", targets="riesgo"
        )
        mlflow.log_input(dataset, context="training")
        print(nombre, metricas)


def main():
    viajes, estaciones = cargar_datos()
    malla = preparar_malla(viajes, estaciones)
    malla_features = agregar_features_temporales_p2(malla)
    entrenamiento, validacion, prueba = partir_por_fecha(malla_features)

    # Entrenamiento y validación juntos (antes de INICIO_PRUEBA); la prueba no se toca
    entrenamiento = pd.concat([entrenamiento, validacion])

    datos = (
        entrenamiento[VARIABLES_RF_FEATURES],
        entrenamiento["riesgo"],
        prueba[VARIABLES_RF_FEATURES],
        prueba["riesgo"],
    )

    version_datos = calcular_hash(config.MANIFEST_PATH)

    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENTO)

    params = parametros_bosque(N_ESTIMADORES, PROFUNDIDAD_MAXIMA)
    modelo = RandomForestClassifier(**params)
    params["entrenado_hasta"] = config.INICIO_PRUEBA

    correr_run_reentrenado(
        "p2-random-forest-reentrenado",
        modelo,
        params,
        datos,
        version_datos,
    )


if __name__ == "__main__":
    main()
