import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mibici import config
from mibici.dataset import calcular_hash
from mibici.features import agregar_features_temporales_p2
from mibici.modeling.train_p2 import (
    VARIABLES_RF_FEATURES,
    cargar_datos,
    correr_run,
    parametros_bosque,
    partir_por_fecha,
    preparar_malla,
)

# Versión alternativa de P2 para el A/B testing: el mismo bosque del champion
# (100 árboles y profundidad 15), entrenado con los datos hasta junio de 2026
N_ESTIMADORES = 100
PROFUNDIDAD_MAXIMA = 15


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
        validacion[VARIABLES_RF_FEATURES],
        validacion["riesgo"],
        prueba[VARIABLES_RF_FEATURES],
        prueba["riesgo"],
    )

    version_datos = calcular_hash(config.MANIFEST_PATH)

    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENTO)

    params = parametros_bosque(N_ESTIMADORES, PROFUNDIDAD_MAXIMA)
    modelo = RandomForestClassifier(**params)
    params["entrenado_hasta"] = config.INICIO_PRUEBA

    # El umbral no se elige aquí: la validación ya forma parte del entrenamiento
    correr_run(
        "p2-random-forest-reentrenado",
        modelo,
        params,
        datos,
        version_datos,
        model_family="random_forest",
        stage_of_work="retrained",
    )


if __name__ == "__main__":
    main()
