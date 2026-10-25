import argparse

import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from mibici import config
from mibici.dataset import calcular_hash
from mibici.features import construir_malla, etiqueta_riesgo, features_p2,agregar_features_temporales_p2
from mibici.modeling.tracking import obtener_git_commit
from mibici.plots import graficar_curva_pr, graficar_matriz_confusion

# P2: baseline y modelo candidato registrados en MLflow

VARIABLES = [
    "inventario_previo",
    "razon_inventario",
    "saldo_1h",
    "saldo_3h",
    "salidas_hora_previa",
    "llegadas_hora_previa",
]

VARIABLES_RF_FEATURES = VARIABLES + [
    "hora_dia",
    "dia_semana",
    "es_fin_semana",
]


def cargar_datos():
    # Junta los viajes en Parquet y lee el catálogo de estaciones
    archivos = sorted(config.PROCESSED_DATA_DIR.glob("datos_abiertos_*.parquet"))
    viajes = pd.concat([pd.read_parquet(archivo) for archivo in archivos])
    estaciones = pd.read_csv(config.RAW_DATA_DIR / "nomenclatura_2026_09.csv")
    return viajes, estaciones


def preparar_malla(viajes, estaciones):
    # Malla con la etiqueta y las variables de P2, sin las filas con NaN
    # (las primeras horas de cada estación y las últimas 2)
    malla = construir_malla(viajes, estaciones)
    malla["hora"] = pd.to_datetime(malla["hora"])

    # Las estaciones sin capacidad (fuera de servicio y sin viajes) no tienen inventario que estimar
    malla = malla[malla["capacidad"] > 0]

    # Todas las estaciones deben traer las mismas horas, si no los rezagos salen mal
    if malla.groupby("estacion_id").size().nunique() > 1:
        raise ValueError("La malla debe traer todas las horas de cada estación")

    malla = etiqueta_riesgo(malla)
    malla = features_p2(malla)
    return malla.dropna(subset=VARIABLES + ["riesgo"])


def partir_por_fecha(malla):
    # Se parte por fecha para no usar información del futuro
    entrenamiento = malla[malla["hora"] < config.INICIO_VALIDACION]
    validacion = malla[
        (malla["hora"] >= config.INICIO_VALIDACION) & (malla["hora"] < config.INICIO_PRUEBA)
    ]
    prueba = malla[malla["hora"] >= config.INICIO_PRUEBA]
    return entrenamiento, validacion, prueba


def evaluar(modelo, X, y, nombre, prefijo):
    # Métricas del conjunto (val o test), su curva precision-recall y su matriz de confusión
    yhat = modelo.predict(X)
    ruta_pr = config.FIGURES_DIR / f"{nombre}_{prefijo}_pr.png"
    ruta_matriz = config.FIGURES_DIR / f"{nombre}_{prefijo}_cm.png"

    metricas = {
        f"{prefijo}_precision": precision_score(y, yhat),
        f"{prefijo}_recall": recall_score(y, yhat),
        f"{prefijo}_f1": f1_score(y, yhat),
        f"{prefijo}_pr_auc": graficar_curva_pr(modelo, X, y, nombre, ruta_pr),
    }
    graficar_matriz_confusion(modelo, X, y, ruta_matriz)
    return metricas, [ruta_pr, ruta_matriz]


def correr_run(nombre, modelo, params, datos, version_datos,model_family,stage_of_work,nested=False,):
    # Entrena el modelo y registra el run en MLflow
    X_train, y_train, X_val, y_val, X_test, y_test = datos

    with mlflow.start_run(run_name=nombre,nested=nested):
        modelo.fit(X_train, y_train)

        metricas, figuras_val = evaluar(modelo, X_val, y_val, nombre, "val")
        metricas_test, figuras_test = evaluar(modelo, X_test, y_test, nombre, "test")
        metricas.update(metricas_test)

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
        mlflow.set_tag("model_family", model_family)
        mlflow.set_tag("stage_of_work", stage_of_work)
        mlflow.set_tag("version_datos", version_datos)
        for figura in figuras_val + figuras_test:
            mlflow.log_artifact(figura, artifact_path="plots")

        # Versión de los datos: manifiesto de la ingesta y conjunto de entrenamiento
        mlflow.log_artifact(config.MANIFEST_PATH, artifact_path="data")
        train_df = pd.concat([X_train, y_train], axis=1)
        dataset = mlflow.data.from_pandas(
            train_df, source=str(config.PROCESSED_DATA_DIR), name="p2-train", targets="riesgo"
        )
        mlflow.log_input(dataset, context="training")
        print(nombre, metricas)

def parse_args():
    """
    Lee el modelo y los hiperparametros de P2.
    """
    parser = argparse.ArgumentParser(
        description="Entrena el baseline o candidato de P2."
    )

    parser.add_argument(
        "--model",
        choices=["baseline", "rf", "lr", "rf-search", "all"],
        default="all",
        help="Modelo de P2 que se desea ejecutar.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="Numero de arboles del Random Forest.",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Profundidad maxima del Random Forest.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Numero maximo de iteraciones de la regresion logistica.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    viajes, estaciones = cargar_datos()
    malla = preparar_malla(viajes, estaciones)
    entrenamiento, validacion, prueba = partir_por_fecha(malla)

    datos = (
        entrenamiento[VARIABLES], entrenamiento["riesgo"],
        validacion[VARIABLES], validacion["riesgo"],
        prueba[VARIABLES], prueba["riesgo"],
    )

    malla_features = agregar_features_temporales_p2(malla)

    entrenamiento_v3, validacion_v3, prueba_v3 = partir_por_fecha(
        malla_features
    )

    datos_features = (
        entrenamiento_v3[VARIABLES_RF_FEATURES],
        entrenamiento_v3["riesgo"],
        validacion_v3[VARIABLES_RF_FEATURES],
        validacion_v3["riesgo"],
        prueba_v3[VARIABLES_RF_FEATURES],
        prueba_v3["riesgo"],
    )


    version_datos = calcular_hash(config.MANIFEST_PATH)

    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENTO)

    # Baseline: siempre predice la clase más frecuente
    if args.model in ("baseline", "all"):
        params = {
            "strategy": "most_frequent",
        }

        baseline = DummyClassifier(
            **params
        )

        correr_run(
            "p2-baseline-clase-mas-frecuente",
            baseline,
            params,
            datos,
            version_datos,
            model_family="dummy_classifier",
            stage_of_work="baseline",
        )

    # Candidato: bosque aleatorio, con más peso a la clase de riesgo
    if args.model in ("rf", "all"):
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "class_weight": {
                0: 0.1,
                1: 0.9,
            },
            "n_jobs": -1,
            "random_state": config.SEMILLA,
        }

        candidato = RandomForestClassifier(
            **params
        )

        correr_run(
            "p2-random-forest",
            candidato,
            params,
            datos,
            version_datos,
            model_family="random_forest",
            stage_of_work="candidate",
        )

    # Candidato: regresión logística con las variables escaladas y el mismo peso de clases
    if args.model in ("lr", "all"):
        params = {
            "max_iter": args.max_iter,
            "class_weight": {
                0: 0.1,
                1: 0.9,
            },
            "random_state": config.SEMILLA,
        }

        candidato = Pipeline([
            ("escalador", MinMaxScaler()),
            ("modelo", LogisticRegression(**params)),
        ])

        correr_run(
            "p2-regresion-logistica",
            candidato,
            params,
            datos,
            version_datos,
            model_family="logistic_regression",
            stage_of_work="candidate",
        )

    # Búsqueda de hiperparámetros del bosque: un run padre y un run hijo por combinación
    if args.model in ("rf-search", "all"):
        param_grid = [
            {"n_estimators": 100, "max_depth": 5},
            {"n_estimators": 100, "max_depth": 15},
            {"n_estimators": 200, "max_depth": 5},
            {"n_estimators": 200, "max_depth": 15},
        ]

        with mlflow.start_run(run_name="p2-rf-busqueda"):
            mlflow.set_tags({
                "equipo": "06",
                "pregunta": "P2",
                "model_family": "random_forest",
                "stage_of_work": "search",
                "git_commit": obtener_git_commit(),
            })

            for i, combinacion in enumerate(param_grid, start=1):
                params = {
                    **combinacion,
                    "class_weight": {
                        0: 0.1,
                        1: 0.9,
                    },
                    "n_jobs": -1,
                    "random_state": config.SEMILLA,
                }

                correr_run(
                    f"p2-rf-trial-{i}",
                    RandomForestClassifier(**params),
                    params,
                    datos,
                    version_datos,
                    model_family="random_forest",
                    stage_of_work="search",
                    nested=True,
                )
    if args.model in ("rf-features-search", "all"):
        param_grid = [
            {"n_estimators": 100, "max_depth": 5},
            {"n_estimators": 100, "max_depth": 15},
            {"n_estimators": 200, "max_depth": 5},
            {"n_estimators": 200, "max_depth": 15},
        ]

        # CORRIDA PADRE
        with mlflow.start_run(
            run_name="p2-rf-features-busqueda"
        ):
            mlflow.set_tags({
                "equipo": "06",
                "pregunta": "P2",
                "model_family": "random_forest",
                "stage_of_work": "search",
                "git_commit": obtener_git_commit(),
                "version_datos": version_datos,
            })

            mlflow.log_params({
                "variables": ",".join(VARIABLES_RF_FEATURES),
                "numero_features": len(VARIABLES_RF_FEATURES),
                "numero_combinaciones": len(param_grid),
            })

            # CORRIDAS HIJAS
            for i, combinacion in enumerate(param_grid, start=1):
                params = {
                    **combinacion,
                    "class_weight": {
                        0: 0.1,
                        1: 0.9,
                    },
                    "n_jobs": -1,
                    "random_state": config.SEMILLA,
                }

                correr_run(
                    f"p2-rf-features-trial-{i}",
                    RandomForestClassifier(**params),
                    params,
                    datos_features,
                    version_datos,
                    model_family="random_forest",
                    stage_of_work="search",
                    nested=True,
                )


if __name__ == "__main__":
    main()
