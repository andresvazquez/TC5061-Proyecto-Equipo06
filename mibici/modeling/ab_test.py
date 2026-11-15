import json

import mlflow
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from mibici import config
from mibici.modeling.train_p2 import VARIABLES_RF_FEATURES
from mibici.monitoring.datos_drift import conjunto_actual

MODEL_NAME = "mibici-equipo06-p2-riesgo"

MODELOS = {
    "champion": f"models:/{MODEL_NAME}@champion",
    "challenger": f"models:/{MODEL_NAME}@challenger",
}

REPORTES_DIR = config.REPORTS_DIR / "ab_test"

RUTA_CSV = REPORTES_DIR / "comparacion_p2.csv"
RUTA_JSON = REPORTES_DIR / "comparacion_p2.json"

COLUMNAS_FLOTANTES = VARIABLES_RF_FEATURES[:6]
COLUMNAS_ENTERAS = VARIABLES_RF_FEATURES[6:]


def preparar_conjunto_actual():
    """
    Construye una sola muestra actual para evaluar ambos modelos.
    """
    actual = conjunto_actual()

    X_actual = actual[VARIABLES_RF_FEATURES].copy()
    y_actual = actual["riesgo"].astype("int64")

    X_actual[COLUMNAS_FLOTANTES] = X_actual[COLUMNAS_FLOTANTES].astype("float64")

    X_actual[COLUMNAS_ENTERAS] = X_actual[COLUMNAS_ENTERAS].astype("int64")

    return X_actual, y_actual


def evaluar_modelo(modelo, X, y):
    """
    Calcula las mismas métricas utilizadas por P2.
    """
    predicciones = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)[:, 1]

    return {
        "precision": float(
            precision_score(
                y,
                predicciones,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predicciones,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predicciones,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                probabilidades,
            )
        ),
        "falsas_alarmas": int(((predicciones == 1) & (y.to_numpy() == 0)).sum()),
    }


def comparar_modelos():
    """
    Evalúa Champion y Challenger sobre las mismas observaciones.
    """
    X_actual, y_actual = preparar_conjunto_actual()

    resultados = {}

    for alias, uri in MODELOS.items():
        modelo = mlflow.sklearn.load_model(uri)

        resultados[alias] = evaluar_modelo(
            modelo,
            X_actual,
            y_actual,
        )

    return resultados, len(X_actual)


def guardar_resultados(resultados, filas):
    """
    Guarda la comparación en CSV y JSON.
    """
    REPORTES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tabla = pd.DataFrame.from_dict(
        resultados,
        orient="index",
    )

    tabla.index.name = "version"
    tabla.to_csv(RUTA_CSV)

    contenido = {
        "filas_evaluadas": filas,
        "modelo_registrado": MODEL_NAME,
        "resultados": resultados,
    }

    RUTA_JSON.write_text(
        json.dumps(
            contenido,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return tabla


def registrar_comparacion(resultados, filas):
    """
    Registra las métricas del A/B testing en MLflow.
    """
    mlflow.set_experiment(config.EXPERIMENTO)

    with mlflow.start_run(run_name="p2-ab-test-champion-challenger"):
        mlflow.set_tags(
            {
                "equipo": "06",
                "pregunta": "P2",
                "stage_of_work": "ab_test",
                "champion_uri": MODELOS["champion"],
                "challenger_uri": MODELOS["challenger"],
            }
        )

        mlflow.log_param(
            "filas_evaluadas",
            filas,
        )

        mlflow.log_param(
            "muestra_semilla",
            config.SEMILLA,
        )

        for alias, metricas in resultados.items():
            for nombre, valor in metricas.items():
                mlflow.log_metric(
                    f"{alias}_{nombre}",
                    valor,
                )

        mlflow.log_artifact(
            str(RUTA_CSV),
            artifact_path="ab_test",
        )

        mlflow.log_artifact(
            str(RUTA_JSON),
            artifact_path="ab_test",
        )


def main():
    mlflow.set_tracking_uri(config.TRACKING_URI)

    resultados, filas = comparar_modelos()

    tabla = guardar_resultados(
        resultados,
        filas,
    )

    registrar_comparacion(
        resultados,
        filas,
    )

    print()
    print("Comparación Champion vs Challenger")
    print()
    print(tabla.to_string())
    print()
    print("Filas evaluadas:", filas)
    print("CSV:", RUTA_CSV)
    print("JSON:", RUTA_JSON)


if __name__ == "__main__":
    main()
