import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from mibici import config
from mibici.dataset import calcular_hash
from mibici.features import construir_malla, etiqueta_riesgo, features_p2
from mibici.plots import graficar_curva_pr

# P2: baseline y modelo candidato registrados en MLflow

VARIABLES = [
    "inventario_previo",
    "razon_inventario",
    "saldo_1h",
    "saldo_3h",
    "salidas_hora_previa",
    "llegadas_hora_previa",
]


def cargar_datos():
    # Junta los viajes en Parquet y lee el catálogo de estaciones
    archivos = sorted(config.PROCESSED_DATA_DIR.glob("datos_abiertos_*.parquet"))
    viajes = pd.concat([pd.read_parquet(archivo) for archivo in archivos])
    estaciones = pd.read_csv(config.PROCESSED_DATA_DIR / "nomenclatura_2026_09.csv")
    return viajes, estaciones


def preparar_malla(viajes, estaciones):
    # Malla con la etiqueta y las variables de P2, sin las filas con NaN
    # (las primeras horas de cada estación y las últimas 2)
    malla = construir_malla(viajes, estaciones)
    malla["hora"] = pd.to_datetime(malla["hora"])

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
    # Métricas del conjunto (val o test) y su curva precision-recall
    yhat = modelo.predict(X)
    ruta_figura = config.FIGURES_DIR / f"{nombre}_{prefijo}_pr.png"

    metricas = {
        f"{prefijo}_precision": precision_score(y, yhat),
        f"{prefijo}_recall": recall_score(y, yhat),
        f"{prefijo}_f1": f1_score(y, yhat),
        f"{prefijo}_pr_auc": graficar_curva_pr(modelo, X, y, nombre, ruta_figura),
    }
    return metricas, ruta_figura


def correr_run(nombre, modelo, params, datos, version_datos):
    # Entrena el modelo y registra el run en MLflow
    X_train, y_train, X_val, y_val, X_test, y_test = datos

    with mlflow.start_run(run_name=nombre):
        modelo.fit(X_train, y_train)

        metricas, figura_val = evaluar(modelo, X_val, y_val, nombre, "val")
        metricas_test, figura_test = evaluar(modelo, X_test, y_test, nombre, "test")
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
        mlflow.log_artifact(figura_val)
        mlflow.log_artifact(figura_test)
        print(nombre, metricas)


def main():
    viajes, estaciones = cargar_datos()
    malla = preparar_malla(viajes, estaciones)
    entrenamiento, validacion, prueba = partir_por_fecha(malla)

    datos = (
        entrenamiento[VARIABLES], entrenamiento["riesgo"],
        validacion[VARIABLES], validacion["riesgo"],
        prueba[VARIABLES], prueba["riesgo"],
    )
    version_datos = calcular_hash(config.MANIFEST_PATH)

    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENTO)

    # Baseline: siempre predice la clase más frecuente
    params = {"strategy": "most_frequent"}
    baseline = DummyClassifier(**params)
    correr_run("p2-baseline-clase-mas-frecuente", baseline, params, datos, version_datos)

    # Candidato: bosque aleatorio, con más peso a la clase de riesgo
    params = {
        "n_estimators": 50,
        "max_depth": 10,
        "class_weight": {0: 0.1, 1: 0.9},
        "n_jobs": -1,
        "random_state": config.SEMILLA,
    }
    candidato = RandomForestClassifier(**params)
    correr_run("p2-random-forest", candidato, params, datos, version_datos)


if __name__ == "__main__":
    main()
