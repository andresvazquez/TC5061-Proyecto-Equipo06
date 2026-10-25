import argparse
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mibici import config
from mibici.dataset import calcular_hash
from mibici.features import construir_features_p1
from mibici.modeling.tracking import RunMLflow


def evaluar_conjuntos(
    y_val_real: pd.Series,
    y_val_pred: pd.Series,
    y_test_real: pd.Series,
    y_test_pred: pd.Series,
) -> dict[str, float]:
    """
    Calcula MAE y RMSE para Validación y Prueba, devolviendo
    las métricas en la estructura solicitada por la rúbrica.
    """
    val_mae = mean_absolute_error(y_val_real, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_real, y_val_pred))

    test_mae = mean_absolute_error(y_test_real, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_real, y_test_pred))

    return {
        "val_mae": float(val_mae),
        "test_mae": float(test_mae),
        "val_rmse": float(val_rmse),
        "test_rmse": float(test_rmse),
    }


def guardar_grafica_predicciones(
    y_real: pd.Series, y_pred: pd.Series, titulo: str, nombre_archivo: str, n_muestras: int = 1000
) -> str:
    """Genera y guarda una gráfica de Dispersión y Residuos representativa mediante un muestreo aleatorio."""
    temp_dir = tempfile.mkdtemp()
    ruta_imagen = os.path.join(temp_dir, nombre_archivo)

    # Convertir a Series de pandas si vienen como arrays para usar sample
    if not isinstance(y_real, pd.Series):
        y_real = pd.Series(y_real)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_real.index)

    # Muestreo aleatorio con semilla fija para representatividad y reproducibilidad
    tamano_muestra = min(len(y_real), n_muestras)
    indices_muestra = y_real.sample(n=tamano_muestra, random_state=config.SEMILLA).index

    y_real_sub = y_real.loc[indices_muestra].to_numpy()
    y_pred_sub = y_pred.loc[indices_muestra].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Real vs Predicho
    max_val = max(y_real_sub.max(), y_pred_sub.max()) + 1
    ax[0].scatter(y_real_sub, y_pred_sub, alpha=0.4, color="teal", edgecolors="none")
    ax[0].plot([0, max_val], [0, max_val], "r--", lw=2, label="Ideal")
    ax[0].set_xlim(-0.5, max_val)
    ax[0].set_ylim(-0.5, max_val)
    ax[0].set_xlabel("Salidas Reales")
    ax[0].set_ylabel("Salidas Predichas")
    ax[0].set_title(f"{titulo} - Real vs Predicho")
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Residuos (sobre la totalidad de los datos para no perder eventos extremos)
    residuos = y_real - y_pred
    ax[1].hist(residuos, bins=50, range=(-15, 15), color="coral", edgecolor="black", alpha=0.7)
    ax[1].set_xlabel("Residuo (Real - Predicho)")
    ax[1].set_ylabel("Frecuencia")
    ax[1].set_title(f"{titulo} - Distribución de Residuos (-15 a 15)")
    ax[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(ruta_imagen, dpi=150)
    plt.close()

    return ruta_imagen


def dividir_train_val_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separa temporalmente el dataset de forma explícita utilizando
    las fechas de corte definidas en config.py.
    """
    # Convertir a Timestamp para asegurar compatibilidad de tipos con la columna 'hora'
    inicio_val = pd.Timestamp(config.INICIO_VALIDACION)
    inicio_test = pd.Timestamp(config.INICIO_PRUEBA)

    train = df[df["hora"] < inicio_val].copy()
    val = df[(df["hora"] >= inicio_val) & (df["hora"] < inicio_test)].copy()
    test = df[df["hora"] >= inicio_test].copy()

    print(f"Corte Train -> Val: {inicio_val} | Corte Val -> Test: {inicio_test}")
    return train, val, test

def parse_args():
    """
    Lee el modelo y los hiperparametros de P1.
    """
    parser = argparse.ArgumentParser(
        description="Entrena el baseline o candidato de P1."
    )

    parser.add_argument(
        "--model",
        choices=["baseline", "hgb", "all"],
        default="all",
        help="Modelo de P1 que se desea ejecutar.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Numero maximo de iteraciones del HistGradientBoosting.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Tasa de aprendizaje del HistGradientBoosting.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    print("Iniciando el entrenamiento y evaluación para el problema P1...")

    # 1. Obtener versión de datos
    if config.MANIFEST_PATH.exists():
        version_datos = calcular_hash(config.MANIFEST_PATH)
    else:
        version_datos = "v1.0-raw"

    # 2. Cargar datos
    archivos_parquet = list(config.PROCESSED_DATA_DIR.glob("datos_abiertos_*.parquet"))

    if not archivos_parquet:
        raise FileNotFoundError(
            f"No se encontraron archivos .parquet en {config.PROCESSED_DATA_DIR}. "
            "Asegúrate de haber ejecutado 'python -m mibici.dataset' primero."
        )

    df_viajes = pd.concat([pd.read_parquet(f) for f in archivos_parquet], ignore_index=True)

    nomenclatura_clean = config.RAW_DATA_DIR / "nomenclatura_2026_09.csv"
    if not nomenclatura_clean.exists():
        nomenclatura_clean = list(config.RAW_DATA_DIR.glob("*nomenclatura*.csv"))[0]

    df_features = construir_features_p1(df_viajes, nomenclatura_clean)

    # 3. División train test val
    train, val, test = dividir_train_val_test(df_features)

    X_cols = [
        "hora_del_dia",
        "dia_de_semana",
        "es_fin_de_semana",
        "mes",
        "salidas_lag_1",
        "salidas_lag_2",
        "salidas_lag_24",
        "salidas_lag_168",
        "salidas_roll_mean_3h",
        "salidas_roll_mean_24h",
        "capacidad",
    ]

    X_train = train[X_cols]
    y_train = train["salidas"]

    X_val = val[X_cols]
    y_val = val["salidas"]

    X_test = test[X_cols]
    y_test = test["salidas"]

    # RUN 1: BASELINE (Demanda de la semana pasada)
    if args.model in ("baseline", "all"):
        with RunMLflow(pregunta="P1", nombre_run="Baseline_P1_Lag168", version_datos=version_datos,
                       model_family="baseline_lag_168",stage_of_work="baseline") as tracker:
            tracker.registrar_parametros({"modelo_tipo": "Baseline_Lag_168"})

            # Predicciones
            y_val_pred_base = val["salidas_lag_168"]
            y_test_pred_base = test["salidas_lag_168"]

            # Métricas agrupadas según la rúbrica
            metricas_baseline = evaluar_conjuntos(
                y_val_real=y_val,
                y_val_pred=y_val_pred_base,
                y_test_real=y_test,
                y_test_pred=y_test_pred_base,
            )
            tracker.registrar_metricas(metricas_baseline)

            # Artefacto gráfico
            img_baseline = guardar_grafica_predicciones(
                y_test, y_test_pred_base, "Baseline", "grafica_baseline.png"
            )
            tracker.registrar_artefacto(img_baseline)

            print("[Baseline] Métricas registradas:", metricas_baseline)

    # RUN 2: MODELO CANDIDATO (HistGradientBoosting)
    if args.model in ("hgb", "all"):
        with RunMLflow(pregunta="P1", nombre_run="Candidato_HistGradientBoosting", version_datos=version_datos,
                       model_family="hist_gradient_boosting",stage_of_work="candidate",) as tracker:
            tracker.registrar_parametros(
                {
                    "modelo_tipo": "HistGradientBoostingRegressor",
                    "random_state": config.SEMILLA,
                    "n_features": len(X_cols),
                    "inicio_validacion": str(config.INICIO_VALIDACION),
                    "inicio_prueba": str(config.INICIO_PRUEBA),
                    "max_iter": args.max_iter,
                    "learning_rate": args.learning_rate,
                }
            )

            modelo = HistGradientBoostingRegressor(
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                random_state=config.SEMILLA,
            )
            modelo.fit(X_train, y_train)

            # Predicciones en Val y Test
            y_val_pred_cand = modelo.predict(X_val)
            y_test_pred_cand = modelo.predict(X_test)

            # Métricas agrupadas según la rúbrica
            metricas_candidato = evaluar_conjuntos(
                y_val_real=y_val,
                y_val_pred=y_val_pred_cand,
                y_test_real=y_test,
                y_test_pred=y_test_pred_cand,
            )
            tracker.registrar_metricas(metricas_candidato)

            # Artefacto gráfico
            img_candidato = guardar_grafica_predicciones(
                y_test, y_test_pred_cand, "HistGradientBoosting", "grafica_candidato.png"
            )
            tracker.registrar_artefacto(img_candidato)

            # Modelo serializado
            mlflow.sklearn.log_model(
                sk_model=modelo,
                name="modelo_hist_gb",
                skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
            )

            print("[Candidato] Métricas registradas:", metricas_candidato)


if __name__ == "__main__":
    main()


 # PASO 1. DEFINIR LAS CONFIGURACIONES
    # ------------------------------------------------------------
    # Christian debe definir las combinaciones de hiperparametros
    # que desea comparar. Los siguientes valores son ejemplos.
    #
    # combinaciones = [
    #     {"max_iter": 100, "learning_rate": 0.05},
    #     {"max_iter": 200, "learning_rate": 0.05},
    #     {"max_iter": 200, "learning_rate": 0.10},
    # ]


    # ------------------------------------------------------------
    # PASO 2. CREAR LA CORRIDA PADRE
    # ------------------------------------------------------------
    # La corrida padre agrupa toda la busqueda.
    # No necesita entrenar un modelo.
    #
    # nested=False indica que la corrida no se crea como hija.
    #
    # with RunMLflow(
    #     pregunta="P1",
    #     nombre_run="Busqueda_Hiperparametros_P1",
    #     version_datos=version_datos,
    #     nested=False,
    # ) as padre:


        # --------------------------------------------------------
        # PASO 3. REGISTRAR LA CONFIGURACION GENERAL DE LA BUSQUEDA
        # --------------------------------------------------------
        # Registrar en el padre la informacion general:
        # tipo de modelo, numero de combinaciones y semilla.
        #
        # padre.registrar_parametros({
        #     "modelo_tipo": "HistGradientBoostingRegressor",
        #     "tipo_busqueda": "manual",
        #     "n_combinaciones": len(combinaciones),
        #     "random_state": config.SEMILLA,
        # })


        # --------------------------------------------------------
        # PASO 4. RECORRER LAS CONFIGURACIONES
        # --------------------------------------------------------
        # Para cada combinacion se debe crear una corrida hija.
        #
        # for i, params in enumerate(combinaciones, start=1):


            # ----------------------------------------------------
            # PASO 5. CREAR UNA CORRIDA HIJA
            # ----------------------------------------------------
            # nested=True indica a MLflow que esta corrida pertenece
            # a la corrida padre que se encuentra activa.
            #
            # Cada hija entrenara una configuracion diferente.
            #
            # with RunMLflow(
            #     pregunta="P1",
            #     nombre_run=f"P1_HGB_Prueba_{i}",
            #     version_datos=version_datos,
            #     nested=True,
            # ) as tracker:


                # ------------------------------------------------
                # PASO 6. REGISTRAR LOS HIPERPARAMETROS
                # ------------------------------------------------
                # Registrar la configuracion particular de la hija.
                #
                # tracker.registrar_parametros({
                #     "modelo_tipo": "HistGradientBoostingRegressor",
                #     **params,
                #     "random_state": config.SEMILLA,
                #     "n_features": len(X_cols),
                # })


                # ------------------------------------------------
                # PASO 7. CREAR EL MODELO
                # ------------------------------------------------
                # Crear una nueva instancia para cada configuracion.
                # Reutilizar el tipo de modelo del RUN 2.
                #
                # modelo = HistGradientBoostingRegressor(
                #     max_iter=params["max_iter"],
                #     learning_rate=params["learning_rate"],
                #     random_state=config.SEMILLA,
                # )


                # ------------------------------------------------
                # PASO 8. ENTRENAR EL MODELO
                # ------------------------------------------------
                # Reutilizar las variables que main() ya preparo.
                # No repetir la ingesta ni el split temporal.
                #
                # modelo.fit(X_train, y_train)


                # ------------------------------------------------
                # PASO 9. GENERAR PREDICCIONES DE VALIDACION
                # ------------------------------------------------
                # Utilizar validacion para comparar configuraciones.
                # No seleccionar hiperparametros mirando el conjunto test.
                #
                # y_val_pred = modelo.predict(X_val)


                # ------------------------------------------------
                # PASO 10. CALCULAR Y REGISTRAR METRICAS
                # ------------------------------------------------
                # Christian debe calcular las metricas de validacion
                # utilizando la funcion de evaluacion apropiada.
                #
                # Por ejemplo, si el equipo utiliza MAE y RMSE:
                #
                # metricas_val = {
                #     "val_mae": ...,
                #     "val_rmse": ...,
                # }
                #
                # Registrar las metricas en la corrida hija:
                #
                # tracker.registrar_metricas(metricas_val)


                # ------------------------------------------------
                # PASO 11. REGISTRAR LOS ARTEFACTOS
                # ------------------------------------------------
                # Christian debe guardar y registrar los artefactos
                # que respalden el resultado de cada configuracion.
                #
                # Ejemplos:
                # - Grafica de predicciones o residuales.
                # - Configuracion utilizada.
                #
                # Reutilizar las funciones graficas existentes
                # cuando sean compatibles con los datos de validacion.
                #
                # tracker.registrar_artefacto(ruta_grafica)


                # ------------------------------------------------
                # PASO 12. REGISTRAR EL MODELO ENTRENADO
                # ------------------------------------------------
                # Registrar el modelo de la hija con MLflow.
                # Incluir su firma y un ejemplo de entrada.
                #
                # Christian puede reutilizar el patron del RUN 2:
                #
                # mlflow.sklearn.log_model(
                #     sk_model=modelo,
                #     name=f"modelo_hgb_prueba_{i}",
                #     signature=signature,
                #     input_example=input_example,
                # )
                #
                # Adaptar los argumentos a la version de MLflow
                # y al procedimiento que utilice el equipo.
                #
                # El registro en el Model Registry del modelo ganador
                # corresponde a una tarea posterior.


    # ------------------------------------------------------------
    # PASO 13. COMPARAR LAS CONFIGURACIONES
    # ------------------------------------------------------------
    # Despues de ejecutar la busqueda:
    #
    # - Revisar las metricas de validacion de todas las hijas.
    # - Comparar los resultados con el baseline de P1.
    # - Elegir la configuracion con base en las metricas acordadas.
    # - Evaluar en test la configuracion seleccionada.
    # - Justificar la eleccion y documentar los resultados.
    #
    # No utilizar el conjunto test para escoger los hiperparametros.
