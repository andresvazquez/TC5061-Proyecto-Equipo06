import argparse
import os
import tempfile

import matplotlib.pyplot as plt
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

    if not isinstance(y_real, pd.Series):
        y_real = pd.Series(y_real)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_real.index)

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
    ax[0].set_title(f"{titulo} Real vs Predicho")
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Residuos
    residuos = y_real - y_pred
    ax[1].hist(residuos, bins=50, range=(-15, 15), color="coral", edgecolor="black", alpha=0.7)
    ax[1].set_xlabel("Residuo (Real - Predicho)")
    ax[1].set_ylabel("Frecuencia")
    ax[1].set_title(f"{titulo} Distribución de Residuos (-15 a 15)")
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
    inicio_val = pd.Timestamp(config.INICIO_VALIDACION)
    inicio_test = pd.Timestamp(config.INICIO_PRUEBA)

    train = df[df["hora"] < inicio_val].copy()
    val = df[(df["hora"] >= inicio_val) & (df["hora"] < inicio_test)].copy()
    test = df[df["hora"] >= inicio_test].copy()

    print(f"Corte Train -> Val: {inicio_val} | Corte Val -> Test: {inicio_test}")
    return train, val, test


def parse_args():
    """
    Lee el modelo y los hiperparámetros de P1.
    """
    parser = argparse.ArgumentParser(
        description="Entrena el baseline o candidatos de P1."
    )

    parser.add_argument(
        "--model",
        choices=["baseline", "hgb", "search", "features", "all"],
        default="all",
        help="Modelo o experimento de P1 que se desea ejecutar.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Número máximo de iteraciones del HistGradientBoosting.",
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

    # 3. División train, val, test
    train, val, test = dividir_train_val_test(df_features)

    X_cols_full = [
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

    X_train = train[X_cols_full]
    y_train = train["salidas"]

    X_val = val[X_cols_full]
    y_val = val["salidas"]

    X_test = test[X_cols_full]
    y_test = test["salidas"]

    # ------------------------------------------------------------
    # RUN 1: BASELINE (Demanda de la semana pasada)
    # ------------------------------------------------------------
    if args.model in ("baseline", "all"):
        with RunMLflow(
            pregunta="P1",
            nombre_run="Baseline_P1_Lag168",
            version_datos=version_datos,
            model_family="baseline_lag_168",
            stage_of_work="baseline",
        ) as tracker:
            tracker.registrar_parametros({"modelo_tipo": "Baseline_Lag_168"})

            y_val_pred_base = val["salidas_lag_168"]
            y_test_pred_base = test["salidas_lag_168"]

            metricas_baseline = evaluar_conjuntos(
                y_val_real=y_val,
                y_val_pred=y_val_pred_base,
                y_test_real=y_test,
                y_test_pred=y_test_pred_base,
            )
            tracker.registrar_metricas(metricas_baseline)

            img_baseline = guardar_grafica_predicciones(
                y_test, y_test_pred_base, "Baseline", "grafica_baseline.png"
            )
            tracker.registrar_artefacto(img_baseline)

            print("[Baseline] Métricas registradas:", metricas_baseline)

    # ------------------------------------------------------------
    # FILA 7a: MODELO CANDIDATO (Variando Algoritmo - HistGradientBoosting)
    # ------------------------------------------------------------
    if args.model in ("hgb", "all"):
        with RunMLflow(
            pregunta="P1",
            nombre_run="Candidato_HistGradientBoosting",
            version_datos=version_datos,
            model_family="hist_gradient_boosting",
            stage_of_work="candidate",
        ) as tracker:
            tracker.registrar_parametros({
                "modelo_tipo": "HistGradientBoostingRegressor",
                "random_state": config.SEMILLA,
                "n_features": len(X_cols_full),
                "inicio_validacion": str(config.INICIO_VALIDACION),
                "inicio_prueba": str(config.INICIO_PRUEBA),
                "max_iter": args.max_iter,
                "learning_rate": args.learning_rate,
            })

            modelo = HistGradientBoostingRegressor(
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                random_state=config.SEMILLA,
            )
            modelo.fit(X_train, y_train)

            y_val_pred_cand = modelo.predict(X_val)
            y_test_pred_cand = modelo.predict(X_test)

            metricas_candidato = evaluar_conjuntos(
                y_val_real=y_val,
                y_val_pred=y_val_pred_cand,
                y_test_real=y_test,
                y_test_pred=y_test_pred_cand,
            )
            tracker.registrar_metricas(metricas_candidato)

            img_candidato = guardar_grafica_predicciones(
                y_test, y_test_pred_cand, "HistGradientBoosting", "grafica_candidato.png"
            )
            tracker.registrar_artefacto(img_candidato)

            # FILA 11a: Modelo con firma e Input Example
            tracker.log_model(model=modelo, X_sample=X_val, artifact_path="model")

            print("[7a - Candidato Algoritmo] Métricas registradas:", metricas_candidato)

    # ------------------------------------------------------------
    # FILA 7b: BÚSQUEDA DE HIPERPARÁMETROS (Nested Runs)
    # ------------------------------------------------------------
    if args.model in ("search", "all"):
        combinaciones = [
            {"max_iter": 50, "learning_rate": 0.05},
            {"max_iter": 100, "learning_rate": 0.05},
            {"max_iter": 100, "learning_rate": 0.10},
            {"max_iter": 150, "learning_rate": 0.10},
        ]

        # Corrida Padre
        with RunMLflow(
            pregunta="P1",
            nombre_run="Busqueda_Hiperparametros_P1",
            version_datos=version_datos,
            stage_of_work="search",
            nested=False,
        ) as padre:
            padre.registrar_parametros({
                "modelo_tipo": "HistGradientBoostingRegressor",
                "tipo_busqueda": "grid_manual",
                "n_combinaciones": len(combinaciones),
                "random_state": config.SEMILLA,
            })

            # Corridas Hijas
            for i, params in enumerate(combinaciones, start=1):
                with RunMLflow(
                    pregunta="P1",
                    nombre_run=f"P1_HGB_Prueba_{i}",
                    version_datos=version_datos,
                    model_family="hist_gradient_boosting",
                    stage_of_work="search",
                    nested=True,
                ) as hija:
                    hija.registrar_parametros({
                        "modelo_tipo": "HistGradientBoostingRegressor",
                        **params,
                        "random_state": config.SEMILLA,
                        "n_features": len(X_cols_full),
                    })

                    modelo_hija = HistGradientBoostingRegressor(
                        max_iter=params["max_iter"],
                        learning_rate=params["learning_rate"],
                        random_state=config.SEMILLA,
                    )
                    modelo_hija.fit(X_train, y_train)

                    y_val_pred_hija = modelo_hija.predict(X_val)
                    val_mae = mean_absolute_error(y_val, y_val_pred_hija)
                    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_hija))

                    hija.registrar_metricas({
                        "val_mae": float(val_mae),
                        "val_rmse": float(val_rmse),
                    })

                    # FILA 11a: Modelo de la corrida hija con firma
                    hija.log_model(model=modelo_hija, X_sample=X_val, artifact_path="model")

            print(f"[7b - Búsqueda Hiperparámetros] {len(combinaciones)} corridas hijas completadas.")

    # ------------------------------------------------------------
    # FILA 7c: VARIANDO FEATURES (Sin variables de Lag/Ventanas)
    # ------------------------------------------------------------
    if args.model in ("features", "all"):
        # Subconjunto de características sin métricas agregadas móviles
        X_cols_reduced = [
            "hora_del_dia",
            "dia_de_semana",
            "es_fin_de_semana",
            "mes",
            "capacidad",
        ]

        X_train_red = train[X_cols_reduced]
        X_val_red = val[X_cols_reduced]
        X_test_red = test[X_cols_reduced]

        with RunMLflow(
            pregunta="P1",
            nombre_run="Candidato_Features_Reducidas",
            version_datos=version_datos,
            model_family="hist_gradient_boosting",
            stage_of_work="candidate",
        ) as tracker:
            tracker.registrar_parametros({
                "modelo_tipo": "HistGradientBoostingRegressor",
                "variacion": "features_reducidas",
                "random_state": config.SEMILLA,
                "n_features": len(X_cols_reduced),
                "features_usadas": str(X_cols_reduced),
            })

            modelo_feat = HistGradientBoostingRegressor(
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                random_state=config.SEMILLA,
            )
            modelo_feat.fit(X_train_red, y_train)

            y_val_pred_feat = modelo_feat.predict(X_val_red)
            y_test_pred_feat = modelo_feat.predict(X_test_red)

            metricas_feat = evaluar_conjuntos(
                y_val_real=y_val,
                y_val_pred=y_val_pred_feat,
                y_test_real=y_test,
                y_test_pred=y_test_pred_feat,
            )
            tracker.registrar_metricas(metricas_feat)

            img_feat = guardar_grafica_predicciones(
                y_test, y_test_pred_feat, "HGB_Features_Reducidas", "grafica_features_reducidas.png"
            )
            tracker.registrar_artefacto(img_feat)

            # FILA 11a: Registro del modelo con firma
            tracker.log_model(model=modelo_feat, X_sample=X_val_red, artifact_path="model")

            print("[7c - Variación Features] Métricas registradas:", metricas_feat)


if __name__ == "__main__":
    main()