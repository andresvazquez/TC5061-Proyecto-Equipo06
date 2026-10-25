from pathlib import Path
import tempfile
from time import perf_counter

import mlflow
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici import config
from mibici.dataset import calcular_hash
from mibici.features import perfil_uso_estacion
from mibici.modeling.tracking import RunMLflow, obtener_git_commit
from mibici.modeling.train_p3 import (
    VARIABLES_BASELINE,
    cargar_datos_entrenamiento,
    cargar_estaciones,
    listar_parquet_entrenamiento,
)

N_CLUSTERS = 3
N_INIT = 10


def entrenar_con_semilla(perfil, semilla):
    """
    Entrena la configuración final P3 con una semilla específica.

    Returns:
        tuple: Etiquetas, métricas y tiempo de entrenamiento.
    """
    datos_modelo = perfil[
        VARIABLES_BASELINE
    ].copy()

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "kmeans",
                KMeans(
                    n_clusters=N_CLUSTERS,
                    n_init=N_INIT,
                    random_state=semilla,
                ),
            ),
        ]
    )

    inicio = perf_counter()

    etiquetas = pipeline.fit_predict(
        datos_modelo
    )

    train_seconds = perf_counter() - inicio

    datos_escalados = pipeline.named_steps[
        "scaler"
    ].transform(datos_modelo)

    silhouette = silhouette_score(
        datos_escalados,
        etiquetas,
    )

    davies_bouldin = davies_bouldin_score(
        datos_escalados,
        etiquetas,
    )

    return {
        "semilla": semilla,
        "train_silhouette": float(silhouette),
        "train_davies_bouldin": float(davies_bouldin),
        "train_seconds": float(train_seconds),
    }


def main():
    """
    Evalúa estabilidad y costo del modelo final P3.
    """
    rutas = listar_parquet_entrenamiento()

    viajes = cargar_datos_entrenamiento(
        rutas
    )

    estaciones = cargar_estaciones()

    perfil = perfil_uso_estacion(
        viajes,
        estaciones,
    )

    version_datos = calcular_hash(
        config.MANIFEST_PATH
    )

    fecha_fin_entrenamiento = (
        pd.Timestamp(config.INICIO_VALIDACION)
        - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    semillas = [
        config.SEMILLA,
        config.SEMILLA + 1,
    ]

    mlflow.set_tags({
        "mlflow.runName": "p3-kmeans-estabilidad-costo",
        "equipo": "06",
        "pregunta": "P3",
        "git_commit": obtener_git_commit(),
        "model_family": "kmeans",
        "stage_of_work": "stability",
        "version_datos": version_datos,
    })

    mlflow.log_params({
        "n_clusters": N_CLUSTERS,
        "n_init": N_INIT,
        "semillas": ",".join(
            str(semilla)
            for semilla in semillas
        ),
        "variables": ",".join(
            VARIABLES_BASELINE
        ),
        "estaciones_entrenamiento": len(perfil),
    })

    resultados = []

    for semilla in semillas:
        resultado = entrenar_con_semilla(
            perfil,
            semilla,
        )

        resultados.append(
            resultado
        )

        with RunMLflow(
            pregunta="P3",
            nombre_run=f"p3-kmeans-seed-{semilla}",
            version_datos=version_datos,
            fecha_inicio=config.FECHA_INICIO,
            fecha_fin=fecha_fin_entrenamiento,
            nested=True,
            model_family="kmeans",
            stage_of_work="stability",
        ) as run:
            run.registrar_parametros({
                "modelo": "KMeans",
                "n_clusters": N_CLUSTERS,
                "n_init": N_INIT,
                "random_state": semilla,
                "variables": ",".join(
                    VARIABLES_BASELINE
                ),
            })

            run.registrar_metricas({
                "train_silhouette": resultado[
                    "train_silhouette"
                ],
                "train_davies_bouldin": resultado[
                    "train_davies_bouldin"
                ],
                "train_seconds": resultado[
                    "train_seconds"
                ],
            })

    resultados_df = pd.DataFrame(
        resultados
    )

    mlflow.log_metrics({
        "silhouette_mean": float(
            resultados_df["train_silhouette"].mean()
        ),
        "silhouette_std": float(
            resultados_df["train_silhouette"].std(ddof=0)
        ),
        "davies_bouldin_mean": float(
            resultados_df["train_davies_bouldin"].mean()
        ),
        "davies_bouldin_std": float(
            resultados_df["train_davies_bouldin"].std(ddof=0)
        ),
        "train_seconds_mean": float(
            resultados_df["train_seconds"].mean()
        ),
        "train_seconds_max": float(
            resultados_df["train_seconds"].max()
        ),
    })

    with tempfile.TemporaryDirectory() as directorio:
        ruta_resultados = (
            Path(directorio)
            / "estabilidad_costo_p3.csv"
        )

        resultados_df.to_csv(
            ruta_resultados,
            index=False,
        )

        mlflow.log_artifact(
            str(ruta_resultados),
            artifact_path="evaluation",
        )

    print("\nValidación final P3")
    print(resultados_df.to_string(index=False))
    print()
    print(
        "Silhouette promedio:",
        resultados_df["train_silhouette"].mean(),
    )
    print(
        "Davies-Bouldin promedio:",
        resultados_df["train_davies_bouldin"].mean(),
    )
    print(
        "Tiempo promedio:",
        resultados_df["train_seconds"].mean(),
    )


if __name__ == "__main__":
    main()
