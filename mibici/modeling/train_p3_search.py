from itertools import product
from pathlib import Path
import tempfile

import joblib
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

VALORES_N_CLUSTERS = [3, 4, 5, 6]
VALORES_N_INIT = [10, 20]


def entrenar_kmeans(perfil, n_clusters, n_init):
    """
    Entrena KMeans con una combinacion de hiperparametros.

    Args:
        perfil (pd.DataFrame): Perfil agregado por estacion.
        n_clusters (int): Numero de grupos.
        n_init (int): Numero de inicializaciones de KMeans.

    Returns:
        tuple: Pipeline, etiquetas, silhouette y Davies-Bouldin.
    """
    if n_clusters < 2:
        raise ValueError(
            "n_clusters debe ser al menos 2."
        )

    if n_clusters >= len(perfil):
        raise ValueError(
            "n_clusters debe ser menor que el numero de estaciones."
        )

    if n_init < 1:
        raise ValueError(
            "n_init debe ser al menos 1."
        )

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
                    n_clusters=n_clusters,
                    n_init=n_init,
                    random_state=config.SEMILLA,
                ),
            ),
        ]
    )

    etiquetas = pipeline.fit_predict(
        datos_modelo
    )

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

    return (
        pipeline,
        etiquetas,
        float(silhouette),
        float(davies_bouldin),
    )


def construir_resumen(perfil, etiquetas):
    """
    Construye el resumen de estaciones y clusters.
    """
    resumen = perfil[
        [
            "estacion_id",
            "salidas",
            "llegadas",
            "volumen_diario",
            "capacidad",
        ]
    ].copy()

    resumen["cluster"] = etiquetas

    return resumen


def registrar_trial(
    perfil,
    n_clusters,
    n_init,
    version_datos,
    fecha_fin_entrenamiento,
    numero_trial,
):
    """
    Ejecuta y registra una combinacion como run hijo.

    Returns:
        dict: Resultado de la combinacion evaluada.
    """
    (
        modelo,
        etiquetas,
        silhouette,
        davies_bouldin,
    ) = entrenar_kmeans(
        perfil,
        n_clusters=n_clusters,
        n_init=n_init,
    )

    nombre_run = (
        f"p3-kmeans-trial-{numero_trial}"
        f"-k{n_clusters}-init{n_init}"
    )

    with tempfile.TemporaryDirectory() as directorio:
        directorio = Path(directorio)

        ruta_modelo = (
            directorio
            / f"kmeans_k{n_clusters}_init{n_init}.joblib"
        )

        ruta_resumen = (
            directorio
            / f"clusters_k{n_clusters}_init{n_init}.csv"
        )

        joblib.dump(
            modelo,
            ruta_modelo,
        )

        resumen = construir_resumen(
            perfil,
            etiquetas,
        )

        resumen.to_csv(
            ruta_resumen,
            index=False,
        )

        with RunMLflow(
            pregunta="P3",
            nombre_run=nombre_run,
            version_datos=version_datos,
            fecha_inicio=config.FECHA_INICIO,
            fecha_fin=fecha_fin_entrenamiento,
            nested=True,
            model_family="kmeans",
            stage_of_work="search",
        ) as run:
            run.registrar_parametros({
                "modelo": "KMeans",
                "n_clusters": n_clusters,
                "n_init": n_init,
                "random_state": config.SEMILLA,
                "variables": ",".join(
                    VARIABLES_BASELINE
                ),
                "estaciones_entrenamiento": len(perfil),
            })

            run.registrar_metricas({
                "train_silhouette": silhouette,
                "train_davies_bouldin": davies_bouldin,
            })

            run.registrar_artefacto(
                ruta_modelo
            )

            run.registrar_artefacto(
                ruta_resumen
            )

    return {
        "trial": numero_trial,
        "n_clusters": n_clusters,
        "n_init": n_init,
        "train_silhouette": silhouette,
        "train_davies_bouldin": davies_bouldin,
    }


def main():
    """
    Ejecuta la busqueda P3 con un run padre y runs hijos.
    """
    rutas = listar_parquet_entrenamiento()

    print(
        "Archivos de entrenamiento:",
        len(rutas),
    )
    print("Primer archivo:", rutas[0])
    print("Ultimo archivo:", rutas[-1])

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

    resultados = []

    mlflow.set_tags({
        "mlflow.runName": "p3-kmeans-busqueda",
        "equipo": "06",
        "pregunta": "P3",
        "git_commit": obtener_git_commit(),
        "model_family": "kmeans",
        "stage_of_work": "search",
        "version_datos": version_datos,
    })

    mlflow.log_params({
        "valores_n_clusters": ",".join(
            str(valor)
            for valor in VALORES_N_CLUSTERS
        ),
        "valores_n_init": ",".join(
            str(valor)
            for valor in VALORES_N_INIT
        ),
        "numero_combinaciones": (
            len(VALORES_N_CLUSTERS)
            * len(VALORES_N_INIT)
        ),
        "variables": ",".join(
            VARIABLES_BASELINE
        ),
        "archivos_entrenamiento": len(rutas),
        "estaciones_entrenamiento": len(perfil),
    })

    combinaciones = product(
        VALORES_N_CLUSTERS,
        VALORES_N_INIT,
    )

    for numero_trial, combinacion in enumerate(
        combinaciones,
        start=1,
    ):
        n_clusters, n_init = combinacion

        print(
            f"Trial {numero_trial}: "
            f"n_clusters={n_clusters}, "
            f"n_init={n_init}"
        )

        resultado = registrar_trial(
            perfil=perfil,
            n_clusters=n_clusters,
            n_init=n_init,
            version_datos=version_datos,
            fecha_fin_entrenamiento=fecha_fin_entrenamiento,
            numero_trial=numero_trial,
        )

        resultados.append(
            resultado
        )

    resultados_df = pd.DataFrame(
        resultados
    ).sort_values(
        by=[
            "train_silhouette",
            "train_davies_bouldin",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    mejor = resultados_df.iloc[0]

    mlflow.log_params({
        "best_n_clusters": int(
            mejor["n_clusters"]
        ),
        "best_n_init": int(
            mejor["n_init"]
        ),
    })

    mlflow.log_metrics({
        "best_train_silhouette": float(
            mejor["train_silhouette"]
        ),
        "best_train_davies_bouldin": float(
            mejor["train_davies_bouldin"]
        ),
    })

    with tempfile.TemporaryDirectory() as directorio:
        ruta_resultados = (
            Path(directorio)
            / "resultados_busqueda_p3.csv"
        )

        resultados_df.to_csv(
            ruta_resultados,
            index=False,
        )

        mlflow.log_artifact(
            str(ruta_resultados),
            artifact_path="search",
        )

    print("\nBusqueda P3 completada")
    print(resultados_df.to_string(index=False))
    print("\nMejor configuracion:")
    print(
        "n_clusters:",
        int(mejor["n_clusters"]),
    )
    print(
        "n_init:",
        int(mejor["n_init"]),
    )
    print(
        "Silhouette:",
        mejor["train_silhouette"],
    )
    print(
        "Davies-Bouldin:",
        mejor["train_davies_bouldin"],
    )


if __name__ == "__main__":
    main()
