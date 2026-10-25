import argparse
from pathlib import Path
import tempfile

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici import config
from mibici.dataset import calcular_hash
from mibici.features import perfil_uso_estacion
from mibici.modeling.tracking import RunMLflow

VARIABLES_BASELINE = [
    "salidas",
    "llegadas",
    "volumen_diario",
]


def parse_args():
    """
    Lee los argumentos de linea de comandos del baseline P3.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Entrena el baseline P3 con KMeans utilizando "
            "variables de volumen por estacion."
        )
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Numero fijo de grupos para el baseline P3.",
    )

    return parser.parse_args()


def listar_parquet_entrenamiento():
    """
    Obtiene los archivos Parquet anteriores al inicio de validacion.

    Returns:
        list[Path]: Archivos mensuales del periodo de entrenamiento.
    """
    rutas = sorted(
        Path(config.PROCESSED_DATA_DIR).glob(
            "datos_abiertos_*.parquet"
        )
    )

    inicio_validacion = pd.Timestamp(
        config.INICIO_VALIDACION
    )

    rutas_entrenamiento = []

    for ruta in rutas:
        periodo = ruta.stem.replace(
            "datos_abiertos_",
            "",
        ).replace(
            "_",
            "-",
        )

        fecha_mes = pd.Timestamp(
            periodo + "-01"
        )

        if fecha_mes < inicio_validacion:
            rutas_entrenamiento.append(ruta)

    if not rutas_entrenamiento:
        raise FileNotFoundError(
            "No se encontraron Parquet para entrenamiento."
        )

    return rutas_entrenamiento


def cargar_datos_entrenamiento(rutas):
    """
    Carga y concatena los viajes del periodo de entrenamiento.

    Solo carga las columnas necesarias para construir las variables P3.

    Args:
        rutas (list[Path]): Archivos mensuales de entrenamiento.

    Returns:
        pd.DataFrame: Viajes concatenados del periodo de entrenamiento.
    """
    columnas = [
        "viaje_id",
        "inicio_del_viaje",
        "fin_del_viaje",
        "origen_id",
        "destino_id",
    ]

    tablas = [
        pd.read_parquet(
            ruta,
            columns=columnas,
        )
        for ruta in rutas
    ]

    return pd.concat(
        tablas,
        ignore_index=True,
    )


def cargar_estaciones():
    """
    Carga la nomenclatura de estaciones utilizada para P3.

    Returns:
        pd.DataFrame: Catalogo de estaciones de MiBici.
    """
    ruta = (
        Path(config.RAW_DATA_DIR)
        / "nomenclatura_2026_09.csv"
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro la nomenclatura: {ruta}"
        )

    return pd.read_csv(ruta)


def entrenar_baseline(perfil, n_clusters):
    """
    Ajusta el baseline P3 usando variables de volumen y KMeans.

    Args:
        perfil (pd.DataFrame): Perfil agregado por estacion.
        n_clusters (int): Numero fijo de grupos.

    Returns:
        tuple: Pipeline entrenado, etiquetas, silhouette y
        Davies-Bouldin.
    """
    if n_clusters < 2:
        raise ValueError(
            "n_clusters debe ser al menos 2."
        )

    if n_clusters >= len(perfil):
        raise ValueError(
            "n_clusters debe ser menor que el numero de estaciones."
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
                    random_state=config.SEMILLA,
                    n_init=10,
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
        silhouette,
        davies_bouldin,
    )


def main():
    """
    Ejecuta y registra en MLflow el baseline P3.
    """
    args = parse_args()

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

    (
        modelo,
        etiquetas,
        silhouette,
        davies_bouldin,
    ) = entrenar_baseline(
        perfil,
        args.n_clusters,
    )

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

    version_datos = calcular_hash(
        config.MANIFEST_PATH
    )

    fecha_fin_entrenamiento = (
        pd.Timestamp(config.INICIO_VALIDACION)
        - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    with tempfile.TemporaryDirectory() as directorio:
        directorio = Path(directorio)

        ruta_resumen = (
            directorio
            / "resumen_clusters_p3.csv"
        )

        ruta_modelo = (
            directorio
            / "baseline_kmeans_p3.joblib"
        )

        resumen.to_csv(
            ruta_resumen,
            index=False,
        )

        joblib.dump(
            modelo,
            ruta_modelo,
        )

        with RunMLflow(
            pregunta="P3",
            nombre_run="p3-baseline-kmeans-volumen",
            version_datos=version_datos,
            fecha_inicio=config.FECHA_INICIO,
            fecha_fin=fecha_fin_entrenamiento,
            model_family="kmeans",
            stage_of_work="baseline",
        ) as run:
            run.registrar_parametros({
                "modelo": "KMeans",
                "n_clusters": args.n_clusters,
                "random_state": config.SEMILLA,
                "n_init": 10,
                "variables": ",".join(
                    VARIABLES_BASELINE
                ),
                "archivos_entrenamiento": len(rutas),
                "estaciones_entrenamiento": len(perfil),
            })

            run.registrar_metricas({
                "train_silhouette": silhouette,
                "train_davies_bouldin": davies_bouldin,
            })

            run.registrar_artefacto(
                ruta_resumen
            )

            run.registrar_artefacto(
                ruta_modelo
            )

    print("\nBaseline P3 completado")
    print("Estaciones:", len(perfil))
    print("Clusters:", args.n_clusters)
    print("Silhouette:", silhouette)
    print("Davies-Bouldin:", davies_bouldin)
    print("Version de datos:", version_datos)


if __name__ == "__main__":
    main()
