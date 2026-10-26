import argparse
from pathlib import Path
import tempfile

import joblib
import mlflow
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici import config
from mibici.dataset import calcular_hash
from mibici.features import perfil_uso_estacion
from mibici.modeling.tracking import RunMLflow
from mibici.plots import graficar_perfiles_clusters

VARIABLES_BASELINE = [
    "salidas",
    "llegadas",
    "volumen_diario",
]

VARIABLES_PERFIL_COMPLETO = [
    *[
        f"perfil_h{hora:02d}"
        for hora in range(24)
    ],
    "proporcion_fin_semana",
    "volumen_diario",
    "saldo_neto",
    "capacidad",
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
        "--model",
        choices=[
            "baseline",
            "agglomerative",
            "full-profile",
        ],
        default="baseline",
        help="Modelo P3 que se desea ejecutar.",
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

def entrenar_agglomerative(perfil, n_clusters):
    """
    Ajusta la candidata P3 con Agglomerative Clustering.

    Utiliza las mismas variables de volumen y el mismo numero
    de grupos que el baseline para comparar exclusivamente el
    efecto de cambiar el algoritmo.

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
                "agglomerative",
                AgglomerativeClustering(
                    n_clusters=n_clusters,
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

def entrenar_perfil_completo(
    perfil,
    n_clusters,
    n_init=10,
):
    """
    Ajusta KMeans utilizando el perfil completo de P3.

    Mantiene el algoritmo KMeans y utiliza las variables
    temporales, de volumen, saldo y capacidad del perfil
    de estaciones.

    Args:
        perfil (pd.DataFrame): Perfil agregado por estacion.
        n_clusters (int): Numero de grupos.
        n_init (int): Numero de inicializaciones de KMeans.

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
        VARIABLES_PERFIL_COMPLETO
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

    if args.model == "baseline":
        (
            modelo,
            etiquetas,
            silhouette,
            davies_bouldin,
        ) = entrenar_baseline(
            perfil,
            args.n_clusters,
        )

        nombre_run = "p3-baseline-kmeans-volumen"
        nombre_modelo = "KMeans"
        model_family = "kmeans"
        stage_of_work = "baseline"
        nombre_artefacto = "baseline_kmeans_p3.joblib"

    elif args.model == "agglomerative":
        (
            modelo,
            etiquetas,
            silhouette,
            davies_bouldin,
        ) = entrenar_agglomerative(
            perfil,
            args.n_clusters,
        )

        nombre_run = "p3-candidato-agglomerative-volumen"
        nombre_modelo = "AgglomerativeClustering"
        model_family = "agglomerative"
        stage_of_work = "candidate"
        nombre_artefacto = "candidato_agglomerative_p3.joblib"

    elif args.model == "full-profile":
        (
            modelo,
            etiquetas,
            silhouette,
            davies_bouldin,
        ) = entrenar_perfil_completo(
            perfil,
            n_clusters=args.n_clusters,
            n_init=10,
        )

        nombre_run = "p3-candidato-kmeans-perfil-completo"
        nombre_modelo = "KMeansPerfilCompleto"
        model_family = "kmeans"
        stage_of_work = "candidate"
        nombre_artefacto = (
            "candidato_kmeans_perfil_completo_p3.joblib"
        )

    else:
        raise ValueError(
            "Modelo P3 no soportado."
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

        if args.model == "full-profile":
            nombre_resumen = (
                "resumen_clusters_perfil_completo_p3.csv"
            )
        else:
            nombre_resumen = "resumen_clusters_p3.csv"

        ruta_resumen = (
            directorio
            / nombre_resumen
        )

        ruta_modelo = (
            directorio
            / nombre_artefacto
        )

        ruta_perfiles = (
            directorio
            / "perfiles_clusters_p3.png"
        )

        resumen.to_csv(
            ruta_resumen,
            index=False,
        )

        graficar_perfiles_clusters(
            resumen,
            ruta_perfiles,
        )

        joblib.dump(
            modelo,
            ruta_modelo,
        )

        with RunMLflow(
            pregunta="P3",
            nombre_run=nombre_run,
            version_datos=version_datos,
            fecha_inicio=config.FECHA_INICIO,
            fecha_fin=fecha_fin_entrenamiento,
            model_family=model_family,
            stage_of_work=stage_of_work,
        ) as run:
            if args.model == "full-profile":
                variables_modelo = VARIABLES_PERFIL_COMPLETO
            else:
                variables_modelo = VARIABLES_BASELINE
            parametros = {
                "modelo": nombre_modelo,
                "n_clusters": args.n_clusters,
                "variables": ",".join(
                    variables_modelo
                ),
                "archivos_entrenamiento": len(rutas),
                "estaciones_entrenamiento": len(perfil),
            }

            if args.model in (
                "baseline",
                "full-profile",
            ):
                parametros.update({
                    "random_state": config.SEMILLA,
                    "n_init": 10,
                })
            run.registrar_parametros(
                parametros
            )

            run.registrar_metricas({
                "train_silhouette": silhouette,
                "train_davies_bouldin": davies_bouldin,
            })

            X_modelo = perfil[
                variables_modelo
            ].copy()

            if hasattr(modelo, "predict"):
                run.log_model(
                    model=modelo,
                    X_sample=X_modelo,
                    artifact_path="model",
                )

            run.registrar_artefacto(
                config.MANIFEST_PATH,
                dir_remoto="data",
            )

            run.log_config(
                config_path="mibici/config.py",
                artifact_path="config",
            )

            run.registrar_artefacto(
                ruta_resumen
            )

            run.registrar_artefacto(
                ruta_modelo
            )

            run.registrar_artefacto(
                ruta_resumen
            )

            run.registrar_artefacto(
                ruta_modelo
            )

            mlflow.log_artifact(
                str(ruta_perfiles),
                artifact_path="plots",
            )

    print(f"\nP3 completado: {nombre_modelo}")
    print("Estaciones:", len(perfil))
    print("Clusters:", args.n_clusters)
    print("Silhouette:", silhouette)
    print("Davies-Bouldin:", davies_bouldin)
    print("Version de datos:", version_datos)


if __name__ == "__main__":
    main()
