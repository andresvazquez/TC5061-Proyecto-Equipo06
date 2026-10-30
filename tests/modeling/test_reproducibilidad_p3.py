import pandas as pd
import pytest

from mibici import config
from mibici.modeling.train_p3_stability import (
    entrenar_con_semilla,
)


def crear_perfil_p3():
    """
    Construye un perfil pequeno con grupos distinguibles
    para validar reproducibilidad sin utilizar data/.
    """
    return pd.DataFrame({
        "salidas": [
            100,
            110,
            90,
            500,
            520,
            480,
            1000,
            1050,
            950,
        ],
        "llegadas": [
            95,
            115,
            92,
            510,
            500,
            490,
            980,
            1070,
            960,
        ],
        "volumen_diario": [
            10,
            11,
            9,
            50,
            52,
            48,
            100,
            105,
            95,
        ],
    })


def test_p3_misma_semilla_mismas_metricas():
    """
    Dos entrenamientos P3 con la misma semilla deben
    producir exactamente las mismas metricas.
    """
    perfil = crear_perfil_p3()

    resultado_1 = entrenar_con_semilla(
        perfil,
        config.SEMILLA,
    )

    resultado_2 = entrenar_con_semilla(
        perfil,
        config.SEMILLA,
    )

    assert resultado_1["semilla"] == config.SEMILLA
    assert resultado_2["semilla"] == config.SEMILLA

    assert resultado_1[
        "train_silhouette"
    ] == pytest.approx(
        resultado_2["train_silhouette"]
    )

    assert resultado_1[
        "train_davies_bouldin"
    ] == pytest.approx(
        resultado_2["train_davies_bouldin"]
    )


def test_p3_metricas_validas_en_repeticiones():
    """
    Las dos ejecuciones reproducibles deben mantener
    metricas de clustering dentro de rangos validos.
    """
    perfil = crear_perfil_p3()

    resultados = [
        entrenar_con_semilla(
            perfil,
            config.SEMILLA,
        )
        for _ in range(2)
    ]

    for resultado in resultados:
        assert (
            -1
            <= resultado["train_silhouette"]
            <= 1
        )

        assert (
            resultado["train_davies_bouldin"]
            >= 0
        )

        assert resultado["train_seconds"] >= 0


def test_p3_reproducibilidad_no_depende_del_tiempo():
    """
    La reproducibilidad se valida sobre los resultados
    del modelo y no sobre el tiempo de ejecucion.
    """
    perfil = crear_perfil_p3()

    resultado_1 = entrenar_con_semilla(
        perfil,
        config.SEMILLA,
    )

    resultado_2 = entrenar_con_semilla(
        perfil,
        config.SEMILLA,
    )

    metricas_1 = {
        "silhouette": resultado_1[
            "train_silhouette"
        ],
        "davies_bouldin": resultado_1[
            "train_davies_bouldin"
        ],
    }

    metricas_2 = {
        "silhouette": resultado_2[
            "train_silhouette"
        ],
        "davies_bouldin": resultado_2[
            "train_davies_bouldin"
        ],
    }

    assert metricas_1 == pytest.approx(
        metricas_2
    )
