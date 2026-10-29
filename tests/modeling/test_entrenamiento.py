import pandas as pd
import pytest

from mibici import config
from mibici.modeling.train_p1 import dividir_train_val_test
from mibici.modeling.train_p2 import partir_por_fecha
from mibici.modeling.train_p3 import entrenar_baseline
from mibici.modeling.train_p3_search import entrenar_kmeans


def crear_datos_temporales():
    """
    Construye una tabla pequena alrededor de los cortes temporales
    configurados para entrenamiento, validacion y prueba.
    """
    inicio_validacion = pd.Timestamp(
        config.INICIO_VALIDACION
    )

    inicio_prueba = pd.Timestamp(
        config.INICIO_PRUEBA
    )

    return pd.DataFrame({
        "hora": [
            inicio_validacion - pd.Timedelta(hours=1),
            inicio_validacion,
            inicio_prueba - pd.Timedelta(hours=1),
            inicio_prueba,
        ],
        "valor": [1, 2, 3, 4],
    })


def crear_perfil_p3():
    """
    Construye un perfil pequeno con tres grupos distinguibles.
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


def test_dividir_train_val_test_respeta_cortes():
    """
    P1 debe separar correctamente train, validacion y prueba.
    """
    df = crear_datos_temporales()

    train, val, test = dividir_train_val_test(df)

    assert len(train) == 1
    assert len(val) == 2
    assert len(test) == 1

    assert (
        train["hora"]
        < pd.Timestamp(config.INICIO_VALIDACION)
    ).all()

    assert (
        val["hora"]
        >= pd.Timestamp(config.INICIO_VALIDACION)
    ).all()

    assert (
        val["hora"]
        < pd.Timestamp(config.INICIO_PRUEBA)
    ).all()

    assert (
        test["hora"]
        >= pd.Timestamp(config.INICIO_PRUEBA)
    ).all()


def test_dividir_train_val_test_fronteras():
    """
    Una fila exactamente en cada corte debe entrar al conjunto
    posterior correspondiente.
    """
    df = pd.DataFrame({
        "hora": [
            pd.Timestamp(config.INICIO_VALIDACION),
            pd.Timestamp(config.INICIO_PRUEBA),
        ]
    })

    train, val, test = dividir_train_val_test(df)

    assert train.empty
    assert len(val) == 1
    assert len(test) == 1


def test_partir_por_fecha_respeta_cortes():
    """
    P2 debe utilizar los mismos cortes temporales del proyecto.
    """
    malla = crear_datos_temporales()

    entrenamiento, validacion, prueba = partir_por_fecha(
        malla
    )

    assert len(entrenamiento) == 1
    assert len(validacion) == 2
    assert len(prueba) == 1

    assert (
        entrenamiento["hora"]
        < config.INICIO_VALIDACION
    ).all()

    assert (
        validacion["hora"]
        >= config.INICIO_VALIDACION
    ).all()

    assert (
        validacion["hora"]
        < config.INICIO_PRUEBA
    ).all()

    assert (
        prueba["hora"]
        >= config.INICIO_PRUEBA
    ).all()


def test_partir_por_fecha_fronteras():
    """
    P2 debe asignar correctamente observaciones en los cortes.
    """
    malla = pd.DataFrame({
        "hora": [
            pd.Timestamp(config.INICIO_VALIDACION),
            pd.Timestamp(config.INICIO_PRUEBA),
        ]
    })

    entrenamiento, validacion, prueba = partir_por_fecha(
        malla
    )

    assert entrenamiento.empty
    assert len(validacion) == 1
    assert len(prueba) == 1


def test_entrenar_baseline_p3_caso_normal():
    """
    El baseline P3 debe producir una etiqueta por estacion
    y metricas de clustering validas.
    """
    perfil = crear_perfil_p3()

    modelo, etiquetas, silhouette, davies = entrenar_baseline(
        perfil,
        n_clusters=3,
    )

    assert len(etiquetas) == len(perfil)
    assert -1 <= silhouette <= 1
    assert davies >= 0

    assert "scaler" in modelo.named_steps
    assert "kmeans" in modelo.named_steps

    assert (
        modelo.named_steps["kmeans"].n_clusters
        == 3
    )


def test_entrenar_baseline_rechaza_un_cluster():
    """
    P3 debe rechazar n_clusters menor que 2.
    """
    perfil = crear_perfil_p3()

    with pytest.raises(
        ValueError,
        match="n_clusters debe ser al menos 2",
    ):
        entrenar_baseline(
            perfil,
            n_clusters=1,
        )


def test_entrenar_baseline_rechaza_clusters_igual_a_estaciones():
    """
    P3 debe rechazar tantos clusters como estaciones.
    """
    perfil = crear_perfil_p3()

    with pytest.raises(
        ValueError,
        match=(
            "n_clusters debe ser menor que "
            "el numero de estaciones"
        ),
    ):
        entrenar_baseline(
            perfil,
            n_clusters=len(perfil),
        )


def test_entrenar_kmeans_search_caso_normal():
    """
    La funcion reutilizada por la busqueda debe respetar
    los hiperparametros recibidos.
    """
    perfil = crear_perfil_p3()

    modelo, etiquetas, silhouette, davies = entrenar_kmeans(
        perfil,
        n_clusters=3,
        n_init=20,
    )

    assert len(etiquetas) == len(perfil)
    assert -1 <= silhouette <= 1
    assert davies >= 0

    assert (
        modelo.named_steps["kmeans"].n_clusters
        == 3
    )

    assert (
        modelo.named_steps["kmeans"].n_init
        == 20
    )


def test_entrenar_kmeans_search_rechaza_n_init_invalido():
    """
    La busqueda no debe aceptar un numero de inicializaciones
    menor que uno.
    """
    perfil = crear_perfil_p3()

    with pytest.raises(
        ValueError,
        match="n_init debe ser al menos 1",
    ):
        entrenar_kmeans(
            perfil,
            n_clusters=3,
            n_init=0,
        )
