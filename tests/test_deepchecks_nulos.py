from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import MixedNulls, PercentOfNulls
import numpy as np
import pandas as pd
import pytest

from mibici.features import construir_malla, etiqueta_riesgo, features_p2


def malla_pequena():
    """
    Construye una malla pequena y limpia para los checks de nulos.
    """
    estaciones = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3],
            "capacidad": [10, 20, 30],
            "latitud": [20.67, 20.68, 20.69],
            "longitud": [-103.35, -103.36, -103.37],
        }
    )

    inicio = pd.date_range(
        "2026-07-01 00:10",
        periods=60,
        freq="25min",
    )

    viajes = pd.DataFrame(
        {
            "viaje_id": range(60),
            "inicio_del_viaje": inicio,
            "fin_del_viaje": (
                inicio
                + pd.Timedelta(minutes=15)
            ),
            "origen_id": [1, 1, 2, 3] * 15,
            "destino_id": [2, 3, 3, 1] * 15,
        }
    )

    malla = features_p2(
        etiqueta_riesgo(
            construir_malla(
                viajes,
                estaciones,
            )
        )
    )

    return malla.dropna().reset_index(drop=True)


def armar_dataset(malla):
    """
    Construye el Dataset comun de DeepChecks para P2.
    """
    return Dataset(
        malla,
        label="riesgo",
        cat_features=["estacion_id"],
        datetime_name="hora",
    )


def test_percent_of_nulls_sin_nulos():
    """
    Una malla limpia debe reportar cero nulos.
    """
    resultado = PercentOfNulls().run(
        armar_dataset(
            malla_pequena()
        )
    )

    assert (resultado.value == 0).to_numpy().all()


def test_percent_of_nulls_detecta_nulos():
    """
    DeepChecks debe detectar valores NaN introducidos
    intencionalmente.
    """
    malla = malla_pequena()

    malla.loc[
        malla.index[:5],
        "saldo_1h",
    ] = np.nan

    resultado = PercentOfNulls(
        columns=["saldo_1h"]
    ).run(
        armar_dataset(malla)
    )

    proporcion_esperada = (
        5 / len(malla)
    )

    proporcion_observada = resultado.value.loc[
        "saldo_1h",
        "Percent of nulls in sample",
    ]

    assert proporcion_observada == pytest.approx(
        proporcion_esperada
    )


def test_percent_of_nulls_condicion_pasa_sin_nulos():
    """
    La condicion estricta debe pasar cuando no existen nulos.
    """
    check = PercentOfNulls()

    check.add_condition_percent_of_nulls_not_greater_than(
        0
    )

    resultado = check.run(
        armar_dataset(
            malla_pequena()
        )
    )

    assert resultado.passed_conditions()


def test_percent_of_nulls_condicion_falla_con_nulos():
    """
    La misma condicion debe fallar al introducir un NaN.
    """
    malla = malla_pequena()

    malla.loc[
        malla.index[0],
        "saldo_1h",
    ] = np.nan

    check = PercentOfNulls(
        columns=["saldo_1h"]
    )

    check.add_condition_percent_of_nulls_not_greater_than(
        0
    )

    resultado = check.run(
        armar_dataset(malla)
    )

    assert not resultado.passed_conditions()


def test_mixed_nulls_detecta_representaciones_mixtas():
    """
    MixedNulls debe detectar distintas representaciones
    de valores faltantes dentro de una columna.
    """
    malla = malla_pequena()

    malla["columna_nulos_mixtos"] = "valor"

    malla.loc[
        malla.index[0],
        "columna_nulos_mixtos",
    ] = np.nan

    malla.loc[
        malla.index[1],
        "columna_nulos_mixtos",
    ] = "null"

    check = MixedNulls(
        columns=["columna_nulos_mixtos"],
        null_string_list=["null"],
    )

    resultado = check.run(
        armar_dataset(malla)
    )

    assert (
        "columna_nulos_mixtos"
        in resultado.value["columns"]
    )

    representaciones = resultado.value[
        "columns"
    ]["columna_nulos_mixtos"]

    assert len(representaciones) == 2

    assert sum(
        detalle["count"]
        for detalle in representaciones.values()
    ) == 2
