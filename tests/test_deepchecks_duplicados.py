from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates
import pandas as pd
import pytest

from mibici.features import construir_malla, etiqueta_riesgo, features_p2


# Malla chica de P2: 3 estaciones, 60 viajes y las dos clases de riesgo
def malla_pequena():
    estaciones = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3],
            "capacidad": [10, 20, 30],
            "latitud": [20.67, 20.68, 20.69],
            "longitud": [-103.35, -103.36, -103.37],
        }
    )
    inicio = pd.date_range("2026-07-01 00:10", periods=60, freq="25min")
    viajes = pd.DataFrame(
        {
            "viaje_id": range(60),
            "inicio_del_viaje": inicio,
            "fin_del_viaje": inicio + pd.Timedelta(minutes=15),
            "origen_id": [1, 1, 2, 3] * 15,
            "destino_id": [2, 3, 3, 1] * 15,
        }
    )
    malla = features_p2(etiqueta_riesgo(construir_malla(viajes, estaciones)))
    return malla.dropna()


# Dataset de DeepChecks, igual que en el quickstart de Data Integrity
def armar_dataset(malla):
    return Dataset(malla, cat_features=["estacion_id"], datetime_name="hora", label="riesgo")


def test_sin_duplicados():
    ds = armar_dataset(malla_pequena())

    # En la malla limpia no hay filas repetidas
    resultado = DataDuplicates().run(ds)

    assert resultado.value == 0


def test_con_duplicados():
    malla = malla_pequena()

    # Se repiten 10 filas a propósito
    sucia = pd.concat([malla, malla.head(10)], ignore_index=True)
    resultado = DataDuplicates().run(armar_dataset(sucia))

    assert resultado.value > 0
    assert resultado.value == pytest.approx(10 / len(sucia))


def test_condicion_pasa_sin_duplicados():
    ds = armar_dataset(malla_pequena())

    # Condición límite: ningún duplicado permitido
    check = DataDuplicates()
    check.add_condition_ratio_less_or_equal(0)
    resultado = check.run(ds)

    assert resultado.passed_conditions()


def test_condicion_falla_con_duplicados():
    malla = malla_pequena()
    sucia = pd.concat([malla, malla.head(10)], ignore_index=True)

    # La misma condición debe fallar cuando hay duplicados
    check = DataDuplicates()
    check.add_condition_ratio_less_or_equal(0)
    resultado = check.run(armar_dataset(sucia))

    assert not resultado.passed_conditions()
