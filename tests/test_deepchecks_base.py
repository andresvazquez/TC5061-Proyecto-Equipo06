from deepchecks.core import SuiteResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
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


def test_dataset_deepchecks():
    malla = malla_pequena()

    # Dataset con la etiqueta, la categórica y la fecha, como en el quickstart
    ds = Dataset(malla, label="riesgo", cat_features=["estacion_id"], datetime_name="hora")

    assert ds.label_name == "riesgo"
    assert ds.cat_features == ["estacion_id"]
    assert ds.datetime_name == "hora"
    assert len(ds.data) == len(malla)


def test_dataset_etiqueta_inexistente():
    malla = malla_pequena()

    # Caso límite: una etiqueta que no está en la tabla debe fallar
    with pytest.raises(DeepchecksValueError):
        Dataset(malla, label="no_existe", cat_features=["estacion_id"], datetime_name="hora")


def test_suite_data_integrity():
    malla = malla_pequena()
    ds = Dataset(malla, label="riesgo", cat_features=["estacion_id"], datetime_name="hora")

    # La suite de integridad corre y regresa un resultado con sus checks
    resultado = data_integrity().run(ds)

    assert isinstance(resultado, SuiteResult)
    assert len(resultado.results) > 0
