from datetime import timedelta

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import MixedDataTypes
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
            "fin_del_viaje": inicio + timedelta(minutes=15),
            "origen_id": [1, 1, 2, 3] * 15,
            "destino_id": [2, 3, 3, 1] * 15,
        }
    )
    malla = features_p2(etiqueta_riesgo(construir_malla(viajes, estaciones)))
    return malla.dropna().reset_index(drop=True)


# Dataset de DeepChecks, igual que en el quickstart de Data Integrity
def armar_dataset(malla):
    return Dataset(malla, label="riesgo", cat_features=["estacion_id"], datetime_name="hora")


def test_sin_tipos_mixtos():
    ds = armar_dataset(malla_pequena())

    # En la malla limpia ninguna columna mezcla números y texto
    resultado = MixedDataTypes().run(ds)

    assert all(not mezcla for mezcla in resultado.value.values())


def test_detecta_tipos_mixtos():
    malla = malla_pequena()
    total = len(malla)
    n_texto = total // 2

    # Se mezclan a propósito texto y números en la misma columna
    malla["codigo_estacion"] = [
        f"est-{i}" if i < n_texto else i for i in range(total)
    ]

    resultado = MixedDataTypes(columns=["codigo_estacion"]).run(armar_dataset(malla))
    mezcla = resultado.value["codigo_estacion"]

    assert mezcla["strings"] == pytest.approx(n_texto / total)
    assert mezcla["numbers"] == pytest.approx((total - n_texto) / total)
    assert mezcla["strings"] > 0
    assert mezcla["numbers"] > 0


def test_condicion_advierte_contaminacion_pequena():
    malla = malla_pequena()
    total = len(malla)
    n_texto = max(1, round(total * 0.05))

    # Caso límite: ~5% de texto cae en la zona de riesgo (1% a 10%)
    malla["codigo_estacion"] = [
        f"est-{i}" if i < n_texto else i for i in range(total)
    ]

    check = MixedDataTypes(columns=["codigo_estacion"])
    check.add_condition_rare_type_ratio_not_in_range((0.01, 0.1))
    resultado = check.run(armar_dataset(malla))

    assert resultado.value["codigo_estacion"]["strings"] == pytest.approx(n_texto / total)
    assert 0.01 < n_texto / total < 0.1
    assert not resultado.passed_conditions()
