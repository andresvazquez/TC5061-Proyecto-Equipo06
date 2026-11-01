from datetime import timedelta

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici import config
from mibici.features import construir_features_p1
from mibici.modeling.train_p1 import dividir_train_val_test, evaluar_conjuntos

VARIABLES_P1 = [
    "hora_del_dia",
    "dia_de_semana",
    "es_fin_de_semana",
    "mes",
    "salidas_lag_1",
    "salidas_lag_2",
    "salidas_lag_24",
    "salidas_lag_168",
    "salidas_roll_mean_3h",
    "salidas_roll_mean_24h",
    "capacidad",
]


def datos_p1():
    """Construye features de P1 con viajes sintéticos alrededor de los cortes configurados."""
    horas = pd.date_range("2026-02-01", "2026-07-10", freq="h")
    viajes = pd.DataFrame({
        "viaje_id": range(len(horas)),
        "inicio_del_viaje": horas,
        "fin_del_viaje": [hora + timedelta(minutes=10) for hora in horas],
        "origen_id": [1 if i % 2 == 0 else 2 for i in range(len(horas))],
        "destino_id": [2 if i % 2 == 0 else 1 for i in range(len(horas))],
    })
    estaciones = pd.DataFrame({
        "id": [1, 2],
        "dpcapacity": [20, 30],
        "latitud": [20.67, 20.68],
        "longitud": [-103.35, -103.36],
    })

    # Pipeline real de features de P1 y división temporal con los cortes del proyecto
    features = construir_features_p1(viajes, estaciones)
    train, val, test = dividir_train_val_test(features)

    assert not train.empty
    assert not val.empty
    assert not test.empty

    return train, val, test


def entrenar_p1(train):
    """Entrena P1 con los mismos hiperparámetros y semilla de train_p1.py."""
    modelo = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.1,
        random_state=config.SEMILLA,
    )
    modelo.fit(train[VARIABLES_P1], train["salidas"])
    return modelo


def test_mismas_predicciones_con_misma_semilla():
    train, val, test = datos_p1()

    modelo_1 = entrenar_p1(train)
    modelo_2 = entrenar_p1(train)

    assert modelo_1.predict(val[VARIABLES_P1]) == pytest.approx(
        modelo_2.predict(val[VARIABLES_P1])
    )
    assert modelo_1.predict(test[VARIABLES_P1]) == pytest.approx(
        modelo_2.predict(test[VARIABLES_P1])
    )


def test_mismas_metricas_con_misma_semilla():
    train, val, test = datos_p1()

    modelo_1 = entrenar_p1(train)
    modelo_2 = entrenar_p1(train)

    metricas_1 = evaluar_conjuntos(
        y_val_real=val["salidas"],
        y_val_pred=modelo_1.predict(val[VARIABLES_P1]),
        y_test_real=test["salidas"],
        y_test_pred=modelo_1.predict(test[VARIABLES_P1]),
    )
    metricas_2 = evaluar_conjuntos(
        y_val_real=val["salidas"],
        y_val_pred=modelo_2.predict(val[VARIABLES_P1]),
        y_test_real=test["salidas"],
        y_test_pred=modelo_2.predict(test[VARIABLES_P1]),
    )

    assert metricas_1 == pytest.approx(metricas_2)
