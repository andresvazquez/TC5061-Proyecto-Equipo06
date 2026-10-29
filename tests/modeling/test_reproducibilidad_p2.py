import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from mibici.features import agregar_features_temporales_p2
from mibici.modeling.train_p2 import VARIABLES_RF_FEATURES, parametros_bosque, preparar_malla


# Malla chica de P2: 3 estaciones, 60 viajes y las dos clases de riesgo
def malla_pequena():
    # Tres estaciones con capacidades distintas y coordenadas cercanas de Guadalajara
    estaciones = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3],
            "capacidad": [10, 20, 30],
            "latitud": [20.67, 20.68, 20.69],
            "longitud": [-103.35, -103.36, -103.37],
        }
    )
    # 60 viajes de 15 min, uno cada 25 min, con un patrón de 4 rutas repetido 15 veces
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
    return agregar_features_temporales_p2(preparar_malla(viajes, estaciones))


# Entrena el bosque de P2 con los parámetros de train_p2 y regresa el modelo y sus variables
def entrenar_bosque(malla):
    X = malla[VARIABLES_RF_FEATURES]
    y = malla["riesgo"]
    # 50 árboles y profundidad 10 son los valores por defecto de train_p2
    modelo = RandomForestClassifier(**parametros_bosque(50, 10))
    modelo.fit(X, y)
    return modelo, X


def test_mismas_predicciones_con_misma_semilla():
    malla = malla_pequena()

    # Dos entrenamientos con la misma semilla
    modelo_1, X = entrenar_bosque(malla)
    modelo_2, _ = entrenar_bosque(malla)

    assert modelo_1.predict(X) == pytest.approx(modelo_2.predict(X))


def test_mismas_probabilidades_con_misma_semilla():
    malla = malla_pequena()

    # Dos entrenamientos con la misma semilla
    modelo_1, X = entrenar_bosque(malla)
    modelo_2, _ = entrenar_bosque(malla)

    assert modelo_1.predict_proba(X) == pytest.approx(modelo_2.predict_proba(X))
