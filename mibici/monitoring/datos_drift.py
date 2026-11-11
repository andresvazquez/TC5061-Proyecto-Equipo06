from functools import lru_cache

from mibici import config
from mibici.features import agregar_features_temporales_p2
from mibici.modeling.train_p2 import (
    VARIABLES_RF_FEATURES,
    cargar_datos,
    partir_por_fecha,
    preparar_malla,
)

COLUMNAS_DRIFT = [*VARIABLES_RF_FEATURES, "riesgo"]
MUESTRA_DRIFT = 100_000


@lru_cache(maxsize=1)
def _malla_base():
    # La malla se comparte entre los conjuntos de referencia y actual
    viajes, estaciones = cargar_datos()
    return preparar_malla(viajes, estaciones)


def _preparar(malla):
    malla = agregar_features_temporales_p2(malla)
    return malla[COLUMNAS_DRIFT]


def _muestrear(malla):
    n = min(MUESTRA_DRIFT, len(malla))
    return malla.sample(n=n, random_state=config.SEMILLA).reset_index(drop=True)


def conjunto_referencia():
    # Datos de entrenamiento: junio de 2023 a marzo de 2026
    entrenamiento, _, _ = partir_por_fecha(_malla_base())
    return _muestrear(_preparar(entrenamiento))
