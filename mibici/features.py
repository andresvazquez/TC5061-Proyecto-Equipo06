import numpy as np
import pandas as pd

# P2, variables del modelo (15c)
# Trabajan sobre la malla estacion por hora, con las columnas estacion_id,
# hora, salidas, llegadas, capacidad e inventario_estimado. Todas usan solo
# horas anteriores a la fila, para no usar informacion del futuro.

# De la 15b (Juan Pablo) necesito la columna inventario_estimado en la malla,
# con el inventario al final de cada hora.


def _suma_previa(malla, columna, horas):
    # Suma de la columna en las "horas" horas anteriores de la misma estacion,
    # sin contar la fila actual (misma ventana data[i-window:i] de series de tiempo).
    # Si todavia no hay esas horas, queda NaN.
    resultado = pd.Series(np.nan, index=malla.index)

    for estacion in malla["estacion_id"].unique():
        datos = malla[malla["estacion_id"] == estacion].sort_values("hora")
        valores = datos[columna].values

        sumas = [np.nan] * horas
        for i in range(horas, len(valores)):
            sumas.append(sum(valores[i - horas : i]))

        resultado[datos.index] = sumas

    return resultado


def inventario_previo(malla):
    """
    Inventario estimado de la hora anterior.

    Sirve para P2 porque es lo que se sabe del inventario al predecir.
    """
    return _suma_previa(malla, "inventario_estimado", 1)


def razon_inventario(malla):
    """
    Inventario de la hora anterior entre la capacidad de la estacion.

    Sirve para P2 porque una estacion casi vacia o casi llena tiene mas riesgo.
    """
    return inventario_previo(malla) / malla["capacidad"]


def salidas_hora_previa(malla):
    """
    Salidas de la estacion en la hora anterior.

    Sirve para P2 porque muchas salidas seguidas vacian la estacion.
    """
    return _suma_previa(malla, "salidas", 1)


def llegadas_hora_previa(malla):
    """
    Llegadas a la estacion en la hora anterior.

    Sirve para P2 porque muchas llegadas seguidas llenan la estacion.
    """
    return _suma_previa(malla, "llegadas", 1)


def saldo_neto(malla, horas):
    """
    Llegadas menos salidas de las ultimas "horas" horas, sin la hora actual.

    Sirve para P2 porque muestra si la estacion se esta llenando o vaciando.
    Las primeras horas de cada estacion quedan en NaN por falta de historia.
    """
    con_saldo = malla.copy()
    con_saldo["saldo"] = con_saldo["llegadas"] - con_saldo["salidas"]
    return _suma_previa(con_saldo, "saldo", horas)


def features_p2(malla):
    """
    Agrega a la malla las variables de P2 y devuelve una copia.

    Columnas nuevas: inventario_previo, razon_inventario, saldo_1h, saldo_3h,
    salidas_hora_previa y llegadas_hora_previa.
    """
    resultado = malla.copy()
    resultado["inventario_previo"] = inventario_previo(malla)
    resultado["razon_inventario"] = razon_inventario(malla)
    resultado["saldo_1h"] = saldo_neto(malla, 1)
    resultado["saldo_3h"] = saldo_neto(malla, 3)
    resultado["salidas_hora_previa"] = salidas_hora_previa(malla)
    resultado["llegadas_hora_previa"] = llegadas_hora_previa(malla)
    return resultado
