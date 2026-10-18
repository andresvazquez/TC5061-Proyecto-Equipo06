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


# P3, perfil de uso por estacion


def perfil_uso_estacion(viajes, estaciones):
    """
    Construye el perfil de uso de cada estacion para P3.

    Resume los viajes en una fila por estacion utilizando:
    - Perfil horario normalizado de salidas.
    - Proporcion de salidas realizadas en fin de semana.
    - Volumen diario promedio de salidas.
    - Saldo neto entre llegadas y salidas.
    - Capacidad de la estacion.

    Para evitar fuga de informacion, la funcion debe recibir unicamente
    viajes pertenecientes al periodo que se desea utilizar para clustering.

    Args:
        viajes (pd.DataFrame): Viajes normalizados con las columnas
            viaje_id, inicio_del_viaje, fin_del_viaje, origen_id
            y destino_id.
        estaciones (pd.DataFrame): Nomenclatura normalizada con las columnas
            estacion_id y capacidad.

    Returns:
        pd.DataFrame: Perfil agregado con una fila por estacion.
    """

    columnas_viajes = {
        "viaje_id",
        "inicio_del_viaje",
        "fin_del_viaje",
        "origen_id",
        "destino_id",
    }

    columnas_estaciones = {
        "estacion_id",
        "capacidad",
    }

    faltantes_viajes = columnas_viajes - set(viajes.columns)
    faltantes_estaciones = columnas_estaciones - set(estaciones.columns)

    if faltantes_viajes:
        raise ValueError(
            "Faltan columnas requeridas en viajes: "
            + ", ".join(sorted(faltantes_viajes))
        )

    if faltantes_estaciones:
        raise ValueError(
            "Faltan columnas requeridas en estaciones: "
            + ", ".join(sorted(faltantes_estaciones))
        )

    viajes = viajes.copy()

    # Variables temporales necesarias para construir el perfil
    viajes["hora"] = viajes["inicio_del_viaje"].dt.hour
    viajes["fecha"] = viajes["inicio_del_viaje"].dt.normalize()
    viajes["fin_semana"] = (
        viajes["inicio_del_viaje"].dt.dayofweek >= 5
    )

    # Salidas por estacion
    salidas = (
        viajes.groupby("origen_id")
        .size()
        .rename("salidas")
    )

    # Llegadas por estacion
    llegadas = (
        viajes.groupby("destino_id")
        .size()
        .rename("llegadas")
    )

    # Volumen diario promedio de salidas
    salidas_diarias = (
        viajes.groupby(["origen_id", "fecha"])
        .size()
        .groupby("origen_id")
        .mean()
        .rename("volumen_diario")
    )

    # Proporcion de salidas realizadas en fin de semana
    proporcion_fin_semana = (
        viajes.groupby("origen_id")["fin_semana"]
        .mean()
        .rename("proporcion_fin_semana")
    )

    # Perfil horario de salidas
    perfil_horario = (
        viajes.groupby(["origen_id", "hora"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )

    # Normalizar cada perfil horario por el total de salidas de la estacion
    totales_horarios = perfil_horario.sum(axis=1)

    perfil_horario = perfil_horario.div(
        totales_horarios.replace(0, np.nan),
        axis=0,
    ).fillna(0)

    perfil_horario.columns = [
        f"perfil_h{hora:02d}"
        for hora in perfil_horario.columns
    ]

    # Unir las variables agregadas
    perfil = pd.concat(
        [
            salidas,
            llegadas,
            salidas_diarias,
            proporcion_fin_semana,
            perfil_horario,
        ],
        axis=1,
    ).fillna(0)

    # Saldo positivo indica mas llegadas que salidas
    perfil["saldo_neto"] = (
        perfil["llegadas"] - perfil["salidas"]
    )

    perfil.index.name = "estacion_id"
    perfil = perfil.reset_index()

    # Incorporar capacidad desde la nomenclatura
    perfil = perfil.merge(
        estaciones[["estacion_id", "capacidad"]],
        on="estacion_id",
        how="left",
        validate="one_to_one",
    )

    return perfil
