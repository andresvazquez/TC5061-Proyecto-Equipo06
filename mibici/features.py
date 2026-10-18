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

    Incluye solo las estaciones que aparecen como origen o destino en los
    viajes recibidos. El volumen_diario es el promedio de salidas por dia,
    contando solo los dias con al menos una salida.

    Para P3, el perfil horario distingue las estaciones segun el momento
    del dia en que se usan; la proporcion de fin de semana permite comparar
    patrones entre semana y fin de semana; el volumen diario representa la
    intensidad de uso; el saldo neto indica si hay mas llegadas que salidas
    o viceversa; y la capacidad representa el tamano de la estacion.

    Para evitar fuga de informacion, la funcion debe recibir unicamente
    viajes pertenecientes al periodo que se desea utilizar para clustering.

    Args:
        viajes (pd.DataFrame): Viajes normalizados con las columnas
            viaje_id, inicio_del_viaje, fin_del_viaje, origen_id
            y destino_id.
        estaciones (pd.DataFrame): Nomenclatura de estaciones con las
        columnas id y dpcapacity, o estacion_id y capacidad.

    Returns:
        pd.DataFrame: Perfil agregado con una fila por estacion presenta
            en los viajes recibidos.
    """

    columnas_viajes = {
        "viaje_id",
        "inicio_del_viaje",
        "fin_del_viaje",
        "origen_id",
        "destino_id",
    }

    estaciones = estaciones.rename(
        columns={
            "id": "estacion_id",
            "dpcapacity": "capacidad",
        }
    )

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


# P1 y base, malla estacion por hora (15a, Christian)
def construir_malla(viajes, estaciones):
    """
    Arma la malla de una fila por estacion y hora (todas las horas, con ceros si no hubo viajes).
    Recibe viajes y estaciones, devuelve estacion_id, hora (fecha y hora), salidas, llegadas y capacidad.
    """
    raise NotImplementedError("Pendiente por trabajar")


# P2, etiqueta de riesgo (15b, Juan Pablo)
def etiqueta_riesgo(malla):
    """
    Agrega a la malla el inventario estimado y el riesgo.

    Recibe la malla y devuelve una copia con:
    - inventario_estimado: bicicletas estimadas al final de cada hora.
    - riesgo: 1 si la estacion estara casi vacia o casi llena
      al finalizar la siguiente hora; 0 en caso contrario.

    Supuestos:
    - Inventario inicial: 50% de la capacidad.
    - Umbral de riesgo: ocupacion <= 10% o >= 90%.
    - Horizonte de prediccion: 1 hora.
    """

    resultado = malla.copy()

    # Validar columnas necesarias
    columnas = {
        "estacion_id", "hora", "salidas",
        "llegadas", "capacidad"
    }

    faltantes = columnas - set(resultado.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    # Ordenar por estacion y hora
    resultado["hora"] = pd.to_datetime(resultado["hora"])
    resultado = resultado.sort_values(
        ["estacion_id", "hora"]
    )

    # Validar informacion
    if (
        resultado["capacidad"].isna().any()
        or (resultado["capacidad"] <= 0).any()
    ):
        raise ValueError("Hay capacidades faltantes o invalidas")

    if resultado[["salidas", "llegadas"]].isna().any().any():
        raise ValueError("Hay movimientos faltantes")

    if resultado.duplicated(["estacion_id", "hora"]).any():
        raise ValueError("Hay estaciones-hora duplicadas")

    # Estimar inventario al final de cada hora
    inventarios = pd.Series(index=resultado.index, dtype=float)

    for _, grupo in resultado.groupby("estacion_id"):

        inventario = grupo["capacidad"].iloc[0] * 0.50

        for indice, fila in grupo.iterrows():

            inventario += fila["llegadas"] - fila["salidas"]

            inventario = np.clip(
                inventario, 0, fila["capacidad"]
            )

            inventarios.loc[indice] = inventario

    resultado["inventario_estimado"] = inventarios

    # Calcular ocupacion estimada
    ocupacion = (
        resultado["inventario_estimado"]
        / resultado["capacidad"]
    )

    # Identificar estados criticos
    resultado["critico"] = (
        (ocupacion <= 0.10)
        | (ocupacion >= 0.90)
    ).astype(int)

    # Target: riesgo en la siguiente hora
    resultado["riesgo"] = (
        resultado.groupby("estacion_id")["critico"]
        .shift(-1)
        .astype("Int64")
    )

    # Verificar que la siguiente fila sea exactamente una hora despues
    siguiente_hora = (
        resultado.groupby("estacion_id")["hora"].shift(-1)
    )

    diferencia = siguiente_hora - resultado["hora"]

    resultado.loc[
        diferencia != pd.Timedelta(hours=1), "riesgo"
    ] = pd.NA

    # Eliminar columna auxiliar
    resultado = resultado.drop(columns=["critico"])

    return resultado