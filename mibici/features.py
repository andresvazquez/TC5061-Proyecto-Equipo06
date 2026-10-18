import numpy as np
import pandas as pd
from pathlib import Path

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
def construir_malla(
    viajes: pd.DataFrame, estaciones: str | Path | pd.DataFrame
) -> pd.DataFrame:
    """Arma la malla espacio-temporal (estación x hora) con salidas, llegadas y metadatos."""
    if isinstance(estaciones, (str, Path)):
        df_estaciones = pd.read_csv(estaciones)
    else:
        df_estaciones = estaciones.copy()

    # Normalizar id de estación
    if "origen_id" in df_estaciones.columns:
        df_estaciones = df_estaciones.rename(columns={"origen_id": "estacion_id"})
    elif "id" in df_estaciones.columns:
        df_estaciones = df_estaciones.rename(
            columns={"id": "estacion_id", "dpcapacity": "capacidad"}
        )

    estaciones_ids = df_estaciones["estacion_id"].unique()

    # Normalizar viajes
    df_viajes = viajes.copy()
    col_origen = "origen_id" if "origen_id" in df_viajes.columns else "origen_estacion"
    col_destino = "destino_id" if "destino_id" in df_viajes.columns else "destino_estacion"

    df_viajes["inicio_del_viaje"] = pd.to_datetime(df_viajes["inicio_del_viaje"])
    df_viajes["fin_del_viaje"] = pd.to_datetime(df_viajes["fin_del_viaje"])

    min_fecha = df_viajes["inicio_del_viaje"].min().floor("h")
    max_fecha = df_viajes["inicio_del_viaje"].max().ceil("h")
    rango_horas = pd.date_range(start=min_fecha, end=max_fecha, freq="h")

    # Producto cartesiano
    malla_index = pd.MultiIndex.from_product(
        [estaciones_ids, rango_horas], names=["estacion_id", "hora"]
    )
    malla = pd.DataFrame(index=malla_index).reset_index()

    # Conteo de salidas y llegadas
    df_viajes["hora_inicio"] = df_viajes["inicio_del_viaje"].dt.floor("h")
    salidas = (
        df_viajes.groupby([col_origen, "hora_inicio"])
        .size()
        .reset_index(name="salidas")
    )

    df_viajes["hora_fin"] = df_viajes["fin_del_viaje"].dt.floor("h")
    llegadas = (
        df_viajes.groupby([col_destino, "hora_fin"])
        .size()
        .reset_index(name="llegadas")
    )

    malla = malla.merge(
        salidas,
        left_on=["estacion_id", "hora"],
        right_on=[col_origen, "hora_inicio"],
        how="left",
    ).drop(columns=[col_origen, "hora_inicio"], errors="ignore")

    malla = malla.merge(
        llegadas,
        left_on=["estacion_id", "hora"],
        right_on=[col_destino, "hora_fin"],
        how="left",
    ).drop(columns=[col_destino, "hora_fin"], errors="ignore")

    malla["salidas"] = malla["salidas"].fillna(0).astype(int)
    malla["llegadas"] = malla["llegadas"].fillna(0).astype(int)

    # Integración de capacidad y coordenadas
    cols_estaticas = ["estacion_id", "capacidad", "latitud", "longitud"]
    cols_existentes = [col for col in cols_estaticas if col in df_estaciones.columns]

    malla = malla.merge(
        df_estaciones[cols_existentes], on="estacion_id", how="left"
    )

    return malla


def agregar_variables_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae variables de fecha y hora a partir de la columna 'hora'."""
    df = df.copy()
    df["hora_del_dia"] = df["hora"].dt.hour
    df["dia_de_semana"] = df["hora"].dt.dayofweek
    df["es_fin_de_semana"] = df["dia_de_semana"].isin([5, 6]).astype(int)
    df["mes"] = df["hora"].dt.month
    return df


def agregar_rezagos_y_moviles(
    df: pd.DataFrame, lags: list[int] = [1, 2, 24, 168]
) -> pd.DataFrame:
    """Agrega rezagos (lags) y promedios móviles agrupando por estación."""
    df = df.copy()
    df = df.sort_values(by=["estacion_id", "hora"]).reset_index(drop=True)

    # Rezagos horarios (1h atras, 2h atras, 24h atras / mismo dia, 168h atras / misma hora semana pasada)
    for lag in lags:
        df[f"salidas_lag_{lag}"] = df.groupby("estacion_id")["salidas"].shift(lag)

    # Promedio móvil de las últimas 3 y 24 horas
    df["salidas_roll_mean_3h"] = df.groupby("estacion_id")["salidas"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["salidas_roll_mean_24h"] = df.groupby("estacion_id")["salidas"].transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=1).mean()
    )

    return df


def construir_features_p1(
    df_viajes: pd.DataFrame,
    estaciones: str | Path | pd.DataFrame,
    eliminar_nans: bool = True,
) -> pd.DataFrame:
    """Pipeline orquestador para generar las características del problema P1."""
    malla = construir_malla(df_viajes, estaciones)
    malla = agregar_variables_calendario(malla)
    dataset_features = agregar_rezagos_y_moviles(malla)

    if eliminar_nans:
        dataset_features = dataset_features.dropna().reset_index(drop=True)

    return dataset_features


# P2, etiqueta de riesgo (15b, Juan Pablo)
def etiqueta_riesgo(malla):
    """
    Agrega a la malla el inventario estimado y el riesgo.
    Recibe la malla, devuelve una copia con inventario_estimado (al final de cada hora) y riesgo (0 o 1).
    """
    raise NotImplementedError("Pendiente por trabajar")
