import numpy as np
import pandas as pd
import pytest

from mibici.features import (
    agregar_features_temporales_p2,
    agregar_rezagos_y_moviles,
    agregar_variables_calendario,
    construir_features_p1,
    construir_malla,
    etiqueta_riesgo,
    features_p2,
    inventario_previo,
    llegadas_hora_previa,
    perfil_uso_estacion,
    razon_inventario,
    saldo_neto,
    salidas_hora_previa,
)


def _crear_estaciones():
    """Nomenclatura pequena con el formato original (id, dpcapacity, name)."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "dpcapacity": [10, 20, 30],
        "name": ["A", "B", "C"],
        "latitud": [20.1, 20.2, 20.3],
        "longitud": [-103.1, -103.2, -103.3],
    })


def _crear_viajes():
    """Tres viajes entre las estaciones 1 y 2; la estacion 3 no tiene viajes."""
    return pd.DataFrame({
        "viaje_id": [1, 2, 3],
        "inicio_del_viaje": [
            "2026-01-05 08:10:00",
            "2026-01-05 08:50:00",
            "2026-01-05 09:20:00",
        ],
        "fin_del_viaje": [
            "2026-01-05 08:40:00",
            "2026-01-05 09:05:00",
            "2026-01-05 09:30:00",
        ],
        "origen_id": [1, 1, 2],
        "destino_id": [2, 2, 1],
    })


def _crear_viajes_largos():
    """Un viaje por hora durante 200 horas, de la estacion 1 a la 2."""
    inicio = pd.date_range("2026-01-01 00:10", periods=200, freq="h")
    return pd.DataFrame({
        "viaje_id": range(200),
        "inicio_del_viaje": inicio,
        "fin_del_viaje": inicio + pd.Timedelta(minutes=30),
        "origen_id": 1,
        "destino_id": 2,
    })


def _crear_viajes_perfil():
    """
    Seis viajes con fechas conocidas (el 5 de enero de 2026 es lunes).

    Estacion 1 (origen de 4 viajes): salidas el lunes 08:00 y 08:30, el martes
    17:00 y el sabado 17:30.
    Estacion 2 (origen de 2 viajes): salidas el lunes 09:00 y el domingo 09:30.
    """
    inicio = pd.to_datetime([
        "2026-01-05 08:00",  # lunes, sale de la 1
        "2026-01-05 08:30",  # lunes, sale de la 1
        "2026-01-06 17:00",  # martes, sale de la 1
        "2026-01-10 17:30",  # sabado, sale de la 1
        "2026-01-05 09:00",  # lunes, sale de la 2
        "2026-01-11 09:30",  # domingo, sale de la 2
    ])
    return pd.DataFrame({
        "viaje_id": [1, 2, 3, 4, 5, 6],
        "inicio_del_viaje": inicio,
        "fin_del_viaje": inicio + pd.Timedelta(minutes=15),
        "origen_id": [1, 1, 1, 1, 2, 2],
        "destino_id": [2, 2, 2, 2, 1, 1],
    })


def _crear_malla_riesgo(
    salidas,
    llegadas,
    capacidad=10,
    inicio="2026-01-05 00:00",
    estacion=1,
):
    """Malla de una estacion con horas consecutivas, lista para etiqueta_riesgo."""
    return pd.DataFrame({
        "estacion_id": estacion,
        "hora": pd.date_range(inicio, periods=len(salidas), freq="h"),
        "salidas": salidas,
        "llegadas": llegadas,
        "capacidad": capacidad,
    })


def _crear_malla_features():
    """
    Dos estaciones con 4 horas cada una y el inventario ya calculado.

    Estacion 1: saldo (llegadas - salidas) = [-1, -1, -3, -2].
    """
    horas = pd.date_range("2026-01-05 00:00", periods=4, freq="h")
    estacion_1 = pd.DataFrame({
        "estacion_id": 1,
        "hora": horas,
        "salidas": [1, 2, 3, 4],
        "llegadas": [0, 1, 0, 2],
        "capacidad": 10,
        "inventario_estimado": [5.0, 4.0, 1.0, 2.0],
    })
    estacion_2 = estacion_1.assign(
        estacion_id=2,
        capacidad=20,
        inventario_estimado=[10.0, 10.0, 12.0, 14.0],
    )
    return pd.concat([estacion_1, estacion_2], ignore_index=True)


def _crear_salidas_por_estacion():
    """Dos estaciones con 5 horas de salidas, entregadas desordenadas a proposito."""
    horas = pd.date_range("2026-01-05 00:00", periods=5, freq="h")
    estacion_1 = pd.DataFrame({
        "estacion_id": 1,
        "hora": horas,
        "salidas": [1, 2, 3, 4, 5],
    })
    estacion_2 = estacion_1.assign(
        estacion_id=2,
        salidas=[10, 20, 30, 40, 50],
    )
    return pd.concat([estacion_2, estacion_1], ignore_index=True)


# 1. Pruebas para construir_malla
def test_construir_malla_caso_normal_tamano_y_columnas():
    """Caso Normal: La malla tiene una fila por estacion y por hora, con sus columnas."""
    malla = construir_malla(_crear_viajes(), _crear_estaciones())

    # 3 estaciones x 3 horas (08:00, 09:00, 10:00)
    assert len(malla) == 9
    assert {"estacion_id", "hora", "salidas", "llegadas", "capacidad"} <= set(malla.columns)
    assert {"latitud", "longitud"} <= set(malla.columns)
    assert not malla.duplicated(["estacion_id", "hora"]).any()
    assert malla["hora"].min() == pd.Timestamp("2026-01-05 08:00")
    assert malla["hora"].max() == pd.Timestamp("2026-01-05 10:00")


def test_construir_malla_caso_normal_conteos():
    """Caso Normal: Las salidas y llegadas se cuentan en la estacion y hora correctas."""
    malla = construir_malla(_crear_viajes(), _crear_estaciones())
    malla = malla.set_index(["estacion_id", "hora"])

    h08 = pd.Timestamp("2026-01-05 08:00")
    h09 = pd.Timestamp("2026-01-05 09:00")

    assert malla.loc[(1, h08), "salidas"] == 2
    assert malla.loc[(2, h09), "salidas"] == 1
    # Los viajes 1 (fin 08:40) y 2 (fin 09:05) llegan a la estacion 2
    assert malla.loc[(2, h08), "llegadas"] == 1
    assert malla.loc[(2, h09), "llegadas"] == 1
    assert malla.loc[(1, h09), "llegadas"] == 1


def test_construir_malla_caso_normal_conserva_totales():
    """Caso Normal: La suma de salidas y de llegadas es igual al numero de viajes."""
    viajes = _crear_viajes()

    malla = construir_malla(viajes, _crear_estaciones())

    assert malla["salidas"].sum() == len(viajes)
    assert malla["llegadas"].sum() == len(viajes)


def test_construir_malla_caso_normal_tipos_sin_nulos():
    """Caso Normal: Salidas y llegadas son enteros y no hay valores faltantes."""
    malla = construir_malla(_crear_viajes(), _crear_estaciones())

    assert pd.api.types.is_integer_dtype(malla["salidas"])
    assert pd.api.types.is_integer_dtype(malla["llegadas"])
    assert malla[["salidas", "llegadas", "capacidad"]].notna().all().all()


def test_construir_malla_caso_normal_columnas_alternativas():
    """Caso Normal: Acepta origen_estacion/destino_estacion y origen_id en estaciones."""
    viajes = _crear_viajes().rename(columns={
        "origen_id": "origen_estacion",
        "destino_id": "destino_estacion",
    })
    estaciones = _crear_estaciones().rename(columns={"id": "origen_id"})

    malla = construir_malla(viajes, estaciones)

    assert malla["salidas"].sum() == 3
    assert set(malla["estacion_id"]) == {1, 2, 3}


def test_construir_malla_caso_normal_estaciones_desde_csv(tmp_path):
    """Caso Normal: Leer las estaciones desde un CSV da lo mismo que pasar el DataFrame."""
    ruta_csv = tmp_path / "estaciones.csv"
    _crear_estaciones().to_csv(ruta_csv, index=False)

    desde_csv = construir_malla(_crear_viajes(), ruta_csv)
    desde_df = construir_malla(_crear_viajes(), _crear_estaciones())

    pd.testing.assert_frame_equal(desde_csv, desde_df)


def test_construir_malla_caso_limite_estacion_sin_viajes():
    """Caso Límite: Una estacion sin viajes aparece en la malla con ceros."""
    malla = construir_malla(_crear_viajes(), _crear_estaciones())

    sin_viajes = malla[malla["estacion_id"] == 3]

    assert len(sin_viajes) == 3
    assert (sin_viajes["salidas"] == 0).all()
    assert (sin_viajes["llegadas"] == 0).all()
    assert (sin_viajes["capacidad"] == 30).all()


def test_construir_malla_caso_limite_un_solo_viaje():
    """Caso Límite: Un solo viaje que inicia en punto produce una sola hora."""
    viajes = _crear_viajes().iloc[[0]].copy()
    viajes["inicio_del_viaje"] = "2026-01-05 08:00:00"

    malla = construir_malla(viajes, _crear_estaciones())

    assert malla["hora"].nunique() == 1
    assert len(malla) == 3
    assert malla["salidas"].sum() == 1


def test_construir_malla_caso_limite_no_modifica_entradas():
    """Caso Límite: La funcion no modifica los DataFrames que recibe."""
    viajes = _crear_viajes()
    estaciones = _crear_estaciones()
    viajes_antes = viajes.copy()
    estaciones_antes = estaciones.copy()

    construir_malla(viajes, estaciones)

    pd.testing.assert_frame_equal(viajes, viajes_antes)
    pd.testing.assert_frame_equal(estaciones, estaciones_antes)


# 2. Pruebas para etiqueta_riesgo
def test_etiqueta_riesgo_caso_normal_inventario_y_etiqueta():
    """Caso Normal: Calcula el inventario hora por hora y etiqueta la siguiente hora."""
    # Capacidad 10, inventario inicial 5.
    # h0: 5-4 = 1 (10%)  h1: 1 (10%)  h2: 1+8 = 9 (90%)  h3: 9-5 = 4 (40%)
    malla = _crear_malla_riesgo(salidas=[4, 0, 0, 5], llegadas=[0, 0, 8, 0])

    resultado = etiqueta_riesgo(malla)

    assert resultado["inventario_estimado"].tolist() == [1.0, 1.0, 9.0, 4.0]
    # riesgo(t) = critico(t+1); la ultima hora no tiene siguiente
    assert resultado["riesgo"].iloc[:3].tolist() == [1, 1, 0]
    assert pd.isna(resultado["riesgo"].iloc[3])


def test_etiqueta_riesgo_caso_normal_devuelve_copia_con_columnas_nuevas():
    """Caso Normal: Devuelve una copia con inventario_estimado y riesgo, sin columnas auxiliares."""
    malla = _crear_malla_riesgo([1, 1], [0, 0])
    columnas_antes = list(malla.columns)

    resultado = etiqueta_riesgo(malla)

    assert list(malla.columns) == columnas_antes
    assert {"inventario_estimado", "riesgo"} <= set(resultado.columns)
    assert "critico" not in resultado.columns
    assert "fecha" not in resultado.columns
    assert len(resultado) == len(malla)


def test_etiqueta_riesgo_caso_normal_estaciones_independientes_y_ordenadas():
    """Caso Normal: Cada estacion lleva su propio inventario y el resultado sale ordenado."""
    estacion_1 = _crear_malla_riesgo([4, 0], [0, 0], estacion=1)
    estacion_2 = _crear_malla_riesgo([0, 0], [0, 0], estacion=2, capacidad=20)
    malla = pd.concat([estacion_2, estacion_1], ignore_index=True)
    malla = malla.sample(frac=1, random_state=0)

    resultado = etiqueta_riesgo(malla)

    assert resultado["estacion_id"].tolist() == [1, 1, 2, 2]
    assert resultado["inventario_estimado"].tolist() == [1.0, 1.0, 10.0, 10.0]
    assert resultado["riesgo"].iloc[0] == 1
    assert resultado["riesgo"].iloc[2] == 0


def test_etiqueta_riesgo_caso_limite_umbrales_exactos():
    """Caso Límite: Una ocupacion exacta de 10% y de 90% cuenta como riesgo."""
    malla = _crear_malla_riesgo(salidas=[4, 0, 0, 0], llegadas=[0, 0, 8, 0])

    resultado = etiqueta_riesgo(malla)

    # Inventarios: 1, 1, 9, 9 (exactamente 10% y 90%)
    assert resultado["inventario_estimado"].tolist() == [1.0, 1.0, 9.0, 9.0]
    assert resultado["riesgo"].iloc[:3].tolist() == [1, 1, 1]


def test_etiqueta_riesgo_caso_limite_justo_dentro_del_rango():
    """Caso Límite: Una ocupacion de 11% y de 89% no cuenta como riesgo."""
    malla = _crear_malla_riesgo(
        salidas=[39, 0, 0],
        llegadas=[0, 0, 78],
        capacidad=100,
    )

    resultado = etiqueta_riesgo(malla)

    assert resultado["inventario_estimado"].tolist() == [11.0, 11.0, 89.0]
    assert resultado["riesgo"].iloc[:2].tolist() == [0, 0]


def test_etiqueta_riesgo_caso_limite_recorta_inventario():
    """Caso Límite: El inventario nunca baja de cero ni sube de la capacidad."""
    malla = _crear_malla_riesgo(salidas=[50, 0, 0], llegadas=[0, 0, 100])

    resultado = etiqueta_riesgo(malla)

    assert resultado["inventario_estimado"].tolist() == [0.0, 0.0, 10.0]
    assert resultado["inventario_estimado"].between(0, 10).all()


def test_etiqueta_riesgo_caso_limite_cruce_de_dia():
    """Caso Límite: No se etiqueta la transicion de 23:00 a 00:00 del dia siguiente."""
    malla = pd.concat([
        _crear_malla_riesgo([0, 0], [0, 0], inicio="2026-01-05 22:00"),
        _crear_malla_riesgo([0, 0], [0, 0], inicio="2026-01-06 00:00"),
    ], ignore_index=True)

    resultado = etiqueta_riesgo(malla)

    assert resultado["riesgo"].notna().tolist() == [True, False, True, False]


def test_etiqueta_riesgo_caso_limite_reinicia_inventario_cada_dia():
    """Caso Límite: El inventario vuelve al 50% de la capacidad al empezar un dia nuevo."""
    malla = pd.concat([
        _crear_malla_riesgo([10], [0], inicio="2026-01-05 23:00"),
        _crear_malla_riesgo([0], [0], inicio="2026-01-06 00:00"),
    ], ignore_index=True)

    resultado = etiqueta_riesgo(malla)

    assert resultado["inventario_estimado"].tolist() == [0.0, 5.0]


def test_etiqueta_riesgo_caso_limite_horas_faltantes():
    """Caso Límite: Si falta una hora, la hora anterior no se etiqueta."""
    malla = _crear_malla_riesgo([0, 0, 0], [0, 0, 0])
    malla = malla.drop(index=1).reset_index(drop=True)

    resultado = etiqueta_riesgo(malla)

    assert resultado["riesgo"].isna().all()


def test_etiqueta_riesgo_caso_limite_columnas_faltantes():
    """Caso Límite: Una malla sin la columna capacidad produce un ValueError."""
    malla = _crear_malla_riesgo([1], [1]).drop(columns=["capacidad"])

    with pytest.raises(ValueError, match="Faltan columnas"):
        etiqueta_riesgo(malla)


def test_etiqueta_riesgo_caso_limite_valores_faltantes():
    """Caso Límite: Una malla con valores nulos produce un ValueError."""
    malla = _crear_malla_riesgo([1, 1], [0, 0]).astype({"salidas": float})
    malla.loc[0, "salidas"] = np.nan

    with pytest.raises(ValueError, match="valores faltantes"):
        etiqueta_riesgo(malla)


@pytest.mark.parametrize("capacidad", [0, -5])
def test_etiqueta_riesgo_caso_limite_capacidad_invalida(capacidad):
    """Caso Límite: Una capacidad de cero o negativa produce un ValueError."""
    malla = _crear_malla_riesgo([1], [1], capacidad=capacidad)

    with pytest.raises(ValueError, match="capacidades invalidas"):
        etiqueta_riesgo(malla)


def test_etiqueta_riesgo_caso_limite_movimientos_negativos():
    """Caso Límite: Salidas o llegadas negativas producen un ValueError."""
    malla = _crear_malla_riesgo([-1, 0], [0, 0])

    with pytest.raises(ValueError, match="negativos"):
        etiqueta_riesgo(malla)


def test_etiqueta_riesgo_caso_limite_estacion_hora_duplicada():
    """Caso Límite: Una estacion con la misma hora repetida produce un ValueError."""
    malla = _crear_malla_riesgo([1, 1], [0, 0])
    malla = pd.concat([malla, malla.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicadas"):
        etiqueta_riesgo(malla)


# 3. Pruebas para las variables de P2 (inventario, razon, salidas, llegadas y saldo)
def test_inventario_previo_caso_normal():
    """Caso Normal: El inventario previo es el de la hora anterior de la misma estacion."""
    resultado = inventario_previo(_crear_malla_features())

    np.testing.assert_allclose(resultado.iloc[:4], [np.nan, 5.0, 4.0, 1.0])
    np.testing.assert_allclose(resultado.iloc[4:], [np.nan, 10.0, 10.0, 12.0])


def test_razon_inventario_caso_normal():
    """Caso Normal: La razon es el inventario previo entre la capacidad de la estacion."""
    resultado = razon_inventario(_crear_malla_features())

    np.testing.assert_allclose(resultado.iloc[:4], [np.nan, 0.5, 0.4, 0.1])
    np.testing.assert_allclose(resultado.iloc[4:], [np.nan, 0.5, 0.5, 0.6])


def test_salidas_y_llegadas_hora_previa_caso_normal():
    """Caso Normal: Devuelven las salidas y llegadas de la hora anterior."""
    malla = _crear_malla_features()

    salidas = salidas_hora_previa(malla)
    llegadas = llegadas_hora_previa(malla)

    np.testing.assert_allclose(salidas.iloc[:4], [np.nan, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(llegadas.iloc[:4], [np.nan, 0.0, 1.0, 0.0])


def test_saldo_neto_caso_normal_ventanas_de_1_y_3_horas():
    """Caso Normal: El saldo suma llegadas menos salidas de las horas previas."""
    malla = _crear_malla_features()

    saldo_1h = saldo_neto(malla, 1)
    saldo_3h = saldo_neto(malla, 3)

    # saldo por hora = [-1, -1, -3, -2]
    np.testing.assert_allclose(saldo_1h.iloc[:4], [np.nan, -1.0, -1.0, -3.0])
    np.testing.assert_allclose(saldo_3h.iloc[:4], [np.nan, np.nan, np.nan, -5.0])


def test_saldo_neto_caso_limite_no_modifica_la_malla():
    """Caso Límite: Calcular el saldo no agrega columnas a la malla original."""
    malla = _crear_malla_features()
    columnas_antes = list(malla.columns)

    saldo_neto(malla, 3)

    assert list(malla.columns) == columnas_antes


def test_inventario_previo_caso_limite_una_sola_hora_por_estacion():
    """Caso Límite: Con una sola hora por estacion no hay hora previa y todo es NaN."""
    malla = _crear_malla_features().groupby("estacion_id").head(1)

    resultado = inventario_previo(malla)

    assert resultado.isna().all()


# 4. Pruebas para features_p2
def test_features_p2_caso_normal_columnas_nuevas_y_tamano():
    """Caso Normal: Agrega las seis variables de P2 y conserva el numero de filas."""
    malla = _crear_malla_features()

    resultado = features_p2(malla)

    nuevas = {
        "inventario_previo",
        "razon_inventario",
        "saldo_1h",
        "saldo_3h",
        "salidas_hora_previa",
        "llegadas_hora_previa",
    }
    assert nuevas <= set(resultado.columns)
    assert len(resultado) == len(malla)
    assert "inventario_previo" not in malla.columns


def test_features_p2_caso_normal_valores_estacion_1():
    """Caso Normal: Las seis variables coinciden con los valores calculados a mano."""
    resultado = features_p2(_crear_malla_features())
    estacion_1 = resultado[resultado["estacion_id"] == 1].reset_index(drop=True)

    esperado = {
        "inventario_previo": [np.nan, 5.0, 4.0, 1.0],
        "razon_inventario": [np.nan, 0.5, 0.4, 0.1],
        "salidas_hora_previa": [np.nan, 1.0, 2.0, 3.0],
        "llegadas_hora_previa": [np.nan, 0.0, 1.0, 0.0],
        "saldo_1h": [np.nan, -1.0, -1.0, -3.0],
        "saldo_3h": [np.nan, np.nan, np.nan, -5.0],
    }
    for columna, valores in esperado.items():
        np.testing.assert_allclose(estacion_1[columna], valores, err_msg=columna)


def test_features_p2_caso_normal_estaciones_no_se_mezclan():
    """Caso Normal: Las variables de una estacion se calculan solo con sus propias horas."""
    resultado = features_p2(_crear_malla_features())
    estacion_2 = resultado[resultado["estacion_id"] == 2].reset_index(drop=True)

    np.testing.assert_allclose(estacion_2["inventario_previo"], [np.nan, 10.0, 10.0, 12.0])
    np.testing.assert_allclose(estacion_2["razon_inventario"], [np.nan, 0.5, 0.5, 0.6])


def test_features_p2_caso_limite_primera_hora_es_nan():
    """Caso Límite: La primera hora de cada estacion no tiene historia y queda en NaN."""
    resultado = features_p2(_crear_malla_features())
    primeras = resultado.groupby("estacion_id").head(1)

    columnas = ["inventario_previo", "saldo_1h", "saldo_3h", "salidas_hora_previa"]
    assert primeras[columnas].isna().all().all()


def test_features_p2_caso_limite_no_usa_informacion_de_la_hora_actual():
    """Caso Límite: Cambiar los datos de una hora no altera las variables de esa misma hora."""
    malla = _crear_malla_features()
    base = features_p2(malla)

    alterada = malla.copy()
    alterada.loc[2, ["salidas", "llegadas", "inventario_estimado"]] = [99, 99, 7.0]
    cambio = features_p2(alterada)

    columnas = [
        "inventario_previo",
        "razon_inventario",
        "salidas_hora_previa",
        "llegadas_hora_previa",
        "saldo_1h",
        "saldo_3h",
    ]
    pd.testing.assert_series_equal(base.loc[2, columnas], cambio.loc[2, columnas])



def test_features_p2_caso_limite_estacion_con_menos_horas_que_la_ventana():
    """Caso Límite: Una estacion con 1 hora y ventana de 3 horas da NaN en lugar de fallar."""
    malla = _crear_malla_features().groupby("estacion_id").head(1)

    resultado = features_p2(malla)

    assert resultado["saldo_3h"].isna().all()


# 5. Pruebas para agregar_variables_calendario
def test_agregar_variables_calendario_caso_normal():
    """Caso Normal: Extrae hora, dia de semana, fin de semana y mes de fechas conocidas."""
    df = pd.DataFrame({
        "hora": pd.to_datetime([
            "2026-10-30 08:00",  # viernes
            "2026-10-31 23:00",  # sabado
            "2026-11-01 00:00",  # domingo
            "2026-11-02 12:00",  # lunes
        ])
    })

    resultado = agregar_variables_calendario(df)

    assert resultado["hora_del_dia"].tolist() == [8, 23, 0, 12]
    assert resultado["dia_de_semana"].tolist() == [4, 5, 6, 0]
    assert resultado["es_fin_de_semana"].tolist() == [0, 1, 1, 0]
    assert resultado["mes"].tolist() == [10, 10, 11, 11]


def test_agregar_variables_calendario_caso_limite_frontera_de_fin_de_semana():
    """Caso Límite: Viernes 23:00 es entre semana y sabado 00:00 ya es fin de semana."""
    df = pd.DataFrame({
        "hora": pd.to_datetime(["2026-10-30 23:00", "2026-10-31 00:00"])
    })

    resultado = agregar_variables_calendario(df)

    assert resultado["es_fin_de_semana"].tolist() == [0, 1]


def test_agregar_variables_calendario_caso_limite_no_modifica_entrada():
    """Caso Límite: Conserva las demas columnas y no modifica el DataFrame original."""
    df = pd.DataFrame({
        "hora": pd.to_datetime(["2026-01-05 10:00"]),
        "salidas": [3],
    })

    resultado = agregar_variables_calendario(df)

    assert list(df.columns) == ["hora", "salidas"]
    assert resultado["salidas"].tolist() == [3]


def test_agregar_variables_calendario_caso_limite_tabla_vacia():
    """Caso Límite: Una tabla vacia regresa vacia pero con las columnas nuevas."""
    df = pd.DataFrame({"hora": pd.to_datetime([])})

    resultado = agregar_variables_calendario(df)

    assert len(resultado) == 0
    assert {"hora_del_dia", "dia_de_semana", "es_fin_de_semana", "mes"} <= set(resultado.columns)


def test_agregar_variables_calendario_caso_limite_sin_columna_hora():
    """Caso Límite: Una tabla sin la columna hora produce un KeyError."""
    with pytest.raises(KeyError):
        agregar_variables_calendario(pd.DataFrame({"otra": [1]}))


# 6. Pruebas para agregar_features_temporales_p2
def test_agregar_features_temporales_p2_caso_normal():
    """Caso Normal: Agrega hora_dia, dia_semana y es_fin_semana a partir de la hora."""
    malla = pd.DataFrame({
        "hora": [
            "2026-10-30 08:00",  # viernes
            "2026-10-31 23:00",  # sabado
            "2026-11-01 00:00",  # domingo
            "2026-11-02 12:00",  # lunes
        ]
    })

    resultado = agregar_features_temporales_p2(malla)

    assert resultado["hora_dia"].tolist() == [8, 23, 0, 12]
    assert resultado["dia_semana"].tolist() == [4, 5, 6, 0]
    assert resultado["es_fin_semana"].tolist() == [0, 1, 1, 0]
    assert pd.api.types.is_integer_dtype(resultado["hora_dia"])


def test_agregar_features_temporales_p2_caso_limite_no_modifica_entrada():
    """Caso Límite: Devuelve una copia y la malla original no recibe columnas nuevas."""
    malla = pd.DataFrame({"hora": ["2026-01-05 10:00"]})

    agregar_features_temporales_p2(malla)

    assert list(malla.columns) == ["hora"]


# 7. Pruebas para agregar_rezagos_y_moviles
def test_agregar_rezagos_y_moviles_caso_normal_valores_por_estacion():
    """Caso Normal: Calcula rezagos y promedios moviles de cada estacion con la hora previa."""
    resultado = agregar_rezagos_y_moviles(_crear_salidas_por_estacion(), lags=[1, 2])
    estacion_1 = resultado[resultado["estacion_id"] == 1].reset_index(drop=True)
    estacion_2 = resultado[resultado["estacion_id"] == 2].reset_index(drop=True)

    # Estacion 1: salidas = [1, 2, 3, 4, 5]
    np.testing.assert_allclose(estacion_1["salidas_lag_1"], [np.nan, 1, 2, 3, 4])
    np.testing.assert_allclose(estacion_1["salidas_lag_2"], [np.nan, np.nan, 1, 2, 3])
    np.testing.assert_allclose(estacion_1["salidas_roll_mean_3h"], [np.nan, 1, 1.5, 2, 3])
    np.testing.assert_allclose(estacion_1["salidas_roll_mean_24h"], [np.nan, 1, 1.5, 2, 2.5])
    # Estacion 2: salidas = [10, 20, 30, 40, 50]
    np.testing.assert_allclose(estacion_2["salidas_lag_1"], [np.nan, 10, 20, 30, 40])
    np.testing.assert_allclose(estacion_2["salidas_roll_mean_3h"], [np.nan, 10, 15, 20, 30])


def test_agregar_rezagos_y_moviles_caso_normal_ordena_por_estacion_y_hora():
    """Caso Normal: El resultado queda ordenado por estacion y hora con indice reiniciado."""
    resultado = agregar_rezagos_y_moviles(_crear_salidas_por_estacion(), lags=[1])

    assert resultado["estacion_id"].tolist() == [1] * 5 + [2] * 5
    assert resultado.index.tolist() == list(range(10))
    assert resultado[resultado["estacion_id"] == 1]["hora"].is_monotonic_increasing
    assert resultado[resultado["estacion_id"] == 2]["hora"].is_monotonic_increasing


def test_agregar_rezagos_y_moviles_caso_normal_lags_por_defecto():
    """Caso Normal: Sin indicar lags crea los rezagos de 1, 2, 24 y 168 horas."""
    resultado = agregar_rezagos_y_moviles(_crear_salidas_por_estacion())

    for lag in (1, 2, 24, 168):
        assert f"salidas_lag_{lag}" in resultado.columns


def test_agregar_rezagos_y_moviles_caso_limite_no_usa_la_hora_actual():
    """Caso Límite: Cambiar las salidas de una hora no altera sus propios rezagos ni promedios."""
    horas = pd.date_range("2026-01-05 00:00", periods=5, freq="h")
    base = pd.DataFrame({"estacion_id": 1, "hora": horas, "salidas": [1, 2, 3, 4, 5]})
    alterada = base.assign(salidas=[1, 2, 3, 4, 500])

    resultado_base = agregar_rezagos_y_moviles(base, lags=[1, 2])
    resultado_alterado = agregar_rezagos_y_moviles(alterada, lags=[1, 2])

    columnas = [
        "salidas_lag_1",
        "salidas_lag_2",
        "salidas_roll_mean_3h",
        "salidas_roll_mean_24h",
    ]
    pd.testing.assert_series_equal(
        resultado_base.loc[4, columnas],
        resultado_alterado.loc[4, columnas],
    )


def test_agregar_rezagos_y_moviles_caso_limite_no_modifica_entrada():
    """Caso Límite: La funcion no agrega columnas al DataFrame original."""
    malla = _crear_salidas_por_estacion()
    columnas_antes = list(malla.columns)

    agregar_rezagos_y_moviles(malla, lags=[1])

    assert list(malla.columns) == columnas_antes


# 8. Pruebas para construir_features_p1
def test_construir_features_p1_caso_normal_con_historia_suficiente():
    """Caso Normal: Con 200 horas de viajes, al quitar NaN quedan las horas con rezago de 168."""
    resultado = construir_features_p1(_crear_viajes_largos(), _crear_estaciones())

    # 3 estaciones x 201 horas, menos las primeras 168 horas de cada una
    assert len(resultado) == 3 * (201 - 168)
    assert resultado.isna().sum().sum() == 0
    assert resultado["hora"].min() == pd.Timestamp("2026-01-01 00:00") + pd.Timedelta(hours=168)
    assert {"hora_del_dia", "salidas_lag_168", "salidas_roll_mean_24h"} <= set(resultado.columns)


def test_construir_features_p1_caso_normal_conserva_filas_sin_eliminar_nans():
    """Caso Normal: Con eliminar_nans=False conserva todas las filas de la malla."""
    resultado = construir_features_p1(
        _crear_viajes(),
        _crear_estaciones(),
        eliminar_nans=False,
    )

    assert len(resultado) == 9
    assert {"salidas", "llegadas", "mes", "salidas_lag_168"} <= set(resultado.columns)


def test_construir_features_p1_caso_limite_poca_historia_queda_vacio():
    """Caso Límite: Con solo 3 horas de datos el rezago de 168 es NaN y no queda ninguna fila."""
    resultado = construir_features_p1(_crear_viajes(), _crear_estaciones())

    assert resultado.empty
    assert "salidas_lag_168" in resultado.columns


# 9. Pruebas para perfil_uso_estacion
def test_perfil_uso_estacion_caso_normal_estacion_1():
    """Caso Normal: Calcula salidas, llegadas, saldo, volumen, fin de semana y capacidad."""
    perfil = perfil_uso_estacion(_crear_viajes_perfil(), _crear_estaciones())
    estacion_1 = perfil.set_index("estacion_id").loc[1]

    assert estacion_1["salidas"] == 4
    assert estacion_1["llegadas"] == 2
    assert estacion_1["saldo_neto"] == -2
    # Salidas en 3 dias distintos (2 + 1 + 1): promedio de 4 / 3
    assert estacion_1["volumen_diario"] == pytest.approx(4 / 3)
    # Una de las cuatro salidas fue en sabado
    assert estacion_1["proporcion_fin_semana"] == pytest.approx(0.25)
    assert estacion_1["capacidad"] == 10
    # Perfil horario: mitad de las salidas a las 08 y mitad a las 17
    assert estacion_1["perfil_h08"] == pytest.approx(0.5)
    assert estacion_1["perfil_h17"] == pytest.approx(0.5)
    assert estacion_1["perfil_h00"] == 0


def test_perfil_uso_estacion_caso_normal_estacion_2():
    """Caso Normal: La estacion 2 tiene mas llegadas que salidas y saldo positivo."""
    perfil = perfil_uso_estacion(_crear_viajes_perfil(), _crear_estaciones())
    estacion_2 = perfil.set_index("estacion_id").loc[2]

    assert estacion_2["salidas"] == 2
    assert estacion_2["llegadas"] == 4
    assert estacion_2["saldo_neto"] == 2
    assert estacion_2["volumen_diario"] == pytest.approx(1.0)
    # Una de las dos salidas fue en domingo
    assert estacion_2["proporcion_fin_semana"] == pytest.approx(0.5)
    assert estacion_2["capacidad"] == 20
    # Las dos salidas (09:00 y 09:30) caen en la hora 9
    assert estacion_2["perfil_h09"] == pytest.approx(1.0)


def test_perfil_uso_estacion_caso_normal_perfil_horario_suma_uno():
    """Caso Normal: Hay 24 columnas de perfil horario y cada estacion con salidas suma 1."""
    perfil = perfil_uso_estacion(_crear_viajes_perfil(), _crear_estaciones())
    columnas_perfil = [col for col in perfil.columns if col.startswith("perfil_h")]

    assert len(columnas_perfil) == 24
    assert columnas_perfil[0] == "perfil_h00"
    assert columnas_perfil[-1] == "perfil_h23"
    np.testing.assert_allclose(perfil[columnas_perfil].sum(axis=1), [1.0, 1.0])


def test_perfil_uso_estacion_caso_normal_nombres_de_columnas_alternativos():
    """Caso Normal: Acepta estaciones con estacion_id/capacidad en lugar de id/dpcapacity."""
    estaciones = _crear_estaciones().rename(columns={
        "id": "estacion_id",
        "dpcapacity": "capacidad",
    })

    desde_alternativo = perfil_uso_estacion(_crear_viajes_perfil(), estaciones)
    desde_original = perfil_uso_estacion(_crear_viajes_perfil(), _crear_estaciones())

    pd.testing.assert_frame_equal(desde_alternativo, desde_original)


def test_perfil_uso_estacion_caso_limite_estacion_solo_destino():
    """Caso Límite: Una estacion que solo recibe viajes queda con salidas y perfil en cero."""
    viajes = _crear_viajes_perfil().iloc[[0]]

    perfil = perfil_uso_estacion(viajes, _crear_estaciones())
    estacion_2 = perfil.set_index("estacion_id").loc[2]

    assert estacion_2["salidas"] == 0
    assert estacion_2["llegadas"] == 1
    assert estacion_2["saldo_neto"] == 1
    assert estacion_2["volumen_diario"] == 0
    assert estacion_2["proporcion_fin_semana"] == 0
    assert estacion_2["perfil_h08"] == 0


def test_perfil_uso_estacion_caso_limite_solo_estaciones_con_viajes():
    """Caso Límite: Las estaciones sin ningun viaje no aparecen en el perfil."""
    perfil = perfil_uso_estacion(_crear_viajes_perfil(), _crear_estaciones())

    # La estacion 3 existe en la nomenclatura pero no tiene viajes
    assert sorted(perfil["estacion_id"]) == [1, 2]
    assert len(perfil) == 2


def test_perfil_uso_estacion_caso_limite_no_modifica_entrada():
    """Caso Límite: La funcion no agrega columnas a los viajes que recibe."""
    viajes = _crear_viajes_perfil()
    columnas_antes = list(viajes.columns)

    perfil_uso_estacion(viajes, _crear_estaciones())

    assert list(viajes.columns) == columnas_antes


def test_perfil_uso_estacion_caso_limite_columnas_de_viajes_faltantes():
    """Caso Límite: Unos viajes sin la columna destino_id producen un ValueError."""
    viajes = _crear_viajes_perfil().drop(columns=["destino_id"])

    with pytest.raises(ValueError, match="Faltan columnas requeridas en viajes"):
        perfil_uso_estacion(viajes, _crear_estaciones())


def test_perfil_uso_estacion_caso_limite_columnas_de_estaciones_faltantes():
    """Caso Límite: Unas estaciones sin capacidad producen un ValueError."""
    estaciones = _crear_estaciones().drop(columns=["dpcapacity"])

    with pytest.raises(ValueError, match="Faltan columnas requeridas en estaciones"):
        perfil_uso_estacion(_crear_viajes_perfil(), estaciones)


def test_perfil_uso_estacion_caso_limite_estaciones_duplicadas():
    """Caso Límite: Una estacion repetida en la nomenclatura produce un MergeError."""
    estaciones = pd.concat(
        [_crear_estaciones(), _crear_estaciones().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(pd.errors.MergeError):
        perfil_uso_estacion(_crear_viajes_perfil(), estaciones)