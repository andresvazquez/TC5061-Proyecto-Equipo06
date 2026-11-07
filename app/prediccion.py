import pandas as pd

from app.schemas import PrediccionEntrada

COLUMNAS = [
    "inventario_previo",
    "razon_inventario",
    "saldo_1h",
    "saldo_3h",
    "salidas_hora_previa",
    "llegadas_hora_previa",
    "hora_dia",
    "dia_semana",
    "es_fin_semana",
]

COLUMNAS_FLOTANTES = COLUMNAS[:6]
COLUMNAS_ENTERAS = COLUMNAS[6:]


def predecir(modelo, entrada: PrediccionEntrada) -> int | float:
    datos = entrada.model_dump()
    fila = pd.DataFrame(
        [[datos[columna] for columna in COLUMNAS]],
        columns=COLUMNAS,
    )
    fila[COLUMNAS_FLOTANTES] = fila[COLUMNAS_FLOTANTES].astype("float64")
    fila[COLUMNAS_ENTERAS] = fila[COLUMNAS_ENTERAS].astype("int64")

    resultado = modelo.predict(fila)

    if isinstance(resultado, pd.DataFrame):
        valor = resultado.iloc[0, 0]
    elif isinstance(resultado, pd.Series):
        valor = resultado.iloc[0]
    elif isinstance(resultado, (list, tuple)):
        valor = resultado[0]
    elif hasattr(resultado, "flat"):
        valor = resultado.flat[0]
    else:
        valor = resultado

    if hasattr(valor, "item"):
        valor = valor.item()

    if isinstance(valor, bool):
        return int(valor)
    if isinstance(valor, (int, float)):
        return valor
    return float(valor)
