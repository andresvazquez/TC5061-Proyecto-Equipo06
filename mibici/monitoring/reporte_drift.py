from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from mibici.monitoring.datos_drift import conjunto_actual, conjunto_referencia

DEFINICION = DataDefinition(
    numerical_columns=[
        "inventario_previo",
        "razon_inventario",
        "saldo_1h",
        "saldo_3h",
        "salidas_hora_previa",
        "llegadas_hora_previa",
        "hora_dia",
        "dia_semana",
    ],
    categorical_columns=[
        "es_fin_semana",
        "riesgo",
    ],
)


def crear_datasets():
    actual = Dataset.from_pandas(
        conjunto_actual(),
        data_definition=DEFINICION,
    )
    referencia = Dataset.from_pandas(
        conjunto_referencia(),
        data_definition=DEFINICION,
    )
    return actual, referencia


def ejecutar_reporte(presets=None):
    actual, referencia = crear_datasets()
    if presets is None:
        presets = [DataDriftPreset()]
    reporte = Report(presets)
    return reporte.run(actual, referencia)


def main():
    snapshot = ejecutar_reporte()
    for metrica in snapshot.dict()["metrics"]:
        print(metrica["metric_name"], metrica["value"])


if __name__ == "__main__":
    main()
