from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from mibici import config
from mibici.monitoring.datos_drift import conjunto_actual, conjunto_referencia

REPORTES_DIR = Path("reports/evidently")

RUTA_HTML = (
    REPORTES_DIR
    / "data_drift.html"
)

RUTA_JSON = (
    REPORTES_DIR
    / "data_drift.json"
)

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

def guardar_reporte(snapshot):
    """
    Guarda el reporte de Data Drift en HTML y JSON.
    """
    REPORTES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot.save_html(
        str(RUTA_HTML)
    )

    snapshot.save_json(
        str(RUTA_JSON)
    )

def ejecutar_reporte_calidad():
    actual, referencia = crear_datasets()
    reporte = Report([DataSummaryPreset()], include_tests=True)
    return reporte.run(actual, referencia)


def guardar_reporte_calidad(snapshot):
    carpeta = config.REPORTS_DIR / "evidently"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "calidad_datos.html"
    snapshot.save_html(str(ruta))
    ruta_json = carpeta / "calidad_datos.json"
    ruta_json.write_text(snapshot.json(), encoding="utf-8")
    return ruta

def main():
    snapshot = ejecutar_reporte()

    guardar_reporte(snapshot)

    for metrica in snapshot.dict()["metrics"]:
        print(
            metrica["metric_name"],
            metrica["value"],
        )

    print()
    print(
        "Reporte HTML:",
        RUTA_HTML,
    )
    print(
        "Reporte JSON:",
        RUTA_JSON,
    )

    ruta = guardar_reporte_calidad(
        ejecutar_reporte_calidad()
    )
    print(
        "Reporte de calidad guardado en",
        ruta,
    )


if __name__ == "__main__":
    main()
