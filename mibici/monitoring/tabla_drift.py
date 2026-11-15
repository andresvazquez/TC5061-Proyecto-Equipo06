import json

import pandas as pd

from mibici import config

RUTA_JSON = config.REPORTS_DIR / "evidently" / "data_drift.json"
RUTA_TABLA = config.REPORTS_DIR / "evidently" / "tabla_drift.csv"


def leer_metricas(ruta=RUTA_JSON):
    return json.loads(ruta.read_text(encoding="utf-8"))["metrics"]


def tabla_drift(metricas):
    # Una fila por variable: método, score, umbral y si hay drift
    filas = []
    for metrica in metricas:
        configuracion = metrica["config"]
        if not configuracion["type"].endswith("ValueDrift"):
            continue
        score = metrica["value"]
        umbral = configuracion["threshold"]
        filas.append(
            {
                "variable": configuracion["column"],
                "metodo": configuracion["method"],
                "score": score,
                "umbral": umbral,
                "drift": score >= umbral,
            }
        )
    return pd.DataFrame(filas).sort_values("score", ascending=False)


def resumen_drift(metricas):
    # Columnas con drift y su proporción
    for metrica in metricas:
        if metrica["config"]["type"].endswith("DriftedColumnsCount"):
            return metrica["value"]


def main():
    metricas = leer_metricas()
    tabla = tabla_drift(metricas)
    resumen = resumen_drift(metricas)

    print(tabla.to_string(index=False))
    print()
    print("Columnas con drift:", int(resumen["count"]), "| proporción:", resumen["share"])

    tabla.to_csv(RUTA_TABLA, index=False)
    print("Tabla guardada en", RUTA_TABLA)


if __name__ == "__main__":
    main()