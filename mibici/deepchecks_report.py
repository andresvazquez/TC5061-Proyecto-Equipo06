import json

import numpy as np
from sklearn.metrics import _scorer

# Compatibilidad de DeepChecks 0.19.1 con NumPy 2 y
# la version fijada de scikit-learn del proyecto.
np.Inf = np.inf
_scorer._SCORERS["max_error"] = _scorer._SCORERS["neg_max_error"]

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    FeatureDrift,
    LabelDrift,
    MixedDataTypes,
    PercentOfNulls,
)
from deepchecks.tabular.suite import Suite

from mibici import config
from mibici.modeling.train_p2 import (
    VARIABLES,
    cargar_datos,
    partir_por_fecha,
    preparar_malla,
)

COLUMNAS_ANALISIS = [
    "estacion_id",
    "hora",
    "riesgo",
    *VARIABLES,
]

RESULTADOS_DIR = (
    config.PROJ_ROOT
    / "reports"
    / "deepchecks"
)

MUESTRA_DRIFT = 100_000


def construir_dataset(df):
    """
    Construye el Dataset de DeepChecks utilizado en el reporte.
    """
    return Dataset(
        df[COLUMNAS_ANALISIS].copy(),
        label="riesgo",
        cat_features=["estacion_id"],
        datetime_name="hora",
    )


def muestrear(df):
    """
    Reduce de forma reproducible el conjunto utilizado
    por los checks de drift.
    """
    if len(df) <= MUESTRA_DRIFT:
        return df.copy()

    return df.sample(
        n=MUESTRA_DRIFT,
        random_state=config.SEMILLA,
    ).sort_values("hora")


def extraer_percent_nulos(resultado):
    """
    Extrae porcentajes de nulos del resultado de DeepChecks.
    """
    tabla = resultado.value

    columna_resultado = (
        "Percent of nulls in sample"
    )

    return {
        str(indice): float(valor)
        for indice, valor in tabla[
            columna_resultado
        ].items()
    }


def extraer_tipos_mixtos(resultado):
    """
    Devuelve las columnas donde DeepChecks detecta
    mezcla de tipos.
    """
    return {
        str(columna): valor
        for columna, valor in resultado.value.items()
        if valor
    }


def extraer_feature_drift(resultado):
    """
    Extrae drift score y metodo para cada feature.
    """
    return {
        str(columna): {
            "drift_score": float(
                detalle["Drift score"]
            ),
            "method": str(
                detalle["Method"]
            ),
        }
        for columna, detalle
        in resultado.value.items()
    }


def ejecutar_integridad(dataset):
    """
    Ejecuta los checks de integridad sobre un Dataset.
    """
    suite = Suite(
        "Integridad de datos MiBici",
        PercentOfNulls(),
        MixedDataTypes(),
        DataDuplicates(),
    )

    return suite.run(dataset)


def ejecutar_drift(train_dataset, test_dataset):
    """
    Ejecuta FeatureDrift y LabelDrift.
    """
    suite = Suite(
        "Drift temporal MiBici",
        FeatureDrift(
            columns=VARIABLES,
        ),
        LabelDrift(),
    )

    return suite.run(
        train_dataset,
        test_dataset,
    )


def main():
    """
    Ejecuta DeepChecks sobre los datos reales de MiBici
    y guarda evidencia HTML y un resumen JSON.
    """
    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Cargando datos reales...")

    viajes, estaciones = cargar_datos()

    print("Construyendo malla P2...")

    malla = preparar_malla(
        viajes,
        estaciones,
    )

    entrenamiento, _, prueba = (
        partir_por_fecha(malla)
    )

    print(
        "Filas entrenamiento:",
        len(entrenamiento),
    )
    print(
        "Filas prueba:",
        len(prueba),
    )

    train_integridad = construir_dataset(
        entrenamiento
    )

    test_integridad = construir_dataset(
        prueba
    )

    train_muestra = muestrear(
        entrenamiento
    )

    test_muestra = muestrear(
        prueba
    )

    train_drift = construir_dataset(
        train_muestra
    )

    test_drift = construir_dataset(
        test_muestra
    )

    print(
        "Muestra train para drift:",
        len(train_muestra),
    )
    print(
        "Muestra test para drift:",
        len(test_muestra),
    )

    print("Ejecutando integridad train...")

    integridad_train = ejecutar_integridad(
        train_integridad
    )

    print("Ejecutando integridad test...")

    integridad_test = ejecutar_integridad(
        test_integridad
    )

    print("Ejecutando drift train vs test...")

    drift = ejecutar_drift(
        train_drift,
        test_drift,
    )

    ruta_integridad_train = (
        RESULTADOS_DIR
        / "integridad_train.html"
    )

    ruta_integridad_test = (
        RESULTADOS_DIR
        / "integridad_test.html"
    )

    ruta_drift = (
        RESULTADOS_DIR
        / "drift_train_test.html"
    )

    integridad_train.save_as_html(
        str(ruta_integridad_train),
        as_widget=False,
    )

    integridad_test.save_as_html(
        str(ruta_integridad_test),
        as_widget=False,
    )

    drift.save_as_html(
        str(ruta_drift),
        as_widget=False,
    )

    percent_train = (
        PercentOfNulls().run(
            train_integridad
        )
    )

    percent_test = (
        PercentOfNulls().run(
            test_integridad
        )
    )

    duplicados_train = (
        DataDuplicates().run(
            train_integridad
        )
    )

    duplicados_test = (
        DataDuplicates().run(
            test_integridad
        )
    )

    tipos_train = (
        MixedDataTypes().run(
            train_integridad
        )
    )

    tipos_test = (
        MixedDataTypes().run(
            test_integridad
        )
    )

    feature_drift = FeatureDrift(
        columns=VARIABLES,
    ).run(
        train_drift,
        test_drift,
    )

    label_drift = LabelDrift().run(
        train_drift,
        test_drift,
    )

    resumen = {
        "filas": {
            "train": len(entrenamiento),
            "test": len(prueba),
            "train_drift_sample": len(
                train_muestra
            ),
            "test_drift_sample": len(
                test_muestra
            ),
        },
        "percent_nulls_train": (
            extraer_percent_nulos(
                percent_train
            )
        ),
        "percent_nulls_test": (
            extraer_percent_nulos(
                percent_test
            )
        ),
        "duplicate_ratio_train": float(
            duplicados_train.value
        ),
        "duplicate_ratio_test": float(
            duplicados_test.value
        ),
        "mixed_types_train": (
            extraer_tipos_mixtos(
                tipos_train
            )
        ),
        "mixed_types_test": (
            extraer_tipos_mixtos(
                tipos_test
            )
        ),
        "feature_drift": (
            extraer_feature_drift(
                feature_drift
            )
        ),
        "label_drift": {
            "drift_score": float(
                label_drift.value[
                    "Drift score"
                ]
            ),
            "method": str(
                label_drift.value[
                    "Method"
                ]
            ),
        },
    }

    ruta_json = (
        RESULTADOS_DIR
        / "resumen_deepchecks.json"
    )

    with ruta_json.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            resumen,
            archivo,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("DeepChecks completado")
    print(
        "Resultados:",
        RESULTADOS_DIR,
    )
    print()
    print(
        json.dumps(
            resumen,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
