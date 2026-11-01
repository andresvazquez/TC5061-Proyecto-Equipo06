from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift, LabelDrift
import numpy as np
import pandas as pd
import pytest


def crear_datasets_sin_drift():
    """
    Construye train y test con las mismas distribuciones.
    """
    rng = np.random.default_rng(42)

    feature = rng.normal(
        loc=0,
        scale=1,
        size=200,
    )

    estaciones = (
        ["A", "B"] * 100
    )

    riesgo = (
        [0, 1] * 100
    )

    train_df = pd.DataFrame({
        "feature": feature,
        "estacion_id": estaciones,
        "riesgo": riesgo,
    })

    test_df = train_df.copy()

    return train_df, test_df


def crear_datasets_con_drift():
    """
    Construye train y test con drift controlado
    en una feature numerica y en la etiqueta.
    """
    rng = np.random.default_rng(42)

    train_df = pd.DataFrame({
        "feature": rng.normal(
            loc=0,
            scale=1,
            size=200,
        ),
        "estacion_id": ["A", "B"] * 100,
        "riesgo": [0, 1] * 100,
    })

    test_df = pd.DataFrame({
        "feature": rng.normal(
            loc=5,
            scale=1,
            size=200,
        ),
        "estacion_id": ["A", "B"] * 100,
        "riesgo": (
            [1] * 180
            + [0] * 20
        ),
    })

    return train_df, test_df


def armar_dataset(df):
    """
    Construye el Dataset comun utilizado para los checks.
    """
    return Dataset(
        df,
        label="riesgo",
        cat_features=["estacion_id"],
    )


def test_feature_drift_cero_con_distribuciones_iguales():
    """
    Train y test identicos no deben presentar drift
    en la feature numerica.
    """
    train_df, test_df = crear_datasets_sin_drift()

    resultado = FeatureDrift(
        columns=["feature"],
    ).run(
        armar_dataset(train_df),
        armar_dataset(test_df),
    )

    drift = resultado.value[
        "feature"
    ]["Drift score"]

    assert drift == pytest.approx(0.0)


def test_feature_drift_detecta_distribucion_desplazada():
    """
    Una diferencia fuerte en la distribucion numerica
    debe producir un drift score alto.
    """
    train_df, test_df = crear_datasets_con_drift()

    resultado = FeatureDrift(
        columns=["feature"],
    ).run(
        armar_dataset(train_df),
        armar_dataset(test_df),
    )

    drift = resultado.value[
        "feature"
    ]["Drift score"]

    assert drift > 0.2


def test_feature_drift_condicion_pasa_sin_drift():
    """
    La condicion de drift debe pasar para distribuciones
    identicas.
    """
    train_df, test_df = crear_datasets_sin_drift()

    check = FeatureDrift(
        columns=["feature"],
    )

    check.add_condition_drift_score_less_than(
        0.2
    )

    resultado = check.run(
        armar_dataset(train_df),
        armar_dataset(test_df),
    )

    assert resultado.passed_conditions()


def test_feature_drift_condicion_falla_con_drift():
    """
    La condicion debe fallar cuando la feature presenta
    un desplazamiento fuerte.
    """
    train_df, test_df = crear_datasets_con_drift()

    check = FeatureDrift(
        columns=["feature"],
    )

    check.add_condition_drift_score_less_than(
        0.2
    )

    resultado = check.run(
        armar_dataset(train_df),
        armar_dataset(test_df),
    )

    assert not resultado.passed_conditions()


def test_label_drift_cero_con_distribuciones_iguales():
    """
    La misma distribucion de la etiqueta debe producir
    drift cercano a cero.
    """
    train_df, test_df = crear_datasets_sin_drift()

    resultado = LabelDrift().run(
        armar_dataset(train_df),
        armar_dataset(test_df),
    )

    drift = resultado.value[
        "Drift score"
    ]

    assert drift == pytest.approx(0.0)


def test_label_drift_detecta_cambio_de_proporcion():
    """
    Un cambio deliberado en la proporcion de clases debe
    ser detectado como label drift.
    """
    train_df, test_df = crear_datasets_con_drift()

    resultado = LabelDrift().run(
        armar_dataset(train_df),
        armar_dataset(test_df),
    )

    drift = resultado.value[
        "Drift score"
    ]

    assert drift > 0.2


def test_label_drift_condicion_pasa_sin_drift():
    """
    La condicion debe pasar cuando la distribucion de
    etiquetas permanece estable.
    """
    train_df, test_df = crear_datasets_sin_drift()

    check = LabelDrift()

    check.add_condition_drift_score_less_than(
        0.2
    )

    resultado = check.run(
        armar_dataset(train_df),
        armar_dataset(test_df),
    )

    assert resultado.passed_conditions()


def test_label_drift_condicion_falla_con_drift():
    """
    La condicion debe fallar ante un cambio importante
    en la distribucion de la etiqueta.
    """
    train_df, test_df = crear_datasets_con_drift()

    check = LabelDrift()

    check.add_condition_drift_score_less_than(
        0.2
    )

    resultado = check.run(
        armar_dataset(train_df),
        armar_dataset(test_df),
    )

    assert not resultado.passed_conditions()
