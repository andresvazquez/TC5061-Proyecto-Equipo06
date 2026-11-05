import importlib
import sys
from unittest.mock import MagicMock

import mlflow
import mlflow.pyfunc

from mibici import config


def test_carga_modelo_champion(monkeypatch):
    """
    El modulo debe configurar MLflow y cargar el champion de P2.
    """
    modelo_falso = MagicMock()

    set_tracking_uri = MagicMock()
    load_model = MagicMock(
        return_value=modelo_falso
    )

    monkeypatch.setattr(
        mlflow,
        "set_tracking_uri",
        set_tracking_uri,
    )

    monkeypatch.setattr(
        mlflow.pyfunc,
        "load_model",
        load_model,
    )

    sys.modules.pop(
        "app.modelo",
        None,
    )

    modulo = importlib.import_module(
        "app.modelo"
    )

    assert modulo.MODEL_NAME == (
        "mibici-equipo06-p2-riesgo"
    )

    assert modulo.MODEL_ALIAS == "champion"

    assert modulo.MODEL_URI == (
        "models:/mibici-equipo06-p2-riesgo"
        "@champion"
    )

    set_tracking_uri.assert_called_once_with(
        config.TRACKING_URI
    )

    load_model.assert_called_once_with(
        modulo.MODEL_URI
    )

    assert modulo.modelo is modelo_falso
