import mlflow
from loguru import logger

from mibici import config

MODEL_NAME = "mibici-equipo06-p2-riesgo"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


def cargar_modelo():
    """
    Carga el modelo P2 promovido como champion en MLflow.

    Returns:
        mlflow.pyfunc.PyFuncModel: Modelo listo para inferencia.
    """
    mlflow.set_tracking_uri(
        config.TRACKING_URI
    )

    return mlflow.pyfunc.load_model(
        MODEL_URI
    )


try:
    modelo = cargar_modelo()
except Exception as error:
    logger.error(f"No se pudo cargar el modelo: {error}")
    modelo = None
