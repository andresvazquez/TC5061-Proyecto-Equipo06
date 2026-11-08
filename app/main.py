from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from app.modelo import MODEL_ALIAS, MODEL_NAME, modelo
from app.prediccion import predecir
from app.schemas import PrediccionEntrada, PrediccionSalida

app = FastAPI()


# El esquema valida el JSON y la función predecir devuelve la predicción
@app.post("/predict", response_model=PrediccionSalida)
def predict(entrada: PrediccionEntrada):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return PrediccionSalida(
        prediccion=predecir(modelo, entrada),
        modelo=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=VERSION,
    )

def obtener_version() -> str | None:
    try:
        return MlflowClient().get_model_version_by_alias(
            MODEL_NAME, MODEL_ALIAS
        ).version
    except Exception:
        return None


VERSION = obtener_version()
