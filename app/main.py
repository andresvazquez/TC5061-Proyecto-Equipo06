from fastapi import FastAPI

from app.modelo import modelo
from app.prediccion import predecir
from app.schemas import PrediccionEntrada

app = FastAPI()


# El esquema valida el JSON y la función predecir devuelve la predicción
@app.post("/predict")
def predict(entrada: PrediccionEntrada):
    return {"prediccion": predecir(modelo, entrada)}
