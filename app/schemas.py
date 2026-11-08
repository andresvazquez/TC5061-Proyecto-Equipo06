from pydantic import BaseModel, Field


class PrediccionEntrada(BaseModel):
    inventario_previo: float = Field(ge=0)
    razon_inventario: float = Field(ge=0, le=1)
    saldo_1h: float
    saldo_3h: float
    salidas_hora_previa: float = Field(ge=0)
    llegadas_hora_previa: float = Field(ge=0)
    hora_dia: int = Field(ge=0, le=23)
    dia_semana: int = Field(ge=0, le=6)
    es_fin_semana: int = Field(ge=0, le=1)


class PrediccionSalida(BaseModel):
    prediccion: int | float
    modelo: str
    alias: str
    version: str | None = None
