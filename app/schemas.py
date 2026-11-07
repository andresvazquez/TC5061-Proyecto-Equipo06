from pydantic import BaseModel


class PrediccionEntrada(BaseModel):
    inventario_previo: float
    razon_inventario: float
    saldo_1h: float
    saldo_3h: float
    salidas_hora_previa: float
    llegadas_hora_previa: float
    hora_dia: int
    dia_semana: int
    es_fin_semana: int
