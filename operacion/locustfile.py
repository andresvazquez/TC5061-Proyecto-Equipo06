import random

from locust import HttpUser, between, task

CASOS_VALIDOS = [
    {
        "inventario_previo": 10.0,
        "razon_inventario": 0.50,
        "saldo_1h": 1.0,
        "saldo_3h": 2.0,
        "salidas_hora_previa": 3.0,
        "llegadas_hora_previa": 2.0,
        "hora_dia": 12,
        "dia_semana": 2,
        "es_fin_semana": 0,
    },
    {
        "inventario_previo": 3.0,
        "razon_inventario": 0.15,
        "saldo_1h": -2.0,
        "saldo_3h": -4.0,
        "salidas_hora_previa": 8.0,
        "llegadas_hora_previa": 2.0,
        "hora_dia": 8,
        "dia_semana": 0,
        "es_fin_semana": 0,
    },
    {
        "inventario_previo": 18.0,
        "razon_inventario": 0.90,
        "saldo_1h": 3.0,
        "saldo_3h": 5.0,
        "salidas_hora_previa": 1.0,
        "llegadas_hora_previa": 6.0,
        "hora_dia": 18,
        "dia_semana": 5,
        "es_fin_semana": 1,
    },
    {
        "inventario_previo": 0.0,
        "razon_inventario": 0.0,
        "saldo_1h": 0.0,
        "saldo_3h": -1.0,
        "salidas_hora_previa": 0.0,
        "llegadas_hora_previa": 1.0,
        "hora_dia": 23,
        "dia_semana": 6,
        "es_fin_semana": 1,
    },
]


class UsuarioMiBici(HttpUser):
    wait_time = between(1, 3)

    @task
    def predecir_riesgo(self):
        entrada = random.choice(CASOS_VALIDOS)

        with self.client.post(
            "/predict",
            json=entrada,
            name="/predict",
            catch_response=True,
        ) as respuesta:
            if respuesta.status_code != 200:
                respuesta.failure(f"Respuesta inesperada: HTTP {respuesta.status_code}")
                return

            try:
                contenido = respuesta.json()
            except ValueError:
                respuesta.failure("La respuesta no contiene JSON valido")
                return

            if "prediccion" not in contenido:
                respuesta.failure("La respuesta no contiene prediccion")
                return

            respuesta.success()
