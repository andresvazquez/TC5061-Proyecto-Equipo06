import pandas as pd
import pytest


# Tres estaciones con capacidad y coordenadas, sin depender de data/
@pytest.fixture
def estaciones_pequenas():
    return pd.DataFrame(
        {
            "estacion_id": [1, 2, 3],
            "capacidad": [20, 10, 30],
            "latitud": [20.67, 20.68, 20.69],
            "longitud": [-103.35, -103.36, -103.37],
        }
    )


# Seis viajes en tres horas, con las columnas que deja dataset.py
@pytest.fixture
def viajes_pequenos():
    return pd.DataFrame(
        {
            "viaje_id": [1, 2, 3, 4, 5, 6],
            "inicio_del_viaje": pd.to_datetime(
                [
                    "2026-07-01 08:05",
                    "2026-07-01 08:40",
                    "2026-07-01 09:10",
                    "2026-07-01 09:30",
                    "2026-07-01 10:15",
                    "2026-07-01 10:50",
                ]
            ),
            "fin_del_viaje": pd.to_datetime(
                [
                    "2026-07-01 08:25",
                    "2026-07-01 09:05",
                    "2026-07-01 09:35",
                    "2026-07-01 09:55",
                    "2026-07-01 10:40",
                    "2026-07-01 11:10",
                ]
            ),
            "origen_id": [1, 1, 2, 3, 2, 1],
            "destino_id": [2, 3, 1, 1, 3, 2],
        }
    )
