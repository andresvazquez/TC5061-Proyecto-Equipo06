import argparse
import re

import pandas as pd

from mibici import config

DIR_LOCUST = config.REPORTS_DIR / "locust"
DIR_FINOPS = config.REPORTS_DIR / "finops"
RUTA_RECURSOS = DIR_LOCUST / "recursos.csv"
RUTA_TABLA = DIR_FINOPS / "costo_inferencia.csv"
RUTA_TABLA_REPLICAS = DIR_FINOPS / "costo_inferencia_replicas.csv"

USUARIOS = [10, 50, 100]
REPLICAS = [1, 2, 4]
SEGUNDOS_POR_HORA = 3600
# Instancia de referencia: AWS EC2 t3.medium (2 vCPU, 4 GiB), Linux, US East (N. Virginia)
# Precio por hora bajo demanda de https://aws.amazon.com/ec2/instance-types/t3/
# consultado el 22 de noviembre de 2026
PRECIO_HORA_USD = 0.0418
VCPU_INSTANCIA = 2

# Unidades que imprime docker stats, expresadas en MB
UNIDADES_MB = {
    "B": 1e-6,
    "kB": 1e-3,
    "KB": 1e-3,
    "KiB": 1.024e-3,
    "MB": 1.0,
    "MiB": 1.048576,
    "GB": 1e3,
    "GiB": 1073.741824,
}


def memoria_a_mb(texto):
    # "936.9MB / 15.72GB" -> 936.9: solo el uso, sin el total de la máquina
    uso = texto.split("/")[0].strip()
    coincidencia = re.fullmatch(r"([\d.]+)\s*([A-Za-z]+)", uso)
    if coincidencia is None or coincidencia.group(2) not in UNIDADES_MB:
        raise ValueError(f"Formato de memoria no reconocido: {texto}")
    return float(coincidencia.group(1)) * UNIDADES_MB[coincidencia.group(2)]


def leer_recursos(ruta=RUTA_RECURSOS):
    # recursos.csv con la CPU como número, la memoria en MB y la hora como fecha UTC
    recursos = pd.read_csv(ruta)
    recursos["hora"] = pd.to_datetime(recursos["hora"], utc=True)
    recursos["cpu"] = recursos["CPUPerc"].str.rstrip("%").astype(float)
    recursos["memoria_mb"] = recursos["MemUsage"].map(memoria_a_mb)
    return recursos


def leer_corrida(usuarios):
    # Fila Aggregated de la corrida y su inicio y fin según el historial de Locust
    stats = pd.read_csv(DIR_LOCUST / f"locust_{usuarios}_stats.csv")
    agregada = stats[stats["Name"] == "Aggregated"].iloc[0]

    historial = pd.read_csv(DIR_LOCUST / f"locust_{usuarios}_stats_history.csv")
    historial = historial[historial["Name"] == "Aggregated"]
    marcas = pd.to_numeric(historial["Timestamp"])
    inicio = pd.to_datetime(marcas.min(), unit="s", utc=True)
    fin = pd.to_datetime(marcas.max(), unit="s", utc=True)
    return agregada, inicio, fin


def resumir_recursos(recursos, inicio, fin):
    # CPU y memoria promedio y máxima de cada contenedor dentro de la corrida
    ventana = recursos[(recursos["hora"] >= inicio) & (recursos["hora"] <= fin)]
    return ventana.groupby("contenedor").agg(
        cpu_prom=("cpu", "mean"),
        cpu_max=("cpu", "max"),
        mem_prom_mb=("memoria_mb", "mean"),
        mem_max_mb=("memoria_mb", "max"),
    )


def calcular_fila(usuarios, recursos, precio_hora):
    agregada, inicio, fin = leer_corrida(usuarios)
    resumen = resumir_recursos(recursos, inicio, fin)
    faltantes = {"mibici-api", "mibici-mlflow"} - set(resumen.index)
    if faltantes:
        raise ValueError(f"Sin muestras de {sorted(faltantes)} en la corrida de {usuarios}")
    api = resumen.loc["mibici-api"]
    mlflow = resumen.loc["mibici-mlflow"]

    rps = float(agregada["Requests/s"])
    peticiones_hora = rps * SEGUNDOS_POR_HORA
    # Fracción de la instancia que usan los dos servicios juntos
    fraccion_cpu = (api["cpu_prom"] + mlflow["cpu_prom"]) / (VCPU_INSTANCIA * 100)

    return {
        "usuarios": usuarios,
        "rps": rps,
        "mediana_ms": float(agregada["50%"]),
        "p95_ms": float(agregada["95%"]),
        "api_cpu_prom": api["cpu_prom"],
        "api_cpu_max": api["cpu_max"],
        "api_mem_prom_mb": api["mem_prom_mb"],
        "api_mem_max_mb": api["mem_max_mb"],
        "mlflow_cpu_prom": mlflow["cpu_prom"],
        "mlflow_cpu_max": mlflow["cpu_max"],
        "mlflow_mem_prom_mb": mlflow["mem_prom_mb"],
        "mlflow_mem_max_mb": mlflow["mem_max_mb"],
        "peticiones_por_hora": peticiones_hora,
        "precio_hora": precio_hora,
        "costo_por_inferencia": precio_hora / peticiones_hora,
        "costo_por_inferencia_cpu_usada": precio_hora * fraccion_cpu / peticiones_hora,
    }


def calcular_replicas(filas, precio_hora):
    registros = []
    for fila in filas:
        fraccion_cpu = (fila["api_cpu_prom"] + fila["mlflow_cpu_prom"]) / (VCPU_INSTANCIA * 100)
        for replicas in REPLICAS:
            rps = fila["rps"] * replicas
            peticiones_hora = rps * SEGUNDOS_POR_HORA
            costo_hora = precio_hora * replicas
            registros.append(
                {
                    "usuarios": fila["usuarios"],
                    "replicas": replicas,
                    "rps_por_replica": fila["rps"],
                    "rps": rps,
                    "mediana_ms": fila["mediana_ms"],
                    "p95_ms": fila["p95_ms"],
                    "fraccion_cpu": fraccion_cpu,
                    "peticiones_por_hora": peticiones_hora,
                    "costo_por_hora": costo_hora,
                    "costo_por_inferencia": costo_hora / peticiones_hora,
                    "costo_por_inferencia_cpu_usada": (
                        costo_hora * fraccion_cpu / peticiones_hora
                    ),
                }
            )
    return pd.DataFrame(registros)


def main():
    parser = argparse.ArgumentParser(
        description="Costo por inferencia a partir de la prueba de carga de Locust"
    )
    parser.add_argument(
        "--precio-hora",
        type=float,
        default=PRECIO_HORA_USD,
        help="Precio por hora de la instancia de referencia (USD)",
    )
    args = parser.parse_args()

    recursos = leer_recursos()
    filas = [calcular_fila(usuarios, recursos, args.precio_hora) for usuarios in USUARIOS]
    tabla = pd.DataFrame(filas)
    tabla_replicas = calcular_replicas(filas, args.precio_hora)

    DIR_FINOPS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_TABLA, index=False)
    tabla_replicas.to_csv(RUTA_TABLA_REPLICAS, index=False)

    print(tabla.set_index("usuarios").T.to_string(float_format=lambda v: f"{v:.6g}"))
    print(f"\nTabla guardada en {RUTA_TABLA}")
    print("\nEscalabilidad lineal: capacidad y costo por hora escalan con las réplicas")
    print(
        tabla_replicas.set_index(["usuarios", "replicas"]).to_string(
            float_format=lambda v: f"{v:.6g}"
        )
    )
    print(f"\nTabla guardada en {RUTA_TABLA_REPLICAS}")


if __name__ == "__main__":
    main()
