import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mibici import config

DIR_LOCUST = config.REPORTS_DIR / "locust"
RUTA_HISTORIAL = DIR_LOCUST / "locust_stats_history.csv"

COLUMNAS = ["Timestamp", "User Count", "Name", "Requests/s", "Failures/s", "50%", "95%"]
NUMERICAS = ["Timestamp", "User Count", "Requests/s", "Failures/s", "50%", "95%"]


def leer_historial(ruta=RUTA_HISTORIAL):
    # Historial de Locust: solo la fila Aggregated, con el tiempo en segundos desde el inicio
    historial = pd.read_csv(ruta)

    faltantes = set(COLUMNAS) - set(historial.columns)
    if faltantes:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(sorted(faltantes)))

    historial = historial[historial["Name"] == "Aggregated"].copy()
    historial[NUMERICAS] = historial[NUMERICAS].apply(pd.to_numeric, errors="coerce")
    historial = historial.dropna(subset=["Requests/s"]).reset_index(drop=True)

    historial["segundos"] = historial["Timestamp"] - historial["Timestamp"].min()
    # Tasa de error: fallos entre peticiones, en porcentaje (0 si no hubo peticiones)
    historial["tasa_error"] = (
        (historial["Failures/s"] / historial["Requests/s"] * 100)
        .where(historial["Requests/s"] > 0, 0)
    )
    return historial


def _graficar_serie(historial, columnas, etiquetas, titulo, ylabel, ruta):
    # Línea(s) contra el tiempo y, en el eje derecho, los usuarios activos
    fig, ax = plt.subplots(figsize=(10, 5))
    for columna, etiqueta in zip(columnas, etiquetas, strict=True):
        ax.plot(historial["segundos"], historial[columna], label=etiqueta)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)

    ax_usuarios = ax.twinx()
    ax_usuarios.step(
        historial["segundos"],
        historial["User Count"],
        where="post",
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Usuarios",
    )
    ax_usuarios.set_ylabel("Usuarios")

    lineas = ax.get_lines() + ax_usuarios.get_lines()
    ax.legend(lineas, [linea.get_label() for linea in lineas], loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def graficar_latencia(historial, ruta):
    # Latencia mediana y percentil 95 durante la prueba
    _graficar_serie(
        historial,
        ["50%", "95%"],
        ["Mediana", "Percentil 95"],
        "Latencia de /predict",
        "Latencia (ms)",
        ruta,
    )


def graficar_peticiones_por_segundo(historial, ruta):
    # Peticiones por segundo atendidas durante la prueba
    _graficar_serie(
        historial,
        ["Requests/s"],
        ["Peticiones por segundo"],
        "Peticiones por segundo de /predict",
        "Peticiones por segundo",
        ruta,
    )


def graficar_tasa_error(historial, ruta):
    # Porcentaje de peticiones fallidas durante la prueba
    _graficar_serie(
        historial,
        ["tasa_error"],
        ["Tasa de error"],
        "Tasa de error de /predict",
        "Fallos (%)",
        ruta,
    )


def main():
    parser = argparse.ArgumentParser(description="Gráficas de la prueba de carga de Locust")
    parser.add_argument("--historial", type=Path, default=RUTA_HISTORIAL)
    parser.add_argument("--salida", type=Path, default=DIR_LOCUST)
    args = parser.parse_args()

    historial = leer_historial(args.historial)
    args.salida.mkdir(parents=True, exist_ok=True)

    graficar_latencia(historial, args.salida / "latencia.png")
    graficar_peticiones_por_segundo(historial, args.salida / "peticiones_por_segundo.png")
    graficar_tasa_error(historial, args.salida / "tasa_error.png")
    print(f"Figuras guardadas en {args.salida}")


if __name__ == "__main__":
    main()