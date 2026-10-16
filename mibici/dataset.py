import hashlib
import os

import pandas as pd

from mibici import config  # config.py: FECHA_INICIO, FECHA_FIN, MANIFEST_PATH


# Descargar los meses elegidos y la nomenclatura de estaciones
def listar_enlaces(inicio, fin):
    # Recibe el rango de meses (AAAA-MM) y devuelve la lista de URL de los CSV del portal
    raise NotImplementedError("pendiente de implementar")


def descargar_mes(url, destino):
    # Baja un CSV a la carpeta destino y devuelve la ruta del archivo
    raise NotImplementedError("pendiente de implementar")


def descargar_nomenclatura(destino):
    # Baja la nomenclatura de estaciones y devuelve la ruta del archivo
    raise NotImplementedError("pendiente de implementar")


# Normalizar nombres de columnas, codificación de caracteres y tipos de fecha
def normalizar_mes(ruta_csv):
    # Recibe la ruta de un CSV crudo y devuelve la tabla normalizada, sin datos personales
    raise NotImplementedError("pendiente de implementar")


def guardar_parquet(tabla: pd.DataFrame, ruta_csv: str) -> tuple[str, int]:
    # Define la ruta de salida .parquet
    ruta_parquet = os.path.splitext(ruta_csv)[0] + ".parquet"

    # Guarda el DataFrame en formato Parquet
    tabla.to_parquet(ruta_parquet, index=False)

    #Obtiene el número de filas/registros
    num_registros = len(tabla)

    #Devuelve la ruta y el número de registros según la especificación
    return ruta_parquet, num_registros


# Versionar los datos con el mecanismo propuesto
def calcular_hash(ruta):
    # Hash SHA-256 del archivo, abierto en modo binario
    with open(ruta, "rb") as archivo:
        digest = hashlib.file_digest(archivo, "sha256")
    return digest.hexdigest()


def generar_manifiesto(rutas_csv, urls, registros):
    # Una fila por archivo descargado con su URL, el número de registros y el hash
    manifiesto = pd.DataFrame(
        {
            "url": urls,
            "registros": registros,
            "hash": [calcular_hash(ruta) for ruta in rutas_csv],
        }
    )
    manifiesto.to_csv(config.MANIFEST_PATH, index=False)

    # La versión de los datos es el hash del propio manifiesto
    return calcular_hash(config.MANIFEST_PATH)


# Proceso ejecutable desde la terminal: python mibici/dataset.py
def main():
    print("Iniciando la ingesta de MiBici")

    # El rango de meses y las rutas salen de config.py
    urls = listar_enlaces(config.FECHA_INICIO, config.FECHA_FIN)
    descargar_nomenclatura(config.RAW_DATA_DIR)

    # Por cada mes: descargar, normalizar y guardar en Parquet
    rutas_csv = []
    registros = []
    for url in urls:
        ruta_csv = descargar_mes(url, config.RAW_DATA_DIR)
        tabla = normalizar_mes(ruta_csv)
        ruta_parquet, n_registros = guardar_parquet(tabla, ruta_csv)
        print("Guardado:", ruta_parquet)
        rutas_csv.append(ruta_csv)
        registros.append(n_registros)

    version = generar_manifiesto(rutas_csv, urls, registros)
    print("Ingesta completa, versión de los datos:", version)


if __name__ == "__main__":
    main()
