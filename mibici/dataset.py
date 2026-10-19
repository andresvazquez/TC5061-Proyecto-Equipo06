from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pandas as pd
import requests

from mibici import config  # config.py: FECHA_INICIO, FECHA_FIN, MANIFEST_PATH


# Descargar los meses elegidos y la nomenclatura de estaciones
def listar_enlaces(inicio, fin):
    """
    Obtiene las URLs de los CSV mensuales de MiBici.

    Args:
        inicio (str): Primer mes en formato AAAA-MM.
        fin (str): Ultimo mes en formato AAAA-MM.

    Returns:
        list[str]: URLs encontradas, ordenadas por fecha.

    Advierte si faltan meses dentro del rango solicitado.
    """

    portal = config.PORTAL_URL

    # 1. Validar el rango de fechas
    fecha_inicio = datetime.strptime(inicio, "%Y-%m")
    fecha_fin = datetime.strptime(fin, "%Y-%m")

    if fecha_inicio > fecha_fin:
        raise ValueError(
            "La fecha inicial debe ser anterior a la final"
        )

    # 2. Consultar el portal de MiBici
    headers = {
        "User-Agent": "MiBici-Academic-Research/1.0",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "es-MX,es;q=0.9",
    }

    respuesta = requests.get(
        portal,
        headers=headers,
        timeout=30
    )
    respuesta.raise_for_status()

    # 3. Analizar el HTML
    soup = BeautifulSoup(respuesta.content, "html.parser")

    patron = re.compile(
        r"datos_abiertos_(\d{4})_(\d{2})\.csv",
        re.IGNORECASE
    )

    enlaces = {}

    for etiqueta in soup.select("a[href]"):
        url = urljoin(portal, etiqueta["href"])
        nombre = Path(urlparse(url).path).name

        coincidencia = patron.fullmatch(nombre)

        if coincidencia:
            anio, mes = coincidencia.groups()

            try:
                fecha = datetime(int(anio), int(mes), 1)
            except ValueError:
                continue

            if fecha_inicio <= fecha <= fecha_fin:
                enlaces[fecha] = url

    # 4. Generar todos los meses esperados
    meses_esperados = []

    anio = fecha_inicio.year
    mes = fecha_inicio.month

    while (anio, mes) <= (fecha_fin.year, fecha_fin.month):
        meses_esperados.append(datetime(anio, mes, 1))

        if mes == 12:
            anio += 1
            mes = 1
        else:
            mes += 1

    # 5. Identificar los meses faltantes
    meses_faltantes = [
        fecha.strftime("%Y-%m")
        for fecha in meses_esperados
        if fecha not in enlaces
    ]

    # 6. Mostrar resumen
    print(f"Meses esperados: {len(meses_esperados)}")
    print(f"Meses encontrados: {len(enlaces)}")

    if meses_faltantes:
        print(
            "ADVERTENCIA: No se encontraron enlaces "
            "para los siguientes meses:"
        )

        for mes in meses_faltantes:
            print(f"  - {mes}")
    else:
        print("Todos los meses solicitados fueron encontrados")

    # 7. Devolver URLs ordenadas
    resultado = [
        enlaces[fecha]
        for fecha in sorted(enlaces)
    ]

    if not resultado:
        raise ValueError(
            f"No se encontraron CSV entre {inicio} y {fin}"
        )

    return resultado


def descargar_mes(url, destino):
    """
    Descarga un CSV mensual de MiBici.

    Args:
        url (str): URL del archivo CSV.
        destino (Path): Carpeta donde se guardara.

    Returns:
        Path: Ruta local del CSV descargado.
    """

    # 1. Preparar la carpeta de destino
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    # 2. Obtener el nombre del archivo desde la URL
    nombre = Path(urlparse(url).path).name

    # 3. Validar el nombre del CSV
    patron = r"datos_abiertos_\d{4}_\d{2}\.csv"

    if not re.fullmatch(patron, nombre, re.IGNORECASE):
        raise ValueError(
            f"Nombre de archivo inesperado: {nombre}"
        )

    # 4. Construir la ruta local
    ruta = destino / nombre

    # 5. Evitar descargar archivos existentes
    if ruta.exists() and ruta.stat().st_size > 0:
        print(f"Archivo existente: {nombre}")
        return ruta

    # 6. Crear archivo temporal
    temporal = ruta.with_suffix(".csv.part")

    # 7. Encabezados HTTP
    headers = {
        "User-Agent": "MiBici-Academic-Research/1.0",
        "Accept": "text/csv,application/octet-stream,*/*",
        "Accept-Language": "es-MX,es;q=0.9",
    }

    try:
        # 8. Descargar el archivo por bloques
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(15, 120)
        ) as respuesta:

            respuesta.raise_for_status()

            with open(temporal, "wb") as archivo:
                for bloque in respuesta.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if bloque:
                        archivo.write(bloque)

        # 9. Verificar que el archivo no este vacio
        if temporal.stat().st_size == 0:
            raise ValueError(
                f"El archivo descargado esta vacio: {nombre}"
            )

        # 10. Renombrar el archivo temporal
        temporal.replace(ruta)

    except Exception:
        temporal.unlink(missing_ok=True)
        raise

    print(f"Descarga completada: {nombre}")

    # 11. Devolver la ruta del archivo
    return ruta


def descargar_nomenclatura(destino):
    """
    Descarga el catalogo de estaciones de MiBici.

    Args:
        destino (Path): Carpeta donde se guardara el CSV.

    Returns:
        Path: Ruta local del archivo descargado.
    """

    # 1. Obtener la URL desde config.py
    url = config.NOMENCLATURA_URL

    # 2. Preparar la carpeta de destino
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    # 3. Obtener el nombre y construir la ruta local
    nombre = Path(urlparse(url).path).name
    ruta = destino / nombre

    # 4. Evitar descargas repetidas
    if ruta.exists() and ruta.stat().st_size > 0:
        print(f"Archivo existente: {nombre}")
        return ruta

    # 5. Crear una ruta temporal
    temporal = ruta.with_suffix(".csv.part")

    # 6. Configurar los encabezados HTTP
    headers = {
        "User-Agent": "MiBici-Academic-Research/1.0",
        "Accept": "text/csv,application/octet-stream,*/*",
        "Accept-Language": "es-MX,es;q=0.9",
    }

    try:
        # 7. Descargar el archivo por bloques
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(15, 120)
        ) as respuesta:

            respuesta.raise_for_status()

            with open(temporal, "wb") as archivo:
                for bloque in respuesta.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if bloque:
                        archivo.write(bloque)

        # 8. Verificar que el archivo no este vacio
        if temporal.stat().st_size == 0:
            raise ValueError(
                "El archivo de nomenclatura esta vacio"
            )

        # 9. Renombrar el archivo temporal
        temporal.replace(ruta)

    except Exception:
        # Eliminar archivos incompletos
        temporal.unlink(missing_ok=True)
        raise

    # 10. Confirmar la descarga
    print(f"Nomenclatura descargada: {nombre}")

    return ruta

# Normalizar nombres de columnas, codificación de caracteres y tipos de fecha
def normalizar_mes(ruta_csv):
    """
    Normaliza un archivo mensual de viajes de MiBici.

    La funcion:
    - Lee archivos con codificacion UTF-8 o ISO-8859-1.
    - Normaliza los nombres de las columnas.
    - Convierte las columnas de fecha a datetime.
    - Elimina las variables personales que no se utilizaran en el proyecto.

    Args:
        ruta_csv (str | Path): Ruta al archivo CSV mensual.

    Returns:
        pd.DataFrame: Tabla normalizada sin datos personales.
    """
    ruta_csv = Path(ruta_csv)

    # 1. Verificar que el archivo exista
    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {ruta_csv}"
        )

    # 2. Intentar primero UTF-8 y usar ISO-8859-1 como respaldo
    try:
        tabla = pd.read_csv(
            ruta_csv,
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        tabla = pd.read_csv(
            ruta_csv,
            encoding="iso-8859-1"
        )

    # 3. Normalizar nombres de columnas
    tabla.columns = (
        tabla.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("á", "a", regex=False)
        .str.replace("é", "e", regex=False)
        .str.replace("í", "i", regex=False)
        .str.replace("ó", "o", regex=False)
        .str.replace("ú", "u", regex=False)
        .str.replace("ñ", "n", regex=False)
    )

    # 4. Columnas requeridas por el pipeline
    columnas_requeridas = {
        "viaje_id",
        "inicio_del_viaje",
        "fin_del_viaje",
        "origen_id",
        "destino_id",
    }

    faltantes = columnas_requeridas - set(tabla.columns)

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(sorted(faltantes))
        )

    # 5. Convertir las fechas
    # 2023-11, 2024-09 y 2024-10 vienen como dd/mm/aaaa h:mm
    formato = None
    if "/" in str(tabla["inicio_del_viaje"].iloc[0]):
        formato = "%d/%m/%Y %H:%M"

    tabla["inicio_del_viaje"] = pd.to_datetime(
        tabla["inicio_del_viaje"],
        format=formato,
        errors="coerce"
    )

    tabla["fin_del_viaje"] = pd.to_datetime(
        tabla["fin_del_viaje"],
        format=formato,
        errors="coerce"
    )

    # 6. Eliminar datos personales
    # Se borran por posicion (usuario, genero y año de nacimiento) porque en
    # 2024-02 y 2024-05 el encabezado del año viene corrupto
    tabla = tabla.drop(columns=tabla.columns[[1, 2, 3]])

    return tabla

def guardar_parquet(tabla: pd.DataFrame, ruta_csv: str) -> tuple[str, int]:
    # Extrae solo el nombre base del archivo sin extensión (ej: 'viajes_2023_01')
    nombre_base = os.path.splitext(os.path.basename(ruta_csv))[0]

    # Asegura que la carpeta processed/ exista
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # Construye la nueva ruta en data/processed/ con extensión .parquet
    ruta_parquet = os.path.join(config.PROCESSED_DATA_DIR, f"{nombre_base}.parquet")

    # Guarda el DataFrame en formato Parquet
    tabla.to_parquet(ruta_parquet, index=False)

    # Obtiene el número de filas/registros
    num_registros = len(tabla)

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

def normalizar_estaciones(ruta_csv, ruta_destino=None):
    """
    Normaliza el catálogo de estaciones de MiBici.
    """
    ruta_csv = Path(ruta_csv)

    # 1. Verificar que el archivo exista
    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {ruta_csv}"
        )

    # 2. Intentar primero UTF-8 y usar ISO-8859-1 como respaldo
    try:
        tabla = pd.read_csv(
            ruta_csv,
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        tabla = pd.read_csv(
            ruta_csv,
            encoding="iso-8859-1"
        )

    # 3. Normalizar nombres de columnas
    tabla.columns = (
        tabla.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("á", "a", regex=False)
        .str.replace("é", "e", regex=False)
        .str.replace("í", "i", regex=False)
        .str.replace("ó", "o", regex=False)
        .str.replace("ú", "u", regex=False)
        .str.replace("ñ", "n", regex=False)
    )

    mapeo_columnas = {
        "id": "origen_id",
        "name": "nombre",
        "latitude": "latitud",
        "longitude": "longitud",
        "dpcapacity": "capacidad",
    }
    tabla = tabla.rename(columns=mapeo_columnas)

    # 6. Definir la ruta de salida y guardar el CSV normalizado
    if ruta_destino is None:
        ruta_destino = ruta_csv.parent / f"{ruta_csv.stem}_clean.csv"
    else:
        ruta_destino = Path(ruta_destino)

    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta_destino, index=False, encoding="utf-8")
    print(f"Estaciones normalizadas guardadas en: {ruta_destino}")

    return tabla

# Proceso ejecutable desde la terminal: python mibici/dataset.py
def main():
    print("Iniciando la ingesta de MiBici")

    # El rango de meses y las rutas salen de config.py
    urls = listar_enlaces(config.FECHA_INICIO, config.FECHA_FIN)
    ruta_nomenclatura = descargar_nomenclatura(config.RAW_DATA_DIR)
    normalizar_estaciones(ruta_nomenclatura)

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
