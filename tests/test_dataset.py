from pathlib import Path

import pandas as pd
import pytest

from mibici import config, dataset
from mibici.dataset import (
    calcular_hash,
    descargar_mes,
    descargar_nomenclatura,
    generar_manifiesto,
    guardar_parquet,
    listar_enlaces,
    normalizar_mes,
)

COLUMNAS_CRUDAS = [
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]


def _crear_csv_viajes(ruta, filas, columnas=COLUMNAS_CRUDAS):
    pd.DataFrame(filas, columns=columnas).to_csv(ruta, index=False)


class _RespuestaFalsa:
    def __init__(self, contenido=b"", bloques=None):
        self.content = contenido
        self._bloques = bloques if bloques is not None else [contenido]

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        yield from self._bloques

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

# 1. Pruebas para calcular_hash
def test_calcular_hash_caso_normal(tmp_path):
    """Caso Normal: Calcula el hash SHA-256 de un archivo de texto estándar."""
    archivo_prueba = tmp_path / "test_normal.txt"
    archivo_prueba.write_text("Contenido de prueba para mibici")

    hash_res = calcular_hash(archivo_prueba)

    assert isinstance(hash_res, str)
    assert len(hash_res) == 64  # SHA-256 genera un string hexadecimal de 64 caracteres
    assert hash_res == "fc62d103cd3e48cef79abffbe0e4474a5b094d9dc5159c9dffe0d446bf673589"


def test_calcular_hash_caso_limite_vacio(tmp_path):
    """Caso Límite: Calcula el hash SHA-256 de un archivo de 0 bytes."""
    archivo_vacio = tmp_path / "vacio.txt"
    archivo_vacio.write_text("")

    hash_res = calcular_hash(archivo_vacio)

    assert isinstance(hash_res, str)
    assert len(hash_res) == 64
    assert hash_res == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# 2. Pruebas para guardar_parquet
def test_guardar_parquet_caso_normal(tmp_path, monkeypatch):
    """Caso Normal: Guarda un DataFrame en formato Parquet y verifica integridad."""
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)

    df_sample = pd.DataFrame({
        "estacion_id": [1, 2, 3],
        "demanda": [10, 20, 15]
    })
    ruta_origen = tmp_path / "datos_abiertos_2023_06.csv"
    ruta_origen.write_text("contenido de prueba")

    ruta_parquet, num_registros = guardar_parquet(df_sample, ruta_origen)

    assert num_registros == 3
    assert Path(ruta_parquet) == tmp_path / "datos_abiertos_2023_06.parquet"
    assert Path(ruta_parquet).exists()

    df_cargado = pd.read_parquet(ruta_parquet)
    assert len(df_cargado) == 3
    assert list(df_cargado.columns) == ["estacion_id", "demanda"]
    assert df_cargado["demanda"].tolist() == [10, 20, 15]


def test_guardar_parquet_caso_limite_vacio(tmp_path, monkeypatch):
    """Caso Límite: Guarda un DataFrame vacío con estructura de columnas definida."""
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)

    df_vacio = pd.DataFrame({
        "estacion_id": pd.Series(dtype="int64"),
        "demanda": pd.Series(dtype="int64")
    })
    ruta_origen = tmp_path / "datos_abiertos_2023_07.csv"
    ruta_origen.write_text("contenido de prueba")

    ruta_parquet, num_registros = guardar_parquet(df_vacio, ruta_origen)

    assert num_registros == 0
    assert Path(ruta_parquet) == tmp_path / "datos_abiertos_2023_07.parquet"
    assert Path(ruta_parquet).exists()

    df_cargado = pd.read_parquet(ruta_parquet)
    assert len(df_cargado) == 0
    assert list(df_cargado.columns) == ["estacion_id", "demanda"]

# 3. Pruebas para normalizar_mes
def test_normalizar_mes_caso_normal(tmp_path):
    """Caso Normal: Normaliza un CSV mensual y elimina los datos personales."""
    ruta_csv = tmp_path / "datos_abiertos_2024_01.csv"
    _crear_csv_viajes(ruta_csv, [
        ["1001", "5001", "M", "1990", "2024-01-01 08:00:00", "2024-01-01 08:15:00", "12", "15"],
        ["1002", "5002", "F", "1985", "2024-01-01 08:30:00", "2024-01-01 08:45:00", "15", "20"],
    ])

    df_normalizado = normalizar_mes(ruta_csv)

    assert len(df_normalizado) == 2
    assert set(df_normalizado.columns) == {
        "viaje_id",
        "inicio_del_viaje",
        "fin_del_viaje",
        "origen_id",
        "destino_id",
    }
    assert "usuario_id" not in df_normalizado.columns
    assert "genero" not in df_normalizado.columns
    assert "ano_de_nacimiento" not in df_normalizado.columns
    assert pd.api.types.is_datetime64_any_dtype(df_normalizado["inicio_del_viaje"])
    assert pd.api.types.is_datetime64_any_dtype(df_normalizado["fin_del_viaje"])
    assert df_normalizado["inicio_del_viaje"].iloc[0] == pd.Timestamp("2024-01-01 08:00:00")


def test_normalizar_mes_caso_normal_formato_dd_mm(tmp_path):
    """Caso Normal: Detecta y normaliza fechas en formato dd/mm/aaaa h:mm."""
    ruta_csv = tmp_path / "datos_abiertos_2024_09.csv"
    _crear_csv_viajes(ruta_csv, [
        ["2001", "6001", "M", "1992", "01/09/2024 0:00", "01/09/2024 0:07", "154", "161"],
        ["2002", "6002", "F", "1998", "01/09/2024 0:05", "01/09/2024 0:12", "161", "154"],
    ])

    df_normalizado = normalizar_mes(ruta_csv)

    assert len(df_normalizado) == 2
    assert pd.api.types.is_datetime64_any_dtype(df_normalizado["inicio_del_viaje"])
    assert df_normalizado["inicio_del_viaje"].iloc[0] == pd.Timestamp("2024-09-01 00:00:00")


def test_normalizar_mes_caso_limite_fecha_invalida(tmp_path):
    """Caso Límite: Una fecha inválida se convierte en NaT sin romper el procesamiento."""
    ruta_csv = tmp_path / "datos_abiertos_2024_10.csv"
    _crear_csv_viajes(ruta_csv, [
        ["4001", "8001", "M", "1990", "01/10/2024 08:00", "01/10/2024 08:15", "12", "15"],
        ["4002", "8002", "F", "1995", "fecha-invalida", "01/10/2024 08:45", "15", "20"],
    ])

    df_normalizado = normalizar_mes(ruta_csv)

    assert len(df_normalizado) == 2
    assert pd.api.types.is_datetime64_any_dtype(df_normalizado["inicio_del_viaje"])
    assert df_normalizado["inicio_del_viaje"].iloc[0] == pd.Timestamp("2024-10-01 08:00:00")
    assert pd.isna(df_normalizado["inicio_del_viaje"].iloc[1])


def test_normalizar_mes_caso_limite_columnas_faltantes(tmp_path):
    """Caso Límite: Un CSV sin columnas requeridas produce un ValueError."""
    ruta_csv = tmp_path / "datos_abiertos_incompleto.csv"
    columnas = [col for col in COLUMNAS_CRUDAS if col != "Destino_Id"]
    _crear_csv_viajes(
        ruta_csv,
        [["3001", "7001", "M", "1991", "2024-01-01 08:00:00", "2024-01-01 08:15:00", "12"]],
        columnas,
    )

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        normalizar_mes(ruta_csv)

# 4. Pruebas para generar_manifiesto
def test_generar_manifiesto_caso_normal(tmp_path, monkeypatch):
    """Caso Normal: Genera un CSV de manifiesto con hashes e información relevante."""
    ruta_manifiesto = tmp_path / "data_manifest.csv"
    monkeypatch.setattr(config, "MANIFEST_PATH", ruta_manifiesto)

    f1 = tmp_path / "datos1.parquet"
    f2 = tmp_path / "datos2.parquet"
    f1.write_text("datos simulados 1")
    f2.write_text("datos simulados 2")

    urls = ["http://ejemplo.com/1", "http://ejemplo.com/2"] #no se descarga nada del portal, solo se simula la url
    registros = [10, 20]

    version = generar_manifiesto([f1, f2], urls, registros)

    assert ruta_manifiesto.exists()
    df_manifiesto = pd.read_csv(ruta_manifiesto)

    assert len(df_manifiesto) == 2
    assert list(df_manifiesto.columns) == ["url", "registros", "hash"]
    assert df_manifiesto["url"].tolist() == urls
    assert df_manifiesto["registros"].tolist() == registros
    assert df_manifiesto["hash"].tolist() == [calcular_hash(f1), calcular_hash(f2)]
    assert version == calcular_hash(ruta_manifiesto)


def test_generar_manifiesto_caso_limite_archivo_vacio(tmp_path, monkeypatch):
    """Caso Límite: Registra un archivo de 0 bytes con el hash SHA-256 estándar."""
    ruta_manifiesto = tmp_path / "data_manifest.csv"
    monkeypatch.setattr(config, "MANIFEST_PATH", ruta_manifiesto)

    f1 = tmp_path / "vacio.parquet"
    f1.write_text("")

    version = generar_manifiesto([f1], ["http://ejemplo.com/vacio"], [0]) #no se descarga nada del portal, solo se simula la url

    df_manifiesto = pd.read_csv(ruta_manifiesto)

    assert len(df_manifiesto) == 1
    assert df_manifiesto["registros"].tolist() == [0]
    assert df_manifiesto["hash"].tolist() == [
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ]
    assert version == calcular_hash(ruta_manifiesto)

# 5. Pruebas para listar_enlaces y descargas (sin red)
def test_listar_enlaces_caso_normal(monkeypatch):
    """Caso Normal: Extrae y ordena los enlaces CSV dentro del rango solicitado."""
    monkeypatch.setattr(config, "PORTAL_URL", "https://mibici.test/")
    html = b"""
    <html><body>
        <a href="datos_abiertos_2024_01.csv">enero</a>
        <a href="datos_abiertos_2024_02.csv">febrero</a>
        <a href="datos_abiertos_2023_12.csv">fuera del rango</a>
        <a href="otro_archivo.csv">no coincide</a>
    </body></html>
    """
    monkeypatch.setattr(
        dataset.requests,
        "get",
        lambda *args, **kwargs: _RespuestaFalsa(html),
    )

    enlaces = listar_enlaces("2024-01", "2024-03")

    assert enlaces == [
        "https://mibici.test/datos_abiertos_2024_01.csv",
        "https://mibici.test/datos_abiertos_2024_02.csv",
    ]


def test_listar_enlaces_caso_limite_rango_invalido(monkeypatch):
    """Caso Límite: Un rango invertido falla antes de consultar el portal."""
    monkeypatch.setattr(
        dataset.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("No debe consultar el portal"),
    )

    with pytest.raises(ValueError, match="fecha inicial"):
        listar_enlaces("2024-05", "2024-01")


def test_descargar_mes_caso_normal(tmp_path, monkeypatch):
    """Caso Normal: Descarga un CSV por bloques y lo guarda sin dejar temporales."""
    contenido = b"Viaje_Id,Usuario_Id\n1,2\n"
    llamadas = []

    def fake_get(url, **kwargs):
        llamadas.append((url, kwargs))
        return _RespuestaFalsa(bloques=[contenido[:10], contenido[10:]])

    monkeypatch.setattr(dataset.requests, "get", fake_get)
    url = "https://mibici.test/datos_abiertos_2024_01.csv"

    ruta = descargar_mes(url, tmp_path)

    assert ruta == tmp_path / "datos_abiertos_2024_01.csv"
    assert ruta.read_bytes() == contenido
    assert not (tmp_path / "datos_abiertos_2024_01.csv.part").exists()
    assert llamadas[0][0] == url
    assert llamadas[0][1]["stream"] is True


def test_descargar_mes_caso_limite_nombre_invalido(tmp_path):
    """Caso Límite: Un nombre que no cumple el patrón se rechaza antes de descargar."""
    with pytest.raises(ValueError, match="Nombre de archivo inesperado"):
        descargar_mes("https://mibici.test/archivo.csv", tmp_path)


def test_descargar_mes_caso_limite_archivo_existente(tmp_path, monkeypatch):
    """Caso Límite: Un archivo ya descargado se reutiliza sin volver a pedirlo."""
    ruta_existente = tmp_path / "datos_abiertos_2024_01.csv"
    ruta_existente.write_text("ya descargado")
    monkeypatch.setattr(
        dataset.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("No debe descargar un archivo existente"),
    )

    ruta = descargar_mes("https://mibici.test/datos_abiertos_2024_01.csv", tmp_path)

    assert ruta == ruta_existente
    assert ruta.read_text() == "ya descargado"


def test_descargar_nomenclatura_caso_normal(tmp_path, monkeypatch):
    """Caso Normal: Descarga el catálogo de estaciones en la carpeta destino."""
    contenido = b"id,estacion\n1,Centro\n"
    monkeypatch.setattr(
        config,
        "NOMENCLATURA_URL",
        "https://mibici.test/nomenclatura_2026_09.csv",
    )
    monkeypatch.setattr(
        dataset.requests,
        "get",
        lambda *args, **kwargs: _RespuestaFalsa(bloques=[contenido]),
    )

    ruta = descargar_nomenclatura(tmp_path)

    assert ruta == tmp_path / "nomenclatura_2026_09.csv"
    assert ruta.read_bytes() == contenido
    assert not (tmp_path / "nomenclatura_2026_09.csv.part").exists()


def test_descargar_nomenclatura_caso_limite_archivo_existente(tmp_path, monkeypatch):
    """Caso Límite: Un catálogo ya descargado se reutiliza sin volver a pedirlo."""
    ruta_existente = tmp_path / "nomenclatura_2026_09.csv"
    ruta_existente.write_text("ya descargado")
    monkeypatch.setattr(
        config,
        "NOMENCLATURA_URL",
        "https://mibici.test/nomenclatura_2026_09.csv",
    )
    monkeypatch.setattr(
        dataset.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("No debe descargar un archivo existente"),
    )

    ruta = descargar_nomenclatura(tmp_path)

    assert ruta == ruta_existente
    assert ruta.read_text() == "ya descargado"
