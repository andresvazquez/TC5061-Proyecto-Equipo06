from argparse import Namespace
import subprocess
import sys

import pytest

from mibici.modeling import tracking
from mibici.modeling.tracking import obtener_git_commit
from mibici.modeling.train_dispatcher import (
    MODELOS_VALIDOS,
    construir_comando,
    parse_args,
    validar_modelo,
)


def _crear_args(question, model, **extra):
    """Argumentos del dispatcher con los valores por defecto del proyecto."""
    base = {
        "question": question,
        "model": model,
        "n_estimators": 50,
        "max_depth": 10,
        "max_iter": 100,
        "learning_rate": 0.1,
        "n_clusters": 4,
    }
    base.update(extra)
    return Namespace(**base)


# 1. Pruebas para validar_modelo
def test_validar_modelo_caso_normal_combinaciones_validas():
    """Caso Normal: Todas las combinaciones pregunta-modelo definidas son aceptadas."""
    for pregunta, modelos in MODELOS_VALIDOS.items():
        for modelo in modelos:
            assert validar_modelo(pregunta, modelo) is None


def test_validar_modelo_caso_limite_modelo_inexistente():
    """Caso Límite: Un modelo que no existe produce un ValueError."""
    with pytest.raises(ValueError, match="no valido para p1"):
        validar_modelo("p1", "xgboost")


def test_validar_modelo_caso_limite_modelo_de_otra_pregunta():
    """Caso Límite: Un modelo de P2 no es valido para P1 ni para P3."""
    validar_modelo("p2", "rf")

    for otra_pregunta in ("p1", "p3"):
        with pytest.raises(ValueError):
            validar_modelo(otra_pregunta, "rf")


def test_validar_modelo_caso_limite_mensaje_lista_las_opciones():
    """Caso Límite: El mensaje de error incluye las opciones validas ordenadas."""
    with pytest.raises(ValueError) as error:
        validar_modelo("p3", "nada")

    assert str(sorted(MODELOS_VALIDOS["p3"])) in str(error.value)


def test_validar_modelo_caso_limite_mayusculas_y_espacios():
    """Caso Límite: El nombre del modelo es sensible a mayusculas y espacios."""
    for modelo in ("", "HGB", " hgb"):
        with pytest.raises(ValueError):
            validar_modelo("p1", modelo)


# 2. Pruebas para construir_comando
def test_construir_comando_caso_normal_p1():
    """Caso Normal: P1 usa el trainer de P1 con max-iter y learning-rate."""
    comando = construir_comando(_crear_args("p1", "hgb", max_iter=250, learning_rate=0.05))

    assert comando == [
        sys.executable,
        "-m",
        "mibici.modeling.train_p1",
        "--model",
        "hgb",
        "--max-iter",
        "250",
        "--learning-rate",
        "0.05",
    ]


def test_construir_comando_caso_normal_p2():
    """Caso Normal: P2 usa el trainer de P2 con n-estimators, max-depth y max-iter."""
    comando = construir_comando(
        _crear_args("p2", "rf", n_estimators=200, max_depth=7, max_iter=30)
    )

    assert comando == [
        sys.executable,
        "-m",
        "mibici.modeling.train_p2",
        "--model",
        "rf",
        "--n-estimators",
        "200",
        "--max-depth",
        "7",
        "--max-iter",
        "30",
    ]


def test_construir_comando_caso_normal_p3_traduce_el_nombre_del_modelo():
    """Caso Normal: P3 traduce kmeans, agglomerative y kmeans-full al nombre del trainer."""
    esperados = {
        "kmeans": "baseline",
        "agglomerative": "agglomerative",
        "kmeans-full": "full-profile",
    }

    for modelo, nombre_trainer in esperados.items():
        comando = construir_comando(_crear_args("p3", modelo, n_clusters=6))

        assert comando[1:3] == ["-m", "mibici.modeling.train_p3"]
        assert comando[comando.index("--model") + 1] == nombre_trainer
        assert comando[comando.index("--n-clusters") + 1] == "6"


def test_construir_comando_caso_normal_p3_search_y_stability_sin_argumentos():
    """Caso Normal: kmeans-search y kmeans-stability usan su propio modulo sin argumentos."""
    esperados = {
        "kmeans-search": "mibici.modeling.train_p3_search",
        "kmeans-stability": "mibici.modeling.train_p3_stability",
    }

    for modelo, modulo in esperados.items():
        assert construir_comando(_crear_args("p3", modelo)) == [sys.executable, "-m", modulo]


def test_construir_comando_caso_normal_siempre_lista_de_strings():
    """Caso Normal: Todo comando empieza con el Python actual y contiene solo strings."""
    for pregunta, modelos in MODELOS_VALIDOS.items():
        for modelo in modelos:
            comando = construir_comando(_crear_args(pregunta, modelo))

            assert comando[0] == sys.executable
            assert comando[1] == "-m"
            assert all(isinstance(parte, str) for parte in comando)


def test_construir_comando_caso_limite_modelo_invalido():
    """Caso Límite: Un modelo que no corresponde a la pregunta produce un ValueError."""
    with pytest.raises(ValueError, match="no valido"):
        construir_comando(_crear_args("p1", "kmeans"))


def test_construir_comando_caso_limite_no_mezcla_argumentos_de_otras_preguntas():
    """Caso Límite: El comando de P1 no incluye argumentos de P2 ni de P3."""
    comando = construir_comando(_crear_args("p1", "baseline"))

    assert "--n-estimators" not in comando
    assert "--n-clusters" not in comando


# 3. Pruebas para parse_args
def test_parse_args_caso_normal_valores_por_defecto(monkeypatch):
    """Caso Normal: Sin argumentos opcionales usa los valores por defecto del proyecto."""
    monkeypatch.setattr(sys, "argv", ["train_dispatcher", "--question", "p2", "--model", "rf"])

    args = parse_args()

    assert (args.question, args.model) == ("p2", "rf")
    assert args.n_estimators == 50
    assert args.max_depth == 10
    assert args.max_iter == 100
    assert args.learning_rate == pytest.approx(0.1)
    assert args.n_clusters == 4


def test_parse_args_caso_limite_pregunta_desconocida(monkeypatch):
    """Caso Límite: Una pregunta que no es p1, p2 ni p3 termina el programa con SystemExit."""
    monkeypatch.setattr(sys, "argv", ["train_dispatcher", "--question", "p9", "--model", "x"])

    with pytest.raises(SystemExit):
        parse_args()


# 4. Pruebas para obtener_git_commit
def test_obtener_git_commit_caso_normal_devuelve_hash_sin_espacios(monkeypatch):
    """Caso Normal: Devuelve el hash que imprime git, sin espacios ni saltos de linea."""
    monkeypatch.setattr(tracking.shutil, "which", lambda _nombre: "/usr/bin/git")
    llamadas = []

    def _git_falso(comando, **kwargs):
        llamadas.append((comando, kwargs))
        return subprocess.CompletedProcess(comando, 0, stdout="  a1b2c3d\n", stderr="")

    monkeypatch.setattr(tracking.subprocess, "run", _git_falso)

    assert obtener_git_commit() == "a1b2c3d"

    comando, kwargs = llamadas[0]
    assert comando == ["/usr/bin/git", "rev-parse", "--short", "HEAD"]
    assert kwargs.get("check") is True
    assert "shell" not in kwargs


def test_obtener_git_commit_caso_normal_sin_simular_git():
    """Caso Normal: Con git real (o sin repo) siempre devuelve un texto no vacio."""
    resultado = obtener_git_commit()

    assert isinstance(resultado, str)
    assert resultado == resultado.strip()
    assert resultado != ""


def test_obtener_git_commit_caso_limite_git_no_instalado(monkeypatch):
    """Caso Límite: Si git no esta instalado devuelve unknown."""
    monkeypatch.setattr(tracking.shutil, "which", lambda _nombre: None)

    assert obtener_git_commit() == "unknown"


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(128, ["git", "rev-parse"]),
        FileNotFoundError("git"),
    ],
    ids=["CalledProcessError", "FileNotFoundError"],
)
def test_obtener_git_commit_caso_limite_git_falla(monkeypatch, error):
    """Caso Límite: Si git falla (no es un repo o desaparece el ejecutable) devuelve unknown."""
    monkeypatch.setattr(tracking.shutil, "which", lambda _nombre: "/usr/bin/git")

    def _git_que_falla(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(tracking.subprocess, "run", _git_que_falla)

    assert obtener_git_commit() == "unknown"