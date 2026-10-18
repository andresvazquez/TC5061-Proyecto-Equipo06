import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Manifiesto de datos (registros y hash de cada archivo): va en references/ y sí se sube a Git
MANIFEST_PATH = PROJ_ROOT / "references" / "data_manifest.csv"

# Rango de meses de la ingesta (AAAA-MM)
FECHA_INICIO = "2023-06"
FECHA_FIN = "2026-09"
# MLflow: la URI se puede cambiar con la variable MLFLOW_TRACKING_URI (o en el .env)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENTO = "mibici-equipo06"

# Semilla de los modelos
SEMILLA = 42

# Dónde empiezan la validación y la prueba (antes de la validación es entrenamiento)
INICIO_VALIDACION = "2026-04-01"
INICIO_PRUEBA = "2026-07-01"

# Fuente de datos de MiBici
PORTAL_URL = "https://mibici.net/es/datos-abiertos/"

# Catalogo de estaciones
NOMENCLATURA_URL = (
    "https://mibici.net/site/assets/files/1118/"
    "nomenclatura_2026_09.csv"
)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
