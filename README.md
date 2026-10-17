# TC5061 Proyecto Equipo06

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto integrador MiBici: regresión, clasificación y clustering

## Propósito del proyecto

Por completar.

## Las tres preguntas

Por completar.

## Datos utilizados

Por completar.

## Estructura de carpetas

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Instalación

Por completar.

## Comandos

### Ingesta de datos

Para bajar y preparar los datos hay que tener el entorno virtual activo y las dependencias instaladas (se explica en Instalación). Después, desde la raíz del proyecto, se corre:

```
python mibici/dataset.py
```

Otra opción es `make data`, que primero instala lo que está en requirements.txt y luego corre lo mismo.

No hay que hacer nada a mano. El script toma el rango de meses de mibici/config.py (FECHA_INICIO y FECHA_FIN), baja los archivos de viajes y la nomenclatura de estaciones a data/raw, limpia cada mes (nombres de columnas, codificación y fechas, y quita los datos personales) y guarda el resultado en Parquet junto al CSV.

Al final crea references/data_manifest.csv con la URL, el número de registros y el hash de cada archivo. Ese manifiesto sí se sube a Git y los datos no. La versión de los datos es el hash del manifiesto, se imprime al terminar y se puede sacar otra vez sin descargar nada con calcular_hash(config.MANIFEST_PATH) de dataset.py. Esa es la versión que se puede registrar en MLflow.

Son más de 1 GB, así que la primera vez tarda varios minutos. Si un archivo ya está en data/raw no se vuelve a descargar.

### Experimento base

Por completar.

## Flujo de trabajo con Git

### Rama principal protegida y Pull Requests

La rama main está protegida. Nadie sube directo a main, ni siquiera el dueño del repositorio. Todo cambio entra por un Pull Request y necesita al menos una aprobación de otra persona del equipo antes de fusionarse. Tampoco se permite force push ni borrar la rama.

Seguimos GitHub Flow: cada cambio se hace en una rama nueva que sale de main, se sube, se abre el Pull Request, se revisa y se fusiona a main.

Los Pull Requests se fusionan solo con squash merge, así que cada uno queda como un solo commit en main con el título del Pull Request como mensaje. Por eso el título sigue la convención de commits. Al fusionar, GitHub borra la rama automáticamente.

### Convención de ramas y commits

Por completar.

### Pull Requests y revisiones

Por completar.

## Buenas prácticas

### MLflow Tracking

Por completar.

### Estructura y código

Por completar.

### Datos y Git

Por completar.

## Integrantes y roles

Por completar.
