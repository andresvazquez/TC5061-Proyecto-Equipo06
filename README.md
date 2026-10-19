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

## Estructura de carpetas

```
TC5061-Proyecto-Equipo06/
├── .venv/              <- Entorno virtual local (ignorado en Git)
├── data/
│   ├── external/       <- Datos de fuentes de terceros
│   ├── interim/        <- Datos intermedios transformados
│   ├── processed/      <- Datasets procesados finales para modelado (.parquet)
│   └── raw/            <- Copias locales de viajes y nomenclatura de estaciones
│
├── docs/               <- Documentación adicional del proyecto
├── mibici/             <- Paquete con el código fuente del proyecto
│   ├── __init__.py
│   ├── config.py       <- Variables globales, rutas e intervalos de fechas 
│   ├── dataset.py      <- Ingesta, limpieza, normalización y generación del manifiesto
│   ├── features.py     <- Creacion de funciones para entrenamiento de los 3 modelos
│   ├── tracking.py     <- Clase RunMLflow para trazabilidad centralizada de experimentos
│   └── modeling/       
│       ├── __init__.py
│       ├── predict.py  <- Código para inferencia de modelos
│       ├── train.py    <- Script base de entrenamiento (no usado)
│       ├── train_p1.py <- Script principal para entrenamiento, evaluación y registro en MLflow(P1)
│       ├── train_p2.py <- Script principal para entrenamiento, evaluación y registro en MLflow(P2)
│       └── train_p3.py <- Script principal para entrenamiento, evaluación y registro en MLflow(P3)
│
├── mlartifacts/        <- Almacenamiento local de artefactos de MLflow (modelos, gráficas)
├── mlruns/             <- Registro local de ejecuciones y experimentos de MLflow
├── models/             <- Modelos entrenados y serializados
├── notebooks/          <- Notebooks de exploración inicial y validación
├── references/         <- Manifiesto de datos (data_manifest.csv) y referencias técnicas
├── reports/            <- Reportes generados y gráficas/artefactos de evaluación
│   └── figures/
│
├── .gitignore          <- Reglas para ignorar datos pesados, entornos y carpetas locales
├── Makefile           <- Comandos automatizados para el ciclo de vida del proyecto
├── mlflow.db           <- Base de datos local para el tracking server de MLflow
├── pyproject.toml      <- Configuración del paquete mibici y herramientas del entorno
├── README.md           <- Guía principal de uso y documentación
└── requirements.txt    <- Dependencias del proyecto para la reproducción del entorno
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

Seguimos [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/) para los commits y una convención equivalente para los nombres de las ramas.

#### Commits

Formato: `tipo: descripción corta` o `tipo(alcance): descripción corta`

- La descripción va en español, en imperativo presente (por ejemplo, agrega o corrige), en minúsculas y sin punto final.
- Un commit por cambio con sentido propio.
- Tipos que usamos:
  - `feat`: función o característica nueva (por ejemplo, la descarga de los meses).
  - `fix`: corrección de un error.
  - `docs`: cambios en documentación (README, ML Canvas, diagrama).
  - `refactor`: reorganizar código sin cambiar lo que hace.
  - `chore`: tareas de mantenimiento o de armado (estructura, dependencias, `.gitignore`).

Ejemplos:

- `feat: descarga los meses y la nomenclatura de estaciones`
- `fix: corrige la lectura de archivos con codificación latin-1`
- `docs: agrega el ML Canvas a references`
- `chore: fija las versiones de las dependencias`

#### Ramas

Formato: `tipo/descripcion-corta`

- Mismos tipos que en los commits (`feat/`, `fix/`, `docs/`, etc.).
- En minúsculas, con guiones entre palabras, sin acentos, ñ ni espacios.
- Una rama por tarea, creada a partir de main.

Ejemplos:

- `feat/ingesta-mensual`
- `feat/manifiesto-datos`
- `docs/ml-canvas`
- `chore/gitignore`

### Pull Requests y revisiones

Todo cambio se integra a `main` mediante un Pull Request. El título del PR sigue la misma convención utilizada para los commits y debe describir de forma clara el cambio realizado.

Antes de fusionarse, cada Pull Request requiere al menos una aprobación de otro integrante. El revisor verifica que el cambio corresponda al alcance de la tarea, respete la estructura del proyecto y no introduzca modificaciones innecesarias en componentes asignados a otros integrantes.

Cuando una revisión identifica un ajuste necesario, el cambio se incorpora en la misma rama y el Pull Request se actualiza para una nueva validación. Una vez aprobado, el PR se integra mediante squash merge para conservar un solo commit representativo del cambio en `main`.

Cada integrante participa tanto como autor de Pull Requests como revisor de contribuciones de otro integrante, dejando evidencia de ambas actividades en GitHub.

## Buenas prácticas

### MLflow Tracking

Por completar.

### Estructura y código

Por completar.

### Datos y Git

Por completar.

## Integrantes y roles

### Andrés Eduardo Vázquez Pérez

Responsable del comando que integra el proceso de ingesta, el versionado de datos, las variables de P2 relacionadas con inventario y flujos recientes, el baseline de clasificación P2 y su notebook exploratorio. También participa en la protección de la rama principal y en la documentación de los comandos de ingesta y del experimento base.

### Juan Pablo Castillo Mendoza

Responsable de la configuración central del proyecto en `config.py`, la descarga de los datos mensuales y la nomenclatura de estaciones, la construcción de la etiqueta de riesgo de P2 y el notebook de exploración de datos. También participa en la documentación del propósito, preguntas y datos utilizados, así como en la validación final utilizando un clon limpio del repositorio.

### Christian Ulises Ramírez García

Responsable del almacenamiento de los datos procesados en Parquet, el feature engineering de P1, las funciones de evaluación de los modelos, el baseline de regresión P1 y su notebook exploratorio. También participa en la configuración de `.gitignore`, documentación de la estructura e instalación y buenas prácticas del repositorio.

### Sergio Alejandro Canto Arizpe

Responsable de la normalización de los archivos mensuales, el feature engineering del perfil de uso de estaciones para P3, el componente común de tracking con MLflow, el baseline de clustering P3 y el notebook exploratorio de P3. También participa en la documentación del flujo Git/GitHub, los roles del equipo y la captura de evidencias finales de MLflow.
