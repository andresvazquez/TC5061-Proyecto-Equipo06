# TC5061 Proyecto Equipo06

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto integrador MiBici: regresión, clasificación y clustering

## Propósito del proyecto

El propósito de este proyecto es desarrollar una solución basada en técnicas de Machine Learning para analizar los patrones de uso del sistema de bicicletas públicas MiBici, ubicado en el Área Metropolitana de Guadalajara, con el objetivo de generar información que apoye la toma de decisiones relacionadas con la operación, distribución de bicicletas y planeación del servicio.

A partir del análisis de datos históricos de viajes y de la información de las estaciones, se busca estimar la demanda futura de bicicletas, identificar posibles problemas de disponibilidad y reconocer estaciones que presentan comportamientos de uso similares. Esto permitirá explorar oportunidades para mejorar la asignación de bicicletas, anticipar situaciones que afecten la disponibilidad del servicio y apoyar la planeación de futuras ampliaciones del sistema.

Adicionalmente, el proyecto busca aplicar buenas prácticas de MLOps mediante la construcción de un flujo de trabajo reproducible que incluya la ingesta y preparación de datos, la generación de variables, el entrenamiento y evaluación de modelos, así como el registro y seguimiento de experimentos utilizando MLflow.

## Preguntas del proyecto

El proyecto aborda tres preguntas mediante diferentes técnicas de Machine Learning:

- **P1 - Regresión:** ¿Cuántos viajes se iniciarán en una estación durante la siguiente hora? Se busca anticipar la demanda para mejorar la distribución de bicicletas.

- **P2 - Clasificación:** ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles durante la siguiente hora? Se busca identificar estaciones que requieran rebalanceo.

- **P3 - Clustering:** ¿Qué estaciones presentan patrones de uso similares y dónde se ubican? Se busca agrupar estaciones para apoyar la planeación y expansión del sistema.

## Datos utilizados

Se utilizan datos históricos del portal de [Datos Abiertos de MiBici](https://mibici.net/es/datos-abiertos/), correspondientes al periodo de **junio de 2023 a septiembre de 2026**.

La base de datos comprende:

- **40 meses** de registros históricos.
- **15,011,969 viajes** registrados.
- **484 estaciones** en la nomenclatura, de las cuales 473 aparecen en los viajes.

Se utilizan dos fuentes principales:

- **Registros de viajes:** archivos CSV mensuales con fechas, horarios y estaciones de origen y destino.
- **Nomenclatura de estaciones:** archivo `nomenclatura_2026_09.csv`, con identificadores, nombres, ubicaciones y capacidades.

Los datos se procesan mediante Python y se almacenan en formato Parquet. Se excluye la información personal de los usuarios y se utiliza un manifiesto con hashes SHA-256 para garantizar la trazabilidad y reproducibilidad de los datos.

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
│   └── modeling/       
│       ├── __init__.py
│       ├── predict.py  <- Código para inferencia de modelos
│       ├── tracking.py <- Clase RunMLflow para trazabilidad centralizada de experimentos
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

Para configurar el entorno de desarrollo y ejecutar los componentes del proyecto desde un clon limpio del repositorio, sigue estos pasos:

### Pre-requisitos
- **Python 3.10** o superior instalado en el sistema.
- **Git** instalado.

---

### 1. Clonar el repositorio

Clona el repositorio en tu máquina local y posicionate dentro de la carpeta del proyecto:

```bash
git clone [https://github.com/andresvazquez/TC5061-Proyecto-Equipo06.git](https://github.com/andresvazquez/TC5061-Proyecto-Equipo06.git)
cd TC5061-Proyecto-Equipo06
```
Crea un entorno virtual
En windows
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En Linux/MacOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Para instalar dependencias y el paquete local
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
O si tienes Make
```bash
make requirements
```
Y para verificar la instalación:
```bash
python -c "import mibici; print(mibici.__name__)"
```

### Variables de entorno

La configuración local se puede ajustar con un archivo `.env` en la raíz del proyecto. Para crearlo a partir de la plantilla:

```bash
cp .env.example .env
```

En PowerShell se puede usar `Copy-Item .env.example .env`. El archivo `.env` está incluido en `.gitignore` por lo que no se sube a Git. Si no existe, el proyecto usa los valores por defecto de `mibici/config.py`. No agregues credenciales reales a `.env.example`.

La variable principal es `MLFLOW_TRACKING_URI`, que define la dirección del servidor de tracking (por defecto `http://localhost:5000`). Se puede sobrescribir desde `.env` o desde el entorno antes de ejecutar los comandos.

## Comandos

### Ingesta de datos

Para bajar y preparar los datos hay que tener el entorno virtual activo y las dependencias instaladas (se explica en Instalación). Después, desde la raíz del proyecto, se corre:

```
python mibici/dataset.py
```

Otra opción es `make data`, que primero instala lo que está en requirements.txt y luego corre lo mismo.

No hay que hacer nada a mano. El script toma el rango de meses de mibici/config.py (FECHA_INICIO y FECHA_FIN), baja los archivos de viajes y la nomenclatura de estaciones a data/raw, limpia cada mes (nombres de columnas, codificación y fechas, y quita los datos personales) y guarda el resultado en Parquet en la carpeta data/processed.

Al final crea references/data_manifest.csv con la URL, el número de registros y el hash de cada archivo. Ese manifiesto sí se sube a Git y los datos no. La versión de los datos es el hash del manifiesto, se imprime al terminar y se puede sacar otra vez sin descargar nada con calcular_hash(config.MANIFEST_PATH) de dataset.py. Esa es la versión que se puede registrar en MLflow.

Son más de 1 GB, así que la primera vez tarda varios minutos. Si un archivo ya está en data/raw no se vuelve a descargar.

#### Ingesta con mlflow run

La ingesta también se puede lanzar desde el MLflow Project. Con el entorno virtual activo y desde la raíz del proyecto, se corre:

```
mlflow run . -e ingest
```

El entry point ingest corre dataset.py, así que no recibe parámetros: el rango de meses y las rutas salen de config.py. Al terminar imprime la versión de los datos y deja el manifiesto en references/data_manifest.csv, igual que arriba.

Por defecto mlflow run crea un entorno aparte con python_env.yaml y para eso necesita tener pyenv instalado. Si no se tiene, se puede usar el entorno que ya está instalado:

```
mlflow run . -e ingest --env-manager local
```

Si MLFLOW_TRACKING_URI apunta al servidor local, hay que levantarlo antes (se explica en Servidor de tracking). Sin esa variable, mlflow run guarda el run en un mlflow.db local y no necesita servidor.

#### Entrenamiento con mlflow run

El entry point train recibe la pregunta, el modelo y sus hiperparámetros. Con el servidor de tracking levantado (se explica en Servidor de tracking), el entorno virtual activo y desde la raíz del proyecto, un ejemplo para P2 es:

```
mlflow run . -e train -P question=p2 -P model=rf -P n_estimators=50 -P max_depth=10 --experiment-name mibici-equipo06 --env-manager local
```

El flag --experiment-name es necesario: mlflow run crea su run en el experimento Default y los scripts de entrenamiento registran en mibici-equipo06, así que sin el flag MLflow rechaza la corrida porque el experimento activo no coincide. La ingesta no lo necesita porque dataset.py no usa MLflow. Y --env-manager local se usa por la misma razón que en la ingesta, si no se tiene pyenv.

Cada mlflow run lanza un solo modelo (en P2 por ejemplo: baseline, rf, lr, rf-search o rf-features-search). Los hiperparámetros que no se indican con -P toman el valor por defecto del MLproject.

#### Reproducir el modelo elegido de cada pregunta

Con el servidor de tracking levantado, MLFLOW_TRACKING_URI apuntando a él, el entorno virtual activo y desde la raíz del proyecto, estos comandos reproducen el modelo elegido de cada pregunta. Llevan --experiment-name mibici-equipo06 y --env-manager local por lo explicado arriba.

P1 (demanda), HistGradientBoosting con 100 iteraciones:

```
mlflow run . -e train -P question=p1 -P model=hgb -P max_iter=100 --experiment-name mibici-equipo06 --env-manager local
```

P2 (riesgo), Random Forest con variables temporales, 100 árboles y profundidad máxima de 15. El comando lanza la búsqueda de cuatro combinaciones (una corrida padre y cuatro hijas) y el modelo elegido es la hija de 100 árboles y profundidad máxima de 15:

```
mlflow run . -e train -P question=p2 -P model=rf-features-search --experiment-name mibici-equipo06 --env-manager local
```

P3 (grupos), KMeans con tres clusters y 10 inicializaciones:

```
mlflow run . -e train -P question=p3 -P model=kmeans -P n_clusters=3 --experiment-name mibici-equipo06 --env-manager local
```

El baseline de cada pregunta se lanza con -P model=baseline en P1 y P2, y en P3 con el mismo comando sin n_clusters, que toma 4 por defecto.

### Servidor de tracking

El Model Registry necesita un almacenamiento respaldado por una base de datos, así que el tracking se hace contra un servidor local con SQLite. Cada integrante levanta el suyo en su máquina, con el entorno virtual activo y desde la raíz del proyecto, en una terminal aparte que se deja abierta:

```
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Esto crea mlflow.db en la raíz (está en .gitignore, no se sube a Git) y deja la interfaz en http://localhost:5000. Esa es la dirección que mibici/config.py toma por defecto en TRACKING_URI, así que no hay que configurar nada más. Si se necesita otra, se cambia con la variable MLFLOW_TRACKING_URI (o en el .env):

```
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Experimento base

Con los datos ya descargados (sección anterior), el entorno virtual activo y desde la raíz del proyecto, primero se levanta MLflow en otra terminal:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

La dirección sale de mibici/config.py (http://localhost:5000) y se puede cambiar con la variable MLFLOW_TRACKING_URI. Con MLflow arriba se corren los tres baselines, uno por pregunta, en este orden:

```
python -m mibici.modeling.train_p3
python -m mibici.modeling.train_p1
python -m mibici.modeling.train_p2
```

P3 tarda unos segundos, P1 alrededor de medio minuto y P2 unos minutos. No hay que hacer nada a mano ni abrir notebooks: cada script lee los datos, construye las variables con mibici/features.py, entrena y registra todo en MLflow.

Al terminar, en el experimento mibici-equipo06 quedan 5 runs: Baseline_P1_Lag168 y Candidato_HistGradientBoosting (P1), p2-baseline-clase-mas-frecuente y p2-random-forest (P2) y p3-baseline-kmeans-volumen (P3). Cada uno registra:

- Parámetros para reproducirlo: rango de fechas, versión de los datos, semilla, modelo e hiperparámetros.
- Métricas con el conjunto en el nombre, como val_mae, test_recall o train_silhouette.
- Etiquetas con el equipo, la pregunta (P1, P2 o P3) y el commit de Git con que se ejecutó.
- Artefactos: las gráficas de P1, las curvas precision-recall de P2 y el modelo KMeans con el resumen de grupos de P3.

Con la versión de datos 723b4c4628046549dce640dabba90372588f9ea49fc882fecee42a5e1afedd7a se espera, en P1, un val_mae de 0.7738 en el baseline y de 0.6348 en el candidato; en P2, un recall de validación de 0.8853 en el random forest (el baseline no detecta riesgo); y en P3, una silueta de 0.5985 y un Davies-Bouldin de 0.5064.

### Registro del modelo elegido

Con el servidor de tracking arriba, el modelo elegido de cada pregunta se registra en el Model Registry con mibici/modeling/registry.py. Se le pasa la pregunta y el ID de la corrida ganadora, que se ve en la interfaz de MLflow:

```
python -m mibici.modeling.registry --question p1 --run-id <run_id>
```

Registra el modelo con su descripción, su linaje (corrida, experimento, commit de Git y versión de los datos) y el alias champion. En el registro quedan mibici-equipo06-p1-demanda, mibici-equipo06-p2-riesgo y mibici-equipo06-p3-grupos, y cada uno se carga con models:/<nombre>@champion.

En P2 la corrida también guarda el umbral de las alertas de riesgo (parámetro umbral) y las métricas con ese umbral, con el sufijo _umbral, junto con las falsas alarmas. El umbral es el más alto con el que el recall de validación todavía llega a RECALL_MINIMO de mibici/config.py (0.85), para reducir las falsas alarmas sin dejar de detectar las estaciones en riesgo. El modelo registrado predice con 0.5, así que para usar el umbral se aplica predict_proba >= umbral, y el valor se guarda como etiqueta umbral en la versión registrada.

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

En el proyecto se aplicaron las siguientes buenas prácticas de ingeniería y MLOps:

### Estructura y código
- **Arquitectura modular:** El código fuente se organiza como un paquete ejecutable (`mibici/`) instalado en modo editable (`pip install -e .`), permitiendo importaciones limpias entre módulos.
- **Configuración centralizada:** Todas las rutas, variables globales y ventanas de división temporal (`INICIO_VALIDACION`, `INICIO_PRUEBA`) se gestionan exclusivamente en `mibici/config.py`.
- **Reproducibilidad y limpieza:** Los experimentos se ejecutan mediante invocación de módulos (`python -m ...`).

### Datos y Git
- **Versionado de datos con manifiesto:** Los datasets brutos y procesados no se rastrean en Git por su tamaño. Su versionado se garantiza mediante `references/data_manifest.csv`, que registra las URLs, el conteo de registros y el hash SHA-256 de cada archivo.
- **Estrategia de branching y PRs:** La rama `main` está protegida. Todo cambio requiere una rama propia, seguir la convención de [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/), una revisión aprobada por otro integrante del equipo y fusión mediante *Squash Merge*.
- **Aislamiento de entorno:** Uso de `.gitignore` para excluir datos pesados (`data/raw/`, `data/processed/`), artefactos locales (`mlruns/`, `mlartifacts/`) y entornos virtuales (`.venv/`).

### MLflow Tracking
- **Seguimiento centralizado:** Se utiliza una clase común `RunMLflow` (`mibici/modeling/tracking.py`) que estandariza la creación de experimentos, registro de hiperparámetros, métricas de desempeño y guardado de artefactos.
- **Trazabilidad y artefactos:** Cada corrida registra parámetros del modelo, métricas de validación y prueba, artefactos visuales y la serialización del modelo dentro del servidor local de MLflow (`mlflow.db`).
- **Esquema de Gestión del Ciclo de Vida del Modelo:** Para la promoción a producción de los modelos del proyecto, el equipo acordó utilizar el **esquema vigente de Aliases (`@champion`)** en lugar del esquema tradicional basado en etapas (*Stages: Staging / Production*).

#### Justificación de la Elección del Esquema de Aliases:
1. **Alineación con Estándares Modernos de MLflow:** A partir de MLflow 2.0+, el sistema clásico de *Stages* ha sido marcado como en desuso (*deprecated*). MLflow recomienda la adopción de *Aliases* para la gobernanza de modelos.
2. **Desacoplamiento y Flexibilidad:** Los *Aliases* permiten asignar identificadores mutables (como `@champion` para el modelo en servicio) a versiones específicas e inmutables del registro (`Version 1`), facilitando despliegues continuos sin alterar el historial de versiones.
3. **Carga Programática Directa:** Permite que los scripts de inferencia o servicios de producción consuman la versión vigente del modelo mediante una URI estable: `models:/mibici-equipo06-p1-demanda@champion`.

#### Verificación de Carga Programática del Modelo Champion:
Se validó la capacidad de resolver y cargar el modelo *Champion* directamente desde el registro mediante el servidor local:

  ```python
  import os
  import mlflow

  # Incrementar tiempo de espera HTTP y definir Tracking URI
  os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "300"
  mlflow.set_tracking_uri("http://localhost:5000")

  # Carga del modelo usando el esquema de alias
  champion_uri = "models:/mibici-equipo06-p1-demanda@champion"
  loaded_model = mlflow.sklearn.load_model(champion_uri)

  print("Carga exitosa del modelo Champion:", type(loaded_model))
  ```
  
Nota: "/mibici-equipo06-p1-demanda@champion" cambia de acuerdo con el modelo que quieras verificar

## Integrantes y roles

### Andrés Eduardo Vázquez Pérez

Responsable del comando que integra el proceso de ingesta, el versionado de datos, las variables de P2 relacionadas con inventario y flujos recientes, el baseline de clasificación P2 y su notebook exploratorio. También participa en la protección de la rama principal y en la documentación de los comandos de ingesta y del experimento base.

### Juan Pablo Castillo Mendoza

Responsable de la configuración central del proyecto en `config.py`, la descarga de los datos mensuales y la nomenclatura de estaciones, la construcción de la etiqueta de riesgo de P2 y el notebook de exploración de datos. También participa en la documentación del propósito, preguntas y datos utilizados, así como en la validación final utilizando un clon limpio del repositorio.

### Christian Ulises Ramírez García

Responsable del almacenamiento de los datos procesados en Parquet, el feature engineering de P1, las funciones de evaluación de los modelos, el baseline de regresión P1 y su notebook exploratorio. También participa en la configuración de `.gitignore`, documentación de la estructura e instalación y buenas prácticas del repositorio.

### Sergio Alejandro Canto Arizpe

Responsable de la normalización de los archivos mensuales, el feature engineering del perfil de uso de estaciones para P3, el componente común de tracking con MLflow, el baseline de clustering P3 y el notebook exploratorio de P3. También participa en la documentación del flujo Git/GitHub, los roles del equipo y la captura de evidencias finales de MLflow.
