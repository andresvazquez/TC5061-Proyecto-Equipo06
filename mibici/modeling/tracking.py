from pathlib import Path
import subprocess

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from mibici import config


def obtener_git_commit():
    """
    Obtiene el hash corto del commit Git actual.

    Permite asociar cada run de MLflow con la version exacta
    del codigo utilizada para ejecutarlo.

    Returns:
        str: Hash corto del commit actual o "unknown" si no
        puede obtenerse.
    """
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return resultado.stdout.strip()

    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def configurar_mlflow():
    """
    Configura el servidor de tracking y el experimento comun del equipo.
    """
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENTO)

class RunMLflow:
    """
    Centraliza el registro de experimentos del proyecto en MLflow.

    Registra informacion comun necesaria para reproducir una corrida:
    parametros, metricas, artefactos y etiquetas de trazabilidad.

    La clase no calcula metricas ni entrena modelos. Recibe esos
    resultados desde los scripts de P1, P2 o P3.
    """


    def __init__(
        self,
        pregunta,
        nombre_run,
        version_datos,
        fecha_inicio=None,
        fecha_fin=None,
        nested=False,
        model_family=None,
        stage_of_work=None,
        ):
        self.pregunta = pregunta
        self.nombre_run = nombre_run
        self.version_datos = version_datos
        self.nested = nested
        self.model_family = model_family
        self.stage_of_work = stage_of_work

        self.fecha_inicio = (
            fecha_inicio
            if fecha_inicio is not None
            else config.FECHA_INICIO
        )

        self.fecha_fin = (
            fecha_fin
            if fecha_fin is not None
            else config.FECHA_FIN
        )

        self.run = None


    def iniciar(self):
        """
        Inicia un run y registra parametros y etiquetas comunes.
        """
        configurar_mlflow()

        self.run = mlflow.start_run(
        run_name=self.nombre_run,
        nested=self.nested,  # NUEVO
        )

        mlflow.log_params({
            "fecha_inicio": self.fecha_inicio,
            "fecha_fin": self.fecha_fin,
            "version_datos": self.version_datos,
        })

        mlflow.set_tags({
            "equipo": "06",
            "pregunta": self.pregunta,
            "git_commit": obtener_git_commit(),
            "model_family": self.model_family,
            "stage_of_work": self.stage_of_work,
        })

        return self

    def registrar_parametros(self, parametros):
        """
        Registra parametros especificos de una corrida.

        Args:
            parametros (dict): Parametros del modelo o experimento.
        """
        mlflow.log_params(parametros)

    def registrar_metricas(self, metricas):
        """
        Registra metricas calculadas externamente.

        Args:
            metricas (dict): Metricas de evaluacion del modelo.
        """
        mlflow.log_metrics(metricas)

    def registrar_artefacto(self, ruta):
        """
        Registra un archivo como artefacto del run.

        Args:
            ruta (str | Path): Ruta al archivo que sera registrado.
        """
        ruta = Path(ruta)

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontro el artefacto: {ruta}"
            )

        mlflow.log_artifact(str(ruta))

    def log_model(self, model, X_sample, artifact_path="model"):
        """
        Registra el modelo con firma (signature) y un ejemplo de entrada (input_example) [Fila 11a].
        """
        # Inferir la firma (Signature)
        predictions = model.predict(X_sample)
        signature = infer_signature(X_sample, predictions)

        # Muestra de entrada (Input Example)
        input_example = X_sample.head(5) if hasattr(X_sample, "head") else X_sample[:5]

        # Tipos de confianza necesarios para HistGradientBoosting
        trusted_types = [
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
        ]

        # Registrar el modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_path,  # Se usa 'name' para evitar el warning de 'artifact_path'
            signature=signature,
            input_example=input_example,
            skops_trusted_types=trusted_types,
        )

    def finalizar(self):
        """
        Finaliza el run activo de MLflow.
        """
        if self.run is not None:
            mlflow.end_run()
            self.run = None

    def __enter__(self):
        return self.iniciar()

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.finalizar()

        return False
