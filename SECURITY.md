# Política de seguridad y manejo de credenciales
Este documento describe las reglas de seguridad para el proyecto con el fin de evitar
la exposición de credenciales y secretos en el repositorio. Aplica a todo el código,
notebooks, documentación, archivos de configuración y artefactos que se versionan en Git.

## Reglas para el equipo
- No versionar contraseñas, tokens, claves de API, claves privadas ni cadenas de
  conexión con credenciales.
- Usar variables de entorno o un archivo `.env` local para cualquier valor
  sensible. `.env` y `.env.*` están excluidos por `.gitignore`.
- Los archivos de ejemplo, si se agregan, deben contener únicamente nombres de
  variables y valores ficticios. La regla implementada con `!.env.example` permite versionarlos.
- Revisar antes de cada commit para confirmar que no se incluyan secretos.
- No incluir credenciales en notebooks, celdas de salida, capturas de pantalla,
  mensajes de error ni descripciones de Pull Requests.
- Mantener las credenciales fuera de `config.py`; ese módulo solo debe leer
  variables de entorno y definir valores por defecto que no sean sensibles.

## Variables del proyecto
La configuración central vive en `mibici/config.py`. La única variable de
entorno utilizada actualmente es:

- `MLFLOW_TRACKING_URI`: dirección del servidor de tracking de MLflow. Su valor
  por defecto es `http://localhost:5000` y no es un secreto.

Si en el futuro se usa un servidor remoto de MLflow con autenticación, sus
credenciales deben proporcionarse mediante variables de entorno o `.env`, nunca
directamente en el repositorio.

## Revisión de credenciales
Para asegurar este punto se realizó una revisión del árbol de trabajo y del historial de Git. Estos son los comandos ejecutados para la prueba y los resultados que se obtuvieron:

```bash
# Archivos versionados con nombres sensibles
git ls-files | grep -Ei '(^|/)(\.env|mlflow\.db|mlruns|mlartifacts|data/)|credential|secret|token|\.pem|\.p12|id_rsa'
# No hay coincidencias, esto nos indica que esta correcto.

# Contenido con asignaciones o formatos de credenciales
git grep -nEI "(password|passwd|secret|token|api[_-]?key|private[_-]?key)[[:space:]]*[:=][[:space:]]*[\"']?[^\"'[:space:]]{6,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}|-----BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY-----" -- .
# No hay coincidencias, esto nos indica que esta correcto.

# Archivos sensibles en el historial
git log --all --format='%h %s' -- .env '.env.*' mlflow.db mlruns mlartifacts
# No se encontraron commits de estos archivos por lo que tambien esta correcto
```

También se verificó el comportamiento de `.gitignore`:
| Ruta | Resultado |
| --- | --- |
| `.env` | Ignorado |
| `.env.local` | Ignorado por `.env.*` |
| `.env.example` | No ignorado, es permitido explícitamente |
| `mlflow.db` | Ignorado |
| `mlruns/` | Ignorado |
| `mlartifacts/` | Ignorado |
| `data/` | Ignorado |

`mlflow.db` existe en entornos locales, pero no está versionado. Los
datos, artefactos de MLflow y entornos virtuales tampoco se versionan.

## Procedimiento ante una exposición
Si se detecta una credencial en el repositorio o en su historial necesitamos seguir los siguientes pasos:
1. Revocar o rotar la credencial de inmediato.
2. Eliminar el valor del archivo afectado y agregarlo a `.env` o a las variables
   de entorno del sistema.
3. Coordinar con el equipo antes de reescribir el historial de Git. La limpieza
   del historial requiere `git filter-repo` y un `push --force` coordinado.
4. Documentar el incidente y la rotación sin incluir el valor expuesto.