#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ] || { [ "$1" != "docker" ] && [ "$1" != "podman" ]; }; then
  echo "Uso: $0 {docker|podman}" >&2
  exit 1
fi

comando="$1"
archivo="reports/locust/recursos.csv"
contenedores=(mibici-api mibici-mlflow)

mkdir -p "$(dirname "$archivo")"

if [ ! -f "$archivo" ]; then
  printf 'hora,contenedor,CPUPerc,MemUsage\n' > "$archivo"
fi

trap 'echo "Muestreo de recursos detenido"; exit 0' INT TERM

while true; do
  hora=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  "$comando" stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' "${contenedores[@]}" | while IFS= read -r linea; do
    printf '%s,%s\n' "$hora" "$linea"
  done >> "$archivo" || true
  sleep 2
done
