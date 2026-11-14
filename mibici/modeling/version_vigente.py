from mibici import config

# Regla para decidir la versión vigente de P2 (la que queda con el alias champion).
# Se fija antes de correr el A/B testing y no se cambia después de ver los resultados.
DIFERENCIA_RECALL = 0.01  # mejora mínima de recall (un punto porcentual)
DIFERENCIA_FALSAS_ALARMAS = 0.05  # cambio mínimo de falsas alarmas (5 % del champion)


def decidir_version(champion, challenger):
    # Cada versión es un diccionario con "recall" y "falsas_alarmas" del mismo conjunto de evaluación
    diferencia_recall = round(challenger["recall"] - champion["recall"], 6)
    limite_sube = champion["falsas_alarmas"] * (1 + DIFERENCIA_FALSAS_ALARMAS)
    limite_baja = champion["falsas_alarmas"] * (1 - DIFERENCIA_FALSAS_ALARMAS)

    # P2 es una alerta de riesgo: una versión con menos recall que el mínimo no puede ser vigente
    if challenger["recall"] < config.RECALL_MINIMO:
        return "champion", f"el challenger no llega al recall mínimo de {config.RECALL_MINIMO}"

    # Mejora el recall sin subir de forma importante las falsas alarmas
    if diferencia_recall >= DIFERENCIA_RECALL and challenger["falsas_alarmas"] <= limite_sube:
        return "challenger", "el challenger mejora el recall sin subir de forma importante las falsas alarmas"

    # Empata en recall y baja de forma importante las falsas alarmas
    if abs(diferencia_recall) < DIFERENCIA_RECALL and challenger["falsas_alarmas"] <= limite_baja:
        return "challenger", "el challenger empata en recall y baja de forma importante las falsas alarmas"

    # En cualquier otro caso la diferencia es ruido y se queda el champion
    return "champion", "la diferencia no alcanza la mínima y se trata como ruido"
