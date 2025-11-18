"""Script de evaluación para el modelo de duración de viajes.

Ejemplo de uso:
    python evaluate.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from main import evaluate_model, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos para evaluación."""
    parser = argparse.ArgumentParser(description="Evaluación del modelo")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cli_main() -> None:
    """Punto de entrada CLI para evaluación."""
    args = parse_args()
    cfg = load_config(Path(args.config))
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    try:
        metrics = evaluate_model(cfg, args.seed)
        print(json.dumps(metrics, indent=2))
    except Exception as exc:  # noqa: BLE001
        logging.exception("Error durante la evaluación: %s", exc)
        raise


if __name__ == "__main__":
    cli_main()
