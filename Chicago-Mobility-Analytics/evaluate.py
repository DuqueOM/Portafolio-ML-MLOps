"""Script de evaluación para el modelo de duración de viajes.

Ejemplo de uso:
    python evaluate.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from main import evaluate_model, load_config, setup_logging

from common_utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos para evaluación."""
    parser = argparse.ArgumentParser(description="Evaluación del modelo")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla opcional (CLI > SEED env > 42)",
    )
    return parser.parse_args()


def cli_main() -> None:
    """Punto de entrada CLI para evaluación."""
    args = parse_args()
    cfg = load_config(Path(args.config))
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    seed_used = set_seed(args.seed)

    try:
        metrics = evaluate_model(cfg, seed_used)
        print(json.dumps(metrics, indent=2))
    except Exception as exc:  # noqa: BLE001
        logging.exception("Error durante la evaluación: %s", exc)
        raise


if __name__ == "__main__":
    cli_main()
