"""Preprocesamiento de datos de clima y duración de viajes.

Este script se puede usar como módulo y como CLI:

    python data/preprocess.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml


def load_config(path: Path) -> Dict:
    """Carga la configuración YAML."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logging(level: str = "INFO") -> None:
    """Configura logging básico a consola."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_raw_weather_data(csv_path: Path) -> pd.DataFrame:
    """Carga el dataset crudo de clima y duración."""
    if not csv_path.exists():
        msg = f"No se encontró el archivo de datos en {csv_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(csv_path)


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Genera variables de tiempo y clima a partir del dataframe crudo."""

    data = df.copy()
    data["start_ts"] = pd.to_datetime(data["start_ts"])
    data["hour"] = data["start_ts"].dt.hour
    data["day_of_week"] = data["start_ts"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["weather_is_bad"] = (data["weather_conditions"] == "Bad").astype(int)

    # Filtramos valores no positivos
    data = data[data["duration_seconds"] > 0].copy()

    y = data["duration_seconds"].astype(float)
    feature_cols = ["hour", "day_of_week", "is_weekend", "weather_is_bad"]
    X = data[feature_cols].astype(float)

    return X, y, data


def load_and_preprocess(cfg: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Carga datos crudos y devuelve matrices de features y target.

    Además guarda un CSV procesado para trazabilidad.
    """

    raw_path = Path(cfg["paths"]["weather_data"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_weather_data(raw_path)
    X, y, df_features = engineer_features(df_raw)

    out_path = processed_dir / "trips_weather_features.csv"
    df_features.to_csv(out_path, index=False)
    logging.info("Datos procesados guardados en %s", out_path)

    return X, y


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Preprocesamiento de datos")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def cli_main() -> None:
    """Punto de entrada para el preprocesamiento vía CLI."""
    args = parse_args()
    cfg = load_config(Path(args.config))
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    X, y = load_and_preprocess(cfg)
    logging.info(
        "Dataset procesado con %d filas y %d características",
        X.shape[0],
        X.shape[1],
    )


if __name__ == "__main__":
    cli_main()
