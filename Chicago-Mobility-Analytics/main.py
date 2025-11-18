#!/usr/bin/env python
"""CLI para entrenamiento, evaluación y predicción de duración de viajes en Chicago.

Uso básico:
    python main.py --mode train --config configs/default.yaml --seed 42
    python main.py --mode eval  --config configs/default.yaml
    python main.py --mode predict --config configs/default.yaml \
        --start_ts "2017-11-11 10:00:00" --weather_conditions Good
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from data.preprocess import load_and_preprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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


def train_model(cfg: Dict, seed: int) -> Dict:
    """Entrena el modelo y devuelve métricas en valid/test."""

    X, y = load_and_preprocess(cfg)

    test_size = cfg["training"]["test_size"]
    val_size = cfg["training"]["validation_size"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=seed
    )

    model_cfg = cfg["model"]["params"]
    model = RandomForestRegressor(random_state=seed, **model_cfg)
    model.fit(X_train, y_train)

    def eval_split(X_split, y_split) -> Dict[str, float]:
        preds = model.predict(X_split)
        mae = mean_absolute_error(y_split, preds)
        rmse = mean_squared_error(y_split, preds, squared=False)
        r2 = r2_score(y_split, preds)
        return {"mae": mae, "rmse": rmse, "r2": r2}

    metrics = {
        "val": eval_split(X_val, y_val),
        "test": eval_split(X_test, y_test),
    }

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "duration_model.pkl")
    # Export combined model pack for demo usage
    try:
        joblib.dump(
            {"model": model, "version": "1.0.0"}, models_dir / "model_v1.0.0.pkl"
        )
    except Exception:
        pass

    # Persist metrics for downstream logging
    try:
        art = Path("artifacts")
        art.mkdir(parents=True, exist_ok=True)
        with (art / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

    logging.info("Modelo entrenado y guardado en %s", models_dir)
    logging.info("Métricas valid: %s", metrics["val"])
    logging.info("Métricas test:  %s", metrics["test"])

    return metrics


def evaluate_model(cfg: Dict, seed: int) -> Dict:
    """Carga el modelo entrenado y evalúa sobre el conjunto completo."""

    models_dir = Path(cfg["paths"]["models_dir"])
    model_path = models_dir / "duration_model.pkl"
    if not model_path.exists():
        msg = f"No se encontró el modelo en {model_path}"
        raise FileNotFoundError(msg)

    model: RandomForestRegressor = joblib.load(model_path)

    X, y = load_and_preprocess(cfg)
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    logging.info("Evaluación global: %s", metrics)

    return metrics


def predict_single(cfg: Dict, start_ts: str, weather_conditions: str) -> float:
    """Predice la duración para un solo viaje."""

    models_dir = Path(cfg["paths"]["models_dir"])
    model_path = models_dir / "duration_model.pkl"
    if not model_path.exists():
        msg = "Modelo no entrenado. Ejecuta --mode train primero."
        raise FileNotFoundError(msg)

    model: RandomForestRegressor = joblib.load(model_path)

    # Extraer características temporales a partir del timestamp
    ts = pd.to_datetime(start_ts)
    hour = ts.hour
    day_of_week = ts.dayofweek
    is_weekend = int(day_of_week in (5, 6))
    weather_is_bad = int(weather_conditions == "Bad")

    X = np.array([[hour, day_of_week, is_weekend, weather_is_bad]], dtype=float)
    pred = float(model.predict(X)[0])

    logging.info(
        "Predicción para %s (%s): %.2f segundos",
        start_ts,
        weather_conditions,
        pred,
    )

    return pred


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Chicago Mobility ML pipeline")
    parser.add_argument("--mode", choices=["train", "eval", "predict"], required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_ts", type=str, help="Timestamp para predicción")
    parser.add_argument(
        "--weather_conditions",
        type=str,
        choices=["Good", "Bad"],
        help="Condición climática para predicción",
    )
    return parser.parse_args()


def cli_main() -> None:
    """Punto de entrada principal del CLI."""

    args = parse_args()
    cfg = load_config(Path(args.config))
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    try:
        if args.mode == "train":
            metrics = train_model(cfg, args.seed)
            print(json.dumps(metrics, indent=2))
        elif args.mode == "eval":
            metrics = evaluate_model(cfg, args.seed)
            print(json.dumps(metrics, indent=2))
        elif args.mode == "predict":
            if not args.start_ts or not args.weather_conditions:
                msg = (
                    "Debe proporcionar --start_ts y --weather_conditions para "
                    "predict."
                )
                raise ValueError(msg)
            pred = predict_single(cfg, args.start_ts, args.weather_conditions)
            print(json.dumps({"duration_seconds": pred}, indent=2))
    except Exception as exc:  # noqa: BLE001
        logging.exception("Error en la ejecución: %s", exc)
        raise


if __name__ == "__main__":
    cli_main()
