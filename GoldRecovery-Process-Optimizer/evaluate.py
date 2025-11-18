#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from main import (
    MetallurgicalPredictor,
    ProcessDataLoader,
    symmetric_mean_absolute_percentage_error,
)
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def bootstrap_mae(
    y_true: np.ndarray, y_pred: np.ndarray, n_iter: int = 200, seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    maes = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        maes.append(mean_absolute_error(y_true[idx], y_pred[idx]))
    return float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))


def evaluate(
    test_paths: List[str], model_path: str, target: str, bootstrap_iters: int = 200
) -> Dict[str, Any]:
    loader = ProcessDataLoader()
    predictor = MetallurgicalPredictor()
    predictor.load_models(model_path)

    df = loader.load_process_data(test_paths)
    df = loader.validate_and_clean_data(df)

    X, y = predictor.prepare_features(df, target)
    y_pred = predictor.predict(X)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = float(np.sqrt(mse))
    smape = symmetric_mean_absolute_percentage_error(y.values, y_pred)

    # Baseline
    dummy = DummyRegressor(strategy="mean").fit(X, y)
    y_dummy = dummy.predict(X)
    mae_dummy = mean_absolute_error(y, y_dummy)

    # Bootstrap CI
    ci_low, ci_high = bootstrap_mae(y.values, y_pred, n_iter=bootstrap_iters)

    metrics = {
        "mae": float(mae),
        "mae_ci95": [ci_low, ci_high],
        "mse": float(mse),
        "rmse": rmse,
        "smape": float(smape),
        "mae_dummy": float(mae_dummy),
    }
    return metrics


def cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate metallurgical model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="models/metallurgical_model.pkl")
    parser.add_argument("--input", type=str, nargs="+", default=None)
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    test_paths = args.input or [
        cfg.get("data", {}).get("test_csv", "gold_recovery_test.csv")
    ]
    target = cfg.get("training", {}).get("target", "final.output.recovery")
    bootstrap_iters = cfg.get("evaluation", {}).get("bootstrap_iters", 200)

    metrics = evaluate(
        test_paths=test_paths,
        model_path=args.model,
        target=target,
        bootstrap_iters=bootstrap_iters,
    )

    results_dir = Path(cfg.get("paths", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "metrics_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(results_dir / "metrics_eval.csv", index=False)

    print("\n=== MÉTRICAS (EVALUATE) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    cli()
