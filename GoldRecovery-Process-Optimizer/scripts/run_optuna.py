from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from data.preprocess import preprocess


def load_training_data(
    cfg_path: str = "configs/config.yaml",
) -> tuple[pd.DataFrame, pd.Series, Dict]:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    train_csv = cfg["data"]["train_csv"]
    df = preprocess([train_csv])
    target_col = cfg["training"]["target"]
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y, cfg


def objective(trial: optuna.Trial) -> float:
    X, y, cfg = load_training_data()

    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "random_state": cfg["training"]["random_state"],
        "n_jobs": -1,
    }

    dtrain = xgb.DMatrix(X, label=y)

    # Simple train/validation split
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(cfg["training"].get("random_state", 42))
    rng.shuffle(idx)
    split = int(n * (1 - cfg["training"].get("test_size", 0.2)))
    train_idx, val_idx = idx[:split], idx[split:]

    dtrain_split = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
    dval_split = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])

    evals_result: Dict[str, Any] = {}
    booster = xgb.train(
        params,
        dtrain_split,
        num_boost_round=params["n_estimators"],
        evals=[(dval_split, "val")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    # Usar RMSE como objetivo a minimizar
    rmse_vals = evals_result["val"]["rmse"]
    best_rmse = float(min(rmse_vals))
    return best_rmse


def main() -> None:
    study = optuna.create_study(direction="minimize", study_name="goldrecovery_xgb_hpo")
    study.optimize(objective, n_trials=25)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "optuna_xgb_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
            },
            f,
            indent=2,
        )

    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
