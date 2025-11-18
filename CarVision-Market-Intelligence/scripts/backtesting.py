from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from data.preprocess import build_preprocessor, infer_feature_types
from sklearn.model_selection import TimeSeriesSplit


def time_order(df: pd.DataFrame, by: str = "model_year") -> pd.DataFrame:
    return df.sort_values(by=[by]).reset_index(drop=True)


essential_cols = [
    "price",
    "model_year",
]


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only essential columns + those used by model
    return df


def backtest_price_model(
    df: pd.DataFrame,
    target: str = "price",
    n_splits: int = 5,
    output_json: Path = Path("artifacts/backtesting.json"),
) -> Dict[str, float]:
    df = time_order(df, by="model_year")

    num_cols, cat_cols = infer_feature_types(df, target=target)
    pre = build_preprocessor(num_cols, cat_cols)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline

    model = RandomForestRegressor(random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics: List[Dict[str, float]] = []

    X = df.drop(columns=[target])
    y = df[target]

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, preds)))
        mae = float(mean_absolute_error(yte, preds))
        r2 = float(r2_score(yte, preds))
        metrics.append({"fold": fold, "rmse": rmse, "mae": mae, "r2": r2})

    summary = {
        "rmse_mean": float(np.mean([m["rmse"] for m in metrics])),
        "mae_mean": float(np.mean([m["mae"] for m in metrics])),
        "r2_mean": float(np.mean([m["r2"] for m in metrics])),
        "folds": metrics,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    data_path = Path("vehicles_us.csv")
    df = load_and_prepare(data_path)
    res = backtest_price_model(df)
    print(json.dumps(res, indent=2))
