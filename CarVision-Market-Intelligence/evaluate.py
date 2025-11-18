"""
Model evaluation script for CarVision Market Intelligence.
Usage:
  python evaluate.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from data.preprocess import clean_data, infer_feature_types, load_data, split_data
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)


def evaluate_model(cfg: Dict) -> Dict:
    paths = cfg["paths"]
    tr = cfg["training"]
    prep = cfg["preprocessing"]

    df = clean_data(load_data(paths["data_path"]))

    num_cols, cat_cols = infer_feature_types(
        df,
        target=tr["target"],
        numeric_features=prep.get("numeric_features") or None,
        categorical_features=prep.get("categorical_features") or None,
        drop_columns=prep.get("drop_columns") or None,
    )

    feature_cols = num_cols + cat_cols

    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_data(
        df, tr["target"], tr["test_size"], tr["val_size"], cfg["seed"], tr["shuffle"]
    )

    model = joblib.load(paths["model_path"])
    y_pred = model.predict(X_test)

    metrics = {
        "rmse": rmse(y_test, y_pred),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mape": mape(y_test, y_pred),
        "r2": float(r2_score(y_test, y_pred)),
    }

    # Baseline dummy median
    dummy = DummyRegressor(strategy="median")
    dummy.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    yb = dummy.predict(X_test)
    baseline = {
        "rmse": rmse(y_test, yb),
        "mae": float(mean_absolute_error(y_test, yb)),
        "mape": mape(y_test, yb),
        "r2": float(r2_score(y_test, yb)),
    }

    artifacts_dir = Path(paths["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(paths["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)
    with open(paths["baseline_metrics_path"], "w") as f:
        json.dump(baseline, f, indent=2)

    # Bootstrap significance test (model vs baseline)
    boot_cfg = (cfg.get("evaluation") or {}).get("bootstrap") or {}
    if boot_cfg.get("enabled", False):
        n = int(boot_cfg.get("n_resamples", 200))
        rng = np.random.default_rng(int(boot_cfg.get("random_state", 42)))
        idx = np.arange(len(y_test))
        deltas: list[float] = []
        for _ in range(n):
            bs_idx = rng.choice(idx, size=len(idx), replace=True)
            y_true_bs = np.array(y_test)[bs_idx]
            y_model_bs = np.array(y_pred)[bs_idx]
            y_base_bs = np.array(yb)[bs_idx]
            rmse_model = rmse(y_true_bs, y_model_bs)
            rmse_base = rmse(y_true_bs, y_base_bs)
            deltas.append(rmse_model - rmse_base)

        deltas_arr = np.array(deltas, dtype=float)
        ci_low, ci_high = np.percentile(deltas_arr, [2.5, 97.5])
        # two-sided p-value: proportion of bootstrap deltas > 0 (model worse) or < 0
        p_value = 2 * min(
            float(np.mean(deltas_arr > 0)),
            float(np.mean(deltas_arr < 0)),
        )
        bootstrap = {
            "delta_rmse_mean": float(deltas_arr.mean()),
            "delta_rmse_ci95": [float(ci_low), float(ci_high)],
            "p_value_two_sided": float(p_value),
        }
        with open(artifacts_dir / "metrics_bootstrap.json", "w") as f:
            json.dump(bootstrap, f, indent=2)
    else:
        bootstrap = None

    # Temporal backtesting basado en model_year (si está disponible)
    metrics_temporal = None
    error_by_segment_path = artifacts_dir / "error_by_segment.csv"
    if "model_year" in df.columns:
        df_sorted = df.sort_values("model_year")
        eval_cfg = (cfg.get("evaluation") or {}).get("temporal", {})
        temporal_size = float(eval_cfg.get("test_size", tr["test_size"]))
        n_total = len(df_sorted)
        n_temporal = max(1, int(n_total * temporal_size))
        df_temporal = df_sorted.tail(n_temporal)

        X_temp = df_temporal[feature_cols]
        y_temp = df_temporal[tr["target"]]
        y_temp_pred = model.predict(X_temp)

        metrics_temporal = {
            "rmse": rmse(y_temp, y_temp_pred),
            "mae": float(mean_absolute_error(y_temp, y_temp_pred)),
            "mape": mape(y_temp, y_temp_pred),
            "r2": float(r2_score(y_temp, y_temp_pred)),
            "n_samples": int(len(df_temporal)),
        }

        with open(artifacts_dir / "metrics_temporal.json", "w") as f:
            json.dump(metrics_temporal, f, indent=2)

        # Error por segmento en el backtest temporal
        segment_rows = []
        segment_cols = ["condition", "type", "model_year"]
        for col in segment_cols:
            if col not in df_temporal.columns:
                continue
            for value, group in df_temporal.groupby(col):
                if len(group) < 30:
                    continue
                X_seg = group[feature_cols]
                y_seg = group[tr["target"]]
                y_seg_pred = model.predict(X_seg)
                row = {
                    "segment_col": col,
                    "segment_value": str(value),
                    "n_samples": int(len(group)),
                    "rmse": rmse(y_seg, y_seg_pred),
                    "mae": float(mean_absolute_error(y_seg, y_seg_pred)),
                    "mape": mape(y_seg, y_seg_pred),
                    "r2": float(r2_score(y_seg, y_seg_pred)),
                }
                segment_rows.append(row)

        if segment_rows:
            pd.DataFrame(segment_rows).to_csv(error_by_segment_path, index=False)

    return {
        "model": metrics,
        "baseline": baseline,
        "bootstrap": bootstrap,
        "temporal": metrics_temporal,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results = evaluate_model(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
