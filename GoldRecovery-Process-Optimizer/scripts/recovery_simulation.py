from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from main import symmetric_mean_absolute_percentage_error as smape


def bootstrap_ci(
    values: np.ndarray, n: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(n):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(float(np.mean(sample)))
    lower = float(np.percentile(stats, 100 * (alpha / 2)))
    upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lower, upper


def run_simulation(model_path: Path, csv_path: Path, target_col: str) -> dict:
    model_data = joblib.load(model_path)
    models = (
        model_data["models"]
        if isinstance(model_data, dict) and "models" in model_data
        else None
    )

    df = pd.read_csv(csv_path)
    y = df[target_col].values
    X = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in [target_col] if c in df.columns], errors="ignore"
    )
    X = X.fillna(X.median())

    if models:
        preds = np.mean([m.predict(X) for m in models.values()], axis=0)
    else:
        # fallback single model
        model = joblib.load(model_path)
        preds = model.predict(X)

    errs = np.abs(y - preds)
    mape = float(np.mean(errs / (np.abs(y) + 1e-6))) * 100
    smape_val = float(smape(y, preds))
    ci_low, ci_high = bootstrap_ci(errs)

    return {
        "mape": mape,
        "smape": smape_val,
        "error_ci_mean_low": ci_low,
        "error_ci_mean_high": ci_high,
        "n_samples": int(len(y)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Recovery simulation and CI report")
    ap.add_argument("--model", default="models/model_v1.0.0.pkl")
    ap.add_argument("--csv", default="gold_recovery_test.csv")
    ap.add_argument("--target", default="final.output.recovery")
    ap.add_argument("--out", default="results/simulation_report.json")
    args = ap.parse_args()

    res = run_simulation(Path(args.model), Path(args.csv), args.target)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
