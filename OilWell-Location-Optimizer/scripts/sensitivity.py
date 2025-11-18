from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_region_model(models_dir: Path, region: int):
    p = models_dir / f"region_{region}.joblib"
    if not p.exists():
        raise SystemExit(f"Model for region {region} not found: {p}")
    return joblib.load(p)


def simulate_scenarios(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    revenue_values: list[float],
    investment_values: list[float],
    n_select: int = 200,
) -> list[dict]:
    X = df[feature_cols]
    preds = model.predict(X)
    dfp = df.copy()
    dfp["pred"] = preds

    results = []
    for rev in revenue_values:
        for inv in investment_values:
            top = dfp.nlargest(n_select, "pred")
            total_units = float(top["product"].sum())
            revenue = total_units * rev
            profit = revenue - inv
            results.append(
                {
                    "revenue_per_unit": rev,
                    "total_investment": inv,
                    "expected_profit": profit,
                    "loss": profit < 0,
                }
            )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Scenario sensitivity for OilWell")
    ap.add_argument("--region", type=int, default=1)
    ap.add_argument("--csv", default="geo_data_1.csv")
    ap.add_argument("--models-dir", default="artifacts/models")
    ap.add_argument("--out", default="artifacts/sensitivity.json")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    df = pd.read_csv(base / args.csv)
    model = load_region_model(base / args.models_dir, args.region)
    feature_cols = ["f0", "f1", "f2"]

    revenue_values = list(np.linspace(3000, 6000, 7))
    investment_values = [80_000_000.0, 100_000_000.0, 120_000_000.0]

    res = simulate_scenarios(df, model, feature_cols, revenue_values, investment_values)
    out = {"region": args.region, "grid": res}
    out_path = base / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
