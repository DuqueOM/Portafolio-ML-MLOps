from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import yaml


def load_config() -> Dict:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # defaults
    return {
        "data": {
            "drop_columns": ["RowNumber", "CustomerId", "Surname"],
            "categorical_features": ["Geography", "Gender"],
            "numerical_features": [
                "CreditScore",
                "Age",
                "Tenure",
                "Balance",
                "NumOfProducts",
                "HasCrCard",
                "IsActiveMember",
                "EstimatedSalary",
            ],
        }
    }


def _load_combined(models_dir: Path) -> Tuple[Any, Any] | None:
    pack = models_dir / "model_v1.0.0.pkl"
    if pack.exists():
        obj = joblib.load(pack)
        if isinstance(obj, dict) and "model" in obj and "preprocessor" in obj:
            return obj["preprocessor"], obj["model"]
    return None


def _load_separate(models_dir: Path) -> Tuple[Any, Any] | None:
    model_path = models_dir / "best_model.pkl"
    prep_path = models_dir / "preprocessor.pkl"
    if model_path.exists() and prep_path.exists():
        return joblib.load(prep_path), joblib.load(model_path)
    return None


def load_artifacts(models_dir: Path | None = None) -> Tuple[Any, Any]:
    root = Path(__file__).resolve().parents[1]
    models_dir = models_dir or (root / "models")
    res = _load_combined(models_dir) or _load_separate(models_dir)
    if res is None:
        raise SystemExit("No model artifacts found. Train first: make train")
    return res


def demo_predict() -> None:
    preprocessor, model = load_artifacts()
    cfg = load_config()

    sample = {
        "CreditScore": 650,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 45,
        "Tenure": 5,
        "Balance": 120000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 75000.0,
    }

    # Arrange columns in original order (drop columns removed)
    cols = cfg["data"]["numerical_features"] + cfg["data"]["categorical_features"]
    df = pd.DataFrame([sample], columns=cols)
    X = preprocessor.transform(df)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    print(json.dumps({"prediction": pred, "probability": proba}, indent=2))


if __name__ == "__main__":
    demo_predict()
