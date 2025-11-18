from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_model() -> Any:
    pack = MODELS_DIR / "model_v1.0.0.pkl"
    if pack.exists():
        obj = joblib.load(pack)
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"]
    # fallback
    p = MODELS_DIR / "duration_model.pkl"
    if p.exists():
        return joblib.load(p)
    raise SystemExit(
        "No model artifacts found. Train first: python main.py --mode train --config configs/default.yaml"
    )


def demo_predict() -> None:
    model = load_model()
    # hour, day_of_week, is_weekend, weather_is_bad
    X = np.array([[9.0, 5.0, 1.0, 0.0]], dtype=float)
    pred = float(model.predict(X)[0])
    print(json.dumps({"duration_seconds": pred}, indent=2))


if __name__ == "__main__":
    demo_predict()
