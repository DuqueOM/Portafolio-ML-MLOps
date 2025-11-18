from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "model"


def load_pipeline() -> object:
    pack = MODELS_DIR / "model_v1.0.0.pkl"
    if pack.exists():
        try:
            obj = joblib.load(pack)
            if isinstance(obj, dict) and "pipeline" in obj:
                return obj["pipeline"]
        except Exception:
            pass
    p = MODELS_DIR / "model.joblib"
    if p.exists():
        return joblib.load(p)
    raise SystemExit("No model artifacts found. Train first: make train")


def demo_predict() -> None:
    pipe = load_pipeline()
    sample = {
        "platform": "PS4",
        "year_of_release": 2015,
        "genre": "Action",
        "critic_score": 85,
        "user_score": 8.2,
        "rating": "M",
    }
    X = pd.DataFrame([sample])
    pred = int(pipe.predict(X)[0])
    print(json.dumps({"is_successful": pred}, indent=2))


if __name__ == "__main__":
    demo_predict()
