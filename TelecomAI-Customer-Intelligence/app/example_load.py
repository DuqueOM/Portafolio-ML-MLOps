from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

MODEL_EXPORT = Path("models/model_v1.0.0.pkl")


def load_pipeline():
    if not MODEL_EXPORT.exists():
        raise FileNotFoundError(
            f"Model export not found at {MODEL_EXPORT}. Run 'make train' first."
        )
    return joblib.load(MODEL_EXPORT)


def demo_predict():
    pipe = load_pipeline()
    sample = pd.DataFrame(
        [{"calls": 80, "minutes": 500.0, "messages": 50, "mb_used": 20000.0}]
    )
    pred = int(pipe.predict(sample)[0])
    proba = (
        float(pipe.predict_proba(sample)[0, 1])
        if hasattr(pipe, "predict_proba")
        else None
    )
    print({"prediction": pred, "probability_is_ultra": proba})


if __name__ == "__main__":
    demo_predict()
