from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

CANDIDATE_PATHS = [
    Path("models/model_v1.0.0.pkl"),
    Path("artifacts/model.joblib"),
]


def load_model():
    for p in CANDIDATE_PATHS:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError(
        f"No model found. Tried: {[str(p) for p in CANDIDATE_PATHS]}"
    )


def demo_predict():
    model = load_model()
    # Minimal payload with a couple of fields; others will be handled by preprocessor
    sample = pd.DataFrame(
        [
            {
                "model_year": 2016,
                "model": "ford focus",
                "odometer": 60000,
                "fuel": "gas",
            }
        ]
    )
    pred = float(model.predict(sample)[0])
    print({"prediction": pred})


if __name__ == "__main__":
    demo_predict()
