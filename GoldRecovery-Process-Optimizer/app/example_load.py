from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_pack() -> dict | None:
    p = MODELS_DIR / "model_v1.0.0.pkl"
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None


def load_models() -> dict | None:
    p = MODELS_DIR / "metallurgical_model.pkl"
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None


def demo_predict() -> None:
    pack = load_pack()
    if pack and isinstance(pack, dict) and "models" in pack:
        # Ensemble average prediction on synthetic sample
        models = pack["models"]
        X = pd.DataFrame(np.random.rand(1, len(pack.get("feature_columns", []))))
        preds = []
        for m in models.values():
            try:
                preds.append(float(m.predict(X)[0]))
            except Exception:
                pass
        pred = float(np.mean(preds)) if preds else 0.0
        print(json.dumps({"predicted_recovery": pred}, indent=2))
        return

    models = load_models()
    if models is not None:
        print(json.dumps({"loaded_model_keys": list(models.keys())}, indent=2))
        return

    print("No model artifacts found. Train first: make train")


if __name__ == "__main__":
    demo_predict()
