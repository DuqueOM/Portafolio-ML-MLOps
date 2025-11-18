from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "models"


def load_models(models_dir: Path = MODELS_DIR) -> Dict[int, Any]:
    combined = models_dir / "model_v1.0.0.pkl"
    if combined.exists():
        pack = joblib.load(combined)
        if isinstance(pack, dict) and "models" in pack:
            try:
                return {int(k): v for k, v in pack["models"].items()}
            except Exception:
                return pack["models"]  # type: ignore
    models: Dict[int, Any] = {}
    for i in [0, 1, 2]:
        p = models_dir / f"region_{i}.joblib"
        if p.exists():
            models[i] = joblib.load(p)
    return models


def demo_predict() -> None:
    models = load_models()
    if not models:
        print("No models found. Train first with: python main.py --mode train")
        return
    region = sorted(models.keys())[0]
    model = models[region]
    sample = pd.DataFrame([{"f0": 1.0, "f1": -2.0, "f2": 3.0}])
    pred = float(model.predict(sample)[0])
    print(json.dumps({"region": region, "prediction": pred}, indent=2))


if __name__ == "__main__":
    demo_predict()
