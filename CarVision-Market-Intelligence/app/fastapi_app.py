from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CarVision Inference API", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
_model = None
_feature_columns = None


class VehicleFeatures(BaseModel):
    model_year: int
    model: str
    condition: Optional[str] = None
    cylinders: Optional[float] = None
    fuel: Optional[str] = None
    odometer: Optional[float] = None
    transmission: Optional[str] = None
    drive: Optional[str] = None
    size: Optional[str] = None
    type: Optional[str] = None
    paint_color: Optional[str] = None
    is_4wd: Optional[float] = None


@app.on_event("startup")
def load_model():
    global _model, _feature_columns
    _model = joblib.load(MODEL_PATH)
    feat_path = ARTIFACTS_DIR / "feature_columns.json"
    if feat_path.exists():
        _feature_columns = json.loads(feat_path.read_text())


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(features: VehicleFeatures):
    if _model is None:
        return {"error": "Model not loaded"}
    df = pd.DataFrame([features.dict()])
    # align to training columns if available
    if _feature_columns:
        for col in _feature_columns:
            if col not in df.columns:
                df[col] = None
        df = df[_feature_columns]
    y_pred = _model.predict(df)
    return {"prediction": float(y_pred[0])}
