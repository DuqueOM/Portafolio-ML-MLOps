from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Gaming Market Intelligence API", version="1.0.0")


class GameInput(BaseModel):
    platform: str
    genre: str
    year_of_release: int
    critic_score: Optional[float] = None
    user_score: Optional[float] = None
    rating: Optional[str] = None


@app.on_event("startup")
def load_model() -> None:
    cfg = yaml.safe_load(open("configs/config.yaml", "r", encoding="utf-8"))
    model_path = Path(cfg["paths"]["model_dir"]) / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"Model artifact not found at {model_path}. Train the model first."
        )
    app.state.model = joblib.load(model_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: GameInput) -> dict:
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([payload.dict()])
    pipe = app.state.model
    try:
        pred = int(pipe.predict(df)[0])
        if hasattr(pipe, "predict_proba"):
            proba = float(pipe.predict_proba(df)[0, 1])
        else:
            proba = None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return {"is_successful": pred, "success_probability": proba}
