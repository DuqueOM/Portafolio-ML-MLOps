from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, Field


class WellFeatures(BaseModel):
    f0: float
    f1: float
    f2: float


class PredictRequest(BaseModel):
    region: int = Field(..., ge=0, le=2)
    records: List[WellFeatures]


class PredictResponse(BaseModel):
    region: int
    predictions: List[float]


def _load_config() -> Dict:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_models(models_dir: Path) -> Dict[int, Any]:
    models: Dict[int, Any] = {}
    for i in [0, 1, 2]:
        path = models_dir / f"region_{i}.joblib"
        if path.exists():
            models[i] = load(path)
    return models


cfg = _load_config()
project_root = Path(__file__).resolve().parents[1]
models_dir = project_root / cfg["project"]["models_dir"]
models = _load_models(models_dir)

app = FastAPI(title="OilWell Location Optimizer API", version="1.0.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "models_loaded": ",".join(map(str, models.keys()))}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    if req.region not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for region {req.region} not available. Train first.",
        )
    model = models[req.region]
    X = pd.DataFrame(
        [r.dict() for r in req.records], columns=["f0", "f1", "f2"]
    )  # fixed order
    preds = model.predict(X)
    return PredictResponse(region=req.region, predictions=[float(p) for p in preds])
