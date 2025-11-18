from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from main import MetallurgicalPredictor
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/metallurgical_model.pkl")

app = FastAPI(title="GoldRecovery Inference API", version="1.0.0")


class Instance(BaseModel):
    # Diccionario generico de features numericas
    features: Dict[str, float]


class PredictRequest(BaseModel):
    instances: List[Instance]


class PredictResponse(BaseModel):
    predictions: List[float]


# Carga perezosa
def _load_predictor() -> MetallurgicalPredictor:
    try:
        predictor = MetallurgicalPredictor()
        predictor.load_models(MODEL_PATH)
        return predictor
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo: {e}")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    if not req.instances:
        raise HTTPException(status_code=400, detail="No se enviaron instancias")

    predictor = _load_predictor()

    rows = [inst.features for inst in req.instances]
    df = pd.DataFrame(rows)

    try:
        preds = predictor.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en prediccion: {e}")

    preds = np.asarray(preds).tolist()
    return PredictResponse(predictions=preds)
