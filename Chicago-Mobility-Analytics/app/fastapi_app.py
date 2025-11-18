"""API de inferencia para predicción de duración de viajes.

Ejecutar con:
    uvicorn app.fastapi_app:app --reload
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from main import load_config, predict_single, setup_logging
from pydantic import BaseModel, Field

CONFIG_PATH = Path("configs/default.yaml")

cfg = load_config(CONFIG_PATH)
setup_logging(cfg.get("logging", {}).get("level", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chicago Mobility Duration API",
    description="API para predecir la duración de viajes en Chicago",
    version="1.0.0",
)


class DurationRequest(BaseModel):
    """Esquema de entrada para predicción de duración."""

    start_ts: datetime = Field(..., description="Fecha y hora de inicio del viaje")
    weather_conditions: Literal["Good", "Bad"] = Field(
        ..., description="Condición climática en el momento del viaje"
    )


class DurationResponse(BaseModel):
    """Esquema de salida de la predicción."""

    duration_seconds: float
    start_ts: datetime
    weather_conditions: str


@app.post("/predict_duration", response_model=DurationResponse)
async def predict_duration(payload: DurationRequest) -> DurationResponse:
    """Endpoint para predecir la duración de un viaje."""

    try:
        prediction = predict_single(
            cfg,
            payload.start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            payload.weather_conditions,
        )
    except FileNotFoundError as exc:
        logger.error("Modelo no disponible: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error durante la predicción: %s", exc)
        raise HTTPException(
            status_code=500, detail="Error interno del servidor"
        ) from exc

    return DurationResponse(
        duration_seconds=prediction,
        start_ts=payload.start_ts,
        weather_conditions=payload.weather_conditions,
    )


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Endpoint de chequeo básico de salud."""

    return {"status": "ok"}
