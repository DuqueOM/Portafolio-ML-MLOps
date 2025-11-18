from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from app.fastapi_app import app
from fastapi.testclient import TestClient
from main import load_config, train


def ensure_artifacts() -> None:
    # Train once to ensure artifacts exist for API
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "configs" / "config.yaml")
    # Only train if artifacts are missing to keep CI fast
    if (
        not (project_root / cfg.paths["model_path"]).exists()
        or not (project_root / cfg.paths["preprocessor_path"]).exists()
    ):
        train(cfg)


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_endpoint_smoke():
    ensure_artifacts()
    client = TestClient(app)

    # Load a single sample from dataset
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / "users_behavior.csv")
    features = ["calls", "minutes", "messages", "mb_used"]
    sample = df[features].iloc[0].to_dict()

    resp = client.post("/predict", json=sample)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]


def test_predict_endpoint_invalid_payload_missing_field():
    """Falta una feature obligatoria → debe devolver 422 Unprocessable Entity."""

    client = TestClient(app)
    payload = {"calls": 80, "minutes": 500.0, "messages": 30}  # falta mb_used
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body


def test_predict_endpoint_invalid_payload_bad_type():
    """Tipo incorrecto en una feature numérica → 422 por validación Pydantic."""

    client = TestClient(app)
    payload = {
        "calls": "eighty",  # debería ser numérico
        "minutes": 500.0,
        "messages": 30,
        "mb_used": 20000.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body
