"""Tests básicos para el entrenamiento del modelo."""

from __future__ import annotations

from pathlib import Path

import pytest
from main import load_config, train_model


@pytest.mark.slow
def test_train_model_runs_and_returns_metrics() -> None:
    """Entrena un modelo pequeño y devuelve métricas razonables."""

    cfg = load_config(Path("configs/default.yaml"))
    metrics = train_model(cfg, seed=0)

    assert "val" in metrics and "test" in metrics
    for split in ("val", "test"):
        assert metrics[split]["mae"] > 0
        assert metrics[split]["rmse"] > 0
