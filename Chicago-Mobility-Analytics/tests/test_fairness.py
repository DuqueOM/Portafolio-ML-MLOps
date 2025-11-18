"""Smoke test de fairness por condiciones climáticas.

Entrena un modelo pequeño usando `load_and_preprocess` y verifica que se
pueden calcular errores por subgrupo de `weather_is_bad` sin fallos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from data.preprocess import load_and_preprocess, load_config
from sklearn.ensemble import RandomForestRegressor


def test_error_by_weather_group_is_computable() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    X, y = load_and_preprocess(cfg)

    # Entrenamiento rápido con un modelo pequeño
    model = RandomForestRegressor(n_estimators=16, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape[0] == X.shape[0]
    assert np.isfinite(preds).all()

    # Fairness smoke: el error se puede computar para ambos grupos de clima
    assert "weather_is_bad" in X.columns
    for flag in (0, 1):
        mask = X["weather_is_bad"] == flag
        group_y = y[mask]
        group_preds = preds[mask.to_numpy()]
        # Debe haber al menos algunos ejemplos por grupo
        assert group_y.shape[0] > 0
        # Error MAE básico
        mae = float(np.mean(np.abs(group_y - group_preds)))
        assert np.isfinite(mae)
