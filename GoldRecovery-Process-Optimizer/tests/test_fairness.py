import numpy as np
import pandas as pd
import pytest
from main import MetallurgicalPredictor


@pytest.mark.slow
def test_group_predictions_finite_by_synthetic_flag():
    """Smoke test de fairness en GoldRecovery.

    Usa un flag sintético de grupo para verificar que el ensemble genera
    predicciones finitas para distintos subgrupos de observaciones.
    """

    n = 60
    df = pd.DataFrame(
        {
            "final.output.recovery": np.random.uniform(70, 95, size=n),
            "rougher.output.concentrate_au": np.random.uniform(1, 5, size=n),
            "primary_cleaner.output.concentrate_au": np.random.uniform(1, 6, size=n),
            "rougher.output.concentrate_ag": np.random.uniform(1, 5, size=n),
            "primary_cleaner.output.concentrate_ag": np.random.uniform(1, 6, size=n),
            "some_noise": np.random.normal(size=n),
            # Flag de grupo sintético (p.ej. turno A/B)
            "group_flag": np.random.randint(0, 2, size=n),
        }
    )

    small_cfg = {
        "models": {
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.2,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 0,
            },
            "lightgbm": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.2,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 0,
            },
            "random_forest": {
                "n_estimators": 20,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 0,
            },
        },
        "ensemble_weights": {"xgboost": 0.4, "lightgbm": 0.35, "random_forest": 0.25},
    }

    predictor = MetallurgicalPredictor(config=small_cfg)
    X, y = predictor.prepare_features(df, target_column="final.output.recovery")

    metrics = predictor.train(X, y)
    assert isinstance(metrics, dict)

    preds = predictor.predict(X)
    assert preds.shape[0] == X.shape[0]

    for flag in (0, 1):
        mask = df["group_flag"] == flag
        group_preds = preds[mask.to_numpy()]
        assert group_preds.size > 0
        assert np.isfinite(group_preds).all()
