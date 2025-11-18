import numpy as np
import pandas as pd
from main import MetallurgicalPredictor


def test_prepare_and_train_predictor():
    # Datos sinteticos con columnas clave
    n = 100
    df = pd.DataFrame(
        {
            "final.output.recovery": np.random.uniform(70, 95, size=n),
            "rougher.output.concentrate_au": np.random.uniform(1, 5, size=n),
            "primary_cleaner.output.concentrate_au": np.random.uniform(1, 6, size=n),
            "rougher.output.concentrate_ag": np.random.uniform(1, 5, size=n),
            "primary_cleaner.output.concentrate_ag": np.random.uniform(1, 6, size=n),
            "some_noise": np.random.normal(size=n),
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

    y_hat = predictor.predict(X[:10])
    assert len(y_hat) == 10
