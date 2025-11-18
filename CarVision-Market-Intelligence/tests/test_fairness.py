import numpy as np
import pandas as pd
from data.preprocess import build_preprocessor, infer_feature_types
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def test_group_predictions_finite_by_fuel():
    """Smoke test de fairness: el modelo genera predicciones finitas por grupo de fuel.

    No pretende ser una auditoría de fairness, sólo verificar que el pipeline
    funciona de forma coherente para distintos subgrupos categóricos.
    """

    df = pd.DataFrame(
        {
            "price": [10000, 15000, 12000, 18000, 22000, 13000, 17000, 21000],
            "model_year": [2011, 2014, 2012, 2016, 2017, 2013, 2015, 2018],
            "odometer": [90000, 60000, 80000, 50000, 40000, 75000, 62000, 35000],
            "fuel": ["gas", "gas", "diesel", "gas", "diesel", "diesel", "gas", "gas"],
            "model": [
                "ford focus",
                "ford focus",
                "audi a4",
                "ford fusion",
                "honda civic",
                "audi a4",
                "ford focus",
                "honda civic",
            ],
        }
    )

    num_cols, cat_cols = infer_feature_types(df, target="price")
    pre = build_preprocessor(num_cols, cat_cols)
    model = RandomForestRegressor(n_estimators=16, random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X = df.drop(columns=["price"])
    y = df["price"]
    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape[0] == X.shape[0]
    assert np.isfinite(preds).all()

    for fuel in ["gas", "diesel"]:
        mask = X["fuel"] == fuel
        group_preds = preds[mask.to_numpy()]
        # Debe haber ejemplos para cada grupo y predicciones finitas
        assert group_preds.size > 0
        assert np.isfinite(group_preds).all()
