import numpy as np
import pandas as pd
from data.preprocess import build_preprocessor, infer_feature_types
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def test_model_fit_predict():
    df = pd.DataFrame(
        {
            "price": [10000, 15000, 12000, 18000, 22000, 13000, 17000, 21000],
            "model_year": [2011, 2014, 2012, 2016, 2017, 2013, 2015, 2018],
            "odometer": [90000, 60000, 80000, 50000, 40000, 75000, 62000, 35000],
            "fuel": ["gas", "gas", "diesel", "gas", "gas", "diesel", "gas", "gas"],
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
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    X = df.drop(columns=["price"])
    y = df["price"]
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert np.isfinite(preds).all()
