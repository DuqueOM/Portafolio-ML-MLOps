import pandas as pd
from data.preprocess import build_preprocessor, clean_data, infer_feature_types


def test_clean_data_basic():
    df = pd.DataFrame(
        {
            "price": [500, 2000, 10000, 600000],
            "model_year": [1980, 2005, 2015, 2026],
            "odometer": [0, 10000, 50000, 1000000],
            "model": ["ford focus", "bmw 320i", "honda civic", "audi a4"],
        }
    )
    dfc = clean_data(df)
    # Filtered extremes should be removed
    assert dfc["price"].between(1000, 500000).all()
    assert dfc["model_year"].between(1990, pd.Timestamp.now().year).all()
    assert dfc["odometer"].between(1, 500000).all()


def test_preprocessor_shapes():
    df = pd.DataFrame(
        {
            "price": [10000, 15000, 12000, 20000],
            "model_year": [2012, 2015, 2013, 2016],
            "odometer": [80000, 60000, 70000, 50000],
            "fuel": ["gas", "gas", "diesel", "gas"],
            "model": ["ford focus", "ford focus", "audi a4", "ford fusion"],
        }
    )
    num_cols, cat_cols = infer_feature_types(df, target="price")
    pre = build_preprocessor(num_cols, cat_cols)
    X = df.drop(columns=["price"])
    Xt = pre.fit_transform(X)
    # At least as many rows
    assert Xt.shape[0] == X.shape[0]
