from __future__ import annotations

import pandas as pd
from data.preprocess import clean_deduplicate_and_shuffle, split_features_target


def test_dedup_keeps_highest_product():
    df = pd.DataFrame(
        {
            "id": ["a", "a", "b"],
            "f0": [0.0, 0.0, 1.0],
            "f1": [0.0, 0.0, 1.0],
            "f2": [0.0, 0.0, 1.0],
            "product": [10.0, 5.0, 7.0],
        }
    )
    out = clean_deduplicate_and_shuffle(
        df, id_col="id", target_col="product", random_state=1
    )
    assert len(out) == 2
    # id "a" should keep product=10
    assert out[out["id"] == "a"]["product"].iloc[0] == 10.0


def test_split_features_target():
    df = pd.DataFrame(
        {
            "id": ["x"],
            "f0": [1.0],
            "f1": [2.0],
            "f2": [3.0],
            "product": [4.0],
        }
    )
    X, y = split_features_target(df, ["f0", "f1", "f2"], "product")
    assert list(X.columns) == ["f0", "f1", "f2"]
    assert y.iloc[0] == 4.0
