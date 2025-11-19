from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from evaluate import (
    bootstrap_region_profit,
    evaluate_baseline,
    prepare_with_predictions,
    rmse_score,
    split_train_val,
    train_linear_regression,
)


def _toy_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    w = np.array([5.0, -2.0, 1.0])
    y = X @ w + rng.normal(scale=0.5, size=n) + 50.0
    df = pd.DataFrame(X, columns=["f0", "f1", "f2"]).assign(
        id=[f"id{i}" for i in range(n)], product=y
    )
    return df


def test_train_and_rmse():
    df = _toy_df(200)
    X = df[["f0", "f1", "f2"]]
    y = df["product"]
    Xtr, Xva, ytr, yva = split_train_val(X, y, test_size=0.25, random_state=1)
    model = train_linear_regression(Xtr, ytr)
    y_pred = model.predict(Xva)
    rmse = rmse_score(yva, y_pred)
    baseline = evaluate_baseline(Xtr, ytr, Xva, yva)
    assert rmse < baseline


@pytest.mark.slow
def test_bootstrap_shapes_and_keys():
    df = _toy_df(1000)
    X = df[["f0", "f1", "f2"]]
    y = df["product"]
    model = train_linear_regression(X, y)
    dfp = prepare_with_predictions(df, model, ["f0", "f1", "f2"])
    res = bootstrap_region_profit(
        df_with_preds=dfp,
        n_bootstrap=50,
        n_explore=100,
        n_select=20,
        price_per_unit=4500,
        total_investment=100_000_000,
        random_state=7,
    )
    assert set(res.keys()) == {
        "expected_profit",
        "ci_lower",
        "ci_upper",
        "loss_probability",
    }
    assert isinstance(res["expected_profit"], float)
