from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class TrainResult:
    model_path: Path
    rmse: float
    baseline_rmse: float
    mean_true: float
    mean_pred: float


def rmse_score(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def split_train_val(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_linear_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> float:
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_val)
    return rmse_score(y_val, y_pred)


def prepare_with_predictions(
    df: pd.DataFrame,
    model: LinearRegression,
    feature_cols: list[str],
) -> pd.DataFrame:
    dfc = df.copy()
    dfc["pred"] = model.predict(dfc[feature_cols])
    return dfc


def bootstrap_region_profit(
    df_with_preds: pd.DataFrame,
    n_bootstrap: int,
    n_explore: int,
    n_select: int,
    price_per_unit: float,
    total_investment: float,
    random_state: int,
) -> Dict[str, float]:
    """Bootstrap over wells: sample n_explore with replacement, select top n_select by pred,
    compute profit using ACTUAL target sums and project economics.
    """
    rng = np.random.default_rng(random_state)
    profits = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        # sample with replacement
        idx = rng.integers(0, len(df_with_preds), size=n_explore)
        sample = df_with_preds.iloc[idx]
        top = sample.nlargest(n_select, "pred")
        total_units = float(top["product"].sum())
        revenue = total_units * float(price_per_unit)
        profits[i] = revenue - float(total_investment)
    mean_profit = float(np.mean(profits))
    ci_lower, ci_upper = np.percentile(profits, [2.5, 97.5]).tolist()
    loss_prob = float(np.mean(profits < 0.0))
    return {
        "expected_profit": mean_profit,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "loss_probability": loss_prob,
    }
