from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score


def _make_synth(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    geo = rng.choice(["France", "Spain", "Germany"], n, p=[0.5, 0.25, 0.25])
    age = rng.integers(18, 92, size=n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logits = -0.5 + 0.02 * (age - 50) + 0.3 * (geo == "Germany") + 0.6 * (x1 > 0)
    p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, p)
    df = pd.DataFrame(
        {
            "CreditScore": (x1 * 100).astype(int) + 650,
            "Geography": geo,
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": age,
            "Tenure": rng.integers(0, 11, n),
            "Balance": np.abs(x2 * 50000),
            "NumOfProducts": rng.integers(1, 5, n),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": rng.integers(0, 2, n),
            "EstimatedSalary": rng.uniform(20000, 150000, n),
            "Exited": y,
        }
    )
    return df


def test_fairness_recall_gap_by_geography():
    df = _make_synth(1000, seed=7)
    X = pd.get_dummies(df.drop(columns=["Exited"]), drop_first=True)
    y = df["Exited"].values
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    y_pred = clf.predict(X)

    recalls = {}
    for g in ["France", "Spain", "Germany"]:
        mask = df["Geography"] == g
        if y[mask].sum() == 0:
            continue
        recalls[g] = recall_score(y[mask], y_pred[mask])

    if len(recalls) >= 2:
        gap = max(recalls.values()) - min(recalls.values())
        assert gap < 0.3  # tolerance threshold


def test_fairness_recall_gap_by_gender():
    df = _make_synth(800, seed=11)
    X = pd.get_dummies(df.drop(columns=["Exited"]), drop_first=True)
    y = df["Exited"].values
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    y_pred = clf.predict(X)

    recalls = {}
    for g in ["Male", "Female"]:
        mask = df["Gender"] == g
        if y[mask].sum() == 0:
            continue
        recalls[g] = recall_score(y[mask], y_pred[mask])

    if len(recalls) >= 2:
        gap = max(recalls.values()) - min(recalls.values())
        assert gap < 0.3
