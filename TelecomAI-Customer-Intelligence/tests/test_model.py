from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data.preprocess import build_preprocessor


def test_training_pipeline_runs():
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / "users_behavior.csv")

    features = ["calls", "minutes", "messages", "mb_used"]
    target = "is_ultra"

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(features)
    clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", class_weight="balanced")

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert preds.shape[0] == y_test.shape[0]
