from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from data.preprocess import build_preprocessor  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


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
    clf = LogisticRegression(
        C=1.0, penalty="l2", solver="liblinear", class_weight="balanced"
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert preds.shape[0] == y_test.shape[0]
