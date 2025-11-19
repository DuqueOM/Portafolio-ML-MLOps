from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from data.preprocess import build_preprocessor, clean_data, infer_feature_types, load_data, split_data
from evaluate import evaluate_model, rmse
from main import train_model


def test_load_data_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "cars.csv"
    sample = pd.DataFrame({"price": [1000, 2000], "model": ["a", "b"]})
    sample.to_csv(csv_path, index=False)

    df = load_data(str(csv_path))
    assert df.equals(sample)


def test_clean_data_filters_and_creates_features() -> None:
    current_year = pd.Timestamp.now().year
    df = pd.DataFrame(
        {
            "price": [500, 5000, 800000],
            "model_year": [1980, current_year - 5, current_year + 1],
            "odometer": [0, 10000, 999999],
        }
    )
    cleaned = clean_data(df)
    assert cleaned["price"].between(1000, 500000).all()
    assert cleaned["model_year"].between(1990, current_year).all()
    assert cleaned["odometer"].between(1, 500000).all()
    assert "vehicle_age" in cleaned.columns
    assert "price_per_mile" in cleaned.columns


def test_infer_feature_types_handles_explicit_lists() -> None:
    df = pd.DataFrame(
        {
            "price": [1, 2],
            "num_a": [10.0, 20.0],
            "cat_a": ["gas", "diesel"],
            "drop_me": [1, 2],
        }
    )
    num_cols, cat_cols = infer_feature_types(
        df,
        target="price",
        numeric_features=["num_a"],
        categorical_features=["cat_a"],
        drop_columns=["drop_me"],
    )
    assert num_cols == ["num_a"]
    assert cat_cols == ["cat_a"]


def test_build_preprocessor_transforms_numeric_and_categorical() -> None:
    df = pd.DataFrame(
        {
            "price": [10000, 12000, 15000],
            "num_col": [1.0, np.nan, 3.0],
            "cat_col": ["ford", "audi", None],
        }
    )
    num_cols, cat_cols = infer_feature_types(df, target="price")
    pre = build_preprocessor(num_cols, cat_cols)
    X = df.drop(columns=["price"])
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(df)
    assert not np.isnan(Xt).any()


def test_split_data_returns_consistent_shapes() -> None:
    df = pd.DataFrame(
        {
            "price": np.arange(100, 200, dtype=float),
            "num_col": np.arange(100),
            "cat_col": ["brand"] * 100,
        }
    )
    X_train, X_val, X_test, y_train, y_val, y_test, indices = split_data(
        df,
        target="price",
        test_size=0.2,
        val_size=0.1,
        seed=42,
        shuffle=True,
    )
    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    assert set(indices.keys()) == {"train", "val", "test"}


def test_evaluate_model_creates_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "configs" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    # Override paths to stay inside tmp_path
    data_csv = project_root / cfg["paths"]["data_path"]
    artifacts_dir = tmp_path / "artifacts"
    cfg["paths"] = {
        **cfg["paths"],
        "data_path": str(data_csv),
        "artifacts_dir": str(artifacts_dir),
        "model_path": str(artifacts_dir / "model.joblib"),
        "metrics_path": str(artifacts_dir / "metrics.json"),
        "baseline_metrics_path": str(artifacts_dir / "baseline.json"),
    }

    train_model(cfg)
    results = evaluate_model(cfg)
    assert "model" in results
    assert Path(cfg["paths"]["metrics_path"]).exists()
    assert Path(cfg["paths"]["baseline_metrics_path"]).exists()


def test_rmse_matches_numpy() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    expected = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    assert rmse(y_true, y_pred) == pytest.approx(expected)
