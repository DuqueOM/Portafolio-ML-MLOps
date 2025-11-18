from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from data.preprocess import PreprocessConfig, load_raw_dataset, make_features_and_target
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def main(cfg_path: str = "configs/config.yaml") -> None:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    model_path = Path(cfg["paths"]["model_dir"]) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Train first."
        )

    pipe = joblib.load(model_path)

    df = load_raw_dataset(cfg["paths"]["dataset_path"])
    pre_cfg = PreprocessConfig(
        numeric_imputer_strategy=cfg["preprocessing"]["numeric_imputer_strategy"],
        categorical_imputer_strategy=cfg["preprocessing"][
            "categorical_imputer_strategy"
        ],
        scale_numeric=cfg["preprocessing"]["scale_numeric"],
        one_hot_drop=cfg["preprocessing"]["one_hot_drop"],
        include_features=cfg["preprocessing"]["features"]["include"],
        exclude_features=cfg["preprocessing"]["features"].get("exclude", []),
        target_threshold_million=cfg["project"]["target_success_threshold_million"],
    )
    X, y = make_features_and_target(df, config=pre_cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["project"]["random_seed"],
        stratify=y if cfg["train"].get("stratify", True) else None,
    )

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Majority-class baseline on y_test
    majority = (
        int(np.round(y_test.mean())) if y_test.nunique() == 2 else int(y_test.mode()[0])
    )
    y_base = np.full_like(y_test, fill_value=majority)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "baseline_accuracy": float(accuracy_score(y_test, y_base)),
        "baseline_f1": float(f1_score(y_test, y_base, zero_division=0)),
    }

    out_path = Path(cfg["paths"]["metrics_dir"]) / "metrics_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
