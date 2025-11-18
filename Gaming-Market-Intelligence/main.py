from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from data.preprocess import (
    PreprocessConfig,
    build_preprocessor,
    load_raw_dataset,
    make_features_and_target,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for k in ["artifacts_dir", "model_dir", "processed_dir", "metrics_dir", "logs_dir"]:
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)


def build_model(cfg: Dict[str, Any]) -> RandomForestClassifier:
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "RandomForestClassifier")
    params = model_cfg.get("params", {})

    if model_type != "RandomForestClassifier":
        raise ValueError(f"Unsupported model type: {model_type}")

    return RandomForestClassifier(**params)


def cmd_train(cfg_path: str, seed: int | None = None) -> None:
    cfg = load_config(cfg_path)
    if seed is not None:
        cfg["project"]["random_seed"] = int(seed)
    ensure_dirs(cfg)
    setup_logging(Path(cfg["paths"]["logs_dir"]))

    np.random.seed(cfg["project"]["random_seed"])  # reproducibility baseline

    dataset_path = cfg["paths"]["dataset_path"]
    logging.info("Loading dataset from %s", dataset_path)
    df = load_raw_dataset(dataset_path)

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

    logging.info(
        "Dataset loaded. Shape X=%s, y=%s. Positive rate=%.3f",
        X.shape,
        y.shape,
        y.mean(),
    )

    stratify = y if cfg["train"].get("stratify", True) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["project"]["random_seed"],
        stratify=stratify,
    )

    preprocessor = build_preprocessor(X, config=pre_cfg)
    model = build_model(cfg)

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", model)])
    logging.info("Fitting model pipeline...")
    pipe.fit(X_train, y_train)

    logging.info("Evaluating model on test set...")
    y_pred = pipe.predict(X_test)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    # ROC-AUC (requires probabilities and positive/negative classes present)
    try:
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
    except Exception as e:
        logging.warning("Probabilistic metrics couldn't be computed: %s", e)

    # Persist artifacts
    model_path = Path(cfg["paths"]["model_dir"]) / "model.joblib"
    metrics_path = Path(cfg["paths"]["metrics_dir"]) / "metrics.json"

    joblib.dump(pipe, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Export combined pack for demo
    try:
        pack_path = Path(cfg["paths"]["model_dir"]) / "model_v1.0.0.pkl"
        joblib.dump({"pipeline": pipe, "version": "1.0.0"}, pack_path)
    except Exception:
        pass

    logging.info("Artifacts saved: model=%s, metrics=%s", model_path, metrics_path)


def cmd_eval(cfg_path: str, seed: int | None = None) -> None:
    cfg = load_config(cfg_path)
    if seed is not None:
        cfg["project"]["random_seed"] = int(seed)
    setup_logging(Path(cfg["paths"]["logs_dir"]))

    model_path = Path(cfg["paths"]["model_dir"]) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run --mode train first."
        )

    logging.info("Loading model from %s", model_path)
    pipe: Pipeline = joblib.load(model_path)

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

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["project"]["random_seed"],
        stratify=(y if cfg["train"].get("stratify", True) else None),
    )

    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)


def cmd_predict(cfg_path: str, payload_json: str | None) -> None:
    cfg = load_config(cfg_path)
    model_path = Path(cfg["paths"]["model_dir"]) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run --mode train first."
        )

    pipe: Pipeline = joblib.load(model_path)

    if payload_json is None:
        raise ValueError("--payload is required for predict mode.")

    payload = json.loads(payload_json)
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("payload must be an object or list of objects")

    X = pd.DataFrame(rows)
    preds = pipe.predict(X)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1].tolist()
    else:
        proba = [None] * len(preds)

    results = [
        {
            "is_successful": int(p),
            "success_probability": (float(pr) if pr is not None else None),
        }
        for p, pr in zip(preds, proba)
    ]
    print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gaming-Market-Intelligence CLI. Modes: train|eval|predict.\n"
            "Examples:\n"
            "  python main.py --mode train --config configs/config.yaml\n"
            "  python main.py --mode eval --config configs/config.yaml\n"
            "  python main.py --mode predict --config configs/config.yaml "
            "--payload '{"
            "platform"
            ": "
            "PS4"
            ", "
            "genre"
            ": "
            "Action"
            ", "
            "year_of_release"
            ": 2015, "
            "critic_score"
            ": 85, "
            "user_score"
            ": 8.2, "
            "rating"
            ": "
            "M"
            "}'\n"
        )
    )
    parser.add_argument("--mode", choices=["train", "eval", "predict"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed in config for split/reproducibility",
    )
    parser.add_argument("--payload", help="JSON string for predict mode", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode
    cfg_path = args.config

    if mode == "train":
        cmd_train(cfg_path, seed=args.seed)
    elif mode == "eval":
        cmd_eval(cfg_path, seed=args.seed)
    elif mode == "predict":
        cmd_predict(cfg_path, args.payload)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
