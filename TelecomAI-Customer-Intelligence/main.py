from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import yaml
from data.preprocess import build_preprocessor, get_features_target, load_dataset
from evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Optional MLflow integration
try:
    import mlflow
    import mlflow.sklearn  # noqa: F401
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telecomai")


@dataclass
class Config:
    project_name: str
    random_seed: int
    paths: Dict[str, str]
    features: list
    target: str
    split: Dict[str, Any]
    model: Dict[str, Any]
    threshold: float = 0.5
    mlflow: Optional[Dict[str, Any]] = None


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)


def ensure_dirs(paths: Dict[str, str]) -> None:
    Path(paths["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
    model_export_path = paths.get("model_export_path")
    if model_export_path:
        Path(model_export_path).parent.mkdir(parents=True, exist_ok=True)


def build_model(model_cfg: Dict[str, Any]) -> LogisticRegression:
    if model_cfg["name"].lower() == "logreg":
        params = model_cfg.get("params", {})
        return LogisticRegression(**params, random_state=None)
    raise ValueError(f"Unsupported model: {model_cfg['name']}")


def train(cfg: Config) -> Dict[str, float]:
    logger.info("Starting training with config: %s", cfg)
    ensure_dirs(cfg.paths)

    df = load_dataset(cfg.paths["data_csv"])
    X, y = get_features_target(df, cfg.features, cfg.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg.split.get("test_size", 0.2)),
        stratify=y if cfg.split.get("stratify", True) else None,
        random_state=int(cfg.random_seed),
    )

    preprocessor = build_preprocessor(cfg.features)
    clf = build_model(cfg.model)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    # MLflow setup (optional)
    mlflow_cfg = cfg.mlflow
    use_mlflow = bool(
        mlflow_cfg is not None and mlflow is not None and mlflow_cfg.get("enable", True)
    )
    if use_mlflow:
        # At this point mypy knows mlflow_cfg and mlflow are not None
        assert mlflow is not None
        assert mlflow_cfg is not None

        tracking_uri = (
            mlflow_cfg.get("tracking_uri")
            or os.getenv("MLFLOW_TRACKING_URI")
            or "file:./mlruns"
        )
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = mlflow_cfg.get("experiment")
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    if use_mlflow:
        with mlflow.start_run(run_name="train"):
            mlflow.log_params(
                {
                    "model_name": cfg.model["name"],
                    **cfg.model.get("params", {}),
                }
            )
            pipeline.fit(X_train, y_train)
            # Save artifacts
            joblib.dump(preprocessor, cfg.paths["preprocessor_path"])
            joblib.dump(clf, cfg.paths["model_path"])
            export_path = cfg.paths.get("model_export_path", "models/model_v1.0.0.pkl")
            joblib.dump(pipeline, export_path)

            # Evaluate
            y_pred = pipeline.predict(X_test)
            y_proba = (
                pipeline.predict_proba(X_test)[:, 1]
                if hasattr(pipeline, "predict_proba")
                else None
            )
            metrics_dict = compute_classification_metrics(
                y_test.to_numpy(), y_pred, y_proba
            )
            save_metrics(metrics_dict, cfg.paths["metrics_path"])
            plot_confusion_matrix(
                y_test.to_numpy(), y_pred, cfg.paths["confusion_matrix_path"]
            )
            plot_roc_curve(y_test.to_numpy(), y_proba, cfg.paths["roc_curve_path"])

            # Log metrics and artifacts
            mlflow.log_metrics(metrics_dict)
            for art in [
                cfg.paths["metrics_path"],
                cfg.paths["confusion_matrix_path"],
                cfg.paths["roc_curve_path"],
                export_path,
            ]:
                if Path(art).exists():
                    mlflow.log_artifact(art)
    else:
        pipeline.fit(X_train, y_train)
        # Save artifacts
        joblib.dump(preprocessor, cfg.paths["preprocessor_path"])
        joblib.dump(clf, cfg.paths["model_path"])
        export_path = cfg.paths.get("model_export_path", "models/model_v1.0.0.pkl")
        joblib.dump(pipeline, export_path)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics_dict = compute_classification_metrics(y_test.to_numpy(), y_pred, y_proba)
    save_metrics(metrics_dict, cfg.paths["metrics_path"])
    plot_confusion_matrix(y_test.to_numpy(), y_pred, cfg.paths["confusion_matrix_path"])
    plot_roc_curve(y_test.to_numpy(), y_proba, cfg.paths["roc_curve_path"])

    logger.info("Training done. Metrics: %s", metrics_dict)
    return metrics_dict


def evaluate(cfg: Config) -> Dict[str, float]:
    logger.info("Starting evaluation...")
    df = load_dataset(cfg.paths["data_csv"])
    X, y = get_features_target(df, cfg.features, cfg.target)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg.split.get("test_size", 0.2)),
        stratify=y if cfg.split.get("stratify", True) else None,
        random_state=int(cfg.random_seed),
    )

    preprocessor = joblib.load(cfg.paths["preprocessor_path"])
    clf = joblib.load(cfg.paths["model_path"])

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    y_pred = pipeline.predict(X_test)
    y_proba = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    metrics_dict = compute_classification_metrics(y_test.to_numpy(), y_pred, y_proba)
    save_metrics(metrics_dict, cfg.paths["metrics_path"])
    plot_confusion_matrix(y_test.to_numpy(), y_pred, cfg.paths["confusion_matrix_path"])
    plot_roc_curve(y_test.to_numpy(), y_proba, cfg.paths["roc_curve_path"])

    logger.info("Evaluation done. Metrics: %s", metrics_dict)
    return metrics_dict


def predict(cfg: Config, input_csv: str | None, output_path: str | None) -> None:
    if input_csv is None or output_path is None:
        raise ValueError("For predict mode, --input_csv and --output_path are required")

    preprocessor = joblib.load(cfg.paths["preprocessor_path"])
    clf = joblib.load(cfg.paths["model_path"])
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    df = pd.read_csv(input_csv)
    missing = [c for c in cfg.features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    preds = pipeline.predict(df[cfg.features])
    probas = (
        pipeline.predict_proba(df[cfg.features])[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    out_df = df.copy()
    out_df["pred_is_ultra"] = preds
    if probas is not None:
        out_df["proba_is_ultra"] = probas

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info("Predictions saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TelecomAI-Customer-Intelligence CLI: train | eval | predict\n"
            "Examples:\n"
            "  python main.py --mode train --config configs/config.yaml\n"
            "  python main.py --mode eval --config configs/config.yaml\n"
            "  python main.py --mode predict --config configs/config.yaml --input_csv data.csv --output_path preds.csv\n"
        )
    )
    parser.add_argument("--mode", choices=["train", "eval", "predict"], required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        evaluate(cfg)
    elif args.mode == "predict":
        predict(cfg, args.input_csv, args.output_path)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
