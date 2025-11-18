from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import yaml
from data.preprocess import PreprocessConfig, load_raw_dataset, make_features_and_target
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def compute_investment_kpis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula KPIs de inversión a partir de etiquetas verdaderas y predicciones.

    Supone el siguiente escenario simplificado:
    - Estrategia baseline: invertir en todos los títulos (todas las filas).
    - Estrategia modelo: invertir sólo en títulos con predicción positiva (y_pred=1).
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    n = int(len(y_true))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Baseline: invertir en todos
    baseline_failures = int((y_true == 0).sum())
    invested_all = n
    successes_all = int((y_true == 1).sum())

    # Modelo: invertir sólo en pred=1
    invested_model = int(tp + fp)
    model_failures = int(fp)
    successes_model = int(tp)

    success_rate_all = successes_all / invested_all if invested_all > 0 else 0.0
    success_rate_model = successes_model / invested_model if invested_model > 0 else 0.0

    failure_rate_all = baseline_failures / invested_all if invested_all > 0 else 0.0
    failure_rate_model = model_failures / invested_model if invested_model > 0 else 0.0

    relative_failure_reduction = 0.0
    if failure_rate_all > 0 and invested_model > 0:
        relative_failure_reduction = (
            failure_rate_all - failure_rate_model
        ) / failure_rate_all

    failures_avoided = baseline_failures - model_failures

    return {
        "total_projects": float(n),
        "baseline_failed_if_invest_all": float(baseline_failures),
        "model_failed_if_invest_pred1": float(model_failures),
        "invested_all": float(invested_all),
        "invested_model": float(invested_model),
        "success_rate_all": float(success_rate_all),
        "success_rate_model": float(success_rate_model),
        "failure_rate_all": float(failure_rate_all),
        "failure_rate_model": float(failure_rate_model),
        "relative_failure_reduction": float(relative_failure_reduction),
        "estimated_failures_avoided": float(failures_avoided),
    }


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

    kpis = compute_investment_kpis(y_test.to_numpy(), y_pred)

    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "metrics_business.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2)

    print(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()
