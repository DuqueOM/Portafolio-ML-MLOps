from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import main as bankchurn_main
import numpy as np
import pandas as pd
import pytest


class DummyPredictor:
    def __init__(self, config_path: str | None = None):
        self.config = {
            "training": {
                "test_size": 0.5,
                "random_state": 42,
                "stratify": True,
            }
        }
        self.preprocessor = "preprocessor"
        self.model = "model"
        self.is_fitted = False

    def load_data(self, data_path: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "CreditScore": [650, 700, 720, 680],
                "Geography": ["France", "Spain", "Germany", "France"],
                "Gender": ["Male", "Female", "Female", "Male"],
                "Age": [40, 45, 38, 50],
                "Tenure": [5, 3, 4, 2],
                "Balance": [60000, 80000, 40000, 50000],
                "NumOfProducts": [1, 2, 1, 2],
                "HasCrCard": [1, 0, 1, 1],
                "IsActiveMember": [1, 0, 1, 1],
                "EstimatedSalary": [70000, 50000, 90000, 45000],
                "Exited": [0, 1, 0, 1],
            }
        )

    def preprocess_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame({"feature": [0.1, 0.9, 0.2, 0.8]})
        y = pd.Series([0, 1, 0, 1])
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        self.is_fitted = True
        return {"f1_mean": 0.9, "roc_auc_mean": 0.95}

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        return {
            "metrics": {"f1_score": 0.8, "roc_auc": 0.9},
            "confusion_matrix": [[2, 0], [0, 2]],
            "predictions": np.array([0, 1, 0, 1]),
            "probabilities": np.linspace(0.1, 0.9, len(X)),
        }

    def save_model(self, model_path: str, preprocessor_path: str) -> None:
        Path(model_path).write_text("model")
        Path(preprocessor_path).write_text("preprocessor")
        metadata = Path(model_path).with_name(Path(model_path).stem + "_metadata.json")
        metadata.write_text("{}")

    def load_model(self, model_path: str, preprocessor_path: str) -> None:
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n_rows = len(X)
        return np.zeros(n_rows, dtype=int), np.linspace(0.2, 0.8, n_rows)


def _run_cli(monkeypatch: pytest.MonkeyPatch, args: list[str], cwd: Path) -> None:
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "argv", ["main.py", *args])
    bankchurn_main.main()


@pytest.fixture()
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  test_size: 0.5\n  random_state: 42\n  stratify: true\n")

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"dummy": [1, 2, 3], "Exited": [0, 1, 0]}).to_csv(input_csv, index=False)

    monkeypatch.setattr(bankchurn_main, "BankChurnPredictor", DummyPredictor)
    monkeypatch.setattr(bankchurn_main.joblib, "dump", lambda obj, path: Path(path).write_text("dump"))

    return config_path, input_csv, tmp_path


def test_cli_train_eval_predict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli_env: tuple[Path, Path, Path],
) -> None:
    config_path, input_csv, workdir = cli_env
    model_path = workdir / "model.pkl"
    preprocessor_path = workdir / "preprocessor.pkl"
    output_path = workdir / "predictions.csv"

    _run_cli(
        monkeypatch,
        [
            "--mode",
            "train",
            "--config",
            str(config_path),
            "--input",
            str(input_csv),
            "--model",
            str(model_path),
            "--preprocessor",
            str(preprocessor_path),
        ],
        workdir,
    )

    assert model_path.exists()
    assert preprocessor_path.exists()
    assert (workdir / "results" / "training_results.json").exists()

    _run_cli(
        monkeypatch,
        [
            "--mode",
            "eval",
            "--config",
            str(config_path),
            "--input",
            str(input_csv),
            "--model",
            str(model_path),
            "--preprocessor",
            str(preprocessor_path),
        ],
        workdir,
    )
    captured = capsys.readouterr()
    assert "RESULTADOS DE EVALUACIÓN" in captured.out

    prediction_input = workdir / "predict_input.csv"
    pd.DataFrame({"feature": [0.1, 0.2], "Exited": [0, 1]}).to_csv(prediction_input, index=False)

    _run_cli(
        monkeypatch,
        [
            "--mode",
            "predict",
            "--config",
            str(config_path),
            "--input",
            str(prediction_input),
            "--output",
            str(output_path),
            "--model",
            str(model_path),
            "--preprocessor",
            str(preprocessor_path),
        ],
        workdir,
    )

    assert output_path.exists()
    predictions_df = pd.read_csv(output_path)
    assert {"churn_prediction", "churn_probability", "risk_level"}.issubset(predictions_df.columns)


def test_cli_hyperopt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cli_env: tuple[Path, Path, Path]) -> None:
    config_path, input_csv, workdir = cli_env

    def fake_hyperopt(X: pd.DataFrame, y: pd.Series, n_trials: int = 0) -> dict[str, float]:  # noqa: ARG001
        return {"lr_C": 1.0}

    monkeypatch.setattr(bankchurn_main, "hyperparameter_optimization", fake_hyperopt)

    _run_cli(
        monkeypatch,
        [
            "--mode",
            "hyperopt",
            "--config",
            str(config_path),
            "--input",
            str(input_csv),
            "--n_trials",
            "2",
        ],
        workdir,
    )

    assert (workdir / "results" / "best_hyperparameters.json").exists()
