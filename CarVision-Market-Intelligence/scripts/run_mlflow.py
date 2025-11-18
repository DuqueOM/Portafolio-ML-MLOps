from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "CarVision")

    # Use validation metrics if available
    metrics_path = Path("artifacts/metrics_val.json")
    metrics: dict[str, float] = {"rmse": 0.0, "mae": 0.0}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            pass

    params = {
        "run_type": "demo",
        "note": "CarVision MLflow demo logging",
    }

    if mlflow is None:
        print("MLflow not installed; skipping logging. Metrics:", metrics)
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="demo-logging"):
        mlflow.log_params(params)
        # If metrics were persisted as a dict with str->float
        if isinstance(metrics, dict):
            # Flatten nested metrics (if any)
            flat_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    flat_metrics[k] = float(v)
            if flat_metrics:
                mlflow.log_metrics(flat_metrics)
        # Log artifacts if exist
        for art in ["artifacts/metrics_val.json", "artifacts/feature_columns.json"]:
            p = Path(art)
            if p.exists():
                mlflow.log_artifact(str(p))
        print(f"Logged CarVision run to {tracking_uri} in experiment '{experiment}'")


if __name__ == "__main__":
    main()
