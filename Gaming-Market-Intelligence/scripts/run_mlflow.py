from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "GamingIntelligence")

    metrics_path = Path("artifacts/metrics/metrics.json")
    metrics = {}
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        except Exception:
            pass

    if mlflow is None:
        print("MLflow not installed; skipping logging. Metrics:", metrics)
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="demo-logging"):
        mlflow.log_params({"run_type": "demo", "note": "Gaming MLflow demo"})
        if metrics:
            mlflow.log_metrics(metrics)
        for art in [metrics_path, Path("configs/config.yaml")]:
            if art.exists():
                mlflow.log_artifact(str(art))
        pack = Path("artifacts/model/model_v1.0.0.pkl")
        if pack.exists():
            mlflow.log_artifact(str(pack))
        print(f"Logged Gaming run to {tracking_uri} in experiment '{experiment}'")


if __name__ == "__main__":
    main()
