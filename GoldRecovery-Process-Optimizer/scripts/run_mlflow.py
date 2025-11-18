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
    experiment = os.getenv("MLFLOW_EXPERIMENT", "GoldRecovery")

    metrics = {}
    for p in [Path("results/cv_results.json"), Path("results/metrics.json")]:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                for k, v in d.items():
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
        mlflow.log_params({"run_type": "demo", "note": "GoldRecovery MLflow demo"})
        if metrics:
            mlflow.log_metrics(metrics)
        for art in [Path("results/cv_results.json"), Path("results/metrics.json"), Path("configs/config.yaml")]:
            if art.exists():
                mlflow.log_artifact(str(art))
        pack = Path("models/model_v1.0.0.pkl")
        if pack.exists():
            mlflow.log_artifact(str(pack))
        print(f"Logged GoldRecovery run to {tracking_uri} in experiment '{experiment}'")


if __name__ == "__main__":
    main()
