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
    experiment = os.getenv("MLFLOW_EXPERIMENT", "ChicagoMobility")

    metrics = {}
    for p in [Path("artifacts/metrics.json")]:
        if p.exists():
            try:
                m = json.loads(p.read_text())
                # flatten nested dict
                for split, vals in m.items():
                    if isinstance(vals, dict):
                        for k, v in vals.items():
                            if isinstance(v, (int, float)):
                                metrics[f"{split}_{k}"] = float(v)
            except Exception:
                pass

    if mlflow is None:
        print("MLflow not installed; skipping logging. Metrics:", metrics)
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="demo-logging"):
        mlflow.log_params({"run_type": "demo", "note": "Chicago MLflow demo logging"})
        if metrics:
            mlflow.log_metrics(metrics)
        for art in [Path("artifacts/metrics.json"), Path("configs/default.yaml")]:
            if art.exists():
                mlflow.log_artifact(str(art))
        print(f"Logged Chicago run to {tracking_uri} in experiment '{experiment}'")


if __name__ == "__main__":
    main()
