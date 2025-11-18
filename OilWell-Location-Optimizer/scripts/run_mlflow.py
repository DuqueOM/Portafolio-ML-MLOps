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
    experiment = os.getenv("MLFLOW_EXPERIMENT", "OilWell")

    metrics_path = Path("artifacts/metrics.json")
    risk_path = Path("artifacts/risk_results.json")

    agg_metrics: dict[str, float] = {}
    if metrics_path.exists():
        try:
            metrics_data = json.loads(metrics_path.read_text())
            regions = metrics_data.get("regions", {})
            rmses = [
                v.get("rmse")
                for v in regions.values()
                if isinstance(v.get("rmse"), (int, float))
            ]
            if rmses:
                agg_metrics["rmse_avg"] = float(sum(rmses) / len(rmses))
            for rid, vals in regions.items():
                if isinstance(vals, dict):
                    if "rmse" in vals:
                        agg_metrics[f"rmse_region_{rid}"] = float(vals["rmse"])  # type: ignore[arg-type]
                    if "baseline_rmse" in vals:
                        agg_metrics[f"baseline_rmse_region_{rid}"] = float(vals["baseline_rmse"])  # type: ignore[arg-type]
        except Exception:
            pass

    if mlflow is None:
        print("MLflow not installed; skipping logging. Metrics:", agg_metrics)
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="demo-logging"):
        mlflow.log_params({"run_type": "demo", "note": "OilWell MLflow demo logging"})
        if agg_metrics:
            mlflow.log_metrics(agg_metrics)
        for p in [metrics_path, risk_path, Path("configs/default.yaml")]:
            if p.exists():
                mlflow.log_artifact(str(p))
        print(f"Logged OilWell run to {tracking_uri} in experiment '{experiment}'")


if __name__ == "__main__":
    main()
