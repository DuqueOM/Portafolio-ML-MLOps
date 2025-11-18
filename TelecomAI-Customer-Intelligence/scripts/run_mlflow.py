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
    experiment = os.getenv("MLFLOW_EXPERIMENT", "TelecomAI")

    metrics_path = Path("artifacts/metrics.json")
    metrics: dict[str, float] = {"placeholder_metric": 0.0}
    business_metrics: dict[str, float] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            pass

    params = {
        "run_type": "demo",
        "note": "Example MLflow logging run",
    }

    if mlflow is None:
        print("MLflow not installed; skipping logging. Metrics:", metrics)
        return

    # Intentar derivar una métrica de negocio: clientes elegibles correctamente clasificados
    # A partir de:
    #   - recall sobre la clase is_ultra=1 (si está en metrics.json)
    #   - número total de clientes elegibles en users_behavior.csv (leído vía configs/config.yaml)
    try:
        recall = metrics.get(
            "recall"
        )  # asumimos que `recall` es para la clase positiva
        cfg_path = Path("configs/config.yaml")
        if recall is not None and cfg_path.exists():
            cfg = json.loads(
                json.dumps(__import__("yaml").safe_load(cfg_path.read_text()))
            )
            data_csv = cfg.get("paths", {}).get("data_csv", "users_behavior.csv")
            data_path = Path(data_csv)
            if data_path.exists():
                import pandas as pd

                df = pd.read_csv(data_path)
                if "is_ultra" in df.columns:
                    n_eligible = int(df["is_ultra"].sum())
                    correctly_classified = float(n_eligible * float(recall))
                    business_metrics = {
                        "biz_total_eligible_customers": float(n_eligible),
                        "biz_correctly_classified_eligible": correctly_classified,
                        "biz_correctly_classified_eligible_rate": float(recall),
                    }
    except Exception:
        # Cualquier fallo en el cálculo de la métrica de negocio no debe romper el logging
        business_metrics = {}

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="demo-logging"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if business_metrics:
            mlflow.log_metrics(business_metrics)
        # Log artifacts if exist
        for art in [
            "artifacts/metrics.json",
            "artifacts/confusion_matrix.png",
            "artifacts/roc_curve.png",
        ]:
            p = Path(art)
            if p.exists():
                mlflow.log_artifact(str(p))
        print(f"Logged run to {tracking_uri} in experiment '{experiment}'")


if __name__ == "__main__":
    main()
