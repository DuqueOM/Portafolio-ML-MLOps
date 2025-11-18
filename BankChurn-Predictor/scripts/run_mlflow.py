from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "BankChurn")

    results_path = Path("results/training_results.json")
    metrics: dict[str, float] = {}
    business_metrics: dict[str, float] = {}
    if results_path.exists():
        try:
            data = json.loads(results_path.read_text())
            cv = data.get("cv_results", {})
            for k, v in cv.items():
                if isinstance(v, (int, float)):
                    metrics[f"cv_{k}"] = float(v)

            test_results = data.get("test_results", {})
            test_metrics = test_results.get("metrics", {})
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[f"test_{k}"] = float(v)

            # Derivar métricas de negocio (proxy) a partir de la matriz de confusión
            # Supuestos: CLV medio y tasa de retención efectiva configurables vía entorno.
            cm = test_results.get("confusion_matrix")
            if (
                isinstance(cm, list)
                and len(cm) == 2
                and all(isinstance(row, list) and len(row) == 2 for row in cm)
            ):
                tn, fp = cm[0]
                fn, tp = cm[1]
                total_at_risk = tp + fn
                detected_at_risk = tp

                # Valores por defecto tomados del resumen ejecutivo (pueden overridearse por entorno)
                clv = float(os.getenv("BC_CLV_USD", "2300"))
                retention_rate = float(os.getenv("BC_RETENTION_RATE", "0.3"))

                saved_customers = float(detected_at_risk) * retention_rate
                saved_revenue = saved_customers * clv

                business_metrics = {
                    "biz_total_at_risk_customers": float(total_at_risk),
                    "biz_detected_at_risk_customers": float(detected_at_risk),
                    "biz_saved_customers_proxy": saved_customers,
                    "biz_saved_revenue_proxy_usd": saved_revenue,
                    "biz_false_positives": float(fp),
                    "biz_false_negatives": float(fn),
                }
        except Exception:
            pass

    if mlflow is None:
        print("MLflow not installed; skipping logging. Metrics:", metrics)
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="demo-logging"):
        mlflow.log_params({"run_type": "demo", "note": "BankChurn MLflow demo logging"})
        if metrics:
            mlflow.log_metrics(metrics)
        if business_metrics:
            mlflow.log_metrics(business_metrics)
        for p in [Path("results/training_results.json"), Path("configs/config.yaml")]:
            if p.exists():
                mlflow.log_artifact(str(p))

        # Log combined model pack as artifact if present
        combined = Path("models/model_v1.0.0.pkl")
        if combined.exists():
            mlflow.log_artifact(str(combined))
            # Try to register a sklearn Pipeline for the combined pack (optional)
            try:
                obj = joblib.load(combined)
                if isinstance(obj, dict) and "preprocessor" in obj and "model" in obj:
                    pipe = Pipeline(
                        [
                            ("preprocessor", obj["preprocessor"]),
                            ("model", obj["model"]),
                        ]
                    )
                    registered_name = os.getenv("MLFLOW_REGISTERED_MODEL", "BankChurn")
                    import mlflow.sklearn as mlflow_sklearn  # type: ignore

                    mlflow_sklearn.log_model(
                        pipe,
                        artifact_path="model",
                        registered_model_name=registered_name,
                    )
            except Exception:
                # registry may be unavailable (e.g., file store); ignore
                pass

        print(f"Logged BankChurn run to {tracking_uri} in experiment '{experiment}'")


if __name__ == "__main__":
    main()
