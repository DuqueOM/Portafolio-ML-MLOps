#!/usr/bin/env python3
from __future__ import annotations

"""
OilWell Location Optimizer — CLI

Uso:
  python main.py --mode train --config configs/default.yaml --seed 12345
  python main.py --mode eval --config configs/default.yaml
  python main.py --mode predict --config configs/default.yaml --region 1 --payload '{"records":[{"f0":1.0,"f1":-2.0,"f2":3.0}]}'

Modos:
  - train   Entrena un modelo por región y guarda artefactos.
  - eval    Ejecuta bootstrap por región y guarda métricas de riesgo.
  - predict Predice `product` para registros de una región con el modelo entrenado.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from data.preprocess import (
    clean_deduplicate_and_shuffle,
    load_region_csv,
    split_features_target,
)
from evaluate import (
    bootstrap_region_profit,
    evaluate_baseline,
    prepare_with_predictions,
    rmse_score,
    split_train_val,
    train_linear_regression,
)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_config(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Dict, base: Path) -> Dict[str, Path]:
    proj = cfg["project"]
    artifacts_dir = base / proj["artifacts_dir"]
    models_dir = base / proj["models_dir"]
    preds_dir = base / proj["predictions_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    return {
        "artifacts": artifacts_dir,
        "models": models_dir,
        "predictions": preds_dir,
        "metrics": base / proj["metrics_path"],
        "risk": base / proj["risk_results_path"],
    }


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def cmd_train(cfg: Dict, base: Path) -> None:
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    paths = ensure_dirs(cfg, base)

    region_metrics: Dict[str, Dict] = {}
    all_models: Dict[int, Any] = {}

    for region_key, rel_path in data_cfg["regions"].items():
        region = int(region_key)
        data_path = (base / rel_path).resolve()
        logging.info(f"[train] region={region} file={data_path}")
        df = load_region_csv(data_path)
        df = clean_deduplicate_and_shuffle(
            df,
            id_col=data_cfg["id_column"],
            target_col=data_cfg["target_column"],
            random_state=train_cfg["random_state"],
        )
        X, y = split_features_target(
            df, data_cfg["feature_columns"], data_cfg["target_column"]
        )
        Xtr, Xva, ytr, yva = split_train_val(
            X,
            y,
            test_size=float(train_cfg["test_size"]),
            random_state=int(train_cfg["random_state"]),
        )
        model = train_linear_regression(Xtr, ytr)
        y_pred = model.predict(Xva)
        rmse = rmse_score(yva, y_pred)
        baseline_rmse = evaluate_baseline(Xtr, ytr, Xva, yva)

        model_path = paths["models"] / f"region_{region}.joblib"
        joblib.dump(model, model_path)
        logging.info(
            f"[train] region={region} rmse={rmse:.2f} baseline_rmse={baseline_rmse:.2f} -> {model_path}"
        )

        region_metrics[str(region)] = {
            "rmse": round(rmse, 4),
            "baseline_rmse": round(baseline_rmse, 4),
            "mean_true": round(float(np.mean(yva)), 4),
            "mean_pred": round(float(np.mean(y_pred)), 4),
            "model_path": str(model_path.relative_to(base)),
        }
        # keep in-memory for combined export
        all_models[region] = model

    # export combined model pack for demo usage
    combined_model_path = paths["models"] / "model_v1.0.0.pkl"
    try:
        joblib.dump({"models": all_models, "version": "1.0.0"}, combined_model_path)
        logging.info(f"[train] exported combined model pack -> {combined_model_path}")
    except Exception as e:
        logging.warning(f"[train] failed to export combined model pack: {e}")

    save_json(paths["metrics"], {"regions": region_metrics})
    logging.info(f"Saved metrics -> {paths['metrics']}")


def cmd_eval(cfg: Dict, base: Path, seed: int) -> None:
    data_cfg = cfg["data"]
    boot = cfg["bootstrap"]
    fin = cfg["financial"]
    paths = ensure_dirs(cfg, base)

    results: Dict[str, Dict] = {}

    for region_key, rel_path in data_cfg["regions"].items():
        region = int(region_key)
        model_path = paths["models"] / f"region_{region}.joblib"
        if not model_path.exists():
            logging.warning(
                f"[eval] skipping region={region} (model not found). Run train first."
            )
            continue
        model = joblib.load(model_path)

        df = load_region_csv(base / rel_path)
        df = clean_deduplicate_and_shuffle(
            df,
            id_col=data_cfg["id_column"],
            target_col=data_cfg["target_column"],
            random_state=seed,
        )
        dfp = prepare_with_predictions(df, model, data_cfg["feature_columns"])
        r = bootstrap_region_profit(
            df_with_preds=dfp,
            n_bootstrap=int(boot["n_bootstrap"]),
            n_explore=int(boot["n_explore"]),
            n_select=int(boot["n_select"]),
            price_per_unit=float(fin["revenue_per_unit"]),
            total_investment=float(fin["total_investment"]),
            random_state=int(seed + region),
        )
        results[str(region)] = {
            "expected_profit": round(float(r["expected_profit"]), 2),
            "ci_lower": round(float(r["ci_lower"]), 2),
            "ci_upper": round(float(r["ci_upper"]), 2),
            "loss_probability": round(float(r["loss_probability"]), 4),
            "model_path": str(model_path.relative_to(base)),
        }
        logging.info(
            f"[eval] region={region} expected_profit=${results[str(region)]['expected_profit']:,} "
            f"loss_prob={results[str(region)]['loss_probability']:.2%}"
        )

    save_json(paths["risk"], {"regions": results})
    logging.info(f"Saved risk results -> {paths['risk']}")


def cmd_predict(
    cfg: Dict, base: Path, region: int, payload: str | None, input_file: Path | None
) -> None:
    paths = ensure_dirs(cfg, base)
    model_path = paths["models"] / f"region_{region}.joblib"
    if not model_path.exists():
        raise SystemExit(f"Model for region {region} not found. Run train first.")
    model = joblib.load(model_path)

    if payload:
        data = json.loads(payload)
    elif input_file:
        with Path(input_file).open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise SystemExit("Provide --payload JSON or --input-file path")

    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise SystemExit("Payload must contain non-empty 'records' list")

    X = pd.DataFrame(records, columns=["f0", "f1", "f2"])  # fixed order
    preds = model.predict(X)
    out = {"region": region, "predictions": [float(p) for p in preds]}
    print(json.dumps(out, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OilWell Location Optimizer CLI")
    p.add_argument(
        "--mode",
        required=True,
        choices=["train", "eval", "predict"],
        help="Execution mode",
    )
    p.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config"
    )
    p.add_argument("--seed", type=int, default=12345, help="Random seed")
    p.add_argument("--region", type=int, help="Region id for predict mode")
    p.add_argument("--payload", type=str, help="Inline JSON payload for predict mode")
    p.add_argument(
        "--input-file", type=str, help="Path to JSON payload file for predict mode"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    cfg = load_config(base / args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    if args.mode == "train":
        cmd_train(cfg, base)
    elif args.mode == "eval":
        cmd_eval(cfg, base, seed=args.seed)
    elif args.mode == "predict":
        if args.region is None:
            raise SystemExit("--region is required for predict mode")
        cmd_predict(
            cfg,
            base,
            region=int(args.region),
            payload=args.payload,
            input_file=Path(args.input_file) if args.input_file else None,
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
