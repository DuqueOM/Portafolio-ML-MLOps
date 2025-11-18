from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def ks_stat(x: np.ndarray, y: np.ndarray, n_bins: int = 50) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return 0.0
    bins = np.linspace(np.min(np.concatenate([x, y])), np.max(np.concatenate([x, y])), n_bins + 1)
    x_cdf = np.cumsum(np.histogram(x, bins=bins)[0]) / max(len(x), 1)
    y_cdf = np.cumsum(np.histogram(y, bins=bins)[0]) / max(len(y), 1)
    return float(np.max(np.abs(x_cdf - y_cdf)))


def psi_stat(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return 0.0
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(x, quantiles))
    if len(bins) < 3:
        bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    x_counts, _ = np.histogram(x, bins=bins)
    y_counts, _ = np.histogram(y, bins=bins)
    x_perc = x_counts / max(x_counts.sum(), 1)
    y_perc = y_counts / max(y_counts.sum(), 1)
    x_perc = np.clip(x_perc, 1e-6, None)
    y_perc = np.clip(y_perc, 1e-6, None)
    return float(np.sum((y_perc - x_perc) * np.log(y_perc / x_perc)))


def compute_drift(ref: pd.DataFrame, cur: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in cols:
        x = ref[c].to_numpy(dtype=float)
        y = cur[c].to_numpy(dtype=float)
        out[c] = {"ks": ks_stat(x, y), "psi": psi_stat(x, y)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="GoldRecovery drift check (KS, PSI)")
    ap.add_argument("--ref", required=True, help="Reference CSV path")
    ap.add_argument("--cur", required=True, help="Current CSV path")
    ap.add_argument("--cols", nargs="+", default=["final.output.recovery"], help="Columns to analyze")
    ap.add_argument("--out-json", default="results/drift.json", help="Output JSON path")
    args = ap.parse_args()

    ref = pd.read_csv(args.ref)
    cur = pd.read_csv(args.cur)
    cols = [c for c in args.cols if c in ref.columns and c in cur.columns]
    drift = compute_drift(ref, cur, cols)

    # Resumen y recomendación simple basada en umbrales de PSI/KS
    all_psi = [v["psi"] for v in drift.values()]
    all_ks = [v["ks"] for v in drift.values()]
    max_psi = max(all_psi) if all_psi else 0.0
    max_ks = max(all_ks) if all_ks else 0.0

    # Reglas simples (orientativas):
    # - PSI < 0.1 y KS < 0.1 → drift bajo (OK)
    # - PSI entre 0.1 y 0.25 o KS entre 0.1 y 0.2 → revisar (monitorizar)
    # - PSI >= 0.25 o KS >= 0.2 → recomendar reentrenar / investigar
    if max_psi >= 0.25 or max_ks >= 0.2:
        action = "retrain"
        reason = "high_drift_detected"
    elif max_psi >= 0.1 or max_ks >= 0.1:
        action = "review"
        reason = "moderate_drift_detected"
    else:
        action = "ok"
        reason = "low_drift"

    summary = {
        "max_psi": float(max_psi),
        "max_ks": float(max_ks),
        "recommended_action": action,
        "reason": reason,
    }

    out = {"columns": cols, "drift": drift, "summary": summary}
    print(json.dumps(out, indent=2))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
