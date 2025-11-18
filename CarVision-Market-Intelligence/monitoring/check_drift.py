from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

try:  # optional
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
except Exception:  # pragma: no cover
    Report = None  # type: ignore
    DataDriftPreset = None  # type: ignore


@dataclass
class DriftResult:
    ks: Dict[str, Dict[str, float]]
    psi: Dict[str, float]
    evidently_html: Optional[str] = None


def compute_psi(ref: np.ndarray, cur: np.ndarray, buckets: int = 10) -> float:
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    quantiles = np.percentile(ref, np.linspace(0, 100, buckets + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9

    ref_counts, _ = np.histogram(ref, bins=quantiles)
    cur_counts, _ = np.histogram(cur, bins=quantiles)

    ref_dist = ref_counts / max(ref_counts.sum(), 1)
    cur_dist = cur_counts / max(cur_counts.sum(), 1)

    ref_dist = np.where(ref_dist == 0, 1e-6, ref_dist)
    cur_dist = np.where(cur_dist == 0, 1e-6, cur_dist)

    psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
    return float(psi)


def calc_drift(
    ref_df: pd.DataFrame, cur_df: pd.DataFrame, features: List[str]
) -> DriftResult:
    ks_metrics: Dict[str, Dict[str, float]] = {}
    psi_metrics: Dict[str, float] = {}
    for col in features:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue
        ref = ref_df[col].to_numpy(dtype=float)
        cur = cur_df[col].to_numpy(dtype=float)
        stat, pval = ks_2samp(ref, cur)
        ks_metrics[col] = {"ks_stat": float(stat), "p_value": float(pval)}
        psi_metrics[col] = compute_psi(ref, cur)
    return DriftResult(ks=ks_metrics, psi=psi_metrics)


def maybe_generate_evidently(
    ref_df: pd.DataFrame, cur_df: pd.DataFrame, output_html: Path
) -> Optional[str]:
    if Report is None or DataDriftPreset is None:
        return None
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_html))
        return str(output_html)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="CarVision drift check (KS & PSI)")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--cur", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--out", default="artifacts/drift_report.json")
    parser.add_argument(
        "--evidently_html", default="artifacts/evidently_drift_report.html"
    )
    args = parser.parse_args()

    ref_df = pd.read_csv(args.ref)
    cur_df = pd.read_csv(args.cur)

    res = calc_drift(ref_df, cur_df, args.features)
    html_path = maybe_generate_evidently(ref_df, cur_df, Path(args.evidently_html))
    if html_path:
        res.evidently_html = html_path

    # Resumen simple y sugerencia de reentreno basada en umbrales
    max_psi = max(res.psi.values()) if res.psi else float("nan")
    min_pval = min((v["p_value"] for v in res.ks.values())) if res.ks else float("nan")
    num_high_psi = sum(1 for v in res.psi.values() if v >= 0.1)

    recommend_retrain = False
    reason = "no_drift_detected"
    if not np.isnan(max_psi) and max_psi >= 0.25:
        recommend_retrain = True
        reason = "max_psi_ge_0_25"
    elif num_high_psi >= 3:
        recommend_retrain = True
        reason = "psi_ge_0_1_in_many_features"
    elif not np.isnan(min_pval) and min_pval < 0.01:
        recommend_retrain = True
        reason = "ks_p_value_lt_0_01"

    summary = {
        "max_psi": float(max_psi) if not np.isnan(max_psi) else None,
        "min_p_value": float(min_pval) if not np.isnan(min_pval) else None,
        "num_features_psi_ge_0_1": int(num_high_psi),
        "recommend_retrain": recommend_retrain,
        "recommendation_reason": reason,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ks": res.ks,
                "psi": res.psi,
                "evidently_html": res.evidently_html,
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"Drift report saved to {args.out}")


if __name__ == "__main__":
    main()
