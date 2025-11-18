from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from evidently import ColumnMapping  # type: ignore
    from evidently.metrics import DataDriftPreset  # type: ignore
    from evidently.report import Report  # type: ignore
except Exception:  # pragma: no cover
    ColumnMapping = None  # type: ignore
    Report = None  # type: ignore
    DataDriftPreset = None  # type: ignore


DEFAULT_COLS = ["CreditScore", "Age", "Balance", "EstimatedSalary"]


def ks_stat(x: np.ndarray, y: np.ndarray, n_bins: int = 50) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return 0.0
    bins = np.linspace(
        np.min(np.concatenate([x, y])), np.max(np.concatenate([x, y])), n_bins + 1
    )
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


def compute_drift(
    ref: pd.DataFrame, cur: pd.DataFrame, cols: List[str]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in cols:
        x = ref[c].to_numpy(dtype=float)
        y = cur[c].to_numpy(dtype=float)
        out[c] = {"ks": ks_stat(x, y), "psi": psi_stat(x, y)}
    return out


def maybe_evidently_report(
    ref: pd.DataFrame, cur: pd.DataFrame, cols: List[str], html_path: Path | None
) -> None:
    if html_path is None:
        return
    if Report is None or ColumnMapping is None or DataDriftPreset is None:
        print("Evidently not installed; skipping HTML report generation")
        return
    mapping = ColumnMapping()
    mapping.numerical_features = cols
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref[cols], current_data=cur[cols], column_mapping=mapping)
    report.save_html(str(html_path))


def main() -> None:
    ap = argparse.ArgumentParser(description="BankChurn drift check (KS, PSI)")
    ap.add_argument("--ref", required=True, help="Reference CSV path")
    ap.add_argument("--cur", required=True, help="Current CSV path")
    ap.add_argument(
        "--cols", nargs="*", default=DEFAULT_COLS, help="Columns to analyze"
    )
    ap.add_argument("--out-json", default="results/drift.json", help="Output JSON path")
    ap.add_argument("--report-html", default=None, help="Optional Evidently HTML path")
    args = ap.parse_args()

    ref = pd.read_csv(args.ref)
    cur = pd.read_csv(args.cur)
    cols = [c for c in args.cols if c in ref.columns and c in cur.columns]
    drift = compute_drift(ref, cur, cols)

    out = {"columns": cols, "drift": drift}
    print(json.dumps(out, indent=2))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    html_path = Path(args.report_html) if args.report_html else None
    maybe_evidently_report(ref, cur, cols, html_path)


if __name__ == "__main__":
    main()
