from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(cfg_path: Path | None) -> dict:
    if cfg_path and cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text())
    return {
        "data": {
            "drop_columns": ["RowNumber", "CustomerId", "Surname"],
        }
    }


def preprocess(input_csv: Path, output_csv: Path, cfg_path: Path | None = None) -> None:
    cfg = load_config(cfg_path)
    df = pd.read_csv(input_csv)
    drops = cfg.get("data", {}).get("drop_columns", [])
    df = df.drop(columns=[c for c in drops if c in df.columns], errors="ignore")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess BankChurn dataset")
    ap.add_argument("--input", required=True, help="Path to raw CSV")
    ap.add_argument("--output", required=True, help="Path to processed CSV")
    ap.add_argument("--config", default="configs/config.yaml", help="Config YAML")
    args = ap.parse_args()

    preprocess(Path(args.input), Path(args.output), Path(args.config))


if __name__ == "__main__":
    main()
