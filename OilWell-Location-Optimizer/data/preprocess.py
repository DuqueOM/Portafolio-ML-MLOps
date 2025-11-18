from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_region_csv(path: Path | str) -> pd.DataFrame:
    """Load a region CSV file.

    Args:
        path: Path to CSV file.
    Returns:
        DataFrame with columns including id, features and target.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    df = pd.read_csv(p)
    return df


def clean_deduplicate_and_shuffle(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    random_state: int = 12345,
) -> pd.DataFrame:
    """Deduplicate by id keeping highest target, then shuffle deterministically.

    This mirrors the logic used in the approved notebook.
    """
    if id_col not in df.columns or target_col not in df.columns:
        missing = {id_col, target_col} - set(df.columns)
        raise ValueError(f"Missing columns in df: {missing}")
    # keep row with max product per id
    df_sorted = df.sort_values([id_col, target_col], ascending=[True, False])
    df_clean = df_sorted.drop_duplicates(subset=id_col, keep="first")
    df_clean = df_clean.sample(frac=1.0, random_state=random_state).reset_index(
        drop=True
    )
    return df_clean


def split_features_target(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing from DataFrame")
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def compute_profit(
    predicted_units_sum: float,
    revenue_per_unit: float,
    total_cost: float,
) -> float:
    """Compute profit given sum of production units, revenue per unit and total cost."""
    revenue = predicted_units_sum * float(revenue_per_unit)
    return float(revenue - float(total_cost))
