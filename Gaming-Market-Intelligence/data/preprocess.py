from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessConfig:
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    scale_numeric: bool = True
    one_hot_drop: str | None = "first"
    include_features: List[str] | None = None
    exclude_features: List[str] | None = None
    target_threshold_million: float = 1.0


RAW_COLUMNS = {
    "name": "name",
    "platform": "platform",
    "year": "year_of_release",
    "genre": "genre",
    "na_sales": "na_sales",
    "eu_sales": "eu_sales",
    "jp_sales": "jp_sales",
    "other_sales": "other_sales",
    "critic_score": "critic_score",
    "user_score": "user_score",
    "rating": "rating",
}


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Load CSV dataset and perform minimal normalization of column names/types."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize types
    if "user_score" in df.columns:
        # Replace 'tbd' and coerce to numeric
        s = df["user_score"].replace({"tbd": np.nan})
        df["user_score"] = pd.to_numeric(s, errors="coerce")

    if "critic_score" in df.columns:
        df["critic_score"] = pd.to_numeric(df["critic_score"], errors="coerce")

    if "year_of_release" in df.columns:
        df["year_of_release"] = pd.to_numeric(
            df["year_of_release"], errors="coerce"
        ).astype("Int64")

    # Compute total_sales to derive target
    required_sales = [
        RAW_COLUMNS["na_sales"],
        RAW_COLUMNS["eu_sales"],
        RAW_COLUMNS["jp_sales"],
        RAW_COLUMNS["other_sales"],
    ]
    if all(col in df.columns for col in required_sales):
        df["total_sales"] = df[required_sales].sum(axis=1)
    else:
        missing = [c for c in required_sales if c not in df.columns]
        logging.warning(
            "Missing sales columns: %s. total_sales won't be computed.", missing
        )
        df["total_sales"] = np.nan

    return df


def make_features_and_target(
    df: pd.DataFrame, *, config: PreprocessConfig
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and binary target from raw df.

    Target is_successful := total_sales >= threshold (in millions).
    Sales columns are excluded from features to avoid leakage.
    """
    df = df.copy()

    if "total_sales" not in df.columns:
        raise ValueError("total_sales not found. Ensure load_raw_dataset computed it.")

    # Build target
    threshold = float(config.target_threshold_million)
    y = (df["total_sales"] >= threshold).astype(int)

    # Select features
    include = config.include_features or [
        RAW_COLUMNS["platform"],
        RAW_COLUMNS["year"],
        RAW_COLUMNS["genre"],
        RAW_COLUMNS["critic_score"],
        RAW_COLUMNS["user_score"],
        RAW_COLUMNS["rating"],
    ]
    exclude = set(
        (config.exclude_features or [])
        + [
            RAW_COLUMNS["name"],
            RAW_COLUMNS["na_sales"],
            RAW_COLUMNS["eu_sales"],
            RAW_COLUMNS["jp_sales"],
            RAW_COLUMNS["other_sales"],
            "total_sales",
        ]
    )

    X = df[[c for c in include if c in df.columns and c not in exclude]].copy()

    return X, y


def build_preprocessor(
    X: pd.DataFrame, *, config: PreprocessConfig
) -> ColumnTransformer:
    """Create a ColumnTransformer for numeric/categorical preprocessing."""
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_steps: List[tuple] = [
        ("imputer", SimpleImputer(strategy=config.numeric_imputer_strategy))
    ]
    if config.scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps: List[tuple] = [
        ("imputer", SimpleImputer(strategy=config.categorical_imputer_strategy)),
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore", drop=config.one_hot_drop),
        ),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), numeric_cols),
            ("cat", Pipeline(steps=cat_steps), categorical_cols),
        ]
    )
    return preprocessor
