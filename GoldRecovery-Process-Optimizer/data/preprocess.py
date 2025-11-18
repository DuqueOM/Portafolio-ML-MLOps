from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

TARGET_COL = "final.output.recovery"


def compute_recovery(
    feed: pd.Series, concentrate: pd.Series, tail: pd.Series
) -> pd.Series:
    """Calcula recovery (%) de acuerdo a la formula clasica del proyecto.

    recovery = 100 * (C * (F - T)) / (F * (C - T))

    Donde:
    - F: concentracion en el feed
    - C: concentracion en el concentrado
    - T: concentracion en el relave (tail)
    """
    f_s = feed.astype(float)
    c_s = concentrate.astype(float)
    t_s = tail.astype(float)

    num = c_s * (f_s - t_s)
    den = f_s * (c_s - t_s)

    den_safe = den.copy()
    near_zero = den_safe.abs() < 1e-9
    den_safe[near_zero] = np.nan

    rec = 100.0 * (num / den_safe)
    return rec


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    # Filtrar valores validos de recovery si existe la columna
    if TARGET_COL in df.columns:
        df = df[(df[TARGET_COL] >= 0) & (df[TARGET_COL] <= 100)]

    # Eliminar columnas con mas de 60% de nulos
    null_ratio = df.isna().mean()
    keep_cols = null_ratio[null_ratio <= 0.6].index.tolist()
    df = df[keep_cols]

    return df


def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ratios de recuperacion (si las columnas existen)
    if {
        "rougher.output.concentrate_au",
        "primary_cleaner.output.concentrate_au",
    }.issubset(df.columns):
        df["au_recovery_ratio"] = df["primary_cleaner.output.concentrate_au"] / (
            df["rougher.output.concentrate_au"] + 1e-6
        )

    if {
        "rougher.output.concentrate_ag",
        "primary_cleaner.output.concentrate_ag",
    }.issubset(df.columns):
        df["ag_recovery_ratio"] = df["primary_cleaner.output.concentrate_ag"] / (
            df["rougher.output.concentrate_ag"] + 1e-6
        )

    if "date" in df.columns:
        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

    return df


def align_columns(
    train_df: pd.DataFrame, other_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Alinea columnas entre train y otro dataset (test), agregando faltantes con NaN."""
    train_cols = train_df.columns
    other_aligned = other_df.reindex(columns=train_cols, fill_value=np.nan)
    return train_df, other_aligned


def load_csvs(paths: List[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    return df


def preprocess(paths: List[str], compute_missing_recovery: bool = True) -> pd.DataFrame:
    """Carga y preprocesa datos para entrenamiento/prediccion."""
    df = load_csvs(paths)

    # Si falta rougher.output.recovery, calcularlo si hay columnas necesarias
    if compute_missing_recovery and "rougher.output.recovery" not in df.columns:
        req = {
            "rougher.input.feed_au",
            "rougher.output.concentrate_au",
            "rougher.output.tail_au",
        }
        if req.issubset(df.columns):
            df["rougher.output.recovery_calc"] = compute_recovery(
                df["rougher.input.feed_au"],
                df["rougher.output.concentrate_au"],
                df["rougher.output.tail_au"],
            )

    df = basic_clean(df)
    df = create_features(df)
    df = fill_missing_with_median(df)
    return df
