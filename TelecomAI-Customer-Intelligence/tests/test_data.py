from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_dataset_schema():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "users_behavior.csv"
    assert csv_path.exists(), "users_behavior.csv no encontrado"

    df = pd.read_csv(csv_path)
    expected_cols = {"calls", "minutes", "messages", "mb_used", "is_ultra"}
    assert expected_cols.issubset(
        df.columns
    ), f"Columnas faltantes: {expected_cols - set(df.columns)}"

    # Chequeos simples
    assert df.shape[0] > 100, "Dataset demasiado pequeño"
    for col in ["calls", "minutes", "messages", "mb_used"]:
        assert df[col].notnull().mean() > 0.9, f"Columna {col} con demasiados nulos"
    assert set(df["is_ultra"].unique()).issubset({0, 1}), "Target debe ser binario 0/1"
    # Chequeo de proporciones de clase: evitar clases extremadamente desbalanceadas
    pos_ratio = df["is_ultra"].mean()
    assert (
        0.1 <= pos_ratio <= 0.9
    ), f"Proporción de clase positiva fuera de rango: {pos_ratio:.3f}"
