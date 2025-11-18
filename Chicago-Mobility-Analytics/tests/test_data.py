"""Tests para el módulo de preprocesamiento de datos."""

from __future__ import annotations

from pathlib import Path

from data.preprocess import load_and_preprocess, load_config


def test_preprocess_generates_expected_columns() -> None:
    """El preprocesamiento debe generar columnas y filas válidas."""

    cfg = load_config(Path("configs/default.yaml"))
    X, y = load_and_preprocess(cfg)

    assert not X.empty, "La matriz de características no debe estar vacía"
    assert len(X) == len(y), "Features y target deben tener el mismo número de filas"

    expected = {"hour", "day_of_week", "is_weekend", "weather_is_bad"}
    assert expected.issubset(set(X.columns)), "Faltan columnas esperadas en X"

    assert not X.isnull().any().any(), "No debe haber valores nulos en X"
    assert (y > 0).all(), "La duración debe ser positiva"
