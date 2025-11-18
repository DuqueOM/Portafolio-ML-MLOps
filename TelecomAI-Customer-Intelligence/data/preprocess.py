from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Carga el dataset desde CSV.

    Args:
        csv_path: Ruta al archivo CSV.

    Returns:
        DataFrame con los datos.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Dataset loaded: %s, shape=%s", csv_path, df.shape)
    return df


def get_features_target(
    df: pd.DataFrame, features: List[str], target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features y target.

    Args:
        df: Datos completos.
        features: Lista de columnas de entrada.
        target: Nombre de la columna objetivo.

    Returns:
        (X, y)
    """
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    X = df[features].copy()
    y = df[target].copy()
    return X, y


def build_preprocessor(numeric_features: List[str]) -> ColumnTransformer:
    """Crea el preprocesador para variables numéricas.

    - Imputación mediana
    - Estandarización
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor
