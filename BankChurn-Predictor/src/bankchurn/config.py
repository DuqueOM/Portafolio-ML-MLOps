"""Configuration management for BankChurn predictor.

Handles loading and validation of YAML configuration files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model training configuration."""

    test_size: float = Field(0.2, ge=0.0, le=1.0)
    random_state: int = 42
    cv_folds: int = Field(5, ge=2)
    ensemble_voting: str = Field("soft", pattern="^(hard|soft)$")


class DataConfig(BaseModel):
    """Data preprocessing configuration."""

    target_column: str = "Exited"
    categorical_features: list[str] = []
    numerical_features: list[str] = []
    drop_columns: list[str] = []


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""

    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "bankchurn"
    enabled: bool = True


class BankChurnConfig(BaseModel):
    """Complete BankChurn configuration."""

    model: ModelConfig
    data: DataConfig
    mlflow: MLflowConfig

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> BankChurnConfig:
        """Load configuration from YAML file.

        Parameters
        ----------
        config_path : str or Path
            Path to YAML configuration file.

        Returns
        -------
        config : BankChurnConfig
            Validated configuration object.

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist.
        ValidationError
            If config is invalid.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            # yaml.safe_load can return None for empty files
            config_dict = yaml.safe_load(f) or {}

        # Provide sensible defaults for missing sections so that
        # older/focused configs without an explicit mlflow block
        # still validate correctly, especially in tests/CI.
        if "model" not in config_dict:
            config_dict["model"] = ModelConfig().dict()
        if "data" not in config_dict:
            config_dict["data"] = DataConfig().dict()
        if "mlflow" not in config_dict:
            config_dict["mlflow"] = MLflowConfig().dict()

        logger.info(f"Loaded configuration from {config_path}")
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns
        -------
        dict
            Configuration as nested dictionary.
        """
        return self.dict()
