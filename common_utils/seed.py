"""Centralized seed helper for reproducibility across projects.

This module configures Python's random module, NumPy, and, when
installed, optional ML frameworks such as PyTorch and TensorFlow.
"""

from __future__ import annotations

import os
import random
from typing import Final, Optional

import numpy as np

# Default seed used when callers do not provide one explicitly
DEFAULT_SEED: Final[int] = 42


def set_seed(seed: Optional[int] = None) -> int:
    """Set global random seeds for reproducible experiments.

    Resolution order:
    1. Explicit ``seed`` argument when provided.
    2. Environment variable ``SEED`` if defined.
    3. Fallback to ``DEFAULT_SEED`` (42).

    Returns the seed value that was actually used.
    """

    if seed is None:
        env_seed = os.getenv("SEED")
        seed = int(env_seed) if env_seed is not None else DEFAULT_SEED

    # Core Python / NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Optional: PyTorch
    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:  # ImportError or runtime CUDA issues
        pass

    # Optional: TensorFlow
    try:  # pragma: no cover - optional dependency
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass

    return seed
