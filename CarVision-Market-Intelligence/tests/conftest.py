from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common_utils.seed import set_seed


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """Set a deterministic global seed for every test.

    Resolution order:
    1. TEST_SEED env var if defined.
    2. SEED env var if defined.
    3. Fallback to 42.
    """

    seed = int(os.getenv("TEST_SEED", os.getenv("SEED", "42")))
    set_seed(seed)
    yield
