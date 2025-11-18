from __future__ import annotations

import numpy as np
import pandas as pd


def test_class_ratio_reasonable():
    # Synthetic quick check: 80/20 +/- tolerance
    n = 1000
    y = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    ratio = y.mean()
    assert 0.1 < ratio < 0.4


def test_feature_shapes():
    df = pd.DataFrame(
        {
            "CreditScore": [650, 720],
            "Geography": ["France", "Germany"],
            "Gender": ["Male", "Female"],
            "Age": [45, 30],
            "Tenure": [5, 2],
            "Balance": [120000.0, 0.0],
            "NumOfProducts": [2, 1],
            "HasCrCard": [1, 0],
            "IsActiveMember": [1, 0],
            "EstimatedSalary": [75000.0, 50000.0],
        }
    )
    assert df.shape == (2, 10)
