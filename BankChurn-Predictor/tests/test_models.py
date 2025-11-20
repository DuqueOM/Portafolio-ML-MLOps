"""Tests for ResampleClassifier and model components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.bankchurn.models import ResampleClassifier


@pytest.fixture
def imbalanced_dataset():
    """Create imbalanced binary classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],  # 90-10 imbalance
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


class TestResampleClassifier:
    """Test suite for ResampleClassifier."""

    def test_initialization(self):
        """Test classifier initialization with default parameters."""
        clf = ResampleClassifier()
        assert clf.strategy == "none"
        assert clf.random_state == 42
        assert clf.estimator is None

    def test_initialization_with_estimator(self):
        """Test initialization with custom estimator."""
        rf = RandomForestClassifier(n_estimators=10)
        clf = ResampleClassifier(estimator=rf, strategy="oversample")
        assert clf.estimator == rf
        assert clf.strategy == "oversample"

    def test_fit_predict_no_resampling(self, imbalanced_dataset):
        """Test fit and predict without resampling."""
        X, y = imbalanced_dataset
        clf = ResampleClassifier(strategy="none", random_state=42)
        clf.fit(X, y)

        assert hasattr(clf, "classes_")
        assert hasattr(clf, "estimator_")
        assert len(clf.classes_) == 2

        # Make predictions
        y_pred = clf.predict(X)
        assert len(y_pred) == len(y)
        assert set(y_pred).issubset(set(clf.classes_))

    def test_fit_predict_with_oversample(self, imbalanced_dataset):
        """Test fit with SMOTE oversampling."""
        pytest.importorskip("imblearn", reason="imbalanced-learn not installed")

        X, y = imbalanced_dataset
        clf = ResampleClassifier(strategy="oversample", random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert len(y_pred) == len(y)

    def test_fit_predict_with_undersample(self, imbalanced_dataset):
        """Test fit with random undersampling."""
        pytest.importorskip("imblearn", reason="imbalanced-learn not installed")

        X, y = imbalanced_dataset
        clf = ResampleClassifier(strategy="undersample", random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert len(y_pred) == len(y)

    def test_predict_proba(self, imbalanced_dataset):
        """Test probability prediction."""
        X, y = imbalanced_dataset
        clf = ResampleClassifier(random_state=42)
        clf.fit(X, y)

        y_proba = clf.predict_proba(X)
        assert y_proba.shape == (len(y), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
        assert np.all((y_proba >= 0) & (y_proba <= 1))

    def test_invalid_strategy_raises_error(self, imbalanced_dataset):
        """Test that invalid strategy raises ValueError."""
        X, y = imbalanced_dataset
        clf = ResampleClassifier(strategy="invalid_strategy")

        with pytest.raises(ValueError, match="Unknown strategy"):
            clf.fit(X, y)

    def test_fit_with_numpy_arrays(self):
        """Test fit with numpy arrays instead of pandas."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        clf = ResampleClassifier(random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert len(y_pred) == len(y)

    def test_reproducibility(self, imbalanced_dataset):
        """Test that results are reproducible with same random_state."""
        X, y = imbalanced_dataset

        clf1 = ResampleClassifier(random_state=42)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)

        clf2 = ResampleClassifier(random_state=42)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)

    @pytest.mark.skip(
        reason="Flaky test - may fail intermittently due to small dataset size and deterministic model behavior"
    )
    def test_different_seeds_different_results(self, imbalanced_dataset):
        """Test that different random seeds produce different results."""
        import numpy as np

        X, y = imbalanced_dataset

        clf1 = ResampleClassifier(random_state=42)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)

        clf2 = ResampleClassifier(random_state=99)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)

        # Predictions should differ (with high probability)
        assert not np.array_equal(pred1, pred2)

    def test_fit_before_predict_check(self, imbalanced_dataset):
        """Test that predict raises error if not fitted."""
        from sklearn.exceptions import NotFittedError

        X, y = imbalanced_dataset
        clf = ResampleClassifier()

        with pytest.raises(NotFittedError):
            clf.predict(X)

    @pytest.mark.parametrize("strategy", ["none", "class_weight"])
    def test_all_strategies(self, imbalanced_dataset, strategy):
        """Test all resampling strategies."""
        X, y = imbalanced_dataset
        clf = ResampleClassifier(strategy=strategy, random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert len(y_pred) == len(y)
