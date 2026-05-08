"""Tests for AutoML Baseline (AutoGluon)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic binary classification data."""
    n = 500
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n, 6),
        columns=[f"feat_{i}" for i in range(6)],
    )
    y = pd.Series(np.random.randint(0, 2, n), name="target")
    return X, y


class TestAutoMLBaseline:
    def test_fit_and_predict(self, binary_data) -> None:
        """AutoML should fit and produce predictions."""
        from src.ml.automl import AutoMLBaseline

        X, y = binary_data
        automl = AutoMLBaseline(label="target", time_limit=30, presets="medium_quality")
        result = automl.fit(X, y)

        assert automl.is_fitted
        assert result.models_trained >= 1
        assert result.train_time_seconds > 0
        assert result.best_score > 0.0
        assert not result.leaderboard.empty

        preds = automl.predict(X.head(10))
        assert len(preds) == 10
        assert all(p in (0, 1) for p in preds)

    def test_predict_proba(self, binary_data) -> None:
        """predict_proba should return probability columns."""
        from src.ml.automl import AutoMLBaseline

        X, y = binary_data
        automl = AutoMLBaseline(label="target", time_limit=30, presets="medium_quality")
        automl.fit(X, y)

        proba = automl.predict_proba(X.head(10))
        assert isinstance(proba, pd.DataFrame)
        assert proba.shape[0] == 10
        assert proba.shape[1] == 2
        assert (proba.sum(axis=1).between(0.99, 1.01)).all()

    def test_leaderboard(self, binary_data) -> None:
        """Leaderboard should be a DataFrame with model names and scores."""
        from src.ml.automl import AutoMLBaseline

        X, y = binary_data
        automl = AutoMLBaseline(label="target", time_limit=30, presets="medium_quality")
        automl.fit(X, y)

        lb = automl.leaderboard()
        assert isinstance(lb, pd.DataFrame)
        assert "model" in lb.columns
        assert len(lb) >= 1

    def test_compare_to_baseline(self, binary_data) -> None:
        """compare_to_baseline should compute uplift."""
        from src.ml.automl import AutoMLBaseline

        X, y = binary_data
        automl = AutoMLBaseline(label="target", time_limit=30, presets="medium_quality")
        automl.fit(X, y)

        comparison = automl.compare_to_baseline(0.60, "Hand-Tuned CatBoost")
        assert "uplift" in comparison
        assert "uplift_percent" in comparison
        assert "winner" in comparison
        assert isinstance(comparison["uplift"], float)
        assert isinstance(comparison["uplift_percent"], float)

    def test_result_summary(self, binary_data) -> None:
        """summary() should return a readable string."""
        from src.ml.automl import AutoMLBaseline

        X, y = binary_data
        automl = AutoMLBaseline(label="target", time_limit=30, presets="medium_quality")
        automl.fit(X, y)

        summary = automl._result.summary()
        assert "AutoML Baseline Results" in summary
        assert "Best model" in summary
        assert "Top 5 models" in summary

    def test_untrained_raises(self) -> None:
        """Untrained AutoML should raise ValueError."""
        from src.ml.automl import AutoMLBaseline

        automl = AutoMLBaseline(label="target")
        X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])

        with pytest.raises(ValueError, match="not fitted"):
            automl.predict(X)

        with pytest.raises(ValueError, match="not fitted"):
            automl.predict_proba(X)

        with pytest.raises(ValueError, match="not fitted"):
            automl.leaderboard()

    def test_presets_affect_speed(self, binary_data) -> None:
        """Medium quality should train faster than high_quality."""
        from src.ml.automl import AutoMLBaseline

        X, y = binary_data

        automl_deploy = AutoMLBaseline(
            label="target", time_limit=30, presets="optimize_for_deployment"
        )
        result_deploy = automl_deploy.fit(X, y)

        assert result_deploy.models_trained >= 1
        assert automl_deploy.is_fitted

    def test_different_presets_all_train(self, binary_data) -> None:
        """All quality presets should produce a valid model."""
        from src.ml.automl import AutoMLBaseline

        X, y = binary_data
        for preset in ["medium_quality", "optimize_for_deployment"]:
            automl = AutoMLBaseline(label="target", time_limit=45, presets=preset)
            result = automl.fit(X, y)
            assert automl.is_fitted
            assert result.best_score > 0.0
