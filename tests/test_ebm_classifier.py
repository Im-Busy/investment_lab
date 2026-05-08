"""Tests for EBM Regime Classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic OHLCV data with regime labels."""
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    np.random.seed(42)
    # Three regimes: low vol (0), medium vol (1), high vol (2)
    regime_changes = [0] * 150 + [1] * 150 + [2] * 150 + [0] * 50
    labels = pd.Series(
        [f"regime_{r}" for r in regime_changes],
        index=dates,
        name="regime",
    )

    # Features that differ by regime
    features = pd.DataFrame(
        {
            "rsi": np.where(
                np.array(regime_changes) == 0,
                np.random.normal(50, 5, n),
                np.where(
                    np.array(regime_changes) == 1,
                    np.random.normal(60, 8, n),
                    np.random.normal(40, 12, n),
                ),
            ),
            "atr": np.where(
                np.array(regime_changes) == 0,
                np.random.exponential(0.5, n),
                np.where(
                    np.array(regime_changes) == 1,
                    np.random.exponential(1.5, n),
                    np.random.exponential(3.0, n),
                ),
            ),
            "adx": np.where(
                np.array(regime_changes) == 0,
                np.random.normal(20, 5, n),
                np.where(
                    np.array(regime_changes) == 1,
                    np.random.normal(35, 8, n),
                    np.random.normal(45, 10, n),
                ),
            ),
            "volatility": np.where(
                np.array(regime_changes) == 0,
                np.random.exponential(0.005, n),
                np.where(
                    np.array(regime_changes) == 1,
                    np.random.exponential(0.015, n),
                    np.random.exponential(0.030, n),
                ),
            ),
            "momentum": np.where(
                np.array(regime_changes) == 0,
                np.random.normal(0.001, 0.01, n),
                np.where(
                    np.array(regime_changes) == 1,
                    np.random.normal(0.005, 0.02, n),
                    np.random.normal(-0.005, 0.03, n),
                ),
            ),
            "volume_ratio": np.random.normal(1.0, 0.3, n),
            "ma_distance": np.random.normal(0, 0.02, n),
            "bb_width": np.random.normal(0.05, 0.02, n),
        },
        index=dates,
    )

    return features, labels


class TestEBMRegimeClassifier:
    def test_train_and_predict(self, synthetic_data) -> None:
        """EBM should train and produce predictions."""
        from src.ml.ebm_classifier import EBMRegimeClassifier

        X, y = synthetic_data
        ebm = EBMRegimeClassifier(
            max_rounds=100,
            outer_bags=4,
            interactions=3,
        )
        result = ebm.train(X, y)

        assert ebm.is_trained
        assert result["train_accuracy"] > 0.3
        assert result["test_accuracy"] > 0.3
        assert result["n_train"] > 0
        assert result["n_test"] > 0

        predictions = ebm.predict(X.head(10))
        assert len(predictions) == 10
        assert all(p in ebm.classes_ for p in predictions)

    def test_predict_proba(self, synthetic_data) -> None:
        """predict_proba should return per-class probabilities."""
        from src.ml.ebm_classifier import EBMRegimeClassifier

        X, y = synthetic_data
        ebm = EBMRegimeClassifier(max_rounds=100, outer_bags=4)
        ebm.train(X, y)

        proba = ebm.predict_proba(X.head(10))
        assert proba.shape == (10, len(ebm.classes_))
        assert (proba.sum(axis=1).between(0.99, 1.01)).all()
        assert ((proba >= 0) & (proba <= 1)).all().all()

    def test_explain_local(self, synthetic_data) -> None:
        """explain_local should decompose a prediction."""
        from src.ml.ebm_classifier import EBMRegimeClassifier

        X, y = synthetic_data
        ebm = EBMRegimeClassifier(max_rounds=100, outer_bags=4)
        ebm.train(X, y)

        explanation = ebm.explain_local(X.head(1))
        assert "intercept" in explanation
        assert "features" in explanation
        assert "predictions" in explanation
        assert isinstance(explanation["predictions"], dict)

    def test_explain_global(self, synthetic_data) -> None:
        """explain_global should return shape functions and interactions."""
        from src.ml.ebm_classifier import EBMRegimeClassifier

        X, y = synthetic_data
        ebm = EBMRegimeClassifier(max_rounds=100, outer_bags=4, interactions=3)
        ebm.train(X, y)

        global_explanation = ebm.explain_global()
        assert "intercept" in global_explanation
        assert "feature_importance" in global_explanation
        assert "shape_functions" in global_explanation
        assert "interaction_terms" in global_explanation

    def test_feature_importance_table(self, synthetic_data) -> None:
        """feature_importance_table should return sorted DataFrame."""
        from src.ml.ebm_classifier import EBMRegimeClassifier

        X, y = synthetic_data
        ebm = EBMRegimeClassifier(max_rounds=100, outer_bags=4)
        ebm.train(X, y)

        imp_table = ebm.feature_importance_table()
        assert len(imp_table) > 0
        assert list(imp_table.columns) == [
            "feature",
            "mean_abs_contribution",
            "sign",
        ]
        # Should be sorted by mean_abs_contribution descending
        assert (
            imp_table["mean_abs_contribution"].iloc[0]
            >= imp_table["mean_abs_contribution"].iloc[-1]
        )

    def test_summary(self, synthetic_data) -> None:
        """summary should return readable model overview."""
        from src.ml.ebm_classifier import EBMRegimeClassifier

        X, y = synthetic_data
        ebm = EBMRegimeClassifier(max_rounds=100, outer_bags=4)
        ebm.train(X, y)

        summary = ebm.summary()
        assert summary["model_type"] == "EBM (Explainable Boosting Machine)"
        assert summary["n_features"] == X.shape[1]
        assert summary["n_classes"] == len(ebm.classes_)
        assert "top_features" in summary
        assert len(summary["top_features"]) <= 10

    def test_untrained_raises(self) -> None:
        """Untrained EBM should raise on predict/explain."""
        from src.ml.ebm_classifier import EBMRegimeClassifier

        ebm = EBMRegimeClassifier()
        X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])

        with pytest.raises(ValueError, match="not trained"):
            ebm.predict(X)

        with pytest.raises(ValueError, match="not trained"):
            ebm.explain_local(X)

        with pytest.raises(ValueError, match="not trained"):
            ebm.explain_global()

    def test_save_load(self, synthetic_data, tmp_path) -> None:
        """EBM should roundtrip through save/load."""
        from src.ml.ebm_classifier import EBMRegimeClassifier

        X, y = synthetic_data
        ebm = EBMRegimeClassifier(max_rounds=100, outer_bags=4)
        ebm.train(X, y)

        path = tmp_path / "ebm_model.pkl"
        ebm.save(str(path))
        assert path.exists()

        loaded = EBMRegimeClassifier()
        loaded.load(str(path))
        assert loaded.is_trained
        assert loaded.feature_names_ == ebm.feature_names_
        assert loaded.classes_ == ebm.classes_

        original_preds = ebm.predict(X.head(5))
        loaded_preds = loaded.predict(X.head(5))
        pd.testing.assert_series_equal(original_preds, loaded_preds)

    def test_higher_accuracy_with_more_rounds(self, synthetic_data) -> None:
        """More boosting rounds should not hurt accuracy."""
        from src.ml.ebm_classifier import EBMRegimeClassifier

        X, y = synthetic_data
        ebm_fast = EBMRegimeClassifier(max_rounds=50, outer_bags=2)
        result_fast = ebm_fast.train(X, y)

        ebm_slow = EBMRegimeClassifier(max_rounds=200, outer_bags=4)
        result_slow = ebm_slow.train(X, y)

        assert result_slow["test_accuracy"] >= result_fast["test_accuracy"] - 0.05

    def test_purge_window(self, synthetic_data) -> None:
        """Purge window should produce non-overlapping train/test splits."""
        from src.ml.ebm_classifier import EBMRegimeClassifier

        X, y = synthetic_data
        n = len(X)

        ebm_no_purge = EBMRegimeClassifier(max_rounds=50)
        result_no_purge = ebm_no_purge.train(X, y, purge_window=0)

        ebm_purge = EBMRegimeClassifier(max_rounds=50)
        result_purge = ebm_purge.train(X, y, purge_window=10)

        # With purge, fewer training samples (purged region removed)
        assert result_purge["n_train"] <= result_no_purge["n_train"]
        # Both should still get reasonable accuracy
        assert result_purge["test_accuracy"] > 0.2
        assert result_no_purge["test_accuracy"] > 0.2
