"""
Tests for GMMRegimeDetector (FS5 - GMM Soft Regime Assignments)
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.gmm_regime import GMMRegimeDetector


def make_synthetic_data(n: int = 300, n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create synthetic feature data with 3 distinct clusters."""
    np.random.seed(seed)
    n_per = n // 3

    c0 = np.random.randn(n_per, n_features) * 0.5 + np.array([-2.0, -2.0, 0.5, 0.0, 0.0])
    c1 = np.random.randn(n_per, n_features) * 0.5 + np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    c2 = np.random.randn(n_per, n_features) * 0.5 + np.array([2.0, 2.0, -0.5, 0.0, 0.0])

    X = np.vstack([c0, c1, c2])
    np.random.shuffle(X)

    columns = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=columns)


class TestGMMRegimeDetector:
    """Core functionality tests for GMMRegimeDetector."""

    def test_fit_predict_basic(self):
        """Test fit and predict on synthetic data."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(2, 5), random_state=42)
        detector.fit(data)

        regimes = detector.predict(data)
        assert len(regimes) == len(data)
        assert isinstance(regimes, pd.Series)
        assert regimes.name == "gmm_regime"
        assert detector.is_fitted

    def test_predict_proba_sums_to_one(self):
        """Test GMM predict_proba returns proper probabilities."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(3, 3), random_state=42)
        detector.fit(data)

        probs = detector.predict_proba(data)
        assert len(probs) == len(data)
        assert isinstance(probs, pd.DataFrame)
        assert probs.shape[1] == detector.optimal_k_

        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_no_negative(self):
        """Test all probabilities are non-negative."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(2, 4), random_state=42)
        detector.fit(data)

        probs = detector.predict_proba(data)
        assert (probs >= 0).all().all()
        assert (probs <= 1).all().all()

    def test_predict_labels_are_strings(self):
        """Test predicted labels are strings."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(3, 3), random_state=42)
        detector.fit(data)

        regimes = detector.predict(data)
        for label in regimes:
            assert isinstance(label, str)

    def test_get_regime_summary(self):
        """Test regime summary has expected fields."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(2, 4), random_state=42)
        detector.fit(data)

        summary = detector.get_regime_summary()
        assert summary.name == "GMMRegimeDetector"
        assert summary.n_regimes == detector.optimal_k_
        assert len(summary.regime_labels) > 0
        assert "bic_scores" in summary.metadata
        assert "aic_scores" in summary.metadata
        assert "covariance_type" in summary.metadata
        assert "component_weights" in summary.metadata

    def test_bic_selection(self):
        """Test that BIC selects a reasonable number of components."""
        data = make_synthetic_data(n=300)
        detector = GMMRegimeDetector(n_range=(3, 5), random_state=42)
        detector.fit(data)

        assert 3 <= detector.optimal_k_ <= 5
        assert len(detector.bic_scores_) >= 1
        assert len(detector.aic_scores_) >= 1

        assert detector.bic_scores_[detector.optimal_k_] <= min(detector.bic_scores_.values())

    def test_get_bic_analysis(self):
        """Test BIC analysis DataFrame."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(2, 5), random_state=42)
        detector.fit(data)

        bic_df = detector.get_bic_analysis()
        assert "bic" in bic_df.columns
        assert "aic" in bic_df.columns
        assert len(bic_df) >= 1

    def test_get_component_weights(self):
        """Test component weights sum to 1."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(3, 3), random_state=42)
        detector.fit(data)

        weights = detector.get_component_weights()
        assert len(weights) == detector.optimal_k_
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_get_component_means(self):
        """Test component means DataFrame shape."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(3, 3), random_state=42)
        detector.fit(data)

        means = detector.get_component_means()
        assert means.shape[0] == detector.optimal_k_
        assert means.shape[1] == data.shape[1]

    def test_predict_before_fit_raises(self):
        """Test predict before fit raises ValueError."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector()

        with pytest.raises(ValueError, match="not fitted"):
            detector.predict(data)

    def test_predict_proba_before_fit_raises(self):
        """Test predict_proba before fit raises ValueError."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector()

        with pytest.raises(ValueError, match="not fitted"):
            detector.predict_proba(data)

    def test_empty_data_raises(self):
        """Test empty DataFrame raises ValueError."""
        detector = GMMRegimeDetector()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            detector.fit(empty_df)

    def test_nan_data_raises(self):
        """Test NaN in data raises ValueError."""
        data = make_synthetic_data()
        data.iloc[0, 0] = np.nan
        detector = GMMRegimeDetector()

        with pytest.raises(ValueError, match="NaN"):
            detector.fit(data)

    def test_insufficient_samples_raises(self):
        """Test too few samples raises ValueError."""
        data = make_synthetic_data(n=10)
        detector = GMMRegimeDetector(min_samples=50)

        with pytest.raises(ValueError, match="Insufficient samples"):
            detector.fit(data)

    def test_invalid_n_range_raises(self):
        """Test n_range with min < 2 raises ValueError."""
        with pytest.raises(ValueError, match=">= 2"):
            GMMRegimeDetector(n_range=(1, 5))

    def test_consistent_predictions(self):
        """Test repeated predictions are consistent."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(3, 3), random_state=42)
        detector.fit(data)

        regimes1 = detector.predict(data)
        regimes2 = detector.predict(data)
        pd.testing.assert_series_equal(regimes1, regimes2)

    def test_regime_mapping_stability(self):
        """Test regime mapping is stable across refits with the same seed."""
        data = make_synthetic_data()
        d1 = GMMRegimeDetector(n_range=(3, 3), random_state=42)
        d2 = GMMRegimeDetector(n_range=(3, 3), random_state=42)

        d1.fit(data)
        d2.fit(data)

        probs1 = d1.predict_proba(data)
        probs2 = d2.predict_proba(data)

        np.testing.assert_allclose(probs1.values, probs2.values, atol=1e-6)

    def test_get_raw_components(self):
        """Test raw component indices are integers 0..k-1."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(3, 3), random_state=42)
        detector.fit(data)

        raw = detector.get_raw_components(data)
        assert len(raw) == len(data)
        assert raw.min() >= 0
        assert raw.max() < detector.optimal_k_

    def test_summary_before_fit_raises(self):
        """Test get_regime_summary before fit raises error."""
        detector = GMMRegimeDetector()
        with pytest.raises(ValueError, match="not fitted"):
            detector.get_regime_summary()

    def test_different_covariance_types(self):
        """Test all covariance types fit without error."""
        data = make_synthetic_data()
        for cov_type in ["full", "tied", "diag", "spherical"]:
            detector = GMMRegimeDetector(
                n_range=(3, 3),
                covariance_type=cov_type,
                random_state=42,
            )
            detector.fit(data)
            regimes = detector.predict(data)
            assert len(regimes) == len(data)

    def test_regime_mapping_no_empty_labels(self):
        """Test no empty string labels in regime mapping."""
        data = make_synthetic_data()
        detector = GMMRegimeDetector(n_range=(2, 4), random_state=42)
        detector.fit(data)

        for label in detector.regime_mapping_.values():
            assert isinstance(label, str)
            assert len(label) > 0
