"""Unit tests for GapFillPredictor (P1.2 / FS19)."""

import numpy as np
import pandas as pd
import pytest

from src.ml.gap_fill_predictor import (
    GapFillPredictor,
    GapFillPrediction,
    GapFillTrainingResult,
    GapDetection,
)


@pytest.fixture
def sample_ohlcv():
    """Generate OHLCV data with deliberate gaps."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base = 100.0 + np.cumsum(np.random.randn(n) * 0.2)
    close = pd.Series(base + np.random.randn(n) * 0.1, index=dates)
    open_ = close.copy()

    # Create deliberate gaps
    gap_indices = [50, 120, 210, 300, 380, 450]
    for idx in gap_indices:
        if idx < n:
            open_.iloc[idx] = close.iloc[idx - 1] * (
                1 + np.random.choice([-0.03, 0.03, -0.02, 0.025])
            )
            # Make gap fill for half of them
            if np.random.random() > 0.5 and idx + 3 < n:
                bar_offset = np.random.choice([1, 2, 3, 4, 5])
                target_idx = min(idx + bar_offset, n - 1)
                if open_.iloc[idx] > close.iloc[idx - 1]:
                    close.iloc[target_idx] = close.iloc[idx - 1] + 1.0
                else:
                    close.iloc[target_idx] = close.iloc[idx - 1] - 1.0

    high = pd.Series(
        np.maximum(open_.values, close.values) + np.abs(np.random.randn(n)) * 0.1, index=dates
    )
    low = pd.Series(
        np.minimum(open_.values, close.values) - np.abs(np.random.randn(n)) * 0.1, index=dates
    )
    volume = pd.Series(np.random.randint(100000, 1000000, n), index=dates)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestGapDetection:
    """Test gap detection logic."""

    def test_detects_gaps(self, sample_ohlcv):
        gfp = GapFillPredictor()
        gaps = gfp.detect_gaps(sample_ohlcv)
        assert len(gaps) > 0
        for gap in gaps:
            assert isinstance(gap, GapDetection)
            assert gap.bar_index > 0
            assert gap.gap_type in ("up", "down")
            assert isinstance(gap.filled, bool)

    def test_gap_types_correct(self, sample_ohlcv):
        gfp = GapFillPredictor(gap_threshold_pct=0.0)
        gaps = gfp.detect_gaps(sample_ohlcv)
        up_gaps = [g for g in gaps if g.gap_type == "up"]
        down_gaps = [g for g in gaps if g.gap_type == "down"]
        for g in up_gaps:
            assert g.gap_pct > 0
        for g in down_gaps:
            assert g.gap_pct < 0

    def test_threshold_filters_small_gaps(self, sample_ohlcv):
        gfp_sensitive = GapFillPredictor(gap_threshold_pct=0.0)
        gfp_strict = GapFillPredictor(gap_threshold_pct=5.0)
        gaps_sensitive = gfp_sensitive.detect_gaps(sample_ohlcv)
        gaps_strict = gfp_strict.detect_gaps(sample_ohlcv)
        assert len(gaps_strict) <= len(gaps_sensitive)

    def test_fill_level_is_previous_close(self, sample_ohlcv):
        gfp = GapFillPredictor()
        gaps = gfp.detect_gaps(sample_ohlcv)
        for gap in gaps:
            prev_close = sample_ohlcv["Close"].iloc[gap.bar_index - 1]
            assert abs(gap.fill_level - prev_close) < 1e-10


class TestGapFillPredictorFit:
    """Test GapFillPredictor training."""

    def test_fit_returns_result(self, sample_ohlcv):
        gfp = GapFillPredictor()
        result = gfp.fit(sample_ohlcv)
        assert isinstance(result, GapFillTrainingResult)
        assert result.n_gaps > 0
        assert 0 <= result.baseline_fill_rate <= 1

    def test_fit_sets_feature_names(self, sample_ohlcv):
        gfp = GapFillPredictor()
        gfp.fit(sample_ohlcv)
        assert len(gfp.feature_names_) > 0

    def test_fit_trains_model(self, sample_ohlcv):
        gfp = GapFillPredictor()
        gfp.fit(sample_ohlcv)
        assert gfp.model is not None

    def test_fit_with_no_gaps(self):
        """No gaps should still return a valid result."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = pd.Series(np.linspace(100, 101, n), index=dates)
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.001,
                "Low": close * 0.999,
                "Close": close,
                "Volume": pd.Series(1000, index=dates),
            },
            index=dates,
        )
        gfp = GapFillPredictor(gap_threshold_pct=5.0)
        result = gfp.fit(df)
        assert isinstance(result, GapFillTrainingResult)
        assert result.n_gaps == 0


class TestGapFillPredictorPredict:
    """Test GapFillPredictor prediction."""

    def test_predict_returns_gap_fill_prediction(self, sample_ohlcv):
        gfp = GapFillPredictor()
        gfp.fit(sample_ohlcv)

        if gfp.model is None:
            pytest.skip("Model not trained")

        gaps = gfp.detect_gaps(sample_ohlcv)
        if not gaps:
            pytest.skip("No gaps detected")

        gap_idx = gaps[0].bar_index
        pred = gfp.predict(sample_ohlcv, gap_idx)
        assert isinstance(pred, GapFillPrediction)
        assert 0.0 <= pred.fill_probability <= 1.0
        assert pred.bar_index == gap_idx

    def test_predict_batch_returns_dataframe(self, sample_ohlcv):
        gfp = GapFillPredictor()
        gfp.fit(sample_ohlcv)

        if gfp.model is None:
            pytest.skip("Model not trained")

        result = gfp.predict_batch(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "fill_probability" in result.columns
            assert "gap_pct" in result.columns
            assert "gap_type" in result.columns

    def test_predict_raises_before_fit(self, sample_ohlcv):
        gfp = GapFillPredictor()
        with pytest.raises(ValueError, match="not trained"):
            gfp.predict(sample_ohlcv, 50)

    def test_predict_batch_empty_no_gaps(self, sample_ohlcv):
        gfp = GapFillPredictor(gap_threshold_pct=10.0)
        gfp.fit(sample_ohlcv)
        result = gfp.predict_batch(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)


class TestGapFillPredictorFeatures:
    """Test feature construction."""

    def test_features_no_nan(self, sample_ohlcv):
        gfp = GapFillPredictor()
        gaps = gfp.detect_gaps(sample_ohlcv)
        if not gaps:
            pytest.skip("No gaps detected")
        X, y = gfp._build_features(sample_ohlcv, gaps)
        if len(X) > 0:
            assert not X.isna().any().any()

    def test_features_have_expected_columns(self, sample_ohlcv):
        gfp = GapFillPredictor()
        gaps = gfp.detect_gaps(sample_ohlcv)
        if not gaps:
            pytest.skip("No gaps detected")
        X, y = gfp._build_features(sample_ohlcv, gaps)
        if len(X) > 0:
            expected = [
                "gap_pct_abs",
                "gap_direction",
                "pre_trend_5",
                "pre_trend_20",
                "volatility_20",
                "volume_ratio",
                "atr_ratio",
                "relative_position",
                "day_of_week",
                "is_monday",
                "pre_gap_range",
            ]
            for col in expected:
                assert col in X.columns, f"Missing column: {col}"

    def test_labels_are_binary(self, sample_ohlcv):
        gfp = GapFillPredictor()
        gaps = gfp.detect_gaps(sample_ohlcv)
        if not gaps:
            pytest.skip("No gaps detected")
        X, y = gfp._build_features(sample_ohlcv, gaps)
        if len(y) > 0:
            assert set(y.unique()).issubset({0, 1})

    def test_monday_flag(self, sample_ohlcv):
        gfp = GapFillPredictor()
        gaps = gfp.detect_gaps(sample_ohlcv)
        if not gaps:
            pytest.skip("No gaps detected")
        X, y = gfp._build_features(sample_ohlcv, gaps)
        if len(X) > 0:
            assert X["is_monday"].max() <= 1.0
            assert X["is_monday"].min() >= 0.0


class TestGapFillPredictorSaveLoad:
    """Test model persistence."""

    def test_save_and_load(self, sample_ohlcv, tmp_path):
        gfp = GapFillPredictor()
        gfp.fit(sample_ohlcv)

        if gfp.model is None:
            pytest.skip("Model not trained")

        path = tmp_path / "gap_fill.pkl"
        gfp.save(path)
        assert path.exists()

        gfp2 = GapFillPredictor()
        gfp2.load(path)
        assert gfp2.model is not None
        assert gfp2.feature_names_ == gfp.feature_names_

    def test_load_nonexistent_raises(self):
        from pathlib import Path

        gfp = GapFillPredictor()
        with pytest.raises(FileNotFoundError):
            gfp.load(Path("nonexistent.pkl"))
