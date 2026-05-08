"""
Tests for DrawdownTargetGenerator (FS3 - MAE/Drawdown as Primary Target)
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.drawdown_target import DrawdownTargetGenerator


def make_ohlcv_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data with a downtrend to ensure drawdowns."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    returns = np.random.randn(n) * 0.02
    close = 100 * np.cumprod(1 + returns)

    high = close + np.abs(np.random.randn(n) * 2)
    low = close - np.abs(np.random.randn(n) * 2)
    open_prices = close + np.random.randn(n)
    volume = np.random.randint(1000000, 10000000, n)

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


class TestDrawdownTargetGenerator:
    """Core tests for DrawdownTargetGenerator."""

    def test_generate_labels_basic(self):
        """Test label generation returns expected columns."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20)
        labels = gen.generate_labels(df)

        assert "mae_pct" in labels.columns
        assert "mfe_pct" in labels.columns
        assert "mae_ratio" in labels.columns
        assert "mae_bucket" in labels.columns
        assert "mae_bucket_int" in labels.columns
        assert len(labels) == len(df)

    def test_mae_is_non_negative(self):
        """Test MAE values are non-negative (drawdown >= 0)."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20)
        labels = gen.generate_labels(df)

        valid_mae = labels["mae_pct"].dropna()
        assert (valid_mae >= 0).all(), f"Found negative MAE: {valid_mae[valid_mae < 0]}"

    def test_mfe_is_non_negative(self):
        """Test MFE values are non-negative (runup >= 0)."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20)
        labels = gen.generate_labels(df)

        valid_mfe = labels["mfe_pct"].dropna()
        assert (valid_mfe >= 0).all(), f"Found negative MFE: {valid_mfe[valid_mfe < 0]}"

    def test_last_n_bars_are_nan(self):
        """Test last horizon bars have NaN labels (no future data)."""
        df = make_ohlcv_data(n=100)
        gen = DrawdownTargetGenerator(horizon=20)
        labels = gen.generate_labels(df)

        last_valid = labels["mae_pct"].notna()
        expected_nan_count = gen.horizon
        actual_nan_tail = labels["mae_pct"].iloc[-expected_nan_count:]

        assert actual_nan_tail.isna().all()

    def test_multi_horizon_labels(self):
        """Test multiple horizons generate separate columns."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20, horizons=[5, 10, 20])
        labels = gen.generate_labels(df)

        for h in [5, 10, 20]:
            assert f"mae_pct_{h}" in labels.columns
            assert f"mfe_pct_{h}" in labels.columns
            assert f"mae_ratio_{h}" in labels.columns

    def test_mae_ratio_range(self):
        """Test MAE ratio is non-negative when MFE > 0."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20)
        labels = gen.generate_labels(df)

        valid_ratio = labels["mae_ratio"].dropna()
        assert (valid_ratio >= 0).all()

    def test_mae_bucket_labels(self):
        """Test MAE bucket contains valid labels."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20)
        labels = gen.generate_labels(df)

        valid_buckets = labels["mae_bucket"].dropna()
        expected_labels = set(gen.mae_labels)
        actual_labels = set(valid_buckets.unique())
        assert actual_labels.issubset(expected_labels)

    def test_generate_trade_level_targets(self):
        """Test trade-level MAE target generation."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20)

        signals = pd.DataFrame(
            {
                "strategy": ["EMA Ribbon"] * 3,
                "timestamp": df.index[10:13],
                "entry_price": df["Close"].iloc[10:13].values,
                "confidence": [0.8, 0.7, 0.9],
            }
        )

        trades = gen.generate_trade_level_targets(df, signals)

        assert "mae_pct" in trades.columns
        assert "mfe_pct" in trades.columns
        assert "mae_bucket" in trades.columns
        assert "mae_price" in trades.columns
        assert "mfe_price" in trades.columns
        assert "trade_return" in trades.columns
        assert len(trades) == len(signals)

    def test_trade_level_mae_at_least_zero(self):
        """Test trade-level MAE is non-negative."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20)

        signals = pd.DataFrame(
            {
                "strategy": ["test"] * 5,
                "timestamp": df.index[10:15],
                "entry_price": df["Close"].iloc[10:15].values,
                "confidence": [0.8] * 5,
            }
        )

        trades = gen.generate_trade_level_targets(df, signals)
        valid_mae = trades["mae_pct"].dropna()
        assert (valid_mae >= 0).all()

    def test_suggest_stops_mae_multiplier(self):
        """Test MFE-based stop suggestion."""
        df = make_ohlcv_data(n=100)
        gen = DrawdownTargetGenerator(horizon=20)

        stops = gen.suggest_stops(df, predicted_mae=0.05, multiplier=1.5, method="mae_multiplier")
        assert len(stops) == len(df)
        assert (stops < df["Close"]).all()

    def test_suggest_stops_atr_multiplier(self):
        """Test ATR-based stop suggestion."""
        df = make_ohlcv_data(n=100)
        gen = DrawdownTargetGenerator(horizon=20)

        stops = gen.suggest_stops(df, predicted_mae=0.05, multiplier=2.0, method="atr_multiplier")
        assert len(stops) == len(df)
        valid = stops.dropna()
        assert (valid < df["Close"].loc[valid.index]).all()

    def test_suggest_stops_invalid_method(self):
        """Test invalid stop method raises ValueError."""
        df = make_ohlcv_data(n=100)
        gen = DrawdownTargetGenerator()

        with pytest.raises(ValueError, match="Unknown stop method"):
            gen.suggest_stops(df, method="invalid")

    def test_get_mae_distribution(self):
        """Test MAE distribution statistics."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20, horizons=[10, 20])
        dist = gen.get_mae_distribution(df)

        assert "horizon_10" in dist.index
        assert "horizon_20" in dist.index
        assert "mean" in dist.columns
        assert "median" in dist.columns
        assert "std" in dist.columns
        assert "q95" in dist.columns

    def test_empty_data_raises(self):
        """Test empty DataFrame raises ValueError."""
        gen = DrawdownTargetGenerator()
        with pytest.raises(ValueError, match="Missing columns"):
            gen.generate_labels(pd.DataFrame())

    def test_insufficient_rows_raises(self):
        """Test too few rows raises ValueError."""
        df = make_ohlcv_data(n=5)
        gen = DrawdownTargetGenerator(horizon=20)
        with pytest.raises(ValueError, match="Insufficient rows"):
            gen.generate_labels(df)

    def test_invalid_horizon_raises(self):
        """Test invalid horizon raises ValueError."""
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            DrawdownTargetGenerator(horizon=0)

    def test_atr_normalized_output(self):
        """Test ATR-normalized MAE column when enabled."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20, use_atr_normalized=True)
        labels = gen.generate_labels(df)

        assert "mae_atr" in labels.columns
        assert "atr" in labels.columns
        valid_atr = labels["mae_atr"].dropna()
        assert len(valid_atr) > 0

    def test_generate_targets_at_signals(self):
        """Test MAE target generation at signal timestamps."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(horizon=20)

        signals = pd.DataFrame(
            {
                "strategy": ["EMA Ribbon"] * 2,
                "timestamp": df.index[50:52],
                "confidence": [0.8, 0.7],
            }
        )

        result = gen.generate_targets_at_signals(df, signals)
        assert "mae_pct" in result.columns
        assert "mae_bucket" in result.columns
        assert len(result) == len(signals)

    def test_custom_mae_bins(self):
        """Test custom MAE bin definitions."""
        df = make_ohlcv_data(n=300)
        gen = DrawdownTargetGenerator(
            horizon=20,
            mae_bins=[0.0, 0.03, 0.07, float("inf")],
        )
        labels = gen.generate_labels(df)

        valid_buckets = labels["mae_bucket"].dropna()
        assert len(valid_buckets.unique()) <= 3
