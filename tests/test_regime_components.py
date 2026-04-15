"""
Tests for new components:
- Regime Detector
- Adaptive Strategy Router
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.regime_detector import RegimeDetector, RegimeDetectorConfig, RegimeState
from src.strategies.adaptive_router import AdaptiveRouter, StrategyCategory, STRATEGY_REGIME_MAP


class FakeDataFrame(pd.DataFrame):
    """Helper to create clean OHLCV data frames."""


def make_ohlcv_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Random walk for close
    returns = np.random.randn(n) * 0.02
    close = 100 * np.cumprod(1 + returns)

    # Generate OHLC from close
    high = close + np.abs(np.random.randn(n) * 2)
    low = close - np.abs(np.random.randn(n) * 2)
    open_prices = close + np.random.randn(n)
    volume = np.random.randint(1000, 10000, n)

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


class TestRegimeDetector:
    """Tests for RegimeDetector."""

    def test_classify_basics(self):
        """Test that classification runs and returns correct shape."""
        data = make_ohlcv_data(n=200)
        detector = RegimeDetector()

        regimes = detector.classify(data)

        assert len(regimes) == len(data)
        assert all(isinstance(r, RegimeState) for r in regimes)
        assert isinstance(regimes, pd.Series)
        assert regimes.name == "regime"

    def test_get_regime_series(self):
        """Test regime series includes all expected columns."""
        data = make_ohlcv_data(n=200)
        detector = RegimeDetector()

        series = detector.get_regime_series(data)

        assert "regime" in series.columns
        assert "adx" in series.columns
        assert "atr" in series.columns
        assert "atr_median" in series.columns
        assert "atr_80th_percentile" in series.columns

    def test_regime_values_are_valid(self):
        """Test all regime values are valid RegimeState members."""
        data = make_ohlcv_data(n=300)
        detector = RegimeDetector()

        regimes = detector.classify(data)
        valid_states = set(RegimeState)

        for regime in regimes:
            assert regime in valid_states

    def test_warmup_period(self):
        """Test initial bars return TRANSITION."""
        data = make_ohlcv_data(n=20)
        detector = RegimeDetector()
        detector.config.warmup_bars = 15

        regimes = detector.classify(data)

        # First warmup bars should be TRANSITION
        warmup_count = sum(1 for r in regimes if r == RegimeState.TRANSITION)
        assert warmup_count >= 15

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RegimeDetectorConfig(
            adx_trending_threshold=30.0,
            adx_ranging_threshold=15.0,
            atr_volatile_percentile=0.90,
        )
        detector = RegimeDetector(config=config)

        assert detector.config.adx_trending_threshold == 30.0
        assert detector.config.adx_ranging_threshold == 15.0
        assert detector.config.atr_volatile_percentile == 0.90

    def test_long_dataset(self):
        """Test classification on longer dataset (1000+ bars)."""
        data = make_ohlcv_data(n=1000)
        detector = RegimeDetector()

        regimes = detector.classify(data)

        assert len(regimes) == 1000
        assert all(isinstance(r, RegimeState) for r in regimes)


class TestAdaptiveRouter:
    """Tests for AdaptiveRouter."""

    def setup_method(self):
        self.router = AdaptiveRouter()

    def test_trending_strategies_enabled(self):
        """Test trend-following strategies active in trending regime."""
        enabled = self.router.get_active_strategies(
            list(STRATEGY_REGIME_MAP.keys()), RegimeState.TRENDING
        )

        assert "EMA Ribbon" in enabled
        assert "SMA Crossover" in enabled
        assert "ADX Trend Strength" in enabled
        assert "Parabolic SAR" in enabled

    def test_ranging_strategies_disabled(self):
        """Test trend-following strategies disabled in ranging regime."""
        enabled = self.router.get_active_strategies(
            list(STRATEGY_REGIME_MAP.keys()), RegimeState.RANGING
        )

        assert "EMA Ribbon" not in enabled
        assert "SMA Crossover" not in enabled
        assert "ADX Trend Strength" not in enabled

    def test_volatile_strategies_enabled(self):
        """Test volatility strategies active in volatile regime."""
        enabled = self.router.get_active_strategies(
            list(STRATEGY_REGIME_MAP.keys()), RegimeState.VOLATILE
        )

        assert "Chandelier Exit" in enabled
        assert "VWAP Bounce" in enabled
        assert "Keltner Channel" in enabled
        assert "Bollinger Bands" in enabled

    def test_is_strategy_active_unknown_defaults_true(self):
        """Test unknown strategies default to active."""
        assert self.router.is_strategy_active("Unknown Strategy", RegimeState.TRENDING)

    def test_get_regime_recommendations(self):
        """Test regime recommendations have correct structure."""
        recs = self.router.get_regime_recommendations(RegimeState.TRENDING)

        assert "regime" in recs
        assert recs["regime"] == "Trending"
        assert "enabled" in recs
        assert "disabled" in recs
        assert "advice" in recs
        assert len(recs["enabled"]) > 0
        assert len(recs["disabled"]) > 0

    def test_apply_regime_filter(self):
        """Test signal filtering by regime."""
        signals = pd.DataFrame(
            {
                "strategy": ["EMA Ribbon", "RSI Divergence", "Chandelier Exit", "SMA Crossover"],
                "timestamp": pd.date_range("2020-01-01", periods=4, freq="D"),
                "confidence": [0.8, 0.7, 0.6, 0.5],
            }
        )
        regimes = pd.Series(
            [RegimeState.TRENDING] * 4,
            index=pd.date_range("2020-01-01", periods=4, freq="D"),
        )

        filtered = self.router.apply_regime_filter(signals, regimes)

        # In trending regime, EMA Ribbon and SMA Crossover should be active
        # RSI Divergence should be inactive
        strategies = filtered["strategy"].tolist()
        assert "EMA Ribbon" in strategies
        assert "SMA Crossover" in strategies
        assert "RSI Divergence" not in strategies

    def test_get_regime_time_series(self):
        """Test full time series with active strategy counts."""
        data = make_ohlcv_data(n=200)

        series = self.router.get_regime_time_series(data)

        assert "regime" in series.columns
        assert "active_strategy_count" in series.columns
        assert "active_strategies" in series.columns
        assert all(isinstance(r, RegimeState) for r in series["regime"])
        assert all(c > 0 for c in series["active_strategy_count"] if c > 0)
