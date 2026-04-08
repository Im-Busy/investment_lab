"""Tests for Donchian Channel Breakout pattern detector."""

import numpy as np
import pandas as pd
import pytest

from src.patterns.breakout.donchian import DonchianChannelBreakout


def make_trending_df(
    n_bars: int = 100,
    trend: str = "up",
    volatility: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic trending data with a clear breakout."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")

    if trend == "up":
        base = np.linspace(100, 110, n_bars)
    else:
        base = np.linspace(110, 100, n_bars)

    noise = rng.normal(0, volatility, n_bars)
    close = base + noise
    high = close + rng.uniform(0.5, 2.0, n_bars)
    low = close - rng.uniform(0.5, 2.0, n_bars)
    opens = close - rng.uniform(-1.0, 1.0, n_bars)
    volume = rng.uniform(1e6, 3e6, n_bars)

    return pd.DataFrame(
        {"Open": opens, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestDonchianChannelBreakout:
    """Test Donchian Channel Breakout detector."""

    def test_init_defaults(self) -> None:
        detector = DonchianChannelBreakout()
        assert detector.channel_period == 20
        assert detector.exit_period == 10
        assert detector.volume_filter is True
        assert detector.atr_filter is True
        assert detector.retest_entry is False

    def test_init_custom(self) -> None:
        detector = DonchianChannelBreakout(
            channel_period=50, exit_period=20, volume_filter=False, atr_filter=False
        )
        assert detector.channel_period == 50
        assert detector.exit_period == 20
        assert detector.volume_filter is False
        assert detector.atr_filter is False

    def test_detect_insufficient_data(self) -> None:
        df = make_trending_df(n_bars=10)
        detector = DonchianChannelBreakout(channel_period=20)
        result = detector.detect(df, 5)
        assert result.detected is False

    def test_detect_upward_breakout(self) -> None:
        df = make_trending_df(n_bars=60, trend="up", volatility=0.5)
        last_date = df.index[-1]
        df.loc[last_date, "Close"] = 120.0
        df.loc[last_date, "High"] = 121.0
        df.loc[last_date, "Volume"] = 5e6

        detector = DonchianChannelBreakout(
            channel_period=20, volume_filter=False, atr_filter=False, retest_entry=False
        )
        result = detector.detect(df, len(df) - 1)

        assert result.detected is True
        assert result.signal is not None
        assert result.signal.direction.value == "Long"

    def test_detect_downward_breakout(self) -> None:
        df = make_trending_df(n_bars=60, trend="down", volatility=0.5)
        last_date = df.index[-1]
        df.loc[last_date, "Close"] = 80.0
        df.loc[last_date, "Low"] = 79.0
        df.loc[last_date, "Volume"] = 5e6

        detector = DonchianChannelBreakout(
            channel_period=20, volume_filter=False, atr_filter=False, retest_entry=False
        )
        result = detector.detect(df, len(df) - 1)

        assert result.detected is True
        assert result.signal is not None
        assert result.signal.direction.value == "Short"

    def test_signal_has_metadata(self) -> None:
        df = make_trending_df(n_bars=60, trend="up", volatility=0.5)
        last_date = df.index[-1]
        df.loc[last_date, "Close"] = 120.0
        df.loc[last_date, "High"] = 121.0

        detector = DonchianChannelBreakout(
            channel_period=20, volume_filter=False, atr_filter=False, retest_entry=False
        )
        result = detector.detect(df, len(df) - 1)

        assert result.signal is not None
        meta = result.signal.metadata
        assert "breakout_direction" in meta
        assert "upper_channel" in meta
        assert "lower_channel" in meta
        assert "middle_line" in meta
        assert "quality" in meta
        assert "channel_period" in meta

    def test_no_breakout_in_consolidation(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        close = 100 + rng.uniform(-0.5, 0.5, 60)
        high = close + rng.uniform(0.5, 1.0, 60)
        low = close - rng.uniform(0.5, 1.0, 60)
        opens = close - rng.uniform(-0.3, 0.3, 60)
        volume = rng.uniform(1e6, 2e6, 60)

        df = pd.DataFrame(
            {"Open": opens, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

        detector = DonchianChannelBreakout(
            channel_period=20, volume_filter=False, atr_filter=False, retest_entry=False
        )
        detections = sum(1 for i in range(20, len(df)) if detector.detect(df, i).detected)
        assert detections <= 1

    def test_volume_filter_affects_quality(self) -> None:
        df = make_trending_df(n_bars=60, trend="up", volatility=0.5)
        last_date = df.index[-1]
        df.loc[last_date, "Close"] = 120.0
        df.loc[last_date, "High"] = 121.0
        df.loc[last_date, "Volume"] = 0.5e6  # Low volume

        detector_no_vol = DonchianChannelBreakout(
            channel_period=20, volume_filter=False, atr_filter=False, retest_entry=False
        )
        result_no_vol = detector_no_vol.detect(df, len(df) - 1)

        detector_with_vol = DonchianChannelBreakout(
            channel_period=20,
            volume_filter=True,
            volume_threshold=1.5,
            atr_filter=False,
            retest_entry=False,
        )
        result_with_vol = detector_with_vol.detect(df, len(df) - 1)

        # Both detect the breakout
        assert result_no_vol.detected is True
        assert result_with_vol.detected is True
        assert result_no_vol.signal is not None
        assert result_with_vol.signal is not None

        # Confidence is lower with volume filter (no volume bonus)
        assert result_with_vol.signal.confidence <= result_no_vol.signal.confidence
        assert result_with_vol.pivot_points.get("volume_confirmed") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
