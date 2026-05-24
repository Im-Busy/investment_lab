"""
Tests for new ICT indicator and strategy modules.

Tests: CISD, PO3, CRT, OTE indicators, plus SilverBullet, TurtleSoup,
CameronModel strategies and their plugin wrappers.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_ohlc(n: int = 100, seed: int = 42, trend: int = 0) -> pd.DataFrame:
    """Create synthetic OHLCV data with optional trend."""
    rng = np.random.default_rng(seed)
    base = 100.0
    drift = np.linspace(0, trend, n) if trend else np.zeros(n)
    close = base + drift + rng.normal(0, 0.5, n).cumsum()
    high = close + abs(rng.normal(0.2, 0.1, n))
    low = close - abs(rng.normal(0.2, 0.1, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(1000, 10000, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )
    return df


def _make_swing_df(n: int = 100) -> pd.DataFrame:
    """Create synthetic swing high/low DataFrame."""
    highlow = np.full(n, np.nan)
    level = np.full(n, np.nan)
    rng = np.random.default_rng(42)
    for i in range(5, n - 5, 10):
        if rng.random() > 0.5:
            highlow[i] = 1  # swing high
            level[i] = 105.0 + rng.normal(0, 1)
        else:
            highlow[i] = -1  # swing low
            level[i] = 95.0 + rng.normal(0, 1)
    return pd.DataFrame({"HighLow": highlow, "Level": level})


# ---------------------------------------------------------------------------
# CISD Tests
# ---------------------------------------------------------------------------


class TestCISD:
    def test_empty_dataframe(self):
        from src.indicators.cisd import detect_cisd

        df = _make_ohlc(2)
        results = detect_cisd(df, min_delivery_bars=3)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_vectorized_returns_columns(self):
        from src.indicators.cisd import detect_cisd_vectorized

        df = _make_ohlc(100, trend=5)
        result = detect_cisd_vectorized(df, min_delivery_bars=3)
        assert "bullish_cisd" in result.columns
        assert "bearish_cisd" in result.columns
        assert "cisd_strength" in result.columns
        assert len(result) == 100

    def test_bullish_cisd_detection(self):
        from src.indicators.cisd import detect_cisd

        np.random.seed(42)
        n = 30
        open_ = np.full(n, 100.0)
        close = np.full(n, 100.0)
        # Bullish delivery: 5 bars closing higher
        for i in range(5):
            open_[i] = 100.0 + i * 0.1
            close[i] = open_[i] + 0.5
        # Bearish CISD break: close below first delivery open
        for i in range(5, 10):
            open_[i] = 100.0 - i * 0.1
            close[i] = open_[i] - 0.5
        close[8] = 99.0  # close below first delivery open (100.0)

        high = np.maximum(open_, close) + 0.2
        low = np.minimum(open_, close) - 0.2
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})
        results = detect_cisd(df, min_delivery_bars=3)
        cisd_list = [r for r in results if r.detected]
        assert len(cisd_list) > 0

    def test_find_recent_cisd_no_signal(self):
        from src.indicators.cisd import detect_cisd_vectorized, find_recent_cisd

        df = _make_ohlc(50)
        cisd_df = detect_cisd_vectorized(df, min_delivery_bars=3)
        result = find_recent_cisd(cisd_df, 49, lookback=10)
        # May or may not find a signal - just verify no crash
        assert result is None or len(result) == 3


# ---------------------------------------------------------------------------
# PO3 Tests
# ---------------------------------------------------------------------------


class TestPO3:
    def test_daily_detection(self):
        from src.indicators.power_of_3 import detect_po3_daily

        df = _make_ohlc(100, trend=3)
        result = detect_po3_daily(df, min_accumulation_bars=3)
        assert "phase" in result.columns
        assert "confidence" in result.columns
        assert len(result) == 100
        # At least some bars should be classified
        unique_phases = set(result["phase"].unique())
        assert "none" in unique_phases  # default phase always present

    def test_intraday_detection(self):
        from src.indicators.power_of_3 import detect_po3_intraday

        df = _make_ohlc(100)
        # Create synthetic session data (Asia session active every other block)
        session_active = np.zeros(100)
        session_active[10:30] = 1
        session_active[50:70] = 1
        session_df = pd.DataFrame({"Active": session_active}, index=df.index)
        result = detect_po3_intraday(df, session_df)
        assert len(result) == 100

    def test_get_phase_at_bar(self):
        from src.indicators.power_of_3 import detect_po3_daily, get_po3_phase_at_bar, PO3Phase

        df = _make_ohlc(20)
        result = detect_po3_daily(df, min_accumulation_bars=2)
        phase_info = get_po3_phase_at_bar(result, 5)
        assert phase_info.phase in (
            PO3Phase.ACCUMULATION,
            PO3Phase.MANIPULATION,
            PO3Phase.DISTRIBUTION,
            PO3Phase.NONE,
        )


# ---------------------------------------------------------------------------
# CRT Tests
# ---------------------------------------------------------------------------


class TestCRT:
    def test_no_setup_on_short_df(self):
        from src.indicators.crt import detect_crt

        df = _make_ohlc(5)
        result = detect_crt(df, ref_bar_idx=0, atr=1.0)
        assert not result.detected

    def test_find_setups(self):
        from src.indicators.crt import find_crt_setups

        df = _make_ohlc(100, trend=5)
        atr = pd.Series(np.full(100, 0.5), index=df.index)
        setups = find_crt_setups(df, atr_series=atr, min_ref_range_atr=0.2, max_lookforward=10)
        assert isinstance(setups, list)

    def test_setup_has_correct_fields(self):
        from src.indicators.crt import detect_crt

        df = _make_ohlc(50, trend=3)
        atr = 1.0
        result = detect_crt(df, ref_bar_idx=5, atr=atr, min_ref_range_atr=0.3, max_lookforward=15)
        assert hasattr(result, "detected")
        assert hasattr(result, "ref_bar")
        assert hasattr(result, "ref_high")
        assert hasattr(result, "ref_low")
        assert hasattr(result, "target_direction")
        assert result.to_dict()  # dict conversion works


# ---------------------------------------------------------------------------
# OTE Tests
# ---------------------------------------------------------------------------


class TestOTE:
    def test_bullish_ote_zone(self):
        from src.indicators.ote import calculate_ote_zone, OTEZoneType

        zone = calculate_ote_zone(
            swing_high=110.0,
            swing_low=100.0,
            direction="bullish",
            current_price=103.0,  # 70% retracement = near optimal
        )
        assert zone.direction == "bullish"
        assert zone.swing_high == 110.0
        assert zone.swing_low == 100.0
        assert zone.retracement_pct > 0

    def test_bearish_ote_zone(self):
        from src.indicators.ote import calculate_ote_zone

        zone = calculate_ote_zone(
            swing_high=110.0,
            swing_low=100.0,
            direction="bearish",
            current_price=107.0,  # 70% retracement from low
        )
        assert zone.direction == "bearish"

    def test_golden_zone(self):
        from src.indicators.ote import calculate_ote_zone, OTEZoneType

        # 61.8% retracement of a 10-point range = 6.18 from top
        # For bullish: 110 - 6.18 = 103.82
        zone = calculate_ote_zone(110.0, 100.0, "bullish", 103.82)
        assert zone.zone_type == OTEZoneType.GOLDEN

    def test_no_zone_outside_range(self):
        from src.indicators.ote import calculate_ote_zone, OTEZoneType

        zone = calculate_ote_zone(110.0, 100.0, "bullish", 108.0)  # Only 20% retracement
        assert zone.zone_type == OTEZoneType.NONE

    def test_detect_entries(self):
        from src.indicators.ote import detect_ote_entries

        df = _make_ohlc(100, trend=5)
        swings = _make_swing_df(100)
        entries = detect_ote_entries(df, swings)
        assert isinstance(entries, list)


# ---------------------------------------------------------------------------
# Silver Bullet Strategy Test
# ---------------------------------------------------------------------------


class TestSilverBulletStrategy:
    def test_import(self):
        from src.strategies.silver_bullet import SilverBulletStrategy

        assert SilverBulletStrategy is not None

    def test_plugin_registration(self):
        from src.strategies.strategy_registry import SilverBulletPlugin, StrategyType

        plugin = SilverBulletPlugin(kill_zone="london_open")
        assert plugin.get_name() == "SilverBullet"
        assert plugin.get_type() == StrategyType.SMC
        assert plugin.is_enabled()

    def test_plugin_evaluate(self):
        from src.strategies.strategy_registry import SilverBulletPlugin

        plugin = SilverBulletPlugin(kill_zone="london_open")
        df = _make_ohlc(200, seed=99, trend=2)
        signals = plugin.evaluate(df)
        assert isinstance(signals, list)

    def test_plugin_disabled(self):
        from src.strategies.strategy_registry import SilverBulletPlugin

        plugin = SilverBulletPlugin(kill_zone="london_open")
        plugin.disable()
        df = _make_ohlc(100)
        signals = plugin.evaluate(df)
        assert signals == []


# ---------------------------------------------------------------------------
# Turtle Soup Strategy Test
# ---------------------------------------------------------------------------


class TestTurtleSoupStrategy:
    def test_import(self):
        from src.strategies.turtle_soup import TurtleSoupStrategy

        assert TurtleSoupStrategy is not None

    def test_plugin_registration(self):
        from src.strategies.strategy_registry import TurtleSoupPlugin, StrategyType

        plugin = TurtleSoupPlugin()
        assert plugin.get_name() == "TurtleSoup"
        assert plugin.get_type() == StrategyType.SMC

    def test_plugin_evaluate(self):
        from src.strategies.strategy_registry import TurtleSoupPlugin

        plugin = TurtleSoupPlugin()
        df = _make_ohlc(200, seed=101, trend=3)
        signals = plugin.evaluate(df)
        assert isinstance(signals, list)


# ---------------------------------------------------------------------------
# Cameron Model Strategy Test
# ---------------------------------------------------------------------------


class TestCameronModelStrategy:
    def test_import(self):
        from src.strategies.cameron_model import CameronModelStrategy

        assert CameronModelStrategy is not None

    def test_plugin_registration(self):
        from src.strategies.strategy_registry import CameronModelPlugin, StrategyType

        plugin = CameronModelPlugin()
        assert plugin.get_name() == "CameronModel"
        assert plugin.get_type() == StrategyType.SMC

    def test_plugin_evaluate(self):
        from src.strategies.strategy_registry import CameronModelPlugin

        plugin = CameronModelPlugin()
        df = _make_ohlc(200, seed=103, trend=4)
        signals = plugin.evaluate(df)
        assert isinstance(signals, list)


# ---------------------------------------------------------------------------
# Registry Integration Test
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_all_plugins_register(self):
        from src.strategies.strategy_registry import (
            SilverBulletPlugin,
            TurtleSoupPlugin,
            CameronModelPlugin,
            SMCPlugin,
            StrategyRegistry,
        )

        registry = StrategyRegistry()
        registry.register(SMCPlugin())
        registry.register(SilverBulletPlugin(kill_zone="london_open"))
        registry.register(TurtleSoupPlugin())
        registry.register(CameronModelPlugin())

        names = registry.get_plugin_names()
        assert "SMC" in names
        assert "SilverBullet" in names
        assert "TurtleSoup" in names
        assert "CameronModel" in names

    def test_all_plugins_evaluate(self):
        from src.strategies.strategy_registry import (
            SilverBulletPlugin,
            TurtleSoupPlugin,
            CameronModelPlugin,
            SMCPlugin,
            StrategyRegistry,
        )

        registry = StrategyRegistry()
        registry.register(SMCPlugin())
        registry.register(SilverBulletPlugin(kill_zone="london_open"))
        registry.register(TurtleSoupPlugin())
        registry.register(CameronModelPlugin())

        df = _make_ohlc(200, seed=77, trend=2)
        signals = registry.evaluate(df)
        assert isinstance(signals, list)

    def test_confluence_returns_correctly(self):
        from src.strategies.strategy_registry import (
            SilverBulletPlugin,
            TurtleSoupPlugin,
            StrategyRegistry,
        )

        registry = StrategyRegistry()
        registry.register(SilverBulletPlugin(kill_zone="london_open"))
        registry.register(TurtleSoupPlugin())

        df = _make_ohlc(200, seed=42, trend=2)
        signals = registry.evaluate_confluence(df, min_plugins=2)
        assert isinstance(signals, list)


# ---------------------------------------------------------------------------
# All indicators importable via public API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_indicators_importable(self):
        from src.indicators import (
            CISDInfo,
            CRTSetup,
            OTEZone,
            OTEZoneType,
            PO3Info,
            PO3Phase,
            calculate_ote_zone,
            detect_cisd,
            detect_cisd_vectorized,
            detect_crt,
            detect_ote_entries,
            detect_po3_daily,
            detect_po3_intraday,
            find_crt_setups,
            find_recent_cisd,
            get_crt_signal_for_bar,
            get_po3_phase_at_bar,
        )

        assert True  # imports succeeded
