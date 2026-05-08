"""Hypothesis property-based tests for pattern detectors.

Verifies invariants across random valid price data:
  - detect() never raises an exception
  - confidence is always in [0, 1]
  - pattern_name matches the detector name
  - signal fields are valid when present
"""

from __future__ import annotations

import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings

from src.patterns.base import PatternResult
from src.patterns.basic.matching_lows import MatchingLows
from src.patterns.basic.n_bar_decline import NBarDecline
from src.patterns.basic.nr7id import NR7ID
from src.patterns.breakout.donchian import DonchianChannelBreakout
from src.patterns.classic.double_bottom import DoubleBottom
from src.patterns.complex.cup_handle import CupAndHandle
from src.patterns.complex.head_shoulders import HeadAndShoulders
from src.patterns.complex.parabolic_arc import ParabolicArc
from src.patterns.harmonic.bollinger import BollingerBands
from src.patterns.harmonic.gartley import GartleyPattern

# ── Data generation strategies ───────────────────────────────────────────────

MIN_BARS = 250  # Enough bars for the deepest-lookback detector (Gartley: ~120)


@st.composite
def _ohlc_df(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a valid OHLCV DataFrame with n rows."""
    n = draw(st.integers(min_value=MIN_BARS, max_value=500))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**31 - 1)))

    # Build random walk prices starting from 100
    returns = rng.normal(0, 0.02, size=n).cumsum()
    close = 100.0 + returns * 10.0

    # High >= Close, Low <= Close with realistic ranges
    bar_range = np.abs(rng.normal(0.5, 0.3, size=n)) + 0.1
    high = close + np.abs(bar_range)
    low = close - np.abs(bar_range)

    # Open within [low, high]
    open_frac = np.clip(rng.normal(0.5, 0.3, size=n), 0.0, 1.0)
    open_price = low + (high - low) * open_frac
    volume = np.clip(rng.lognormal(14, 0.8, size=n), 100, 1e9)

    return pd.DataFrame(
        {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


@st.composite
def _detector_and_df(draw: st.DrawFn, detector_cls: type) -> tuple:
    """Generate a detector instance and valid DataFrame."""
    inst = detector_cls()
    df = draw(_ohlc_df())
    return inst, df


# ── Common invariants tested across all detectors ────────────────────────────


def _check_common_invariants(result: PatternResult, detector_name: str) -> None:
    """Verify invariants that hold for every PatternResult."""
    assert isinstance(result, PatternResult), f"Expected PatternResult, got {type(result)}"
    # Some detectors append direction/breakout type (e.g. "NR7ID (Long)")
    assert result.pattern_name.startswith(detector_name), (
        f"pattern_name mismatch: {result.pattern_name} does not start with {detector_name}"
    )

    if result.signal is not None:
        sig = result.signal
        assert 0.0 <= sig.confidence <= 1.0, f"confidence out of bounds: {sig.confidence}"
        assert not np.isnan(sig.stop_loss), "stop_loss is NaN"
        assert not np.isnan(sig.take_profit_1), "take_profit_1 is NaN"
        assert sig.take_profit_1 > 0, f"take_profit_1 <= 0: {sig.take_profit_1}"
        assert sig.pattern_name.startswith(detector_name), (
            f"signal pattern_name mismatch: {sig.pattern_name} does not start with {detector_name}"
        )


# ── Property tests per detector ──────────────────────────────────────────────


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None
)
@given(detector_and_df=_detector_and_df(GartleyPattern))
def test_gartley_no_exceptions(detector_and_df: tuple) -> None:
    """GartleyPattern.detect() never raises for valid data."""
    inst, df = detector_and_df
    for i in range(inst.min_bars_required, len(df)):
        result = inst.detect(df, i)
        _check_common_invariants(result, inst.name)


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None
)
@given(detector_and_df=_detector_and_df(HeadAndShoulders))
def test_head_shoulders_no_exceptions(detector_and_df: tuple) -> None:
    inst, df = detector_and_df
    for i in range(inst.min_bars_required, len(df)):
        result = inst.detect(df, i)
        _check_common_invariants(result, inst.name)


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None
)
@given(detector_and_df=_detector_and_df(MatchingLows))
def test_matching_lows_no_exceptions(detector_and_df: tuple) -> None:
    inst, df = detector_and_df
    for i in range(inst.min_bars_required, len(df)):
        result = inst.detect(df, i)
        _check_common_invariants(result, inst.name)


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None
)
@given(detector_and_df=_detector_and_df(DonchianChannelBreakout))
def test_donchian_no_exceptions(detector_and_df: tuple) -> None:
    inst, df = detector_and_df
    for i in range(inst.min_bars_required, len(df)):
        result = inst.detect(df, i)
        _check_common_invariants(result, inst.name)


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None
)
@given(detector_and_df=_detector_and_df(NBarDecline))
def test_n_bar_decline_no_exceptions(detector_and_df: tuple) -> None:
    inst, df = detector_and_df
    for i in range(inst.min_bars_required, len(df)):
        result = inst.detect(df, i)
        _check_common_invariants(result, inst.name)


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None
)
@given(detector_and_df=_detector_and_df(ParabolicArc))
def test_parabolic_arc_no_exceptions(detector_and_df: tuple) -> None:
    inst, df = detector_and_df
    for i in range(inst.min_bars_required, len(df)):
        result = inst.detect(df, i)
        _check_common_invariants(result, inst.name)


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None
)
@given(detector_and_df=_detector_and_df(BollingerBands))
def test_bollinger_no_exceptions(detector_and_df: tuple) -> None:
    inst, df = detector_and_df
    for i in range(inst.min_bars_required, len(df)):
        result = inst.detect(df, i)
        _check_common_invariants(result, inst.name)


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None
)
@given(detector_and_df=_detector_and_df(NR7ID))
def test_nr7id_no_exceptions(detector_and_df: tuple) -> None:
    inst, df = detector_and_df
    for i in range(inst.min_bars_required, len(df)):
        result = inst.detect(df, i)
        _check_common_invariants(result, inst.name)


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None
)
@given(detector_and_df=_detector_and_df(CupAndHandle))
def test_cup_handle_no_exceptions(detector_and_df: tuple) -> None:
    inst, df = detector_and_df
    for i in range(inst.min_bars_required, len(df)):
        result = inst.detect(df, i)
        _check_common_invariants(result, inst.name)


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None
)
@given(detector_and_df=_detector_and_df(DoubleBottom))
def test_double_bottom_no_exceptions(detector_and_df: tuple) -> None:
    inst, df = detector_and_df
    for i in range(inst.min_bars_required, len(df)):
        result = inst.detect(df, i)
        _check_common_invariants(result, inst.name)
