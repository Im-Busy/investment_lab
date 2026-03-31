"""
Technical Indicators Module

This module provides technical indicators and utility functions for pattern detection.
Includes SMC/ICT-specific indicators for liquidity mapping and structure detection.
"""

# SMC/ICT Indicators
from .asian_range import (
    AsianRange,
    calculate_session_statistics,
    detect_asian_range,
    get_asian_range_for_day,
)
from .fibonacci import fibonacci_extension, fibonacci_retracement
from .ifvg import IFVG, IFVGProximity, detect_ifvg, find_nearest_unfilled_ifvg, update_ifvg_status
from .indicator_cache import IndicatorCache
from .liquidity_sweep import (
    SweepInfo,
    detect_liquidity_sweep,
    find_sweep_candle,
    validate_sweep_volume,
)
from .mss import (
    MSSInfo,
    PivotPoint,
    detect_mss,
    find_pivot_high,
    find_pivot_low,
    validate_mss_with_htf,
)
from .pivots import (
    calculate_pivot_points,
    clear_pivot_cache,
    find_local_extrema,
    find_swing_highs,
    find_swing_highs_cached,
    find_swing_lows,
    find_swing_lows_cached,
    get_pivot_cache_info,
)
from .regime import MarketPhase, MarketRegimeDetector, RegimeState, TrendDirection, VolatilityRegime
from .technical import (
    adx,
    atr,
    average_range,
    clear_indicator_cache,
    ema,
    ema_cached,
    get_indicator_cache_info,
    rsi,
    sma,
    sma_cached,
    true_range,
)

__all__ = [
    # Technical indicators
    "sma",
    "sma_cached",
    "ema",
    "ema_cached",
    "atr",
    "rsi",
    "adx",
    "true_range",
    "average_range",
    "clear_indicator_cache",
    "get_indicator_cache_info",
    # Pivot detection
    "find_local_extrema",
    "find_swing_highs",
    "find_swing_highs_cached",
    "find_swing_lows",
    "find_swing_lows_cached",
    "calculate_pivot_points",
    "clear_pivot_cache",
    "get_pivot_cache_info",
    # Fibonacci
    "fibonacci_retracement",
    "fibonacci_extension",
    # Market Regime
    "MarketRegimeDetector",
    "RegimeState",
    "TrendDirection",
    "VolatilityRegime",
    "MarketPhase",
    # SMC/ICT Indicators - Asian Range
    "detect_asian_range",
    "get_asian_range_for_day",
    "calculate_session_statistics",
    "AsianRange",
    # SMC/ICT Indicators - Liquidity Sweep
    "detect_liquidity_sweep",
    "find_sweep_candle",
    "validate_sweep_volume",
    "SweepInfo",
    # SMC/ICT Indicators - IFVG
    "detect_ifvg",
    "find_nearest_unfilled_ifvg",
    "update_ifvg_status",
    "IFVG",
    "IFVGProximity",
    # SMC/ICT Indicators - MSS
    "detect_mss",
    "find_pivot_high",
    "find_pivot_low",
    "validate_mss_with_htf",
    "MSSInfo",
    "PivotPoint",
    # Performance Optimization
    "IndicatorCache",
]
