"""Twelve ICT single-candlestick patterns with exact OHLC mathematics.

Paper: "Candlestick Brokers Study" — only 3/12 patterns (OWM/OBM/CBM) appear consistently
across brokers, but the mathematical definitions are useful for confluence scoring.

Patterns (12):
- Bullish: WM, CWM, OWM, WDD, WPU, WSS
- Bearish: BM, CBM, OBM, BGD, BPU, BSS

Each detector returns a boolean array (True at pattern bars) and a direction indicator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_candle_ratios(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> dict[str, np.ndarray]:
    eps = 1e-10
    body = np.abs(close - open_)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    total_range = high - low
    body_ratio = body / (total_range + eps)
    upper_wick_ratio = upper_wick / (total_range + eps)
    lower_wick_ratio = lower_wick / (total_range + eps)
    bullish = close > open_
    return {
        "body": body,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "total_range": total_range,
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "bullish": bullish,
    }


# ── Bullish Patterns ──


def detect_white_marubozu(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """White Marubozu: opens at low, closes at high, body > 80% of range."""
    r = _compute_candle_ratios(open_, high, low, close)
    return (r["bullish"] & (r["body_ratio"] > 0.80)).astype(np.int8)


def detect_closing_white_marubozu(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Closing White Marubozu: bullish, body > 70%, upper wick < 10%, lower wick > 10%."""
    r = _compute_candle_ratios(open_, high, low, close)
    return (
        r["bullish"]
        & (r["body_ratio"] > 0.70)
        & (r["upper_wick_ratio"] < 0.10)
        & (r["lower_wick_ratio"] > 0.10)
    ).astype(np.int8)


def detect_opening_white_marubozu(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Opening White Marubozu: bullish, body > 70%, upper wick > 10%, lower wick < 10%."""
    r = _compute_candle_ratios(open_, high, low, close)
    return (
        r["bullish"]
        & (r["body_ratio"] > 0.70)
        & (r["upper_wick_ratio"] > 0.10)
        & (r["lower_wick_ratio"] < 0.10)
    ).astype(np.int8)


def detect_white_dragonfly_doji(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """White Dragonfly Doji: body < 20%, upper wick < 10%, lower wick > 50%, close > open."""
    r = _compute_candle_ratios(open_, high, low, close)
    return (
        r["bullish"]
        & (r["body_ratio"] < 0.20)
        & (r["upper_wick_ratio"] < 0.10)
        & (r["lower_wick_ratio"] > 0.50)
    ).astype(np.int8)


def detect_white_paper_umbrella(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """White Paper Umbrella (Hammer): body 20-50%, lower wick > 2x body, upper wick < 10%, bullish."""
    r = _compute_candle_ratios(open_, high, low, close)
    return (
        r["bullish"]
        & (r["body_ratio"] >= 0.20)
        & (r["body_ratio"] <= 0.50)
        & (r["lower_wick_ratio"] > r["body_ratio"] * 2)
        & (r["upper_wick_ratio"] < 0.10)
    ).astype(np.int8)


def detect_white_spinning_top(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """White Small Shadow (Spinning Top): body 20-40%, upper+lower wick both > body/2, bullish."""
    r = _compute_candle_ratios(open_, high, low, close)
    half_body = r["body_ratio"] * 0.5
    return (
        r["bullish"]
        & (r["body_ratio"] >= 0.20)
        & (r["body_ratio"] <= 0.40)
        & (r["upper_wick_ratio"] > half_body)
        & (r["lower_wick_ratio"] > half_body)
    ).astype(np.int8)


# ── Bearish Patterns ──


def detect_black_marubozu(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Black Marubozu: opens at high, closes at low, body > 80% of range."""
    r = _compute_candle_ratios(open_, high, low, close)
    return (~r["bullish"] & (r["body_ratio"] > 0.80)).astype(np.int8)


def detect_closing_black_marubozu(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Closing Black Marubozu: bearish, body > 70%, upper wick < 10%, lower wick > 10%."""
    r = _compute_candle_ratios(open_, high, low, close)
    return (
        ~r["bullish"]
        & (r["body_ratio"] > 0.70)
        & (r["upper_wick_ratio"] < 0.10)
        & (r["lower_wick_ratio"] > 0.10)
    ).astype(np.int8)


def detect_opening_black_marubozu(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Opening Black Marubozu: bearish, body > 70%, upper wick > 10%, lower wick < 10%."""
    r = _compute_candle_ratios(open_, high, low, close)
    return (
        ~r["bullish"]
        & (r["body_ratio"] > 0.70)
        & (r["upper_wick_ratio"] > 0.10)
        & (r["lower_wick_ratio"] < 0.10)
    ).astype(np.int8)


def detect_black_gravestone_doji(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Black Gravestone Doji: body < 20%, upper wick > 50%, lower wick < 10%, bearish."""
    r = _compute_candle_ratios(open_, high, low, close)
    return (
        ~r["bullish"]
        & (r["body_ratio"] < 0.20)
        & (r["upper_wick_ratio"] > 0.50)
        & (r["lower_wick_ratio"] < 0.10)
    ).astype(np.int8)


def detect_black_paper_umbrella(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Black Paper Umbrella (Hanging Man): body 20-50%, lower wick > 2x body, upper wick < 10%, bearish."""
    r = _compute_candle_ratios(open_, high, low, close)
    return (
        ~r["bullish"]
        & (r["body_ratio"] >= 0.20)
        & (r["body_ratio"] <= 0.50)
        & (r["lower_wick_ratio"] > r["body_ratio"] * 2)
        & (r["upper_wick_ratio"] < 0.10)
    ).astype(np.int8)


def detect_black_spinning_top(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Black Small Shadow (Spinning Top): body 20-40%, upper+lower wick both > body/2, bearish."""
    r = _compute_candle_ratios(open_, high, low, close)
    half_body = r["body_ratio"] * 0.5
    return (
        ~r["bullish"]
        & (r["body_ratio"] >= 0.20)
        & (r["body_ratio"] <= 0.40)
        & (r["upper_wick_ratio"] > half_body)
        & (r["lower_wick_ratio"] > half_body)
    ).astype(np.int8)


# ── Bulk Detection ──


def detect_all_twelve(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> dict[str, np.ndarray]:
    """Detect all 12 ICT candlestick patterns and return as a dictionary.

    Returns:
        Dict mapping pattern_name -> int8 array (1=bullish, -1=bearish for that pattern).
        Many patterns are direction-specific so their arrays contain only 0 or +1.
    """
    bullish_map = {
        "white_marubozu": detect_white_marubozu(open_, high, low, close),
        "closing_white_marubozu": detect_closing_white_marubozu(open_, high, low, close),
        "opening_white_marubozu": detect_opening_white_marubozu(open_, high, low, close),
        "white_dragonfly_doji": detect_white_dragonfly_doji(open_, high, low, close),
        "white_paper_umbrella": detect_white_paper_umbrella(open_, high, low, close),
        "white_spinning_top": detect_white_spinning_top(open_, high, low, close),
    }
    bearish_map = {
        "black_marubozu": detect_black_marubozu(open_, high, low, close),
        "closing_black_marubozu": detect_closing_black_marubozu(open_, high, low, close),
        "opening_black_marubozu": detect_opening_black_marubozu(open_, high, low, close),
        "black_gravestone_doji": detect_black_gravestone_doji(open_, high, low, close),
        "black_paper_umbrella": detect_black_paper_umbrella(open_, high, low, close),
        "black_spinning_top": detect_black_spinning_top(open_, high, low, close),
    }
    return {**bullish_map, **bearish_map}
