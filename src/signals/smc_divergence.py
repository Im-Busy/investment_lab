"""SMT (Smart Money Technique) Divergence detector.

Detects when correlated assets diverge at key SMC structural levels,
signaling market manipulation and impending reversal.

Source: ICT MMXM Model, toaz.info glossary
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SMTDivergence:
    """A single SMT divergence instance."""

    bar_index: int
    direction: int  # +1 bullish divergence, -1 bearish divergence
    primary_symbol: str
    correlated_symbol: str
    primary_level: float
    correlated_level: float
    divergence_strength: float  # 0.0-1.0


SMC_CORRELATED_PAIRS: dict[str, str] = {
    "SPY": "QQQ",
    "QQQ": "SPY",
    "ES=F": "NQ=F",
    "NQ=F": "ES=F",
    "EURUSD=X": "GBPUSD=X",
    "GBPUSD=X": "EURUSD=X",
    "BTC-USD": "ETH-USD",
    "ETH-USD": "BTC-USD",
    "GC=F": "SI=F",
    "SI=F": "GC=F",
}


def detect_smt_divergence(
    primary_df: pd.DataFrame,
    correlated_df: pd.DataFrame,
    lookback_bars: int = 20,
    correlation_window: int = 50,
    min_correlation: float = 0.7,
    divergence_threshold_pct: float = 0.3,
) -> tuple[np.ndarray, list[SMTDivergence]]:
    """Detect SMT divergences between two correlated instruments.

    ICT MMXM:
    - Bullish SMT: Price makes lower low but correlated does NOT → reversal up
    - Bearish SMT: Price makes higher high but correlated does NOT → reversal down

    Args:
        primary_df: OHLCV DataFrame for primary instrument
        correlated_df: OHLCV DataFrame for correlated instrument
        lookback_bars: Lookback window for comparing swing extremes
        correlation_window: Rolling correlation window
        min_correlation: Minimum correlation to consider instruments correlated
        divergence_threshold_pct: Minimum divergence as % of price

    Returns:
        signals: +1 bullish divergence, -1 bearish divergence
        divergences: list of SMTDivergence dataclasses
    """
    p_high = primary_df["High"].to_numpy(dtype=np.float64)
    p_low = primary_df["Low"].to_numpy(dtype=np.float64)
    c_high = correlated_df["High"].to_numpy(dtype=np.float64)
    c_low = correlated_df["Low"].to_numpy(dtype=np.float64)
    p_close = primary_df["Close"].to_numpy(dtype=np.float64)
    c_close = correlated_df["Close"].to_numpy(dtype=np.float64)
    n = min(len(p_high), len(c_high))

    signals = np.zeros(n, dtype=np.int8)
    divergences: list[SMTDivergence] = []

    rolling_corr = (
        pd.Series(p_close)
        .rolling(correlation_window, min_periods=10)
        .corr(pd.Series(c_close))
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )

    for i in range(lookback_bars * 2, n):
        if rolling_corr[i] < min_correlation:
            continue

        window_p_high = p_high[i - lookback_bars : i]
        window_p_low = p_low[i - lookback_bars : i]
        window_c_high = c_high[i - lookback_bars : i]
        window_c_low = c_low[i - lookback_bars : i]

        p_recent_high = float(np.max(window_p_high))
        p_recent_low = float(np.min(window_p_low))
        c_recent_high = float(np.max(window_c_high))
        c_recent_low = float(np.min(window_c_low))

        p_range = p_recent_high - p_recent_low
        if p_range <= 0:
            continue

        # Bearish SMT: primary makes higher high but correlated does NOT
        p_latest_high = p_high[i]
        c_latest_high = c_high[i]
        if p_latest_high > p_recent_high * (1 + divergence_threshold_pct * 0.01):
            if c_latest_high <= c_recent_high:
                strength = (p_latest_high - p_recent_high) / p_recent_high
                divergences.append(
                    SMTDivergence(
                        bar_index=i,
                        direction=-1,
                        primary_symbol="PRIMARY",
                        correlated_symbol="CORRELATED",
                        primary_level=p_latest_high,
                        correlated_level=c_latest_high,
                        divergence_strength=float(np.clip(strength * 10.0, 0.0, 1.0)),
                    )
                )
                signals[i] = -1

        # Bullish SMT: primary makes lower low but correlated does NOT
        p_latest_low = p_low[i]
        c_latest_low = c_low[i]
        if p_latest_low < p_recent_low * (1 - divergence_threshold_pct * 0.01):
            if c_latest_low >= c_recent_low:
                strength = (p_recent_low - p_latest_low) / p_recent_low
                divergences.append(
                    SMTDivergence(
                        bar_index=i,
                        direction=1,
                        primary_symbol="PRIMARY",
                        correlated_symbol="CORRELATED",
                        primary_level=p_latest_low,
                        correlated_level=c_latest_low,
                        divergence_strength=float(np.clip(strength * 10.0, 0.0, 1.0)),
                    )
                )
                signals[i] = 1

    return signals, divergences
