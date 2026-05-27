"""P25: Fuzzy scoring wrapper for strategy integration.

Provides compute_fuzzy_score() that wraps FuzzyInferenceSystem for
easy integration into CombinedStrategy and other strategies.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.signals.fuzzy_system import FuzzyInferenceSystem

logger = logging.getLogger(__name__)


def compute_fuzzy_score(
    df: pd.DataFrame,
    ema_short_col: str = "ema_short",
    ema_long_col: str = "ema_long",
    hbar_col: str = "hbar",
    ivol_col: str = "ivol",
    bb_lower_col: str = "bb_lower",
    bb_upper_col: str = "bb_upper",
    close_col: str = "Close",
) -> pd.Series:
    """Compute fuzzy logic score for each bar. Returns -1..+1 series.

    The fuzzy system evaluates 4 indicators (dEMA, Heikin-Ashi bars,
    implied volatility, BB position) through Mamdani Min inference with
    weighted averaging defuzzification to produce a continuous signal
    in [-1, 1] where positive = bullish, negative = bearish.

    Args:
        df: OHLCV DataFrame with indicator columns precomputed.
            Required columns: ema_short, ema_long, hbar, ivol,
            bb_lower, bb_upper, and a close column.
        ema_short_col: Name of short EMA column.
        ema_long_col: Name of long EMA column.
        hbar_col: Name of Heikin-Ashi bar column.
        ivol_col: Name of implied volatility column.
        bb_lower_col: Name of BB lower band column.
        bb_upper_col: Name of BB upper band column.
        close_col: Name of close price column.

    Returns:
        Series of fuzzy scores in [-1, 1], same index as input df.
    """
    required = [
        ema_short_col,
        ema_long_col,
        hbar_col,
        ivol_col,
        bb_lower_col,
        bb_upper_col,
        close_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for fuzzy scoring: {missing}")

    fis = FuzzyInferenceSystem()
    result_df = fis.evaluate_df(
        df,
        ema_short_col=ema_short_col,
        ema_long_col=ema_long_col,
        hbar_col=hbar_col,
        iv_col=ivol_col,
        bb_lower_col=bb_lower_col,
        bb_upper_col=bb_upper_col,
        close_col=close_col,
    )
    return pd.Series(result_df["signal"].values, index=df.index)


def compute_fuzzy_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute all indicators needed for fuzzy inference system.

    Computes: EMA(12), EMA(26), Heikin-Ashi bars, ATR-based implied
    volatility, and Bollinger Bands (20, 2).

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns.

    Returns:
        DataFrame with added columns: ema_short, ema_long, hbar, ivol,
        bb_lower, bb_upper.
    """
    result = df.copy()

    close = result["Close"].astype(float)
    high = result["High"].astype(float)
    low = result["Low"].astype(float)
    open_ = result["Open"].astype(float)

    result["ema_short"] = close.ewm(span=12, adjust=False).mean()
    result["ema_long"] = close.ewm(span=26, adjust=False).mean()

    ha_close = (open_ + high + low + close) / 4.0
    ha_open = ha_close.copy()
    for i in range(1, len(ha_open)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    result["hbar"] = ha_close - ha_open

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean().bfill()
    result["ivol"] = atr / close.replace(0, np.nan).bfill()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    result["bb_lower"] = bb_mid - 2.0 * bb_std
    result["bb_upper"] = bb_mid + 2.0 * bb_std

    result = result.bfill().fillna(0)
    return result
