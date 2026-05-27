"""P28-13: Fund flow / order flow signals for market microstructure.

Computes institutional fund flows and large-order detection as market
microstructure signals. Combines volume decomposition, money flow analysis,
and smart/dumb money classification from price-volume patterns.

Source: stock-sdk API — fund flow & order flow data feeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FlowSignal:
    """Per-bar fund/order flow signal."""

    money_flow_ratio: float
    """Chaikin-style MF multiplier for the bar."""

    volume_force: float
    """Volume × price_change — raw force behind the move."""

    on_balance_volume: float
    """Running OBV at this bar."""

    accumulation_dist: float
    """Running A/D line value."""

    eom: float
    """Ease of Movement indicator."""

    large_order_detected: bool = False
    """Whether this bar qualifies as a large-order event."""


@dataclass
class FlowSummary:
    """Rolling-window summary of fund flow signals."""

    ticker: str
    net_flow: float
    """Net money flow over the window (positive = accumulation)."""

    flow_divergence: float
    """Correlation between flow and price — negative = divergence."""

    smart_money_index: float
    """Normalized [-1, 1] smart/dumb money classification."""

    accumulation_days: int
    """Count of days with positive money flow."""

    distribution_days: int
    """Count of days with negative money flow."""

    large_order_count: int
    """Number of large-order events in the window."""


def compute_money_flow(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Compute Chaikin Money Flow multiplier per bar.

    MFM = ((Close - Low) - (High - Close)) / (High - Low)
    Returns 0 when High == Low.

    Args:
        high, low, close, volume: Arrays of OHLCV data.
    """
    hl_range = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = np.where(hl_range > 0, ((close - low) - (high - close)) / hl_range, 0.0)
    mfm = np.clip(mfm, -1.0, 1.0)
    return mfm


def compute_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Compute On-Balance Volume.

    OBV_i = OBV_{i-1} + volume_i * sign(close_i - close_{i-1})
    """
    diff = np.diff(close, prepend=close[0])
    obv = np.cumsum(volume * np.sign(diff))
    return obv


def compute_accumulation_distribution(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Compute Accumulation/Distribution Line.

    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    A/D_i = A/D_{i-1} + CLV_i * Volume_i
    """
    mfm = compute_money_flow(high, low, close, volume)
    ad = np.cumsum(mfm * volume)
    return ad


def compute_ease_of_movement(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """Compute Ease of Movement indicator.

    EOM_i = ((H_i + L_i)/2 - (H_{i-1} + L_{i-1})/2) / (Volume_i / (H_i - L_i))

    Smoothed with a simple moving average.
    """
    mid = (high + low) / 2.0
    mid_diff = np.diff(mid, prepend=mid[0])
    hl_range = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        box_ratio = np.where(hl_range > 0, volume / hl_range, np.nan)
        eom_raw = np.where(box_ratio > 0, mid_diff / box_ratio, 0.0)
    eom_raw = np.nan_to_num(eom_raw, nan=0.0)

    if len(eom_raw) <= window:
        return eom_raw

    kernel = np.ones(window) / window
    eom = np.convolve(eom_raw, kernel, mode="same")
    return eom


def detect_large_orders(
    volume: np.ndarray, window: int = 20, threshold_mult: float = 2.0
) -> np.ndarray:
    """Detect bars with unusually high volume (large-order events).

    Uses rolling median absolute deviation for robust outlier detection.

    Args:
        volume: Volume array.
        window: Rolling window size.
        threshold_mult: Multiples of rolling median above which an order is "large".

    Returns:
        Boolean array where True = large order detected.
    """
    if len(volume) < window:
        return np.zeros(len(volume), dtype=bool)

    s = pd.Series(volume)
    rolled = s.rolling(window, center=True, min_periods=window)
    median = rolled.median().to_numpy()
    roll_mean = rolled.mean().to_numpy()
    vol_std = rolled.std().to_numpy()

    def _rolling_mad(arr: np.ndarray) -> float:
        med = np.median(arr)
        abs_dev = np.abs(arr - med)
        result = float(np.median(abs_dev))
        if result == 0.0:
            result = float(np.mean(abs_dev))
        return max(result, 1e-10)

    mad = rolled.apply(_rolling_mad, raw=True).to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        z_median = np.where(
            (mad > 1e-10) & ~np.isnan(median),
            np.abs(volume - median) / (mad * 0.6745),
            0.0,
        )
        z_std = np.where(
            (vol_std > 1e-10) & ~np.isnan(roll_mean),
            np.abs(volume - roll_mean) / vol_std,
            0.0,
        )
        valid_median = ~np.isnan(median) & (mad > 1e-10)
        valid_std = ~np.isnan(roll_mean) & (vol_std > 1e-10)
        z_score = np.where(valid_median, z_median, np.where(valid_std, z_std, 0.0))

    return z_score > threshold_mult


def compute_flow_components(
    df: pd.DataFrame,
    *,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    volume_col: str = "Volume",
    ema_window: int = 14,
) -> list[FlowSignal]:
    """Compute all fund flow components from OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with High, Low, Close, Volume columns.
        high_col, low_col, close_col, volume_col: Column name overrides.
        ema_window: Smoothing window for EoM.

    Returns:
        List of FlowSignal per bar.
    """
    high = df[high_col].values.astype(np.float64)
    low = df[low_col].values.astype(np.float64)
    close = df[close_col].values.astype(np.float64)
    volume = df[volume_col].values.astype(np.float64)

    mfm = compute_money_flow(high, low, close, volume)
    vf = volume * (close - np.roll(close, 1))
    vf[0] = 0.0
    obv = compute_obv(close, volume)
    ad = compute_accumulation_distribution(high, low, close, volume)
    eom = compute_ease_of_movement(high, low, volume, window=ema_window)
    large_orders = detect_large_orders(volume)

    signals: list[FlowSignal] = []
    for i in range(len(df)):
        signals.append(
            FlowSignal(
                money_flow_ratio=float(mfm[i]),
                volume_force=float(vf[i]),
                on_balance_volume=float(obv[i]),
                accumulation_dist=float(ad[i]),
                eom=float(eom[i]),
                large_order_detected=bool(large_orders[i]),
            )
        )
    return signals


def compute_flow_summary(
    close: np.ndarray,
    flow_signals: list[FlowSignal],
    window: int = 20,
) -> FlowSummary:
    """Compute rolling-window summary of fund flow activity.

    Args:
        close: Closing price array.
        flow_signals: List of FlowSignal from compute_flow_components().
        window: Lookback window for summary statistics.

    Returns:
        FlowSummary with aggregated metrics.
    """
    if len(flow_signals) < window:
        window = len(flow_signals)

    recent = flow_signals[-window:]
    recent_close = close[-window:]

    money_flows = np.array([s.money_flow_ratio for s in recent])
    vol_forces = np.array([s.volume_force for s in recent])
    ad_vals = np.array([s.accumulation_dist for s in recent])
    large_cnt = sum(1 for s in recent if s.large_order_detected)

    net_flow = float(np.sum(vol_forces))

    if len(recent_close) >= 3 and np.std(vol_forces) > 1e-10 and np.std(recent_close) > 1e-10:
        flow_div = np.corrcoef(vol_forces, recent_close)[0, 1]
        flow_div = float(np.nan_to_num(flow_div, nan=0.0))
    else:
        flow_div = 0.0

    if len(ad_vals) >= 2:
        ad_change = ad_vals[-1] - ad_vals[0]
        ad_max = max(np.abs(ad_vals).max(), 1e-10)
        smi = float(np.clip(ad_change / ad_max, -1.0, 1.0))
    else:
        smi = 0.0

    accum = int(np.sum(money_flows > 0))
    distr = int(np.sum(money_flows < 0))

    return FlowSummary(
        ticker="",
        net_flow=net_flow,
        flow_divergence=flow_div,
        smart_money_index=smi,
        accumulation_days=accum,
        distribution_days=distr,
        large_order_count=int(large_cnt),
    )


def fund_flow_to_ml_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Convert OHLCV to ML-ready fund flow features.

    Args:
        df: OHLCV DataFrame.
        windows: Rolling windows for summary features (default: [5, 10, 20, 60]).

    Returns:
        DataFrame with fund flow feature columns, same index as df.
    """
    if windows is None:
        windows = [5, 10, 20, 60]

    signals = compute_flow_components(df)
    close = df["Close"].astype(np.float64).values

    all_features: dict[str, np.ndarray] = {
        "money_flow_ratio": np.array([s.money_flow_ratio for s in signals]),
        "volume_force": np.array([s.volume_force for s in signals]),
        "obv_change_1d": np.diff(np.array([s.on_balance_volume for s in signals]), prepend=0),
        "obv_change_5d": np.full(len(signals), np.nan),
        "eom": np.array([s.eom for s in signals]),
        "large_order": np.array([float(s.large_order_detected) for s in signals]),
    }

    for w in windows:
        rolled = pd.Series(close).rolling(w, min_periods=max(1, w // 2))
        all_features[f"price_change_{w}d"] = rolled.apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / max(abs(x.iloc[0]), 1e-10), raw=False
        ).to_numpy()

    obv = np.array([s.on_balance_volume for s in signals])
    obv_5d = np.full(len(signals), np.nan)
    for i in range(len(signals)):
        if i >= 5:
            obv_5d[i] = obv[i] - obv[i - 5]
    all_features["obv_change_5d"] = obv_5d

    result = pd.DataFrame(all_features, index=df.index)
    result = result.interpolate(method="linear").fillna(0.0)
    return result
