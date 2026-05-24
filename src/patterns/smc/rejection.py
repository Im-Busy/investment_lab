"""Rejection Block detector — strong structural rejections with inducement + session context.

ICT Definition (David Woods):
- Rejection Block = price REJECTS from OB/FVG/structural level with inducement + session context
- David Woods: "80% of the time, rejection block + displacement forms real PA confirmation"
- HTF Rejection Block = LTF Algo Candle (multi-timeframe relationship)
- In PD Array Matrix: Top of Premium zone and bottom of Discount zone
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RejectionBlockSignal:
    """A single rejection block signal."""

    bar_index: int
    direction: int  # +1 bullish rejection, -1 bearish rejection
    level: float
    wick_ratio: float
    volume_ratio: float
    session_formed: bool
    inducement_present: bool
    confidence: float  # 0.0-1.0


def detect_rejection_blocks(
    ohlc: pd.DataFrame,
    fvg_result: pd.DataFrame,
    ob_result: pd.DataFrame,
    atr: np.ndarray,
    volume_sma_20: np.ndarray,
    session_active: np.ndarray | None = None,
    wick_body_ratio_min: float = 1.5,
    zone_buffer_atr: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, list[RejectionBlockSignal]]:
    """Detect Rejection Blocks at OB/FVG structural levels.

    Scans for candles that strongly reject from known structural zones (OB, FVG),
    with inducement and session context.

    Args:
        ohlc: OHLCV DataFrame
        fvg_result: Output from smc.fvg()
        ob_result: Output from smc.ob()
        atr: Precomputed ATR(14) array
        volume_sma_20: 20-bar rolling average volume
        session_active: Boolean array for session context
        wick_body_ratio_min: Minimum wick-to-body ratio
        zone_buffer_atr: ATR multiplier for zone proximity

    Returns:
        (bullish_signals, bearish_signals, rejection_signals)
    """
    open_ = ohlc["Open"].to_numpy(dtype=np.float64)
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    volume = ohlc["Volume"].to_numpy(dtype=np.float64)
    n = len(close)

    fvg_top = fvg_result["Top"].to_numpy(dtype=np.float64)
    fvg_bottom = fvg_result["Bottom"].to_numpy(dtype=np.float64)
    fvg_sig = fvg_result["FVG"].to_numpy(dtype=np.float64)
    ob_top = ob_result["Top"].to_numpy(dtype=np.float64)
    ob_bottom = ob_result["Bottom"].to_numpy(dtype=np.float64)
    ob_sig = ob_result["OB"].to_numpy(dtype=np.float64)

    bullish_signals = np.zeros(n, dtype=np.int8)
    bearish_signals = np.zeros(n, dtype=np.int8)
    rejections: list[RejectionBlockSignal] = []

    for i in range(3, n):
        body = abs(close[i] - open_[i])
        upper_wick = high[i] - max(close[i], open_[i])
        lower_wick = min(close[i], open_[i]) - low[i]

        if body <= 0:
            continue

        buffer = zone_buffer_atr * atr[i]

        # Bearish rejection: long upper wick near zone TOP
        if upper_wick / body >= wick_body_ratio_min:
            for j in range(max(0, i - 10), i):
                found = False
                if not np.isnan(fvg_sig[j]) and fvg_sig[j] == -1:
                    zone_top = fvg_top[j]
                    if abs(high[i] - zone_top) <= buffer:
                        vol_ratio = volume[i] / (volume_sma_20[i] + 1e-10)
                        on_session = session_active is not None and bool(session_active[i])
                        has_induce = _has_inducement(high, low, i)
                        conf = _rejection_confidence(
                            upper_wick / body, vol_ratio, on_session, has_induce
                        )
                        rejections.append(
                            RejectionBlockSignal(
                                bar_index=i,
                                direction=-1,
                                level=zone_top,
                                wick_ratio=upper_wick / body,
                                volume_ratio=vol_ratio,
                                session_formed=on_session,
                                inducement_present=has_induce,
                                confidence=conf,
                            )
                        )
                        bearish_signals[i] = -1
                        found = True
                if not found and not np.isnan(ob_sig[j]) and ob_sig[j] == -1:
                    zone_top = ob_top[j]
                    if abs(high[i] - zone_top) <= buffer:
                        vol_ratio = volume[i] / (volume_sma_20[i] + 1e-10)
                        on_session = session_active is not None and bool(session_active[i])
                        has_induce = _has_inducement(high, low, i)
                        conf = _rejection_confidence(
                            upper_wick / body, vol_ratio, on_session, has_induce
                        )
                        rejections.append(
                            RejectionBlockSignal(
                                bar_index=i,
                                direction=-1,
                                level=zone_top,
                                wick_ratio=upper_wick / body,
                                volume_ratio=vol_ratio,
                                session_formed=on_session,
                                inducement_present=has_induce,
                                confidence=conf,
                            )
                        )
                        bearish_signals[i] = -1
                        found = True
                if found:
                    break

        # Bullish rejection: long lower wick near zone BOTTOM
        if lower_wick / body >= wick_body_ratio_min:
            for j in range(max(0, i - 10), i):
                found = False
                if not np.isnan(fvg_sig[j]) and fvg_sig[j] == 1:
                    zone_bottom = fvg_bottom[j]
                    if abs(low[i] - zone_bottom) <= buffer:
                        vol_ratio = volume[i] / (volume_sma_20[i] + 1e-10)
                        on_session = session_active is not None and bool(session_active[i])
                        has_induce = _has_inducement(high, low, i)
                        conf = _rejection_confidence(
                            lower_wick / body, vol_ratio, on_session, has_induce
                        )
                        rejections.append(
                            RejectionBlockSignal(
                                bar_index=i,
                                direction=1,
                                level=zone_bottom,
                                wick_ratio=lower_wick / body,
                                volume_ratio=vol_ratio,
                                session_formed=on_session,
                                inducement_present=has_induce,
                                confidence=conf,
                            )
                        )
                        bullish_signals[i] = 1
                        found = True
                if not found and not np.isnan(ob_sig[j]) and ob_sig[j] == 1:
                    zone_bottom = ob_bottom[j]
                    if abs(low[i] - zone_bottom) <= buffer:
                        vol_ratio = volume[i] / (volume_sma_20[i] + 1e-10)
                        on_session = session_active is not None and bool(session_active[i])
                        has_induce = _has_inducement(high, low, i)
                        conf = _rejection_confidence(
                            lower_wick / body, vol_ratio, on_session, has_induce
                        )
                        rejections.append(
                            RejectionBlockSignal(
                                bar_index=i,
                                direction=1,
                                level=zone_bottom,
                                wick_ratio=lower_wick / body,
                                volume_ratio=vol_ratio,
                                session_formed=on_session,
                                inducement_present=has_induce,
                                confidence=conf,
                            )
                        )
                        bullish_signals[i] = 1
                        found = True
                if found:
                    break

    return bullish_signals, bearish_signals, rejections


def _has_inducement(high: np.ndarray, low: np.ndarray, idx: int, lookback: int = 5) -> bool:
    """Check for small counter-move (inducement) before the rejection."""
    if idx < lookback + 2:
        return False
    recent_range = float(np.max(high[idx - lookback : idx])) - float(
        np.min(low[idx - lookback : idx])
    )
    if recent_range <= 0:
        return False
    counter_move = (
        abs(high[idx] - high[idx - 2])
        if high[idx] > high[idx - 2]
        else abs(low[idx] - low[idx - 2])
    )
    return counter_move > 0.2 * recent_range


def _rejection_confidence(
    wick_body_ratio: float,
    volume_ratio: float,
    session_formed: bool,
    inducement_present: bool,
) -> float:
    """Score rejection confidence 0.0-0.95."""
    score = min(wick_body_ratio / 5.0, 1.0) * 0.35
    score += np.clip(volume_ratio / 3.0, 0.0, 1.0) * 0.25
    if session_formed:
        score += 0.20
    if inducement_present:
        score += 0.20
    return float(np.clip(score, 0.0, 0.95))
