"""Breaker Block detector — failed Order Block that becomes opposite-direction support/resistance.

ICT Definition (14-ICT #4, David Woods):
- Bullish Breaker: Bearish OB that got mitigated, then price breaks through OB bottom → becomes bullish support
- Bearish Breaker: Bullish OB that got mitigated, then price breaks through OB top → becomes bearish resistance
- Strong Breaker: momentum shift aggressive + breaker has inducement + formed on session H/L
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BreakerBlock:
    """A single breaker block instance."""

    bar_index: int
    direction: int  # +1 = bullish breaker, -1 = bearish breaker
    top: float
    bottom: float
    source_ob_index: int
    mitigation_index: int
    breach_index: int
    strength: float  # 0.0-1.0
    active: bool = True
    inducement_present: bool = False
    session_formed: bool = False
    sweep_confirmed: bool = False


def detect_breaker_blocks(
    ohlc: pd.DataFrame,
    swing_highs_lows: pd.DataFrame,
    atr: np.ndarray,
    volume_sma_20: np.ndarray,
    session_active: np.ndarray | None = None,
    sweep_signals: np.ndarray | None = None,
    buffer_atr_mult: float = 0.5,
    min_volume_mult: float = 1.2,
    require_inducement: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[BreakerBlock]]:
    """Detect Breaker Blocks from OB breaches.

    Uses the smartmoneyconcepts library OB detection, then tracks breaches
    to identify breaker blocks.

    Args:
        ohlc: OHLCV DataFrame (columns: open, high, low, close, volume)
        swing_highs_lows: Output from smc.swing_highs_lows()
        atr: Precomputed ATR(14) numpy array
        volume_sma_20: 20-bar rolling average volume
        session_active: Boolean array for active trading sessions
        sweep_signals: Liquidity sweep signal array (+1=bullish, -1=bearish).
            When provided, checks preceding 5 bars for sweep confirmation.
        buffer_atr_mult: ATR multiplier for zone boundary buffer
        min_volume_mult: Minimum volume vs SMA to confirm
        require_inducement: Whether inducement is required for breaker validity

    Returns:
        (bullish_signals, bearish_signals, breaker_list)
    """
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    open_ = ohlc["Open"].to_numpy(dtype=np.float64)
    volume = ohlc["Volume"].to_numpy(dtype=np.float64)
    n = len(close)

    from smartmoneyconcepts import smc as smc_lib

    ob_result = smc_lib.ob(ohlc, swing_highs_lows, close_mitigation=False)
    ob_signal = ob_result["OB"].to_numpy(dtype=np.float64)
    ob_top = ob_result["Top"].to_numpy(dtype=np.float64)
    ob_bottom = ob_result["Bottom"].to_numpy(dtype=np.float64)
    ob_percentage = ob_result["Percentage"].to_numpy(dtype=np.float64)
    ob_mitigated = ob_result["MitigatedIndex"].to_numpy(dtype=np.float64)

    bullish_signals = np.zeros(n, dtype=np.int8)
    bearish_signals = np.zeros(n, dtype=np.int8)
    breakers: list[BreakerBlock] = []

    tracked_obs: list[dict] = []

    for i in range(n):
        if not np.isnan(ob_signal[i]) and ob_signal[i] != 0:
            tracked_obs.append(
                {
                    "index": i,
                    "direction": int(ob_signal[i]),
                    "top": ob_top[i],
                    "bottom": ob_bottom[i],
                    "volume_strength": ob_percentage[i] if not np.isnan(ob_percentage[i]) else 50.0,
                    "mitigated": False,
                    "mitigation_index": -1,
                    "breached": False,
                    "breach_index": -1,
                }
            )

        for ob in tracked_obs:
            if not ob["mitigated"]:
                mit_idx = (
                    int(ob_mitigated[ob["index"]])
                    if not np.isnan(ob_mitigated[ob["index"]])
                    else -1
                )
                if mit_idx > 0 and i >= mit_idx:
                    ob["mitigated"] = True
                    ob["mitigation_index"] = mit_idx

        for ob in tracked_obs:
            if ob["mitigated"] and not ob["breached"]:
                ob_direction = ob["direction"]
                ob_top_val = ob["top"]
                ob_bottom_val = ob["bottom"]

                if ob_direction == -1:
                    if close[i] < ob_bottom_val - buffer_atr_mult * atr[i]:
                        ob["breached"] = True
                        ob["breach_index"] = i
                        vol_ok = volume[i] >= min_volume_mult * volume_sma_20[i]
                        has_induce = _check_inducement(close, low, i, lookback=8)
                        on_session = session_active is not None and bool(session_active[i])
                        sweep_ok = _check_sweep_before(sweep_signals, i, lookback=5)
                        strength = _compute_breaker_strength(
                            ob["volume_strength"], vol_ok, has_induce, on_session
                        )
                        if not sweep_ok:
                            strength *= 0.6
                        if not require_inducement or has_induce:
                            breakers.append(
                                BreakerBlock(
                                    bar_index=i,
                                    direction=1,
                                    top=ob_top_val,
                                    bottom=ob_bottom_val,
                                    source_ob_index=ob["index"],
                                    mitigation_index=ob["mitigation_index"],
                                    breach_index=i,
                                    strength=float(np.clip(strength, 0.0, 1.0)),
                                    inducement_present=has_induce,
                                    session_formed=on_session,
                                    sweep_confirmed=sweep_ok,
                                )
                            )

                elif ob_direction == 1:
                    if close[i] > ob_top_val + buffer_atr_mult * atr[i]:
                        ob["breached"] = True
                        ob["breach_index"] = i
                        vol_ok = volume[i] >= min_volume_mult * volume_sma_20[i]
                        has_induce = _check_inducement(close, high, i, lookback=8)
                        on_session = session_active is not None and bool(session_active[i])
                        sweep_ok = _check_sweep_before(sweep_signals, i, lookback=5)
                        strength = _compute_breaker_strength(
                            ob["volume_strength"], vol_ok, has_induce, on_session
                        )
                        if not sweep_ok:
                            strength *= 0.6
                        if not require_inducement or has_induce:
                            breakers.append(
                                BreakerBlock(
                                    bar_index=i,
                                    direction=-1,
                                    top=ob_top_val,
                                    bottom=ob_bottom_val,
                                    source_ob_index=ob["index"],
                                    mitigation_index=ob["mitigation_index"],
                                    breach_index=i,
                                    strength=float(np.clip(strength, 0.0, 1.0)),
                                    inducement_present=has_induce,
                                    session_formed=on_session,
                                    sweep_confirmed=sweep_ok,
                                )
                            )

    # Detect retests of active breaker zones for entry signals
    active_breakers = [b for b in breakers if b.active]
    for i in range(n):
        for breaker in active_breakers:
            if breaker.bar_index >= i:
                continue
            if breaker.direction == 1:
                zone_bottom = breaker.bottom
                zone_top = breaker.top
                if low[i] <= zone_top + buffer_atr_mult * atr[i] and low[i] >= zone_bottom:
                    if close[i] > open_[i]:
                        bullish_signals[i] = 1
            elif breaker.direction == -1:
                zone_top = breaker.top
                zone_bottom = breaker.bottom
                if high[i] >= zone_bottom - buffer_atr_mult * atr[i] and high[i] <= zone_top:
                    if close[i] < open_[i]:
                        bearish_signals[i] = -1

    return bullish_signals, bearish_signals, breakers


def _check_inducement(
    prices: np.ndarray, extremes: np.ndarray, idx: int, lookback: int = 8
) -> bool:
    """Check for a mini liquidity grab (inducement) before the breach."""
    if idx < lookback + 3:
        return False
    window_extremes = extremes[idx - lookback : idx]
    prev_high = float(np.max(window_extremes))
    prev_low = float(np.min(window_extremes))
    range_size = prev_high - prev_low
    if range_size <= 0:
        return False
    recent_change = abs(prices[idx] - prices[idx - 2])
    return recent_change > 0.3 * range_size


def _check_sweep_before(
    sweep_signals: np.ndarray | None,
    idx: int,
    lookback: int = 5,
) -> bool:
    """Check if any liquidity sweep occurred in the preceding `lookback` bars."""
    if sweep_signals is None:
        return False
    start = max(0, idx - lookback)
    return bool(np.any(sweep_signals[start:idx] != 0))


def _compute_breaker_strength(
    ob_volume_strength: float,
    volume_confirmed: bool,
    inducement_present: bool,
    session_formed: bool,
) -> float:
    """Compute breaker strength score 0.0-1.0."""
    score = ob_volume_strength / 100.0 * 0.35
    if volume_confirmed:
        score += 0.25
    if inducement_present:
        score += 0.25
    if session_formed:
        score += 0.15
    return float(np.clip(score, 0.0, 1.0))
