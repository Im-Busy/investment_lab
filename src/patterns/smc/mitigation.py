"""Mitigation Block detector — mitigated strong swing levels that become reverse polarity zones.

ICT Definition (toaz.info, David Woods):
- Mitigation Block = strong swing point that price "mitigates" (visits), then price reverses
  from it, developing weakness at that level
- Bullish Mitigation Block: Strong Low gets mitigated → price bounces → becomes support
- Bearish Mitigation Block: Strong High gets mitigated → price rejects → becomes resistance
- In PD Array Matrix hierarchy: Bullish/Bearish Mitigation between Breaker Blocks and Equilibrium
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MitigationBlock:
    """A single mitigation block instance."""

    bar_index: int
    direction: int  # +1 bullish, -1 bearish
    level: float
    source_swing_index: int
    mitigation_bar_index: int
    reversal_bar_index: int
    strength: float  # 0.0-1.0
    active: bool = True


def detect_mitigation_blocks(
    ohlc: pd.DataFrame,
    swing_highs_lows: pd.DataFrame,
    atr: np.ndarray,
    session_active: np.ndarray | None = None,
    buffer_atr_mult: float = 0.3,
    min_reversal_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[MitigationBlock]]:
    """Detect Mitigation Blocks from mitigated strong swing levels.

    Args:
        ohlc: OHLCV DataFrame
        swing_highs_lows: Output from smc.swing_highs_lows()
        atr: Precomputed ATR(14) array
        session_active: Boolean array for session context
        buffer_atr_mult: Zone width around mitigated level
        min_reversal_ratio: Minimum reversal fraction from level

    Returns:
        (bullish_signals, bearish_signals, mitigation_blocks)
    """
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    open_ = ohlc["Open"].to_numpy(dtype=np.float64)
    n = len(close)

    shl_signal = swing_highs_lows["HighLow"].to_numpy(dtype=np.float64)
    shl_level = swing_highs_lows["Level"].to_numpy(dtype=np.float64)

    bullish_signals = np.zeros(n, dtype=np.int8)
    bearish_signals = np.zeros(n, dtype=np.int8)
    blocks: list[MitigationBlock] = []

    swing_points: list[dict] = []
    for i in range(n):
        if not np.isnan(shl_signal[i]) and shl_signal[i] != 0:
            swing_points.append(
                {
                    "index": i,
                    "type": int(shl_signal[i]),
                    "level": shl_level[i],
                    "mitigated": False,
                    "mitigation_idx": -1,
                }
            )

    # Scan for mitigation: price touches the level
    for sp in swing_points:
        for j in range(sp["index"] + 1, n):
            if sp["mitigated"]:
                break
            buffer = buffer_atr_mult * atr[j]
            if sp["type"] == 1:
                if high[j] >= sp["level"] - buffer:
                    sp["mitigated"] = True
                    sp["mitigation_idx"] = j
            elif sp["type"] == -1:
                if low[j] <= sp["level"] + buffer:
                    sp["mitigated"] = True
                    sp["mitigation_idx"] = j

    # For each mitigated swing, check for reversal
    for sp in swing_points:
        if not sp["mitigated"]:
            continue
        mit_idx = sp["mitigation_idx"]

        if sp["type"] == 1:  # Mitigated Swing High → Bearish Mitigation Block
            for j in range(mit_idx + 2, min(mit_idx + 12, n)):
                move_from_level = sp["level"] - close[j]
                low_slice = low[mit_idx:j]
                prior_leg = sp["level"] - float(np.min(low_slice)) if len(low_slice) > 0 else atr[j]
                if prior_leg <= 0:
                    prior_leg = atr[j]
                reversal_ratio = move_from_level / prior_leg
                if reversal_ratio > min_reversal_ratio and close[j] < close[j - 1]:
                    on_session = session_active is not None and bool(session_active[j])
                    strength = 0.5 + 0.25 * min(reversal_ratio, 1.0) + (0.25 if on_session else 0.0)
                    blocks.append(
                        MitigationBlock(
                            bar_index=j,
                            direction=-1,
                            level=sp["level"],
                            source_swing_index=sp["index"],
                            mitigation_bar_index=mit_idx,
                            reversal_bar_index=j,
                            strength=float(np.clip(strength, 0.0, 1.0)),
                        )
                    )
                    break

        elif sp["type"] == -1:  # Mitigated Swing Low → Bullish Mitigation Block
            for j in range(mit_idx + 2, min(mit_idx + 12, n)):
                move_from_level = close[j] - sp["level"]
                high_slice = high[mit_idx:j]
                prior_leg = (
                    float(np.max(high_slice)) - sp["level"] if len(high_slice) > 0 else atr[j]
                )
                if prior_leg <= 0:
                    prior_leg = atr[j]
                reversal_ratio = move_from_level / prior_leg
                if reversal_ratio > min_reversal_ratio and close[j] > close[j - 1]:
                    on_session = session_active is not None and bool(session_active[j])
                    strength = 0.5 + 0.25 * min(reversal_ratio, 1.0) + (0.25 if on_session else 0.0)
                    blocks.append(
                        MitigationBlock(
                            bar_index=j,
                            direction=1,
                            level=sp["level"],
                            source_swing_index=sp["index"],
                            mitigation_bar_index=mit_idx,
                            reversal_bar_index=j,
                            strength=float(np.clip(strength, 0.0, 1.0)),
                        )
                    )
                    break

    # Detect retests for entry signals
    for i in range(n):
        for block in blocks:
            if not block.active or block.bar_index > i:
                continue
            buffer = buffer_atr_mult * atr[i]
            if block.direction == 1:
                if low[i] <= block.level + buffer and close[i] > open_[i]:
                    bullish_signals[i] = 1
                    block.active = False
            elif block.direction == -1:
                if high[i] >= block.level - buffer and close[i] < open_[i]:
                    bearish_signals[i] = -1
                    block.active = False

    return bullish_signals, bearish_signals, blocks
