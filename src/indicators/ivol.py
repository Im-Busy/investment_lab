"""P24-13: iV volume indicator — short-period vol / long-period vol ratio.

iV >> 1 → growing activity supports price direction.
iV << 1 → contracting activity weakens signal conviction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ivol(
    data: pd.DataFrame,
    short_period: int = 5,
    long_period: int = 20,
) -> np.ndarray:
    """Compute iV = rolling short-term vol / rolling long-term vol.

    Uses log returns for vol computation. Returns ratio > 0.
    """
    log_ret = np.log(data["Close"] / data["Close"].shift(1))
    short_vol = log_ret.rolling(short_period, min_periods=max(3, short_period // 2)).std()
    long_vol = log_ret.rolling(long_period, min_periods=max(5, long_period // 2)).std()

    long_vol_safe = long_vol.where(long_vol > 1e-10, 1e-10)
    ivol = short_vol / long_vol_safe
    return ivol.fillna(1.0).values


def ivol_signal(ivol: np.ndarray, idx: int, threshold: float = 0.8) -> float:
    """Generate iV signal: multiplier that amplifies/attenuates base signal.

    Returns:
        1.15 if iV > 1.5 (expanding activity — confirm)
        0.85 if iV < threshold (contracting — attenuate)
        1.0 otherwise
    """
    if idx < 0 or idx >= len(ivol):
        return 1.0
    val = float(ivol[idx])
    if val > 1.5:
        return 1.15
    if val < threshold:
        return 0.85
    return 1.0
