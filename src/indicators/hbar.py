"""P24-12: HBar indicator — (Close−Open)/(High−Low) normalized ratio.

HBar ∈ [−1, 1]. > 0.7 → bullish strength, < −0.7 → bearish selling.
Normalized by rolling average spread to handle varying volatility regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_hbar(
    o: np.ndarray,
    h: np.ndarray,
    lo: np.ndarray,
    c: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """Compute HBar = (C − O) / (H − L) smoothed by rolling average spread."""
    spread = h - lo
    avg_spread = pd.Series(spread).rolling(window, min_periods=1).mean().values
    safe_denom = np.where(spread > 0, spread, avg_spread)
    safe_denom = np.where(safe_denom < 1e-10, 1e-10, safe_denom)
    raw = (c - o) / safe_denom
    return np.clip(raw, -1.5, 1.5)


def hbar_signal(hbar: np.ndarray, idx: int, lookback: int = 3) -> float:
    """Generate HBar trading signal: mean over recent bars.

    Returns 1.0 (bullish) if HBar > 0.3, -1.0 (bearish) if < -0.3, else 0.
    """
    if idx < lookback:
        return 0.0
    mean_hbar = float(np.mean(hbar[idx - lookback : idx + 1]))
    if mean_hbar > 0.3:
        return 1.0
    if mean_hbar < -0.3:
        return -1.0
    return 0.0
