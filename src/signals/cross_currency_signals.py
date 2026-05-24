"""Cross-currency implied signals via rolling correlation propagation.

Paper: "Heuristic Based Trading System" (Öztürk 2015) - B18 in MASTER_COMPARISON_REPORT
When two currency pairs exhibit positive rolling correlation, a Buy signal on pair A
implies a Buy signal on pair B. Signal strength scales with correlation magnitude.

Common pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF,
              EURGBP, EURJPY, GBPJPY, EURCHF, EURAUD, GBPAUD, GBPCAD
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_cross_currency_signals(
    pairs_data: dict[str, pd.DataFrame],
    primary_signals: dict[str, np.ndarray],
    correlation_window: int = 30,
    min_correlation: float = 0.60,
    correlation_weight: float = 0.5,
) -> dict[str, np.ndarray]:
    """Propagate trading signals across correlated currency pairs.

    For each pair, computes rolling correlation against all other pairs.
    When primary_signal fires on a correlated pair, an implied signal is generated
    on this pair with strength = correlation_sign * primary_signal * correlation_weight.

    Args:
        pairs_data: Dict mapping pair name → DataFrame with 'Close' column.
        primary_signals: Dict mapping pair name → int8 signal array (+1 buy, -1 sell, 0 none).
        correlation_window: Rolling window in bars for correlation (default 30).
        min_correlation: Minimum |correlation| to propagate signals (default 0.60).
        correlation_weight: Scaling factor for implied signal strength (default 0.5).

    Returns:
        Dict mapping pair name → implied signal array (float, in [-1, 1]).
    """
    pair_names = list(pairs_data.keys())
    n = min(len(df) for df in pairs_data.values())

    returns: dict[str, np.ndarray] = {}
    for name in pair_names:
        close = pairs_data[name]["Close"].to_numpy(dtype=np.float64)[:n]
        r = np.diff(close, prepend=close[0]) / np.where(close > 0, close, 1e-10)
        returns[name] = r

    implied: dict[str, np.ndarray] = {name: np.zeros(n, dtype=np.float64) for name in pair_names}

    for i in range(correlation_window, n):
        for pair_a in pair_names:
            sig = float(primary_signals[pair_a][i]) if i < len(primary_signals[pair_a]) else 0.0
            if sig == 0.0:
                continue
            for pair_b in pair_names:
                if pair_a == pair_b:
                    continue
                ra = returns[pair_a][i - correlation_window : i]
                rb = returns[pair_b][i - correlation_window : i]
                ra_std = float(np.std(ra))
                rb_std = float(np.std(rb))
                if ra_std < 1e-10 or rb_std < 1e-10:
                    continue
                corr = float(np.corrcoef(ra, rb)[0, 1])
                if abs(corr) < min_correlation:
                    continue
                implied[pair_b][i] += sig * corr * correlation_weight

    for name in pair_names:
        implied[name] = np.clip(implied[name], -1.0, 1.0)

    return implied


def implied_signal_scalar(
    pair_name: str,
    implied_signals: dict[str, np.ndarray],
    idx: int,
) -> float:
    """Get the implied cross-currency multiplier for a specific bar.

    Returns a scalar in [0.85, 1.15] that can multiply a base signal.
    """
    if pair_name not in implied_signals:
        return 1.0
    arr = implied_signals[pair_name]
    if idx >= len(arr):
        return 1.0
    val = float(arr[idx])
    return 1.0 + val * 0.15
