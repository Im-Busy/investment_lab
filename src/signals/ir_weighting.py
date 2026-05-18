"""
R1: IR-Weighted Pattern Synthesis.

Replaces hardcoded PATTERN_RELIABILITY with rolling Information Ratio weights
computed from IS performance per pattern.

Source: 华泰多因子系列1 §2.3

Formula:
    IR_p = mean(return_p) / std(return_p)  over rolling window
    weight_p = max(IR_p, 0) / Σ(max(IR_k, 0))   (only positive IR patterns contribute)

Architecture: consumed by RulesFirstStrategy.sigmoid() scoring and combined_strategy.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Pattern-to-price allocation ──
# Each pattern's signal direction maps to: long_side / short_side / neutral
_DIRECTION_SYMBOLS = {
    "long": 1,
    "short": -1,
    "both": 0,  # handled separately: long signal=+1, short=-1
    "neutral": 0,
}


class IRWeighting:
    """Rolling Information Ratio weighting for dynamic pattern reliability.

    Parameters:
        window: Rolling window size in bars for IR computation (default 252 ~ 1 year).
        min_bars: Minimum bars required before producing non-zero weights.
        ir_floor: IR values below this are set to 0 (excluded from weighting).
        mode: 'gate' (default) — zero-out negative-IR patterns entirely.
              'scalar' — map IR to [scalar_min, scalar_max] multiplier on base weights.

    Modes:
        gate:   original behavior. IR < 0 → weight = 0 (pattern excluded).
        scalar: low-IR patterns get reduced but non-zero multipliers (0.3x-0.7x).
                Preserves trade count while filtering noise.
    """

    def __init__(
        self,
        window: int = 252,
        min_bars: int = 50,
        ir_floor: float = 0.0,
        mode: str = "gate",
        scalar_min: float = 0.3,
        scalar_max: float = 0.7,
    ) -> None:
        self._window = window
        self._min_bars = min_bars
        self._ir_floor = ir_floor
        self._mode = mode
        self._scalar_min = scalar_min
        self._scalar_max = scalar_max
        self._returns: Optional[np.ndarray] = None
        self._signals: Optional[Dict[str, np.ndarray]] = None
        self._ir_history: Dict[str, np.ndarray] = {}
        self._weight_history: Dict[str, np.ndarray] = {}
        self._scalar_history: Dict[str, np.ndarray] = {}

    def fit(
        self,
        close_prices: np.ndarray,
        signals: Dict[str, np.ndarray],
        pattern_directions: Optional[Dict[str, str]] = None,
    ) -> "IRWeighting":
        """Precompute IR time series for all patterns (look-ahead safe).

        Computes 1-bar forward returns from close prices, then calculates
        rolling Information Ratio per pattern. At bar ``idx``, only returns
        realized through bar ``idx`` are used (no look-ahead).

        Args:
            close_prices: (N,) array of close prices.
            signals: dict mapping pattern_name -> (N,) int8 array of -1/0/1 signals.
            pattern_directions: optional dict of pattern_name -> 'long'/'short'/'both'.
                If None, pattern signal sign is used directly.

        Returns:
            self
        """
        close = np.asarray(close_prices, dtype=np.float64)
        n = len(close)
        forward_ret = np.zeros(n, dtype=np.float64)
        forward_ret[:-1] = close[1:] / close[:-1] - 1.0

        self._signals = signals
        self._pattern_directions = pattern_directions or {}

        for name, sig_arr in signals.items():
            sig_arr = np.asarray(sig_arr, dtype=np.int8)
            direction = self._pattern_directions.get(name, "both")

            if direction == "long":
                aligned_ret = np.where(sig_arr == 1, forward_ret, 0.0)
            elif direction == "short":
                aligned_ret = np.where(sig_arr == -1, -forward_ret, 0.0)
            else:
                aligned_ret = sig_arr * forward_ret

            # Shift by 1: at bar idx, ret[idx] = forward return from idx->idx+1
            # is NOT known yet.  Only aligned_ret[0..idx-1] is known.
            ir_series = np.full(n, 0.0, dtype=np.float64)
            for idx in range(self._min_bars, n):
                start = max(0, idx - self._window)
                window = aligned_ret[start:idx]  # up to idx-1, inclusive
                nonzero = window[window != 0.0]
                if len(nonzero) >= 3:
                    mu = np.mean(nonzero)
                    sigma = np.std(nonzero, ddof=1)
                    if sigma > 1e-10:
                        ir_series[idx] = mu / sigma

            self._ir_history[name] = ir_series

        self._compute_weights(self._ir_history)
        self._returns = forward_ret
        return self

    def _compute_weights(self, ir_history: Dict[str, np.ndarray]) -> None:
        """Normalize positive IR values into weights [0, 1] at each bar.

        In 'gate' mode: negative IR patterns get 0 weight (excluded).
        In 'scalar' mode: also compute _scalar_history mapping IR → [scalar_min, scalar_max].
        """
        if not ir_history:
            return

        patterns = list(ir_history.keys())
        n_bars = len(next(iter(ir_history.values())))
        weight_matrix = np.zeros((n_bars, len(patterns)), dtype=np.float64)
        scalar_matrix = np.zeros((n_bars, len(patterns)), dtype=np.float64)

        for j, name in enumerate(patterns):
            ir = ir_history[name]
            ir_clipped = np.maximum(ir, self._ir_floor)
            weight_matrix[:, j] = ir_clipped

            # Scalar mode: sigmoid-like mapping IR → [scalar_min, scalar_max]
            if self._mode == "scalar":
                sigmoid = 1.0 / (1.0 + np.exp(-ir * 2.0))
                scalar_matrix[:, j] = self._scalar_min + sigmoid * (
                    self._scalar_max - self._scalar_min
                )

        # Row-wise normalization: weight_p = max(IR_p,0) / sum(max(IR_k,0))
        row_sums = weight_matrix.sum(axis=1)
        mask = row_sums > 1e-10
        for j in range(len(patterns)):
            weight_matrix[mask, j] /= row_sums[mask]

        for j, name in enumerate(patterns):
            self._weight_history[name] = weight_matrix[:, j].copy()
            if self._mode == "scalar":
                self._scalar_history[name] = scalar_matrix[:, j].copy()

    def get_weights(self, idx: int) -> Dict[str, float]:
        """Get normalized IR weights at a specific bar index.

        Args:
            idx: Bar index (0-based).

        Returns:
            Dict of pattern_name -> weight (0.0 to 1.0, sums to 1.0).
        """
        if not self._weight_history:
            if self._signals:
                return {k: 1.0 / len(self._signals) for k in self._signals}
            return {}
        return {name: float(arr[idx]) for name, arr in self._weight_history.items()}

    def get_scalars(self, idx: int) -> Dict[str, float]:
        """Get scalar multipliers for each pattern (scalar mode only).

        Returns values in [scalar_min, scalar_max] based on IR sigmoid mapping.
        In gate mode, returns empty dict.
        """
        if not self._scalar_history:
            return {}
        return {name: float(arr[idx]) for name, arr in self._scalar_history.items()}

    @property
    def mode(self) -> str:
        return self._mode

    def get_weights_array(self, idx: int, pattern_names: list[str]) -> np.ndarray:
        """Get weights as numpy array in pattern_names order."""
        return np.array([self.get_weights(idx).get(n, 0.0) for n in pattern_names], dtype=float)

    def get_ir(self, idx: int) -> Dict[str, float]:
        """Get raw IR values at a specific bar index."""
        if not self._ir_history:
            return {}
        return {name: float(arr[idx]) for name, arr in self._ir_history.items()}

    @property
    def weight_series(self) -> pd.DataFrame:
        """Full weight history as DataFrame (rows=bars, cols=patterns)."""
        if not self._weight_history:
            return pd.DataFrame()
        idx = pd.RangeIndex(len(next(iter(self._weight_history.values()))))
        return pd.DataFrame(self._weight_history, index=idx)

    @property
    def ir_series(self) -> pd.DataFrame:
        """Full IR history as DataFrame (rows=bars, cols=patterns)."""
        if not self._ir_history:
            return pd.DataFrame()
        idx = pd.RangeIndex(len(next(iter(self._ir_history.values()))))
        return pd.DataFrame(self._ir_history, index=idx)


def ir_from_pattern_signals(
    close_prices: np.ndarray,
    signals: Dict[str, np.ndarray],
    pattern_directions: Optional[Dict[str, str]] = None,
    window: int = 252,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Convenience: compute end-of-period IR weights for all patterns.

    Args:
        close_prices: Close price array.
        signals: Pattern signal arrays.
        pattern_directions: Pattern direction annotations.
        window: Rolling IR window.

    Returns:
        Tuple of (final_weights_dict, full_weight_dataframe).
    """
    irw = IRWeighting(window=window)
    irw.fit(close_prices, signals, pattern_directions)
    idx = len(close_prices) - 1
    final_weights = irw.get_weights(idx)
    df = irw.weight_series
    return final_weights, df
