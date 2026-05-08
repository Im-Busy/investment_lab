"""
Triple Barrier Labeling for financial ML.

Implements the triple-barrier method from López de Prado (2018) "Advances in
Financial Machine Learning", Chapter 3. Labels each observation based on which
barrier is touched first:

    +1: Upper barrier (take-profit) hit before lower barrier and time limit
    -1: Lower barrier (stop-loss) hit before upper barrier and time limit
     0: Time limit (vertical barrier) expires before either price barrier

This fixes the critical disconnect between ML training labels and actual trade
mechanics: current labels use naive "profitable within N bars?", but real trades
exit when TP or SL levels are hit within a time limit.

The implementation supports two modes:
    1. Fixed-horizon barriers: TP/SL as absolute prices with a vertical time barrier
    2. Dynamic barriers: TP/SL derived from ATR multiples, typical of pattern detectors

Usage:
    from src.ml.triple_barrier import TripleBarrierLabeler

    labeler = TripleBarrierLabeler()
    labels = labeler.fit(
        close=price_series,
        high=high_series,
        low=low_series,
        take_profit=125.0,       # absolute or None for dynamic
        stop_loss=95.0,          # absolute or None for dynamic
        time_limit=20,           # bars
        atr_mult_tp=1.5,         # used if take_profit is None
        atr_mult_sl=1.0,         # used if stop_loss is None
        atr_series=atr_values,   # required for dynamic barriers
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import NamedTuple


class BarrierLabel(NamedTuple):
    """Result of a single triple-barrier label computation.

    Attributes:
        label: +1 (TP hit first), -1 (SL hit first), 0 (time limit expired).
        return_pct: Percentage return at barrier hit or time limit.
        barrier: Which barrier was hit: 'tp', 'sl', 'time'.
        bars_to_exit: Number of bars until exit.
    """

    label: int
    return_pct: float
    barrier: str
    bars_to_exit: int


class TripleBarrierLabeler:
    """Generate labels using the triple-barrier method.

    For each time index t, the label is determined by the first barrier crossed
    in the forward window [t+1, t+time_limit]:

    - Upper barrier (take-profit): If high reaches take_profit before stop_loss
      and within time_limit, label = +1.

    - Lower barrier (stop-loss): If low reaches stop_loss before take_profit
      and within time_limit, label = -1.

    - Time barrier: If neither price barrier is hit within time_limit, label = 0
      and the return is the return at time_limit.

    Attributes:
        use_dynamic_barriers: Whether TP/SL are derived from ATR.
        atr_mult_tp: Default ATR multiplier for take-profit.
        atr_mult_sl: Default ATR multiplier for stop-loss.
    """

    def __init__(
        self,
        atr_mult_tp: float = 1.5,
        atr_mult_sl: float = 1.0,
    ) -> None:
        self.atr_mult_tp = atr_mult_tp
        self.atr_mult_sl = atr_mult_sl

    def fit(
        self,
        close: pd.Series | np.ndarray,
        high: pd.Series | np.ndarray | None = None,
        low: pd.Series | np.ndarray | None = None,
        take_profit: float | pd.Series | np.ndarray | None = None,
        stop_loss: float | pd.Series | np.ndarray | None = None,
        time_limit: int = 20,
        atr_series: pd.Series | np.ndarray | None = None,
        entry_prices: pd.Series | np.ndarray | None = None,
        side: str = "long",
    ) -> pd.Series:
        """Generate triple-barrier labels.

        Args:
            close: Close price series.
            high: High price series (optional — uses close if None for dynamic).
            low: Low price series (optional — uses close if None for dynamic).
            take_profit: Absolute TP price, per-bar Series, or None for dynamic.
            stop_loss: Absolute SL price, per-bar Series, or None for dynamic.
            time_limit: Maximum bars to hold (vertical barrier).
            atr_series: ATR values for dynamic TP/SL (required if TP/SL is None).
            entry_prices: Entry prices for dynamic TP/SL (defaults to close).

        Returns:
            Series with index matching close, values: +1, -1, or 0.
        """
        close_arr = np.asarray(close, dtype=np.float64)
        n = len(close_arr)

        if high is None:
            high_arr = close_arr.copy()
        else:
            high_arr = np.asarray(high, dtype=np.float64)

        if low is None:
            low_arr = close_arr.copy()
        else:
            low_arr = np.asarray(low, dtype=np.float64)

        if entry_prices is None:
            entry_arr = close_arr.copy()
        else:
            entry_arr = np.asarray(entry_prices, dtype=np.float64)

        if atr_series is not None:
            atr_arr = np.asarray(atr_series, dtype=np.float64)
        else:
            atr_arr = None

        tp_arr = self._resolve_barrier(
            take_profit, entry_arr, atr_arr, self.atr_mult_tp, n, side=side, is_tp=True
        )
        sl_arr = self._resolve_barrier(
            stop_loss, entry_arr, atr_arr, self.atr_mult_sl, n, side=side, is_tp=False
        )

        labels = np.full(n, 0, dtype=np.float64)
        returns = np.full(n, 0.0, dtype=np.float64)
        barriers = np.full(n, "time", dtype=object)
        bars_to_exit = np.full(n, time_limit, dtype=np.float64)

        for i in range(n):
            entry = close_arr[i]
            tp = tp_arr[i]
            sl = sl_arr[i]

            if tp <= entry or sl >= entry:
                continue

            end = min(i + time_limit + 1, n)

            for j in range(i + 1, end):
                h = high_arr[j]
                l = low_arr[j]
                c = close_arr[j]

                if h >= tp:
                    labels[i] = 1
                    returns[i] = (tp - entry) / entry
                    barriers[i] = "tp"
                    bars_to_exit[i] = j - i
                    break
                elif l <= sl:
                    labels[i] = -1
                    returns[i] = (sl - entry) / entry
                    barriers[i] = "sl"
                    bars_to_exit[i] = j - i
                    break
            else:
                last_idx = end - 1
                returns[i] = (close_arr[last_idx] - entry) / entry
                bars_to_exit[i] = last_idx - i

        result = pd.Series(labels, index=close.index if isinstance(close, pd.Series) else None)
        result.attrs["return_pct"] = returns
        result.attrs["barrier"] = barriers
        result.attrs["bars_to_exit"] = bars_to_exit
        return result

    def fit_single(
        self,
        entry_price: float,
        close_arr: np.ndarray,
        high_arr: np.ndarray | None = None,
        low_arr: np.ndarray | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        time_limit: int = 20,
    ) -> BarrierLabel:
        """Generate a single triple-barrier label for one trade context.

        Args:
            entry_price: Entry price at time t.
            close_arr: Future close prices starting at t+1.
            high_arr: Future high prices starting at t+1.
            low_arr: Future low prices starting at t+1.
            take_profit: Absolute TP level.
            stop_loss: Absolute SL level.
            time_limit: Maximum bars forward to check.

        Returns:
            BarrierLabel with label, return_pct, barrier, bars_to_exit.
        """
        if high_arr is None:
            high_arr = close_arr.copy()
        if low_arr is None:
            low_arr = close_arr.copy()

        n_forward = min(time_limit, len(close_arr))

        for j in range(n_forward):
            h = high_arr[j]
            l = low_arr[j]

            if take_profit is not None and h >= take_profit:
                return BarrierLabel(
                    label=1,
                    return_pct=float((take_profit - entry_price) / entry_price),
                    barrier="tp",
                    bars_to_exit=j + 1,
                )
            if stop_loss is not None and l <= stop_loss:
                return BarrierLabel(
                    label=-1,
                    return_pct=float((stop_loss - entry_price) / entry_price),
                    barrier="sl",
                    bars_to_exit=j + 1,
                )

        if n_forward > 0:
            last_close = close_arr[n_forward - 1]
            return_value = (last_close - entry_price) / entry_price
        else:
            return_value = 0.0

        return BarrierLabel(
            label=0,
            return_pct=float(return_value),
            barrier="time",
            bars_to_exit=n_forward,
        )

    def _resolve_barrier(
        self,
        barrier_value: float | np.ndarray | pd.Series | None,
        entry_arr: np.ndarray,
        atr_arr: np.ndarray | None,
        default_mult: float,
        n: int,
        side: str = "long",
        is_tp: bool = True,
    ) -> np.ndarray:
        """Resolve barrier to a per-bar array.

        Args:
            barrier_value: Scalar, per-bar array, or None (dynamic from ATR).
            entry_arr: Per-bar entry prices.
            atr_arr: Per-bar ATR values.
            default_mult: Default ATR multiplier for dynamic barriers.
            n: Number of bars.
            side: Trade direction ("long" or "short").
            is_tp: True for take-profit, False for stop-loss.

        Returns:
            Per-bar barrier array.
        """
        if barrier_value is not None:
            if isinstance(barrier_value, pd.Series):
                return np.asarray(barrier_value.values, dtype=np.float64)
            if np.ndim(barrier_value) > 0:
                return np.asarray(barrier_value, dtype=np.float64)
            return np.full(n, float(barrier_value), dtype=np.float64)

        if atr_arr is None:
            raise ValueError("atr_series required when take_profit or stop_loss is None")

        sign = 1.0 if is_tp else -1.0
        if side == "short":
            sign = -sign
        return entry_arr + sign * atr_arr * default_mult

    @staticmethod
    def label_summary(labels: pd.Series) -> dict:
        """Compute summary statistics for triple-barrier labels.

        Args:
            labels: Series with values +1, -1, 0.

        Returns:
            Dict with counts and ratios.
        """
        counts = labels.value_counts()
        total = len(labels)
        return {
            "n_total": total,
            "n_positive": int(counts.get(1, 0)),
            "n_negative": int(counts.get(-1, 0)),
            "n_timeout": int(counts.get(0, 0)),
            "pct_positive": float(counts.get(1, 0) / max(total, 1)),
            "pct_negative": float(counts.get(-1, 0) / max(total, 1)),
            "pct_timeout": float(counts.get(0, 0) / max(total, 1)),
            "event_rate": float((counts.get(1, 0) + counts.get(-1, 0)) / max(total, 1)),
        }

    @staticmethod
    def from_pattern_signal(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        take_profit_series: pd.Series,
        stop_loss_series: pd.Series,
        signal_mask: pd.Series | None = None,
        time_limit: int = 20,
    ) -> pd.Series:
        """Generate labels only where signals exist, using per-signal TP/SL.

        Convenience method for applying triple-barrier labels to pattern
        detector outputs, where each signal carries its own TP and SL levels.

        Args:
            close: Close price series.
            high: High price series.
            low: Low price series.
            take_profit_series: Per-bar TP levels (NaN where no signal).
            stop_loss_series: Per-bar SL levels (NaN where no signal).
            signal_mask: Boolean mask of signal bars (auto-detected if None).
            time_limit: Maximum bars to hold.

        Returns:
            Series of labels aligned with close index, NaN where no signal.
        """
        if signal_mask is None:
            signal_mask = take_profit_series.notna() & stop_loss_series.notna()

        n = len(close)
        labels = pd.Series(np.nan, index=close.index)
        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        tp_arr = np.asarray(take_profit_series, dtype=np.float64)
        sl_arr = np.asarray(stop_loss_series, dtype=np.float64)

        signal_indices = np.where(signal_mask.values)[0]

        for idx in signal_indices:
            tp = tp_arr[idx]
            sl = sl_arr[idx]
            entry = close_arr[idx]

            if np.isnan(tp) or np.isnan(sl) or tp <= entry or sl >= entry:
                continue

            end = min(idx + time_limit + 1, n)
            for j in range(idx + 1, end):
                if high_arr[j] >= tp:
                    labels.iloc[idx] = 1
                    break
                elif low_arr[j] <= sl:
                    labels.iloc[idx] = -1
                    break
            else:
                labels.iloc[idx] = 0

        return labels
