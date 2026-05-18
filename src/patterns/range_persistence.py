"""
Range-Persistence Pattern Detectors.

Implements three range-based patterns derived from the key empirical finding
in "From Hypotheses to Factors" (arXiv:2604.26747v1):

    "Small, liquidity-scarce tokens with persistent intraday range and
     positive trend tend to outperform."

Range (high-low/close) proxies for speculative attention — active but
not yet crowded. These are the paper's top-performing single-factor signals
(Pure OOS Sharpe 1.7-2.4 on crypto panel).

Detectors (discrete pattern triggers):
    PersistentRange: Short MA of hl_range crosses above long MA while range > median
    ContractingRange: Range moves below rolling low-percentile + trend confirmation
    ExpandingRange: Range spikes above expansion_factor × rolling mean + trend confirmation

All implement the BasePattern interface for per-bar detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from src.patterns.base import (
    BasePattern,
    PatternResult,
    PatternType,
    SignalDirection,
    TradeSignal,
)

if TYPE_CHECKING:
    from src.indicators.indicator_cache import IndicatorCache


@dataclass
class PersistentRange(BasePattern):
    """Detects persistent elevated intraday range as a bullish continuation signal.

    Trigger: hl_range short MA > long MA AND current range > median range.
    This signals that speculative attention is actively increasing.

    Source: h5_smallcap_low_volume_range20 (Pure OOS Sharpe +2.410)
    """

    name: str = "PersistentRange"
    pattern_type: PatternType = PatternType.CONTINUATION
    min_bars_required: int = 20
    short_ma: int = 10
    long_ma: int = 20
    atr_period: int = 14

    def __post_init__(self) -> None:
        super().__post_init__()

    def detect(
        self,
        df: pd.DataFrame,
        i: int,
        window_start: Optional[int] = None,
    ) -> PatternResult:
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values

        if i < self.long_ma:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        hl_range = (high - low) / np.where(close > 0, close, np.nan)
        range_ma_short = np.nanmean(hl_range[i - self.short_ma + 1 : i + 1])
        range_ma_long = np.nanmean(hl_range[i - self.long_ma + 1 : i + 1])
        median_range = np.nanmedian(hl_range[max(0, i - self.long_ma) : i + 1])

        if np.isnan(range_ma_short) or np.isnan(range_ma_long):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        detected = range_ma_short > range_ma_long and hl_range[i] > median_range

        signal = None
        if detected:
            signal = self.generate_signal(df, i)

        return PatternResult(
            detected=detected,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            start_index=i,
            end_index=i,
            pivot_points={
                "hl_range": float(hl_range[i]),
                "range_ma_short": float(range_ma_short),
                "range_ma_long": float(range_ma_long),
            },
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        close = float(df["Close"].values[i])
        atr_vals = np.abs(np.diff(df["Close"].values[max(0, i - self.atr_period) : i + 1]))
        atr = float(np.nanmean(atr_vals)) if len(atr_vals) > 0 else close * 0.02

        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.LONG,
            entry_price=close,
            stop_loss=close - 2 * atr,
            take_profit_1=close + 2 * atr,
            confidence=min(1.0, 0.55),
            timestamp=df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None,
        )


@dataclass
class ContractingRange(BasePattern):
    """Detects range contraction (coiling) as a breakout precursor.

    Trigger: hl_range falls below rolling low-percentile → volatility compression.
    Direction signaled by recent price trend.

    This is the typical "volatility contraction → expansion" setup.
    """

    name: str = "ContractingRange"
    pattern_type: PatternType = PatternType.BREAKOUT
    min_bars_required: int = 20
    lookback: int = 20
    percentile_threshold: float = 0.25
    trend_window: int = 5
    atr_period: int = 14

    def __post_init__(self) -> None:
        super().__post_init__()

    def detect(
        self,
        df: pd.DataFrame,
        i: int,
        window_start: Optional[int] = None,
    ) -> PatternResult:
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        if i < self.lookback:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values

        hl_range = (high - low) / np.where(close > 0, close, np.nan)
        window = hl_range[i - self.lookback : i + 1]
        if np.any(np.isnan(window)):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        threshold = float(np.nanpercentile(window, self.percentile_threshold * 100))
        current_range = hl_range[i]
        contracting = float(current_range) <= threshold

        if not contracting:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        signal = self.generate_signal(df, i)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            start_index=i,
            end_index=i,
            pivot_points={
                "hl_range": float(current_range),
                "threshold": threshold,
            },
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        close_arr = df["Close"].values
        close = float(close_arr[i])

        start = max(0, i - self.trend_window)
        trend_close = close_arr[start : i + 1]
        trend = (trend_close[-1] - trend_close[0]) / max(trend_close[0], 1e-9)

        atr_vals = np.abs(np.diff(close_arr[max(0, i - self.atr_period) : i + 1]))
        atr = float(np.nanmean(atr_vals)) if len(atr_vals) > 0 else close * 0.02

        if trend > 0:
            return TradeSignal(
                pattern_name=self.name,
                direction=SignalDirection.LONG,
                entry_price=close,
                stop_loss=close - 2.5 * atr,
                take_profit_1=close + 3 * atr,
                confidence=min(1.0, 0.60),
                timestamp=df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None,
            )
        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.SHORT,
            entry_price=close,
            stop_loss=close + 2.5 * atr,
            take_profit_1=close - 3 * atr,
            confidence=min(1.0, 0.60),
            timestamp=df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None,
        )


@dataclass
class ExpandingRange(BasePattern):
    """Detects range expansion (volatility breakout) with trend confirmation.

    Trigger: hl_range > expansion_factor × rolling mean of hl_range.
    Direction confirmed by recent price trend.

    This is a volatility breakout/continuation pattern.
    """

    name: str = "ExpandingRange"
    pattern_type: PatternType = PatternType.VOLATILITY
    min_bars_required: int = 20
    lookback_mean: int = 20
    expansion_factor: float = 1.5
    trend_window: int = 3
    atr_period: int = 14

    def __post_init__(self) -> None:
        super().__post_init__()

    def detect(
        self,
        df: pd.DataFrame,
        i: int,
        window_start: Optional[int] = None,
    ) -> PatternResult:
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        if i < self.lookback_mean:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values

        hl_range = (high - low) / np.where(close > 0, close, np.nan)
        range_ma = np.nanmean(hl_range[i - self.lookback_mean + 1 : i + 1])

        if np.isnan(range_ma) or np.isnan(hl_range[i]) or hl_range[i] <= 0:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        expanding = float(hl_range[i]) > self.expansion_factor * float(range_ma)

        if not expanding:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                start_index=i,
                end_index=i,
            )

        signal = self.generate_signal(df, i)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            start_index=i,
            end_index=i,
            pivot_points={
                "hl_range": float(hl_range[i]),
                "range_ma": float(range_ma),
                "expansion_ratio": float(hl_range[i] / range_ma),
            },
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        close_arr = df["Close"].values
        close = float(close_arr[i])

        start = max(0, i - self.trend_window)
        trend_close = close_arr[start : i + 1]
        trend = (trend_close[-1] - trend_close[0]) / max(trend_close[0], 1e-9)

        atr_vals = np.abs(np.diff(close_arr[max(0, i - self.atr_period) : i + 1]))
        atr = float(np.nanmean(atr_vals)) if len(atr_vals) > 0 else close * 0.02

        if trend > 0:
            return TradeSignal(
                pattern_name=self.name,
                direction=SignalDirection.LONG,
                entry_price=close,
                stop_loss=close - 2 * atr,
                take_profit_1=close + 2.5 * atr,
                confidence=min(1.0, 0.55),
                timestamp=df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None,
            )
        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.SHORT,
            entry_price=close,
            stop_loss=close + 2 * atr,
            take_profit_1=close - 2.5 * atr,
            confidence=min(1.0, 0.55),
            timestamp=df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None,
        )
