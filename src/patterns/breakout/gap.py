"""
Gap Pattern (Explosion Gap Pivot) with 4-type hierarchy classification.

Gap Classification (adapted from Duddella, NCFE):
  - Common Gap: Small gap within range, low volume change. Low reliability → skip.
  - Breakaway Gap: Gap from consolidation, high volume. Trade direction → almost never fills.
  - Continuation Gap: Mid-trend gap, moderate volume. Trade direction.
  - Exhaustion Gap: End of trend, high volume then reversal. Fade (counter-trend).

Detection Logic:
- Gap Up: open > prior high (gap zone between prior high and current open)
- Gap Down: open < prior low (gap zone between current open and prior low)
- Explosion Gap Pivot Strategy:
  1. Wait for gap to occur
  2. Monitor for throwback/pullback toward gap
  3. If price retraces and STOPS (does not fill gap) -> Pivot Point identified
      - Pivot Low (for gap up): lowest point of retracement that holds above gap
      - Pivot High (for gap down): highest point of retracement that holds below gap
  4. Entry: Buy stop above high of gap candle (for gap up)
  5. Protective Stop: Initially at gap low, then move to below pivot low

Entry Rules:
- Long Entry: Buy Stop = high[breakout_bar] + filter (for gap up with pivot confirmation)
- Short Entry: Sell Stop = low[breakdown_bar] - filter (for gap down with pivot confirmation)
- Signal only if pivot confirmation occurs (gap not filled during retracement)
- Common gaps → NOT traded (filtered out)
- Exhaustion gaps → faded (reversal trade)

Stop Loss Rules:
- Long Stop: Below gap zone or pivot low
- Short Stop: Above gap zone or pivot high

Take Profit Rules:
- Target: Based on gap size and prior swing levels
- Gap Size = |current_open - prior_high| (gap up) or |prior_low - current_open| (gap down)

Warning: Gaps that fully "fill" (price returns through gap) invalidate signal
"""

from typing import Dict, Optional

import enum
import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class GapPattern(BasePattern):
    """
    Gap Pattern Detector (Explosion Gap Pivot Strategy)

    Detects gap breakouts with pivot confirmation for entry signals.
    """

    def __init__(
        self,
        min_gap_pct: float = 0.01,
        max_lookback: int = 20,
        pivot_bars: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        confirmation_filter: float = 0.005,
    ):
        """
        Initialize Gap pattern detector.

        Args:
            min_gap_pct: Minimum gap size as percentage of price (default 1%)
            max_lookback: Maximum bars to look back for gap detection
            pivot_bars: Bars to check for pivot confirmation
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            confirmation_filter: Minimum breakout percentage for confirmation
        """
        super().__init__(name="Gap Pattern", pattern_type=PatternType.BREAKOUT, min_bars_required=3)
        self.min_gap_pct = min_gap_pct
        self.max_lookback = max_lookback
        self.pivot_bars = pivot_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.confirmation_filter = confirmation_filter
        self._atr_gap_skip: float = 2.5
        self._atr_lookback: int = 10

    def _classify_gap(self, gap: dict, arrays: dict, i: int) -> str:
        """Classify gap type: common, breakaway, continuation, exhaustion."""
        if i < self._atr_lookback:
            return "breakaway"

        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        recent_close = [
            float(close_arr[j])
            for j in range(max(0, i - self._atr_lookback), i)
            if j < len(close_arr)
        ]
        if not recent_close:
            return "breakaway"

        recent_high = max(
            float(high_arr[j])
            for j in range(max(0, i - self._atr_lookback), i)
            if j < len(high_arr)
        )
        recent_low = min(
            float(low_arr[j]) for j in range(max(0, i - self._atr_lookback), i) if j < len(low_arr)
        )
        avg_range = recent_high - recent_low

        gap_size = gap["gap_size"]
        gap_type = gap["type"]

        # Size filter: gaps > 2.5x range are noise/event-driven → skip
        if avg_range > 0 and gap_size > self._atr_gap_skip * avg_range:
            return "skip"

        # Common gap: small gap within normal range
        if avg_range > 0 and gap_size < 0.25 * avg_range:
            return "common"

        # Trend context: check directional bias over lookback
        price_change = (
            (recent_close[-1] - recent_close[0]) / recent_close[0] if recent_close[0] > 0 else 0
        )

        if gap_type == "gap_up":
            if price_change > 0.02:
                return "continuation"
            elif price_change < -0.02:
                return "exhaustion"
            else:
                return "breakaway"
        else:
            if price_change < -0.02:
                return "continuation"
            elif price_change > 0.02:
                return "exhaustion"
            else:
                return "breakaway"

    def _detect_gap(self, arrays: dict, i: int) -> Optional[Dict]:
        """
        Detect gap at bar i.

        Returns:
            Dictionary with gap details or None
        """
        if i < 1:
            return None

        open_arr = arrays["open"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        close_arr = arrays["close"]

        current_open = float(open_arr[i])
        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        current_close = float(close_arr[i])

        prior_high = float(high_arr[i - 1])
        prior_low = float(low_arr[i - 1])
        prior_close = float(close_arr[i - 1])

        # Gap Up: current open > prior high
        if current_open > prior_high:
            gap_size = current_open - prior_high
            gap_pct = gap_size / prior_close if prior_close > 0 else 0

            if gap_pct >= self.min_gap_pct:
                return {
                    "type": "gap_up",
                    "gap_start": prior_high,  # Bottom of gap
                    "gap_end": current_open,  # Top of gap
                    "gap_size": gap_size,
                    "gap_pct": gap_pct,
                    "gap_bar": i,
                    "gap_high": current_high,
                    "gap_low": current_low,
                    "prior_high": prior_high,
                    "prior_low": prior_low,
                }

        # Gap Down: current open < prior low
        if current_open < prior_low:
            gap_size = prior_low - current_open
            gap_pct = gap_size / prior_close if prior_close > 0 else 0

            if gap_pct >= self.min_gap_pct:
                return {
                    "type": "gap_down",
                    "gap_start": current_open,  # Bottom of gap
                    "gap_end": prior_low,  # Top of gap
                    "gap_size": gap_size,
                    "gap_pct": gap_pct,
                    "gap_bar": i,
                    "gap_high": current_high,
                    "gap_low": current_low,
                    "prior_high": prior_high,
                    "prior_low": prior_low,
                }

        return None

    def _find_pivot_confirmation(
        self, arrays: dict, gap: Dict, gap_bar: int, current_bar: int
    ) -> Optional[Dict]:
        """
        Find pivot confirmation after gap.

        For gap up: Look for pivot low above gap
        For gap down: Look for pivot high below gap
        """
        if current_bar <= gap_bar:
            return None

        low_arr = arrays["low"]
        high_arr = arrays["high"]
        close_arr = arrays["close"]

        gap_type = gap["type"]
        gap_start = gap["gap_start"]
        gap_end = gap["gap_end"]

        # Check bars between gap and current for pivot confirmation
        start_bar = gap_bar + 1
        end_bar = min(current_bar, gap_bar + self.pivot_bars + 1)

        if end_bar <= start_bar:
            return None

        if gap_type == "gap_up":
            # For gap up, look for pivot low above gap
            # Price should have pulled back but not filled the gap
            pivot_low = None
            pivot_bar_idx = None

            for bar_idx in range(start_bar, end_bar + 1):
                if bar_idx >= len(low_arr):
                    break

                bar_low = float(low_arr[bar_idx])
                bar_close = float(close_arr[bar_idx])

                # Check if gap is still intact (low stays above gap start)
                if bar_low > gap_start:
                    if pivot_low is None or bar_low < pivot_low:
                        pivot_low = bar_low
                        pivot_bar_idx = bar_idx

                # If gap is filled, no valid pivot
                if bar_low <= gap_start:
                    return None

            if pivot_low is not None and pivot_bar_idx is not None:
                # Check for breakout above gap high
                current_close = float(close_arr[current_bar])
                current_high = float(high_arr[current_bar])

                if current_close > gap["gap_high"] * (1 + self.confirmation_filter):
                    return {
                        "pivot_type": "pivot_low",
                        "pivot_price": pivot_low,
                        "pivot_bar": pivot_bar_idx,
                        "breakout_bar": current_bar,
                    }

        elif gap_type == "gap_down":
            # For gap down, look for pivot high below gap
            # Price should have thrown back but not filled the gap
            pivot_high = None
            pivot_bar_idx = None

            for bar_idx in range(start_bar, end_bar + 1):
                if bar_idx >= len(high_arr):
                    break

                bar_high = float(high_arr[bar_idx])
                bar_close = float(close_arr[bar_idx])

                # Check if gap is still intact (high stays below gap end)
                if bar_high < gap_end:
                    if pivot_high is None or bar_high > pivot_high:
                        pivot_high = bar_high
                        pivot_bar_idx = bar_idx

                # If gap is filled, no valid pivot
                if bar_high >= gap_end:
                    return None

            if pivot_high is not None and pivot_bar_idx is not None:
                # Check for breakout below gap low
                current_close = float(close_arr[current_bar])
                current_low = float(low_arr[current_bar])

                if current_close < gap["gap_low"] * (1 - self.confirmation_filter):
                    return {
                        "pivot_type": "pivot_high",
                        "pivot_price": pivot_high,
                        "pivot_bar": pivot_bar_idx,
                        "breakout_bar": current_bar,
                    }

        return None

    def _find_pattern(self, arrays: dict, i: int) -> Optional[Dict]:
        """
        Find Gap pattern with pivot confirmation.

        Args:
            arrays: Dictionary with NumPy arrays
            i: Current bar index

        Returns:
            Dictionary with pattern details or None
        """
        # Look for recent gaps within lookback period
        lookback = min(self.max_lookback, i)

        for gap_bar in range(i - lookback, i):
            if gap_bar < 1:
                continue

            gap = self._detect_gap(arrays, gap_bar)
            if gap is None:
                continue

            # Look for pivot confirmation
            pivot = self._find_pivot_confirmation(arrays, gap, gap_bar, i)
            if pivot is None:
                continue

            return {
                **gap,
                **pivot,
                "pattern_start": gap_bar,
                "pattern_end": i,
                "gap_class": self._classify_gap(gap, arrays, gap_bar),
            }

        return None

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized detection of Gap patterns across the entire DataFrame.

        Returns:
            np.ndarray of np.int8: 0=no signal, 1=LONG (gap up), -1=SHORT (gap down)
        """
        n = len(df)
        result = np.zeros(n, dtype=np.int8)
        if n < 11:
            return result

        open_a = df["Open"].to_numpy()
        high_a = df["High"].to_numpy()
        low_a = df["Low"].to_numpy()
        close_a = df["Close"].to_numpy()

        min_gap = self.min_gap_pct
        lookback = min(self.max_lookback, n - 1)
        pivot_bars = self.pivot_bars
        confirm = self.confirmation_filter
        atr_skip = self._atr_gap_skip

        for i in range(10, n):
            gap_start_idx = max(1, i - lookback)

            for gap_bar in range(gap_start_idx, i):
                current_open = open_a[gap_bar]
                prior_high = high_a[gap_bar - 1]
                prior_low = low_a[gap_bar - 1]
                prior_close = close_a[gap_bar - 1]
                gap_type = None
                gap_start = 0.0
                gap_end = 0.0
                gap_high = 0.0
                gap_low = 0.0
                gap_size = 0.0

                if current_open > prior_high:
                    gap_size = current_open - prior_high
                    if prior_close > 0 and (gap_size / prior_close) >= min_gap:
                        gap_type = "gap_up"
                        gap_start = prior_high
                        gap_end = current_open
                        gap_high = high_a[gap_bar]
                        gap_low = low_a[gap_bar]
                elif current_open < prior_low:
                    gap_size = prior_low - current_open
                    if prior_close > 0 and (gap_size / prior_close) >= min_gap:
                        gap_type = "gap_down"
                        gap_start = current_open
                        gap_end = prior_low
                        gap_high = high_a[gap_bar]
                        gap_low = low_a[gap_bar]

                if gap_type is None:
                    continue

                # Size filter: skip oversized gaps
                if gap_bar >= 10:
                    recent_high = max(high_a[max(0, gap_bar - 10) : gap_bar])
                    recent_low = min(low_a[max(0, gap_bar - 10) : gap_bar])
                    avg_range = recent_high - recent_low
                    if avg_range > 0 and gap_size > atr_skip * avg_range:
                        continue

                # Scan ahead for pivot confirmation
                start_bar = gap_bar + 1
                end_bar = min(i, gap_bar + pivot_bars + 1)
                if end_bar <= start_bar:
                    continue

                if gap_type == "gap_up":
                    pivot_low = None
                    gap_filled = False
                    for bar_idx in range(start_bar, end_bar + 1):
                        if bar_idx >= n:
                            break
                        bar_low = low_a[bar_idx]
                        if bar_low <= gap_start:
                            gap_filled = True
                            break
                        if pivot_low is None or bar_low < pivot_low:
                            pivot_low = bar_low
                    if not gap_filled and pivot_low is not None:
                        if close_a[i] > gap_high * (1.0 + confirm):
                            result[i] = 1
                else:  # gap_down
                    pivot_high = None
                    gap_filled = False
                    for bar_idx in range(start_bar, end_bar + 1):
                        if bar_idx >= n:
                            break
                        bar_high = high_a[bar_idx]
                        if bar_high >= gap_end:
                            gap_filled = True
                            break
                        if pivot_high is None or bar_high > pivot_high:
                            pivot_high = bar_high
                    if not gap_filled and pivot_high is not None:
                        if close_a[i] < gap_low * (1.0 - confirm):
                            result[i] = -1

        return result

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Gap pattern at bar index i.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            window_start: Optional window start for bounds checking

        Returns:
            PatternResult with detection status and signal
        """
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        arrays = self._extract_arrays(df)
        pattern = self._find_pattern(arrays, i)

        if pattern is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self._generate_signal(df, i, pattern, arrays)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "gap_type": pattern["type"],
                "gap_class": pattern.get("gap_class", "breakaway"),
                "gap_size": pattern["gap_size"],
                "gap_pct": pattern["gap_pct"],
                "gap_start": pattern["gap_start"],
                "gap_end": pattern["gap_end"],
                "pivot_price": pattern["pivot_price"],
                "pivot_bar": pattern["pivot_bar"],
            },
            bars_since_detection=0,
            start_index=pattern["pattern_start"],
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, df: pd.DataFrame, i: int, pattern: Dict, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate Gap pattern signal."""
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        gap_size = pattern["gap_size"]
        gap_type = pattern["type"]
        gap_class = pattern.get("gap_class", "breakaway")
        pivot_price = pattern["pivot_price"]

        # Common gaps and oversized gaps → no signal
        if gap_class in ("common", "skip"):
            return None

        # Exhaustion gap: fade (trade opposite direction)
        if gap_class == "exhaustion":
            if gap_type == "gap_up":
                # Exhaustion gap up → short the reversal
                entry_price = current_low - self.entry_offset
                stop_loss = current_high + self.stop_offset
                take_profit_1 = entry_price - (gap_size * 1.0)
                confidence = 0.50
                return TradeSignal(
                    pattern_name="Exhaustion Gap Up",
                    direction=SignalDirection.SHORT,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    confidence=confidence,
                    metadata={
                        "gap_type": "gap_up",
                        "gap_class": "exhaustion",
                        "gap_size": gap_size,
                    },
                )
            else:
                # Exhaustion gap down → long the reversal
                entry_price = current_high + self.entry_offset
                stop_loss = current_low - self.stop_offset
                take_profit_1 = entry_price + (gap_size * 1.0)
                confidence = 0.50
                return TradeSignal(
                    pattern_name="Exhaustion Gap Down",
                    direction=SignalDirection.LONG,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    confidence=confidence,
                    metadata={
                        "gap_type": "gap_down",
                        "gap_class": "exhaustion",
                        "gap_size": gap_size,
                    },
                )

        if gap_type == "gap_up":
            # Gap up with pivot low confirmation - LONG signal
            entry_price = current_high + self.entry_offset
            stop_loss = pivot_price - self.stop_offset
            take_profit_1 = entry_price + (gap_size * 1.0)
            take_profit_2 = entry_price + (gap_size * 1.5)
            take_profit_3 = entry_price + (gap_size * 2.0)

            confidence = 0.60

            return TradeSignal(
                pattern_name="Gap Up",
                direction=SignalDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                confidence=confidence,
                timestamp=df.index[i] if hasattr(df, "index") else None,
                metadata={
                    "gap_type": "gap_up",
                    "gap_class": gap_class,
                    "gap_size": gap_size,
                    "pivot_price": pivot_price,
                    "entry_type": "buy_stop",
                },
            )
        else:
            # Gap down with pivot high confirmation - SHORT signal
            entry_price = current_low - self.entry_offset
            stop_loss = pivot_price + self.stop_offset
            take_profit_1 = entry_price - (gap_size * 1.0)
            take_profit_2 = entry_price - (gap_size * 1.5)
            take_profit_3 = entry_price - (gap_size * 2.0)

            confidence = 0.60

            return TradeSignal(
                pattern_name="Gap Down",
                direction=SignalDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                confidence=confidence,
                timestamp=df.index[i] if hasattr(df, "index") else None,
                metadata={
                    "gap_type": "gap_down",
                    "gap_class": gap_class,
                    "gap_size": gap_size,
                    "pivot_price": pivot_price,
                    "entry_type": "sell_stop",
                },
            )


# ── Gap type classification utilities ─────────────────────────


class GapType(enum.Enum):
    """Gap classification per Duddella 2007 / NCFE."""

    COMMON = "common"
    BREAKAWAY = "breakaway"
    CONTINUATION = "measuring"
    EXHAUSTION = "exhaustion"


def classify_gap_type(ohlc: pd.DataFrame, gap_indices: np.ndarray) -> dict[int, GapType]:
    """Classify each detected gap by type using context rules.

    Rules (Duddella 2007, NCFE):
    - Breakaway: gap at trendline penetration + volume surge + after consolidation
    - Continuation: mid-trend, volume lower than breakaway
    - Exhaustion: wide gap + heavy volume + near end of extended move
    - Common: within trading range, low volume, narrow gap
    """
    classifications: dict[int, GapType] = {}
    open_ = ohlc["Open"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    volume = ohlc["Volume"].to_numpy(dtype=np.float64)
    n = len(close)

    for gap_idx in gap_indices:
        if gap_idx < 1 or gap_idx >= n:
            continue
        gap_size = abs(open_[gap_idx] - close[gap_idx - 1])
        vol_ratio = volume[gap_idx] / (
            float(np.mean(volume[max(0, gap_idx - 20) : gap_idx])) + 1e-10
        )

        prev_close = close[max(0, gap_idx - 10) : gap_idx]
        if len(prev_close) < 5:
            classifications[gap_idx] = GapType.COMMON
            continue

        price_range = float(np.max(prev_close)) - float(np.min(prev_close))
        avg_price = float(np.mean(prev_close))
        if avg_price <= 0:
            classifications[gap_idx] = GapType.COMMON
            continue

        relative_gap = gap_size / avg_price

        if vol_ratio > 1.8 and relative_gap > 0.02:
            if relative_gap > 0.05 and vol_ratio > 3.0:
                classifications[gap_idx] = GapType.EXHAUSTION
            elif price_range / avg_price < 0.03:
                classifications[gap_idx] = GapType.BREAKAWAY
            else:
                classifications[gap_idx] = GapType.CONTINUATION
        elif vol_ratio > 1.2 and relative_gap > 0.01:
            classifications[gap_idx] = GapType.CONTINUATION
        else:
            classifications[gap_idx] = GapType.COMMON

    return classifications


def is_gap_tradable(ohlc: pd.DataFrame, gap_idx: int, atr: np.ndarray) -> bool:
    """Duddella gap validity check: gap must be <= 2.5x 10-day ATR."""
    if gap_idx < 1 or gap_idx >= len(ohlc):
        return False
    gap_size = abs(ohlc.iloc[gap_idx]["Open"] - ohlc.iloc[gap_idx - 1]["Close"])
    avg_atr_10 = (
        float(np.mean(atr[max(0, gap_idx - 10) : gap_idx]))
        if gap_idx >= 10
        else float(atr[gap_idx])
    )
    if avg_atr_10 <= 0:
        avg_atr_10 = gap_size
    return gap_size <= 2.5 * avg_atr_10
