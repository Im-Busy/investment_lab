"""
Symmetric Triangle Pattern

Detection Logic:
- Identify at least 2 Lower Highs: high[i] < high[i-2] ... high[j] < high[i]
- Identify at least 2 Higher Lows: low[k] > low[k-2] ... low[m] > low[k]
- Convergence: Slope of Highs < 0 and Slope of Lows > 0
- Pattern Complete when price approaches apex (intersection of trendlines)
- Volatility Contraction: Range[i] < SMA(Range, 10) during formation

Entry Rules:
- Long Entry: Buy Stop = high[breakout_bar] + 0.01 (Close above upper trendline)
- Short Entry: Sell Stop = low[breakdown_bar] - 0.01 (Close below lower trendline)
- Confirmation: Close must be outside trendline by at least 1 tick

Stop Loss Rules:
- Long Stop: low[recent_swing_low_within_triangle] - 0.01
- Short Stop: high[recent_swing_high_within_triangle] + 0.01

Take Profit Rules:
- Target 1: Entry + 0.50 * Triangle_Depth (50% of pattern height)
- Target 2: Entry + 1.00 * Triangle_Depth (100% of pattern height)
- Triangle_Depth = Max(Highs of pattern) - Min(Lows of pattern)
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.technical import average_range, volume_sma
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class SymmetricTriangle(BasePattern):
    """
    Symmetric Triangle Pattern Detector

    A continuation pattern characterized by converging trendlines
    with lower highs and higher lows.
    """

    def __init__(
        self,
        min_highs: int = 2,
        min_lows: int = 2,
        max_pattern_bars: int = 50,
        min_pattern_bars: int = 10,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = False,
    ):
        """
        Initialize Symmetric Triangle pattern detector.

        Args:
            min_highs: Minimum number of lower highs required
            min_lows: Minimum number of higher lows required
            max_pattern_bars: Maximum bars for pattern formation
            min_pattern_bars: Minimum bars for pattern formation
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            volume_filter: Require volume confirmation on breakout
        """
        super().__init__(
            name="Symmetric Triangle",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=min_pattern_bars,
        )
        self.min_highs = min_highs
        self.min_lows = min_lows
        self.max_pattern_bars = max_pattern_bars
        self.min_pattern_bars = min_pattern_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.volume_filter = volume_filter

    def _find_lower_highs(
        self, df: pd.DataFrame, i: int, lookback: int = 30
    ) -> List[Tuple[int, float]]:
        """
        Find sequence of lower highs.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            lookback: Maximum bars to look back

        Returns:
            List of (index, price) tuples for lower highs
        """
        start = max(0, i - lookback)
        highs: List[Tuple[int, float]] = []

        # Find all local highs (bars with higher high than neighbors)
        for j in range(start + 2, i + 1):
            prev_high = self._safe_float(df.iloc[j - 1]["High"])
            curr_high = self._safe_float(df.iloc[j]["High"])
            next_high = self._safe_float(df.iloc[j + 1]["High"]) if j + 1 <= i else curr_high

            if curr_high > prev_high and curr_high > next_high:
                highs.append((j, curr_high))

        # Find sequence of lower highs
        if len(highs) < self.min_highs:
            return []

        lower_highs: List[Tuple[int, float]] = []
        for k in range(len(highs) - 1, 0, -1):
            if highs[k][1] < highs[k - 1][1]:
                if not lower_highs:
                    lower_highs.insert(0, highs[k])
                lower_highs.insert(0, highs[k - 1])
            else:
                break

        return lower_highs if len(lower_highs) >= self.min_highs else []

    def _find_higher_lows(
        self, df: pd.DataFrame, i: int, lookback: int = 30
    ) -> List[Tuple[int, float]]:
        """
        Find sequence of higher lows.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            lookback: Maximum bars to look back

        Returns:
            List of (index, price) tuples for higher lows
        """
        start = max(0, i - lookback)
        lows: List[Tuple[int, float]] = []

        # Find all local lows
        for j in range(start + 2, i + 1):
            prev_low = self._safe_float(df.iloc[j - 1]["Low"])
            curr_low = self._safe_float(df.iloc[j]["Low"])
            next_low = self._safe_float(df.iloc[j + 1]["Low"]) if j + 1 <= i else curr_low

            if curr_low < prev_low and curr_low < next_low:
                lows.append((j, curr_low))

        # Find sequence of higher lows
        if len(lows) < self.min_lows:
            return []

        higher_lows: List[Tuple[int, float]] = []
        for k in range(len(lows) - 1, 0, -1):
            if lows[k][1] > lows[k - 1][1]:
                if not higher_lows:
                    higher_lows.insert(0, lows[k])
                higher_lows.insert(0, lows[k - 1])
            else:
                break

        return higher_lows if len(higher_lows) >= self.min_lows else []

    def _calculate_trendline_slope(self, points: List[Tuple[int, float]]) -> float:
        """
        Calculate slope of trendline through points.

        Args:
            points: List of (index, price) tuples

        Returns:
            Slope of trendline
        """
        if len(points) < 2:
            return 0

        # Simple linear regression
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x**2)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

        return float(slope)

    def _calculate_triangle_depth(
        self, lower_highs: List[Tuple[int, float]], higher_lows: List[Tuple[int, float]]
    ) -> float:
        """
        Calculate triangle depth (max height).

        Args:
            lower_highs: List of lower high points
            higher_lows: List of higher low points

        Returns:
            Triangle depth
        """
        if not lower_highs or not higher_lows:
            return 0.0

        max_high: float = max(h[1] for h in lower_highs)
        min_low: float = min(pt[1] for pt in higher_lows)

        return float(max_high - min_low)

    def _check_volatility_contraction(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """
        Check if volatility is contracting during pattern formation.

        Args:
            df: DataFrame with OHLC data
            start_idx: Pattern start index
            end_idx: Pattern end index

        Returns:
            True if volatility is contracting
        """
        if end_idx - start_idx < 5:
            return False

        # Calculate average range for pattern period
        pattern_ranges = []
        for j in range(start_idx, end_idx + 1):
            high = self._safe_float(df.iloc[j]["High"])
            low = self._safe_float(df.iloc[j]["Low"])
            pattern_ranges.append(high - low)

        avg_pattern_range: float = float(np.mean(pattern_ranges))

        # Compare with overall average range
        avg_range = average_range(df, 20)
        if end_idx < len(avg_range):
            overall_avg = self._safe_float(avg_range.iloc[end_idx])
            return bool(avg_pattern_range < overall_avg)

        return True

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Symmetric Triangle pattern at bar index i.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index

        Returns:
            PatternResult with detection status and signal
        """
        if not self._validate_data(df, i):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Find lower highs and higher lows
        lower_highs = self._find_lower_highs(df, i, self.max_pattern_bars)
        higher_lows = self._find_higher_lows(df, i, self.max_pattern_bars)

        if len(lower_highs) < self.min_highs or len(higher_lows) < self.min_lows:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Check convergence (upper slope negative, lower slope positive)
        upper_slope = self._calculate_trendline_slope(lower_highs)
        lower_slope = self._calculate_trendline_slope(higher_lows)

        if upper_slope >= 0 or lower_slope <= 0:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Check volatility contraction
        start_idx = min(lower_highs[0][0], higher_lows[0][0])
        end_idx = max(lower_highs[-1][0], higher_lows[-1][0])

        _ = self._check_volatility_contraction(df, start_idx, end_idx)

        # Calculate triangle depth
        depth = self._calculate_triangle_depth(lower_highs, higher_lows)

        # Check for breakout
        current_close = self._safe_float(df.iloc[i]["Close"])

        # Get upper and lower trendline values at current bar
        upper_trendline = lower_highs[-1][1]  # Simplified
        lower_trendline = higher_lows[-1][1]  # Simplified

        breakout_up = current_close > upper_trendline
        breakout_down = current_close < lower_trendline

        if not (breakout_up or breakout_down):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "lower_highs": [(idx, price) for idx, price in lower_highs],
                    "higher_lows": [(idx, price) for idx, price in higher_lows],
                    "upper_slope": upper_slope,
                    "lower_slope": lower_slope,
                    "depth": depth,
                },
            )

        # Generate signal
        signal = self._generate_signal(
            df, i, lower_highs, higher_lows, depth, breakout_up, breakout_down
        )

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "lower_highs": [(idx, price) for idx, price in lower_highs],
                "higher_lows": [(idx, price) for idx, price in higher_lows],
                "upper_slope": upper_slope,
                "lower_slope": lower_slope,
                "depth": depth,
                "breakout_direction": "up" if breakout_up else "down",
            },
            bars_since_detection=0,
            start_index=start_idx,
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self,
        df: pd.DataFrame,
        i: int,
        lower_highs: List[Tuple[int, float]],
        higher_lows: List[Tuple[int, float]],
        depth: float,
        breakout_up: bool,
        breakout_down: bool,
    ) -> Optional[TradeSignal]:
        """Generate trade signal for triangle breakout."""

        current_high = self._safe_float(df.iloc[i]["High"])
        current_low = self._safe_float(df.iloc[i]["Low"])

        # Find recent swing low/high within triangle for stop loss
        swing_low = min([pt[1] for pt in higher_lows])
        swing_high = max([h[1] for h in lower_highs])

        if breakout_up:
            # Long signal
            entry_price = current_high + self.entry_offset
            stop_loss = swing_low - self.stop_offset

            take_profit_1 = entry_price + (depth * 0.5)
            take_profit_2 = entry_price + depth

            direction = SignalDirection.LONG
            breakout_dir = "up"

        else:  # breakout_down
            # Short signal
            entry_price = current_low - self.entry_offset
            stop_loss = swing_high + self.stop_offset

            take_profit_1 = entry_price - (depth * 0.5)
            take_profit_2 = entry_price - depth

            direction = SignalDirection.SHORT
            breakout_dir = "down"

        # Volume confirmation
        volume_confirmed = True
        if self.volume_filter:
            vol_sma = volume_sma(df["Volume"], 20)
            if i < len(vol_sma):
                current_vol = self._safe_float(df.iloc[i]["Volume"])
                avg_vol = self._safe_float(vol_sma.iloc[i])
                volume_confirmed = current_vol > avg_vol

        # Confidence
        confidence = 0.55
        if volume_confirmed:
            confidence += 0.1

        # Check if breakout is near apex (last 1/3 of pattern)
        pattern_bars = max(lower_highs[-1][0], higher_lows[-1][0]) - min(
            lower_highs[0][0], higher_lows[0][0]
        )
        if pattern_bars > 0:
            apex_proximity = (i - min(lower_highs[0][0], higher_lows[0][0])) / pattern_bars
            if apex_proximity > 0.67:  # Near apex
                confidence += 0.1

        return TradeSignal(
            pattern_name=f"{self.name} ({breakout_dir.capitalize()} Breakout)",
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.index[i],
            metadata={
                "breakout_direction": breakout_dir,
                "triangle_depth": depth,
                "upper_slope": self._calculate_trendline_slope(lower_highs),
                "lower_slope": self._calculate_trendline_slope(higher_lows),
                "volume_confirmed": volume_confirmed,
                "entry_type": "buy_stop" if breakout_up else "sell_stop",
            },
        )


class AscendingTriangle(SymmetricTriangle):
    """
    Ascending Triangle Pattern Detector

    Variation with flat upper trendline and rising lower trendline.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ascending Triangle"

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect Ascending Triangle - flat top, rising bottom."""
        result = super().detect(df, i)

        if not result.detected:
            return result

        # Check for flat upper trendline (slope near 0)
        upper_slope = result.pivot_points.get("upper_slope", 0)
        _ = result.pivot_points.get("lower_slope", 0)

        # Ascending triangle: flat top (slope ~0), rising bottom (slope > 0)
        if upper_slope > -0.01:  # Nearly flat or slightly rising
            result.pattern_name = self.name
            return result

        return PatternResult(detected=False, pattern_name=self.name, pattern_type=self.pattern_type)


class DescendingTriangle(SymmetricTriangle):
    """
    Descending Triangle Pattern Detector

    Variation with flat lower trendline and falling upper trendline.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Descending Triangle"

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect Descending Triangle - falling top, flat bottom."""
        result = super().detect(df, i)

        if not result.detected:
            return result

        # Check for flat lower trendline (slope near 0)
        lower_slope = result.pivot_points.get("lower_slope", 0)

        # Descending triangle: falling top (slope < 0), flat bottom (slope ~0)
        if lower_slope < 0.01:  # Nearly flat or slightly falling
            result.pattern_name = self.name
            return result

        return PatternResult(detected=False, pattern_name=self.name, pattern_type=self.pattern_type)
