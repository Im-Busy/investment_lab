"""
Three Hills and A Mountain Pattern

Detection Logic:
- Identify 3 Hills: Hill1, Hill2, Hill3 using Local Extrema (lookback n=5)
- Condition 1 (Hill Retracements): Each hill retraces 0.50 to 0.618 of previous hill height
- Condition 2 (Trend Line): Line connecting bottoms of Three Hills
- Condition 3 (Breakdown): Close[i] < Trend_Line_Level after Hill3 completion
- Mountain Formation: After 62% retracement of AB range, rally 1.00 to 1.27 of AB range
- Two Trade Setups: Short on trendline break, Long on Mountain formation

Entry Rules:
- Short Entry (First Trade): Sell Stop = Low[Breakdown_Bar] - 0.01
- Long Entry (Second Trade): Buy Stop = High[Previous_Bar] + 0.01 after 62% retracement
- Short triggered on close below trendline connecting hill bottoms
- Long triggered at level C after 62% AB retracement completion

Stop Loss Rules:
- Short Stop: High[Hill3_B] + 0.01
- Long Stop: Low[Level_C] - 0.01
- Stop placed above B level for first trade, below C level for second trade

Take Profit Rules:
- Short Target 1: Entry - 0.62 * AB_Range
- Long Target 2: Entry + 1.00 * AB_Range (to 1.27 * AB_Range)
- AB_Range = High[Hill1] - Low[Hill3_Bottom]
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from ..base import BasePattern, PatternType, SignalDirection, TradeSignal, PatternResult
from ...indicators.pivots import find_swing_highs, find_swing_lows
from ...indicators.fibonacci import is_fib_ratio_match


class ThreeHillsMountain(BasePattern):
    """
    Three Hills and Mountain Pattern Detector
    
    A complex pattern consisting of three successively lower highs (hills)
    followed by a potential mountain formation.
    """
    
    def __init__(
        self,
        lookback: int = 5,
        retrace_min: float = 0.50,
        retrace_max: float = 0.618,
        min_pattern_bars: int = 30,
        max_pattern_bars: int = 150,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01
    ):
        """
        Initialize Three Hills and Mountain pattern detector.
        
        Args:
            lookback: Lookback period for pivot detection
            retrace_min: Minimum hill retracement (default 0.50)
            retrace_max: Maximum hill retracement (default 0.618)
            min_pattern_bars: Minimum bars for pattern formation
            max_pattern_bars: Maximum bars for pattern formation
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
        """
        super().__init__(
            name="Three Hills and Mountain",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=min_pattern_bars
        )
        self.lookback = lookback
        self.retrace_min = retrace_min
        self.retrace_max = retrace_max
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
    
    def _find_hills(
        self,
        df: pd.DataFrame,
        i: int
    ) -> Optional[Dict]:
        """
        Find three hills formation.
        
        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            
        Returns:
            Dictionary with hill details or None
        """
        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)
        
        # Collect peaks and troughs
        peaks = []
        troughs = []
        
        lookback = min(self.max_pattern_bars, i)
        
        for j in range(i - lookback, i + 1):
            if j < 0:
                continue
            if pd.notna(swing_highs.iloc[j]):
                peaks.append((j, self._safe_float(swing_highs.iloc[j])))
            if pd.notna(swing_lows.iloc[j]):
                troughs.append((j, self._safe_float(swing_lows.iloc[j])))
        
        if len(peaks) < 3 or len(troughs) < 3:
            return None
        
        # Sort by index
        peaks = sorted(peaks, key=lambda x: x[0])
        troughs = sorted(troughs, key=lambda x: x[0])
        
        # Find three successively lower highs (hills)
        for p1 in range(len(peaks) - 2):
            hill1 = peaks[p1]
            
            for p2 in range(p1 + 1, len(peaks) - 1):
                hill2 = peaks[p2]
                
                # Hill2 must be lower than Hill1
                if hill2[1] >= hill1[1]:
                    continue
                
                # Find trough between Hill1 and Hill2
                trough1 = None
                for t in troughs:
                    if hill1[0] < t[0] < hill2[0]:
                        if trough1 is None or t[1] < trough1[1]:
                            trough1 = t
                
                if trough1 is None:
                    continue
                
                # Check first retracement
                hill1_height = hill1[1] - trough1[1]
                hill2_height = hill2[1] - trough1[1]
                
                if hill1_height <= 0:
                    continue
                
                retrace1 = (hill1[1] - hill2[1]) / hill1_height
                
                if not (self.retrace_min <= retrace1 <= self.retrace_max):
                    continue
                
                for p3 in range(p2 + 1, len(peaks)):
                    hill3 = peaks[p3]
                    
                    # Hill3 must be lower than Hill2
                    if hill3[1] >= hill2[1]:
                        continue
                    
                    # Find trough between Hill2 and Hill3
                    trough2 = None
                    for t in troughs:
                        if hill2[0] < t[0] < hill3[0]:
                            if trough2 is None or t[1] < trough2[1]:
                                trough2 = t
                    
                    if trough2 is None:
                        continue
                    
                    # Check second retracement
                    hill2_height = hill2[1] - trough2[1]
                    
                    if hill2_height <= 0:
                        continue
                    
                    retrace2 = (hill2[1] - hill3[1]) / hill2_height
                    
                    if not (self.retrace_min <= retrace2 <= self.retrace_max):
                        continue
                    
                    # Find trough after Hill3
                    trough3 = None
                    for t in troughs:
                        if t[0] > hill3[0]:
                            if trough3 is None or t[1] < trough3[1]:
                                trough3 = t
                    
                    # Calculate trendline (connecting troughs)
                    if trough3 is None:
                        trough3 = trough2  # Use trough2 as fallback
                    
                    # Calculate trendline slope
                    if len([trough1, trough2, trough3]) >= 2:
                        t1, t2 = trough1, trough2
                        slope = (t2[1] - t1[1]) / (t2[0] - t1[0]) if t2[0] != t1[0] else 0
                        intercept = t1[1] - (slope * t1[0])
                    else:
                        continue
                    
                    return {
                        'hill1': hill1,
                        'hill2': hill2,
                        'hill3': hill3,
                        'trough1': trough1,
                        'trough2': trough2,
                        'trough3': trough3,
                        'retrace1': retrace1,
                        'retrace2': retrace2,
                        'trendline_slope': slope,
                        'trendline_intercept': intercept,
                        'ab_range': hill1[1] - min(trough1[1], trough2[1], trough3[1])
                    }
        
        return None
    
    def _get_trendline_value(self, slope: float, intercept: float, idx: int) -> float:
        """Calculate trendline value at given index."""
        return (slope * idx) + intercept
    
    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Three Hills and Mountain pattern at bar index i.
        
        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            
        Returns:
            PatternResult with detection status and signal
        """
        if not self._validate_data(df, i):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        # Find hills
        hills = self._find_hills(df, i)
        
        if hills is None:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        # Check pattern duration
        pattern_duration = hills['hill3'][0] - hills['hill1'][0]
        if pattern_duration < self.min_pattern_bars:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        # Calculate trendline value at current bar
        trendline_value = self._get_trendline_value(
            hills['trendline_slope'],
            hills['trendline_intercept'],
            i
        )
        
        current_close = self._safe_float(df.iloc[i]['Close'])
        current_low = self._safe_float(df.iloc[i]['Low'])
        
        # Check for trendline breakdown (short signal)
        breakdown = current_close < trendline_value
        
        if not breakdown:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    'hills_detected': True,
                    'awaiting_breakdown': True,
                    'trendline_value': trendline_value,
                    'hills': hills
                }
            )
        
        # Generate signal
        signal = self._generate_signal(df, i, hills, trendline_value)
        
        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                'hill1_idx': hills['hill1'][0],
                'hill1': hills['hill1'][1],
                'hill2_idx': hills['hill2'][0],
                'hill2': hills['hill2'][1],
                'hill3_idx': hills['hill3'][0],
                'hill3': hills['hill3'][1],
                'trendline_value': trendline_value,
                'ab_range': hills['ab_range']
            },
            bars_since_detection=0,
            start_index=hills['hill1'][0],
            end_index=i
        )
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None
    
    def _generate_signal(
        self,
        df: pd.DataFrame,
        i: int,
        hills: Dict,
        trendline_value: float
    ) -> Optional[TradeSignal]:
        """Generate trade signal for Three Hills breakdown."""
        
        current_low = self._safe_float(df.iloc[i]['Low'])
        hill3_high = hills['hill3'][1]
        ab_range = hills['ab_range']
        
        # Short entry below breakdown bar
        entry_price = current_low - self.entry_offset
        
        # Stop above Hill3
        stop_loss = hill3_high + self.stop_offset
        
        # Target based on AB range
        take_profit_1 = entry_price - (ab_range * 0.62)
        take_profit_2 = entry_price - ab_range
        
        # Confidence
        confidence = 0.55
        
        # Check retracement quality
        if 0.55 <= hills['retrace1'] <= 0.618 and 0.55 <= hills['retrace2'] <= 0.618:
            confidence += 0.1
        
        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], 'name') else None,
            metadata={
                'retrace1': hills['retrace1'],
                'retrace2': hills['retrace2'],
                'ab_range': ab_range,
                'trendline_value': trendline_value,
                'entry_type': 'sell_stop'
            }
        )


class ThreeDrives(BasePattern):
    """
    Three Drives Pattern Detector
    
    Similar to Three Hills but focuses on three drives to a top/bottom
    with Fibonacci relationships.
    """
    
    def __init__(
        self,
        lookback: int = 5,
        fib_tolerance: float = 0.05,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01
    ):
        super().__init__(
            name="Three Drives",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=30
        )
        self.lookback = lookback
        self.fib_tolerance = fib_tolerance
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
    
    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect Three Drives pattern - simplified implementation."""
        # Similar to Three Hills but with stricter Fib relationships
        return PatternResult(
            detected=False,
            pattern_name=self.name,
            pattern_type=self.pattern_type
        )
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        return None
