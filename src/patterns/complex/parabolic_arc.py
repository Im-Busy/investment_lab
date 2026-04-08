"""
Parabolic Arc Pattern

Detection Logic:
- Parabolic Base: Price consolidation before vertical move
- Vertical Move: Accelerated price movement (steep slope)
- Condition 1 (Acceleration Stop): Price momentum decreases after vertical rally
- Condition 2 (Lower Lows): Price shows lower-lows after peak formation
- Condition 3 (Failed Test): Second failed attempt to test previous peak (similar to 2B setup)
- Condition 4 (Trendline Break): Breakdown of trendline connecting major swing lows
- Significant Correction: 62% retracement from top expected

Entry Rules:
- Short Entry: Sell Stop = Low[First_Peak_Test] - 0.01
- Entry triggered on second failed attempt to test peak
- Alternative Entry: Trendline breakdown connecting major swing lows
- Entry valid after acceleration comes to stop and reverses

Stop Loss Rules:
- Stop Loss: High[Second_Peak] + 0.01
- Stop placed few ticks above high point of Parabolic Arc
- Protective stop above close of second peak

Take Profit Rules:
- Target 1: Entry - 0.62 * Parabolic_Range
- Parabolic_Range = High[Parabolic_Peak] - Low[Parabolic_Base]
- Target set at 62% of entire range to first peak
- Alternative Target: Major swing low prior to parabolic move
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from ..base import BasePattern, PatternType, SignalDirection, TradeSignal, PatternResult
from ...indicators.pivots import find_swing_highs, find_swing_lows, get_recent_swing_low
from ...indicators.technical import volume_sma


class ParabolicArc(BasePattern):
    """
    Parabolic Arc Pattern Detector
    
    Identifies parabolic price movements and their eventual reversal
    when momentum decelerates and price fails to make new highs.
    """
    
    def __init__(
        self,
        min_base_bars: int = 10,
        min_vertical_bars: int = 10,
        steep_threshold: float = 0.03,
        failed_test_threshold: float = 0.03,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = False
    ):
        """
        Initialize Parabolic Arc pattern detector.
        
        Args:
            min_base_bars: Minimum bars for consolidation base
            min_vertical_bars: Minimum bars for vertical move
            steep_threshold: Minimum slope for parabolic move (per bar)
            failed_test_threshold: Threshold for failed peak test (3%)
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            volume_filter: Require volume confirmation
        """
        super().__init__(
            name="Parabolic Arc",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=min_base_bars + min_vertical_bars
        )
        self.min_base_bars = min_base_bars
        self.min_vertical_bars = min_vertical_bars
        self.steep_threshold = steep_threshold
        self.failed_test_threshold = failed_test_threshold
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.volume_filter = volume_filter
    
    def _find_parabolic_move(
        self,
        df: pd.DataFrame,
        i: int
    ) -> Optional[Dict]:
        """
        Find parabolic price move.
        
        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            
        Returns:
            Dictionary with parabolic move details or None
        """
        if i < self.min_base_bars + self.min_vertical_bars:
            return None
        
        swing_highs = find_swing_highs(df, 5)
        swing_lows = find_swing_lows(df, 5)
        
        # Collect peaks
        peaks = []
        for j in range(max(0, i - 100), i + 1):
            if pd.notna(swing_highs.iloc[j]):
                peaks.append((j, self._safe_float(swing_highs.iloc[j])))
        
        if len(peaks) < 2:
            return None
        
        # Find the highest peak (parabolic top)
        highest_peak = max(peaks, key=lambda x: x[1])
        peak_idx = highest_peak[0]
        peak_price = highest_peak[1]
        
        # Check if we're after the peak
        if i - peak_idx < 2:
            return None
        
        # Find the base (consolidation before parabolic move)
        base_lows = []
        for j in range(max(0, peak_idx - 60), peak_idx):
            if pd.notna(swing_lows.iloc[j]):
                base_lows.append((j, self._safe_float(swing_lows.iloc[j])))
        
        if not base_lows:
            return None
        
        # Find significant low before the parabolic move
        base_low = min(base_lows, key=lambda x: x[1])
        base_idx = base_low[0]
        base_price = base_low[1]
        
        # Calculate the slope of the move
        vertical_bars = peak_idx - base_idx
        if vertical_bars < self.min_vertical_bars:
            return None
        
        price_change = peak_price - base_price
        avg_slope = price_change / vertical_bars / base_price  # Normalized slope
        
        # Check if slope is steep enough (parabolic)
        if avg_slope < self.steep_threshold:
            return None
        
        # Calculate parabolic range
        parabolic_range = peak_price - base_price
        
        return {
            'peak_idx': peak_idx,
            'peak_price': peak_price,
            'base_idx': base_idx,
            'base_price': base_price,
            'vertical_bars': vertical_bars,
            'avg_slope': avg_slope,
            'parabolic_range': parabolic_range,
            'peaks': peaks
        }
    
    def _check_failed_test(
        self,
        df: pd.DataFrame,
        i: int,
        para: Dict
    ) -> Optional[Dict]:
        """
        Check for failed test of parabolic peak.
        
        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            para: Parabolic move dictionary
            
        Returns:
            Dictionary with failed test details or None
        """
        peak_idx = para['peak_idx']
        peak_price = para['peak_price']
        
        # Look for test of peak after the initial decline
        tests = []
        for j in range(peak_idx + 1, i + 1):
            high = self._safe_float(df.iloc[j]['High'])
            
            # Check if price approached the peak
            approach_ratio = abs(high - peak_price) / peak_price
            
            if approach_ratio < self.failed_test_threshold:
                # This is a test of the peak
                tests.append((j, high))
        
        if len(tests) < 2:
            return None
        
        # Check for failed second test (lower high than first test)
        first_test = tests[0]
        second_test = tests[-1]
        
        # Second test should be lower than first test
        if second_test[1] >= first_test[1]:
            return None
        
        # Check for momentum decrease (lower close on second test)
        first_close = self._safe_float(df.iloc[first_test[0]]['Close'])
        second_close = self._safe_float(df.iloc[second_test[0]]['Close'])
        
        momentum_decreasing = second_close < first_close
        
        return {
            'first_test_idx': first_test[0],
            'first_test_price': first_test[1],
            'second_test_idx': second_test[0],
            'second_test_price': second_test[1],
            'momentum_decreasing': momentum_decreasing
        }
    
    def _check_trendline_break(
        self,
        df: pd.DataFrame,
        i: int,
        para: Dict
    ) -> bool:
        """
        Check for trendline break connecting swing lows.
        
        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            para: Parabolic move dictionary
            
        Returns:
            True if trendline broken
        """
        swing_lows = find_swing_lows(df, 5)
        
        # Find swing lows connecting the parabolic move
        lows = []
        for j in range(para['base_idx'], para['peak_idx']):
            if pd.notna(swing_lows.iloc[j]):
                lows.append((j, self._safe_float(swing_lows.iloc[j])))
        
        if len(lows) < 2:
            return False
        
        # Use first and last low to draw trendline
        first_low = lows[0]
        last_low = lows[-1]
        
        # Calculate trendline
        slope = (last_low[1] - first_low[1]) / (last_low[0] - first_low[0]) if last_low[0] != first_low[0] else 0
        intercept = first_low[1] - (slope * first_low[0])
        
        # Get trendline value at current bar
        trendline_value = (slope * i) + intercept
        
        # Check if close broke below trendline
        current_close = self._safe_float(df.iloc[i]['Close'])
        
        return current_close < trendline_value
    
    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Parabolic Arc pattern at bar index i.
        
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
        
        # Find parabolic move
        para = self._find_parabolic_move(df, i)
        
        if para is None:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        # Check for failed test of peak
        failed_test = self._check_failed_test(df, i, para)
        
        # Check for trendline break
        trendline_break = self._check_trendline_break(df, i, para)
        
        # Pattern confirmed if either failed test or trendline break
        if failed_test is None and not trendline_break:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    'parabolic_detected': True,
                    'peak_idx': para['peak_idx'],
                    'peak_price': para['peak_price'],
                    'awaiting_reversal': True
                }
            )
        
        # Generate signal
        signal = self._generate_signal(df, i, para, failed_test)
        
        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                'peak_idx': para['peak_idx'],
                'peak_price': para['peak_price'],
                'base_idx': para['base_idx'],
                'base_price': para['base_price'],
                'parabolic_range': para['parabolic_range'],
                'failed_test': failed_test is not None,
                'trendline_break': trendline_break
            },
            bars_since_detection=0,
            start_index=para['base_idx'],
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
        para: Dict,
        failed_test: Optional[Dict]
    ) -> Optional[TradeSignal]:
        """Generate trade signal for Parabolic Arc reversal."""
        
        current_low = self._safe_float(df.iloc[i]['Low'])
        current_high = self._safe_float(df.iloc[i]['High'])
        peak_price = para['peak_price']
        parabolic_range = para['parabolic_range']
        
        # Entry below current low (breakdown)
        entry_price = current_low - self.entry_offset
        
        # Stop above the peak or second test high
        if failed_test:
            stop_price = failed_test['second_test_price']
        else:
            stop_price = peak_price
        stop_loss = stop_price + self.stop_offset
        
        # Target 62% of parabolic range
        take_profit_1 = entry_price - (parabolic_range * 0.62)
        take_profit_2 = entry_price - parabolic_range
        
        # Find prior swing low for alternative target
        swing_low = get_recent_swing_low(df, para['base_idx'], lookback=50)
        if swing_low:
            take_profit_3 = swing_low[1]
        else:
            take_profit_3 = None
        
        # Volume confirmation
        volume_confirmed = True
        if self.volume_filter:
            vol_sma = volume_sma(df['Volume'], 20)
            if i < len(vol_sma):
                # Check for volume spike during parabolic move
                peak_vol = self._safe_float(df.iloc[para['peak_idx']]['Volume'])
                avg_vol = self._safe_float(vol_sma.iloc[i])
                volume_confirmed = peak_vol > avg_vol * 1.5
        
        # Confidence
        confidence = 0.55
        if failed_test and failed_test.get('momentum_decreasing'):
            confidence += 0.1
        if volume_confirmed:
            confidence += 0.05
        
        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=min(confidence, 1.0),
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], 'name') else None,
            metadata={
                'peak_price': peak_price,
                'parabolic_range': parabolic_range,
                'vertical_bars': para['vertical_bars'],
                'avg_slope': para['avg_slope'],
                'failed_test': failed_test is not None,
                'entry_type': 'sell_stop'
            }
        )


class ParabolicBase(BasePattern):
    """
    Parabolic Base Pattern Detector
    
    Identifies consolidation bases before parabolic moves.
    Used for potential long entries before vertical price moves.
    """
    
    def __init__(
        self,
        min_base_bars: int = 15,
        max_base_bars: int = 60,
        volatility_threshold: float = 0.02,
        entry_offset: float = 0.01
    ):
        super().__init__(
            name="Parabolic Base",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=min_base_bars
        )
        self.min_base_bars = min_base_bars
        self.max_base_bars = max_base_bars
        self.volatility_threshold = volatility_threshold
        self.entry_offset = entry_offset
    
    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect Parabolic Base pattern."""
        if not self._validate_data(df, i):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        # Look for tight consolidation
        lookback = min(self.max_base_bars, i)
        
        base_highs = []
        base_lows = []
        
        for j in range(i - lookback, i + 1):
            if j < 0:
                continue
            base_highs.append(self._safe_float(df.iloc[j]['High']))
            base_lows.append(self._safe_float(df.iloc[j]['Low']))
        
        if len(base_highs) < self.min_base_bars:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        base_high = max(base_highs)
        base_low = min(base_lows)
        base_range = base_high - base_low
        avg_price = (base_high + base_low) / 2
        
        # Check for tight range (low volatility)
        volatility = base_range / avg_price if avg_price > 0 else 1
        
        if volatility > self.volatility_threshold:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        # Check for breakout from base
        current_close = self._safe_float(df.iloc[i]['Close'])
        current_high = self._safe_float(df.iloc[i]['High'])
        
        breakout = current_close > base_high
        
        if not breakout:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    'base_high': base_high,
                    'base_low': base_low,
                    'volatility': volatility,
                    'awaiting_breakout': True
                }
            )
        
        # Generate signal
        entry_price = current_high + self.entry_offset
        stop_loss = base_low - self.entry_offset
        risk = entry_price - stop_loss
        
        signal = TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=entry_price + (risk * 2),
            take_profit_2=entry_price + (risk * 3),
            take_profit_3=None,
            confidence=0.55,
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], 'name') else None,
            metadata={
                'base_range': base_range,
                'volatility': volatility,
                'entry_type': 'buy_stop'
            }
        )
        
        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                'base_high': base_high,
                'base_low': base_low,
                'base_range': base_range,
                'volatility': volatility
            },
            bars_since_detection=0,
            start_index=i - lookback,
            end_index=i
        )
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None
