"""
Spike and Ledge Pattern

Detection Logic:
- Spike Formation: New High or New Low with climax volume
- Spike Condition: High[i] > Max(High[i-20:i-1]) for bullish spike OR Low[i] < Min(Low[i-20:i-1]) for bearish spike
- Ledge Formation: Series of bars with matching highs and lows
- Ledge Condition: Max(High[ledge_bars]) - Min(Low[ledge_bars]) < 0.02 * Price_Range (tight range)
- Ledge Duration: Minimum 3 bars, Maximum 20 bars (as per text)
- Pattern Complete after ledge formation with breakout/breakdown

Entry Rules:
- Long Entry: Buy Stop = High[Ledge_Breakout_Bar] + 0.01
- Short Entry: Sell Stop = Low[Ledge_Breakdown_Bar] - 0.01
- Entry triggered on breakout/breakdown from ledge formation
- Direction opposite to prior spike trend (reversal pattern)

Stop Loss Rules:
- Long Stop: Low[Ledge] - 0.01
- Short Stop: High[Ledge] + 0.01
- Stop placed on opposite side of ledge pattern

Take Profit Rules:
- Target 1: Prior Major Swing Low (for short trades)
- Target 2: Prior Major Swing High (for long trades)
- Alternative Target: Gap levels if gap existed prior to spike
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from ..base import BasePattern, PatternType, SignalDirection, TradeSignal, PatternResult
from ...indicators.pivots import get_recent_swing_high, get_recent_swing_low
from ...indicators.technical import volume_sma, average_range


class SpikeAndLedge(BasePattern):
    """
    Spike and Ledge Pattern Detector
    
    A reversal pattern where a climax spike is followed by a consolidation
    ledge, then a reversal in the opposite direction.
    """
    
    def __init__(
        self,
        spike_lookback: int = 20,
        min_ledge_bars: int = 3,
        max_ledge_bars: int = 20,
        ledge_range_threshold: float = 0.02,
        volume_threshold: float = 2.0,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01
    ):
        """
        Initialize Spike and Ledge pattern detector.
        
        Args:
            spike_lookback: Lookback period for spike detection
            min_ledge_bars: Minimum bars for ledge formation
            max_ledge_bars: Maximum bars for ledge formation
            ledge_range_threshold: Maximum range as % of price for ledge
            volume_threshold: Volume multiplier for climax detection
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
        """
        super().__init__(
            name="Spike and Ledge",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=spike_lookback + min_ledge_bars
        )
        self.spike_lookback = spike_lookback
        self.min_ledge_bars = min_ledge_bars
        self.max_ledge_bars = max_ledge_bars
        self.ledge_range_threshold = ledge_range_threshold
        self.volume_threshold = volume_threshold
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
    
    def _find_spike(
        self,
        df: pd.DataFrame,
        i: int
    ) -> Optional[Dict]:
        """
        Find spike formation at or before bar i.
        
        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            
        Returns:
            Dictionary with spike details or None
        """
        if i < self.spike_lookback:
            return None
        
        # Look for spike in recent bars
        for j in range(i, max(i - self.max_ledge_bars - 1, self.spike_lookback), -1):
            if j < self.spike_lookback:
                continue
            
            current_high = self._safe_float(df.iloc[j]['High'])
            current_low = self._safe_float(df.iloc[j]['Low'])
            current_vol = self._safe_float(df.iloc[j]['Volume'])
            
            # Check for bullish spike (new high)
            prev_highs = [self._safe_float(df.iloc[k]['High']) for k in range(j - self.spike_lookback, j)]
            is_bullish_spike = current_high > max(prev_highs) if prev_highs else False
            
            # Check for bearish spike (new low)
            prev_lows = [self._safe_float(df.iloc[k]['Low']) for k in range(j - self.spike_lookback, j)]
            is_bearish_spike = current_low < min(prev_lows) if prev_lows else False
            
            if not (is_bullish_spike or is_bearish_spike):
                continue
            
            # Check for climax volume
            vol_sma = volume_sma(df['Volume'], 20)
            if j < len(vol_sma):
                avg_vol = self._safe_float(vol_sma.iloc[j])
                is_climax_vol = current_vol > (avg_vol * self.volume_threshold)
            else:
                is_climax_vol = True  # Assume climax if we can't verify
            
            spike_type = 'bullish' if is_bullish_spike else 'bearish'
            
            return {
                'idx': j,
                'high': current_high,
                'low': current_low,
                'volume': current_vol,
                'type': spike_type,
                'is_climax_vol': is_climax_vol
            }
        
        return None
    
    def _find_ledge(
        self,
        df: pd.DataFrame,
        i: int,
        spike: Dict
    ) -> Optional[Dict]:
        """
        Find ledge formation after spike.
        
        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            spike: Spike formation dictionary
            
        Returns:
            Dictionary with ledge details or None
        """
        spike_idx = spike['idx']
        
        # Ledge should form after spike
        if i - spike_idx < self.min_ledge_bars:
            return None
        
        if i - spike_idx > self.max_ledge_bars:
            return None
        
        # Analyze bars between spike and current
        ledge_bars = list(range(spike_idx + 1, i + 1))
        
        if len(ledge_bars) < self.min_ledge_bars:
            return None
        
        # Calculate ledge range
        ledge_highs = [self._safe_float(df.iloc[j]['High']) for j in ledge_bars]
        ledge_lows = [self._safe_float(df.iloc[j]['Low']) for j in ledge_bars]
        
        ledge_high = max(ledge_highs)
        ledge_low = min(ledge_lows)
        ledge_range = ledge_high - ledge_low
        
        # Average price for threshold calculation
        avg_price = (ledge_high + ledge_low) / 2
        threshold = avg_price * self.ledge_range_threshold
        
        # Check if range is tight enough
        if ledge_range > threshold:
            return None
        
        # Check volume contraction during ledge
        spike_vol = spike['volume']
        ledge_vols = [self._safe_float(df.iloc[j]['Volume']) for j in ledge_bars]
        avg_ledge_vol = np.mean(ledge_vols)
        
        volume_contracting = avg_ledge_vol < spike_vol
        
        return {
            'start_idx': spike_idx + 1,
            'end_idx': i,
            'num_bars': len(ledge_bars),
            'high': ledge_high,
            'low': ledge_low,
            'range': ledge_range,
            'volume_contracting': volume_contracting,
            'bars': ledge_bars
        }
    
    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Spike and Ledge pattern at bar index i.
        
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
        
        # Find spike
        spike = self._find_spike(df, i)
        
        if spike is None:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        # Find ledge
        ledge = self._find_ledge(df, i, spike)
        
        if ledge is None:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={'spike_detected': True, 'spike': spike}
            )
        
        # Check for breakout from ledge (reversal direction)
        current_close = self._safe_float(df.iloc[i]['Close'])
        current_high = self._safe_float(df.iloc[i]['High'])
        current_low = self._safe_float(df.iloc[i]['Low'])
        
        # For bullish spike, look for breakdown (reversal)
        # For bearish spike, look for breakout (reversal)
        if spike['type'] == 'bullish':
            # Expect breakdown for reversal
            breakdown = current_close < ledge['low']
            if not breakdown:
                return PatternResult(
                    detected=False,
                    pattern_name=self.name,
                    pattern_type=self.pattern_type,
                    pivot_points={'spike': spike, 'ledge': ledge, 'awaiting_breakdown': True}
                )
            direction = 'short'
        else:
            # Expect breakout for reversal
            breakout = current_close > ledge['high']
            if not breakout:
                return PatternResult(
                    detected=False,
                    pattern_name=self.name,
                    pattern_type=self.pattern_type,
                    pivot_points={'spike': spike, 'ledge': ledge, 'awaiting_breakout': True}
                )
            direction = 'long'
        
        # Generate signal
        signal = self._generate_signal(df, i, spike, ledge, direction)
        
        return PatternResult(
            detected=True,
            pattern_name=f"{self.name} ({spike['type'].title()} Spike)",
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                'spike_idx': spike['idx'],
                'spike_type': spike['type'],
                'spike_high': spike['high'],
                'spike_low': spike['low'],
                'ledge_high': ledge['high'],
                'ledge_low': ledge['low'],
                'ledge_bars': ledge['num_bars'],
                'direction': direction
            },
            bars_since_detection=0,
            start_index=spike['idx'],
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
        spike: Dict,
        ledge: Dict,
        direction: str
    ) -> Optional[TradeSignal]:
        """Generate trade signal for Spike and Ledge reversal."""
        
        current_high = self._safe_float(df.iloc[i]['High'])
        current_low = self._safe_float(df.iloc[i]['Low'])
        
        if direction == 'long':
            # Breakout from ledge after bearish spike
            entry_price = ledge['high'] + self.entry_offset
            stop_loss = ledge['low'] - self.stop_offset
            
            # Target prior swing high
            swing_high = get_recent_swing_high(df, i, lookback=50)
            if swing_high:
                take_profit_1 = swing_high[1]
            else:
                risk = entry_price - stop_loss
                take_profit_1 = entry_price + (risk * 2)
            
            take_profit_2 = entry_price + (entry_price - stop_loss) * 3
            
            signal_direction = SignalDirection.LONG
        
        else:  # direction == 'short'
            # Breakdown from ledge after bullish spike
            entry_price = ledge['low'] - self.entry_offset
            stop_loss = ledge['high'] + self.stop_offset
            
            # Target prior swing low
            swing_low = get_recent_swing_low(df, i, lookback=50)
            if swing_low:
                take_profit_1 = swing_low[1]
            else:
                risk = stop_loss - entry_price
                take_profit_1 = entry_price - (risk * 2)
            
            take_profit_2 = entry_price - (stop_loss - entry_price) * 3
            
            signal_direction = SignalDirection.SHORT
        
        # Confidence
        confidence = 0.55
        if spike['is_climax_vol']:
            confidence += 0.1
        if ledge['volume_contracting']:
            confidence += 0.1
        
        return TradeSignal(
            pattern_name=f"{self.name} ({direction.title()})",
            direction=signal_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], 'name') else None,
            metadata={
                'spike_type': spike['type'],
                'spike_idx': spike['idx'],
                'ledge_bars': ledge['num_bars'],
                'ledge_range': ledge['range'],
                'volume_contracting': ledge['volume_contracting'],
                'climax_volume': spike['is_climax_vol'],
                'entry_type': 'buy_stop' if direction == 'long' else 'sell_stop'
            }
        )
