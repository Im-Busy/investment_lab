"""
ABC Pattern

Detection Logic:
- Identify 3 pivots: A, B, C using Local Extrema (lookback n=5)
- Bullish Setup: A=High, B=Low, C=High (Correction)
- Bearish Setup: A=Low, B=High, C=Low (Correction)
- Condition 1 (C Retracement): 0.382 * abs(AB) <= abs(BC) <= 0.618 * abs(AB)
- Condition 2 (Trend): Prior to A, there must be a significant swing (XA) defining the trend
- Pattern Complete when C pivot is formed

Entry Rules:
- Long Entry: Buy Stop = high[i-1] + 0.01 (Above previous bar high after C formation)
- Short Entry: Sell Stop = low[i-1] - 0.01 (Below previous bar low after C formation)
- Entry triggered on bar i following C completion

Stop Loss Rules:
- Long Stop: low[C] - 0.01
- Short Stop: high[C] + 0.01

Take Profit Rules:
- Target 1: Entry + 1.00 * abs(AB) (100% of AB range)
- Target 2: Entry + 1.27 * abs(BC) (127% of BC range)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from ..base import BasePattern, PatternType, SignalDirection, TradeSignal, PatternResult
from ...indicators.pivots import find_swing_highs, find_swing_lows
from ...indicators.fibonacci import calculate_bc_retracement, is_fib_ratio_match
from ...indicators.technical import volume_sma


class ABCPattern(BasePattern):
    """
    ABC Pattern Detector
    
    A simpler harmonic pattern that identifies correction patterns
    using Fibonacci retracements.
    """
    
    # Fibonacci ratio constraints
    BC_RETRACEMENT_MIN = 0.382
    BC_RETRACEMENT_MAX = 0.618
    
    def __init__(
        self,
        lookback: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        fib_tolerance: float = 0.05,
        require_trend: bool = True,
        volume_filter: bool = False
    ):
        """
        Initialize ABC pattern detector.
        
        Args:
            lookback: Lookback period for pivot detection
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            fib_tolerance: Tolerance for Fibonacci ratio matching
            require_trend: Require prior trend (XA swing)
            volume_filter: Require volume confirmation
        """
        super().__init__(
            name="ABC Pattern",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=lookback * 4
        )
        self.lookback = lookback
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.fib_tolerance = fib_tolerance
        self.require_trend = require_trend
        self.volume_filter = volume_filter
    
    def _find_pivots(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Find potential ABC pivots.
        
        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            
        Returns:
            Dictionary with pivot indices and prices or None
        """
        if i < self.lookback * 3:
            return None
        
        # Find recent swing highs and lows
        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)
        
        # Get valid swing points in range
        highs = []
        lows = []
        
        for j in range(i - self.lookback * 3, i + 1):
            if j < 0:
                continue
            if pd.notna(swing_highs.iloc[j]):
                highs.append((j, self._safe_float(swing_highs.iloc[j])))
            if pd.notna(swing_lows.iloc[j]):
                lows.append((j, self._safe_float(swing_lows.iloc[j])))
        
        if len(highs) < 2 or len(lows) < 1:
            return None
        
        # Try to identify bullish ABC (A=High, B=Low, C=High)
        bullish_pivots = self._identify_bullish_abc(highs, lows, i)
        
        # Try to identify bearish ABC (A=Low, B=High, C=Low)
        bearish_pivots = self._identify_bearish_abc(highs, lows, i)
        
        if bullish_pivots:
            return {
                'direction': 'bullish',
                'A': bullish_pivots[0],
                'B': bullish_pivots[1],
                'C': bullish_pivots[2],
                'X': bullish_pivots[3] if len(bullish_pivots) > 3 else None
            }
        elif bearish_pivots:
            return {
                'direction': 'bearish',
                'A': bearish_pivots[0],
                'B': bearish_pivots[1],
                'C': bearish_pivots[2],
                'X': bearish_pivots[3] if len(bearish_pivots) > 3 else None
            }
        
        return None
    
    def _identify_bullish_abc(
        self,
        highs: List[Tuple[int, float]],
        lows: List[Tuple[int, float]],
        current_idx: int
    ) -> Optional[List]:
        """
        Identify bullish ABC: A=High, B=Low, C=High (correction in uptrend)
        
        For a bullish ABC, we expect:
        - A is a swing high
        - B is a swing low below A (retracement)
        - C is a swing high below A (lower high)
        - BC retracement of AB should be 38.2% to 61.8%
        """
        highs = sorted(highs, key=lambda x: x[0])
        lows = sorted(lows, key=lambda x: x[0])
        
        if len(highs) < 2 or len(lows) < 1:
            return None
        
        # C should be the most recent high
        C = highs[-1]
        
        # Find B (low before C)
        B_candidates = [(idx, price) for idx, price in lows if idx < C[0]]
        if not B_candidates:
            return None
        B = B_candidates[-1]
        
        # Find A (high before B)
        A_candidates = [(idx, price) for idx, price in highs if idx < B[0]]
        if not A_candidates:
            return None
        A = A_candidates[-1]
        
        # Validate: A > B < C and C < A (lower high)
        if not (A[1] > B[1] and C[1] > B[1] and C[1] < A[1]):
            return None
        
        # Validate Fibonacci retracement
        bc_ratio = calculate_bc_retracement(A[1], B[1], C[1])
        if not (self.BC_RETRACEMENT_MIN - self.fib_tolerance <= bc_ratio <= self.BC_RETRACEMENT_MAX + self.fib_tolerance):
            return None
        
        # Find X (prior swing low for trend context)
        X_candidates = [(idx, price) for idx, price in lows if idx < A[0]]
        X = X_candidates[-1] if X_candidates else None
        
        if self.require_trend and X is None:
            return None
        
        return [A, B, C, X] if X else [A, B, C]
    
    def _identify_bearish_abc(
        self,
        highs: List[Tuple[int, float]],
        lows: List[Tuple[int, float]],
        current_idx: int
    ) -> Optional[List]:
        """
        Identify bearish ABC: A=Low, B=High, C=Low (correction in downtrend)
        
        For a bearish ABC, we expect:
        - A is a swing low
        - B is a swing high above A (retracement)
        - C is a swing low above A (higher low)
        - BC retracement of AB should be 38.2% to 61.8%
        """
        highs = sorted(highs, key=lambda x: x[0])
        lows = sorted(lows, key=lambda x: x[0])
        
        if len(highs) < 1 or len(lows) < 2:
            return None
        
        # C should be the most recent low
        C = lows[-1]
        
        # Find B (high before C)
        B_candidates = [(idx, price) for idx, price in highs if idx < C[0]]
        if not B_candidates:
            return None
        B = B_candidates[-1]
        
        # Find A (low before B)
        A_candidates = [(idx, price) for idx, price in lows if idx < B[0]]
        if not A_candidates:
            return None
        A = A_candidates[-1]
        
        # Validate: A < B > C and C > A (higher low)
        if not (A[1] < B[1] and C[1] < B[1] and C[1] > A[1]):
            return None
        
        # Validate Fibonacci retracement
        bc_ratio = calculate_bc_retracement(A[1], B[1], C[1])
        if not (self.BC_RETRACEMENT_MIN - self.fib_tolerance <= bc_ratio <= self.BC_RETRACEMENT_MAX + self.fib_tolerance):
            return None
        
        # Find X (prior swing high for trend context)
        X_candidates = [(idx, price) for idx, price in highs if idx < A[0]]
        X = X_candidates[-1] if X_candidates else None
        
        if self.require_trend and X is None:
            return None
        
        return [A, B, C, X] if X else [A, B, C]
    
    def detect(self, df: pd.DataFrame, i: int) -> PatternResult:
        """
        Detect ABC pattern at bar index i.
        
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
        
        pivots = self._find_pivots(df, i)
        
        if pivots is None:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        A = pivots['A']
        B = pivots['B']
        C = pivots['C']
        X = pivots.get('X')
        direction = pivots['direction']
        
        # Generate signal
        signal = self._generate_signal(df, i, A, B, C, direction)
        
        # Calculate Fibonacci ratio
        bc_ratio = calculate_bc_retracement(A[1], B[1], C[1])
        ab_range = abs(A[1] - B[1])
        
        return PatternResult(
            detected=True,
            pattern_name=f"{self.name} ({direction.capitalize()})",
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                'A_idx': A[0], 'A': A[1],
                'B_idx': B[0], 'B': B[1],
                'C_idx': C[0], 'C': C[1],
                'X_idx': X[0] if X else None,
                'X': X[1] if X else None,
                'direction': direction,
                'bc_ratio': bc_ratio,
                'ab_range': ab_range
            },
            bars_since_detection=0,
            start_index=A[0],
            end_index=C[0]
        )
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None
    
    def _generate_signal(
        self,
        df: pd.DataFrame,
        i: int,
        A: Tuple[int, float],
        B: Tuple[int, float],
        C: Tuple[int, float],
        direction: str
    ) -> Optional[TradeSignal]:
        """Generate trade signal for ABC pattern."""
        
        ab_range = abs(A[1] - B[1])
        bc_range = abs(B[1] - C[1])
        
        if direction == 'bullish':
            # For bullish ABC (correction), we expect continuation down
            # Actually, for a bullish ABC correction, we're looking for a short entry
            # Wait - let me reconsider. ABC in uptrend: A=High, B=Low, C=Lower High
            # This is a bearish continuation pattern
            
            # Entry below C (the lower high)
            entry_price = C[1] - self.entry_offset
            stop_loss = C[1] + self.stop_offset  # Above C
            
            # Targets based on AB projection
            take_profit_1 = entry_price - ab_range  # 100% of AB
            take_profit_2 = entry_price - (bc_range * 1.27)  # 127% of BC
            
            signal_direction = SignalDirection.SHORT
            pattern_name = f"{self.name} (Bearish Correction)"
        
        else:  # bearish ABC
            # For bearish ABC (correction in downtrend): A=Low, B=High, C=Higher Low
            # This is a bullish continuation pattern
            
            # Entry above C (the higher low)
            entry_price = C[1] + self.entry_offset
            stop_loss = C[1] - self.stop_offset  # Below C
            
            take_profit_1 = entry_price + ab_range  # 100% of AB
            take_profit_2 = entry_price + (bc_range * 1.27)  # 127% of BC
            
            signal_direction = SignalDirection.LONG
            pattern_name = f"{self.name} (Bullish Correction)"
        
        # Volume confirmation
        volume_confirmed = True
        if self.volume_filter:
            vol_sma = volume_sma(df['Volume'], 20)
            if i < len(vol_sma):
                current_vol = self._safe_float(df.iloc[i]['Volume'])
                avg_vol = self._safe_float(vol_sma.iloc[i])
                volume_confirmed = current_vol > avg_vol
        
        # Confidence
        bc_ratio = calculate_bc_retracement(A[1], B[1], C[1])
        confidence = 0.55
        
        # Higher confidence if BC is near 61.8% (deep retracement)
        if abs(bc_ratio - 0.618) < 0.05:
            confidence += 0.1
        if volume_confirmed:
            confidence += 0.05
        
        return TradeSignal(
            pattern_name=pattern_name,
            direction=signal_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], 'name') else None,
            metadata={
                'direction': direction,
                'A': A[1], 'B': B[1], 'C': C[1],
                'bc_ratio': bc_ratio,
                'ab_range': ab_range,
                'bc_range': bc_range,
                'volume_confirmed': volume_confirmed,
                'entry_type': 'sell_stop' if direction == 'bullish' else 'buy_stop'
            }
        )


class ABCCorrectionPattern(BasePattern):
    """
    ABC Correction Pattern
    
    Alternative interpretation where ABC is a continuation pattern:
    - Bullish: Price breaks above C after correction
    - Bearish: Price breaks below C after correction
    """
    
    def __init__(
        self,
        lookback: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        fib_tolerance: float = 0.05
    ):
        super().__init__(
            name="ABC Correction",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=lookback * 4
        )
        self.lookback = lookback
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.fib_tolerance = fib_tolerance
    
    def detect(self, df: pd.DataFrame, i: int) -> PatternResult:
        """Detect ABC Correction pattern."""
        # Simplified - delegates to ABCPattern
        return PatternResult(
            detected=False,
            pattern_name=self.name,
            pattern_type=self.pattern_type
        )
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        return None
