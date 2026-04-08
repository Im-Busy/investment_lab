"""
Gartley Pattern

Detection Logic:
- Identify 5 pivots: X, A, B, C, D using ZigZag or Local Extrema (lookback n=5)
- Bullish Setup: X=Low, A=High, B=Low, C=High, D=Low
- Bearish Setup: X=High, A=Low, B=High, C=Low, D=High
- Condition 1 (AB Retracement): 0.618 * abs(XA) <= abs(AB) <= 0.618 * abs(XA) (Tolerance +/- 0.05)
- Condition 2 (BC Retracement): 0.382 * abs(AB) <= abs(BC) <= 0.886 * abs(AB)
- Condition 3 (CD Extension): 1.27 * abs(BC) <= abs(CD) <= 1.62 * abs(BC)
- Condition 4 (D Level): 0.786 * abs(XA) <= abs(XD) <= 0.786 * abs(XA) (Tolerance +/- 0.05)
- Pattern Complete when D pivot is formed and price enters Potential Reversal Zone (PRZ)

Entry Rules:
- Wait for confirmation bar after D formation
- Long Entry: Buy Stop = high[confirmation_bar] + 0.01
- Short Entry: Sell Stop = low[confirmation_bar] - 0.01
- Entry valid for next 3 to 5 bars

Stop Loss Rules:
- Long Stop: low[D] - 0.01 (Below PRZ)
- Short Stop: high[D] + 0.01 (Above PRZ)

Take Profit Rules:
- Target 1: Price level of Pivot A
- Target 2: Entry + 1.27 * abs(AD) (Long) or Entry - 1.27 * abs(AD) (Short)
- Target 3: Entry + 1.62 * abs(AD) (Long) or Entry - 1.62 * abs(AD) (Short)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from ..base import BasePattern, PatternType, SignalDirection, TradeSignal, PatternResult
from ...indicators.pivots import find_swing_highs, find_swing_lows
from ...indicators.fibonacci import (
    calculate_ab_retracement,
    calculate_bc_retracement,
    calculate_cd_extension,
    calculate_xd_retracement,
    is_fib_ratio_match
)
from ...indicators.technical import volume_sma, average_range


class GartleyPattern(BasePattern):
    """
    Gartley Pattern Detector
    
    A harmonic pattern that uses Fibonacci ratios to identify
    potential reversal points in the market.
    """
    
    # Fibonacci ratio tolerances
    AB_RETRACEMENT_TARGET = 0.618
    AB_TOLERANCE = 0.05
    
    BC_RETRACEMENT_MIN = 0.382
    BC_RETRACEMENT_MAX = 0.886
    
    CD_EXTENSION_MIN = 1.27
    CD_EXTENSION_MAX = 1.618
    
    XD_RETRACEMENT_TARGET = 0.786
    XD_TOLERANCE = 0.05
    
    def __init__(
        self,
        lookback: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        fib_tolerance: float = 0.05,
        require_confirmation: bool = True,
        volume_filter: bool = False
    ):
        """
        Initialize Gartley pattern detector.
        
        Args:
            lookback: Lookback period for pivot detection
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            fib_tolerance: Tolerance for Fibonacci ratio matching
            require_confirmation: Require confirmation bar after D
            volume_filter: Require volume confirmation
        """
        super().__init__(
            name="Gartley Pattern",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=lookback * 6  # Need enough bars for XABCD
        )
        self.lookback = lookback
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.fib_tolerance = fib_tolerance
        self.require_confirmation = require_confirmation
        self.volume_filter = volume_filter
    
    def _find_pivots(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Find potential XABCD pivots for Gartley pattern.
        
        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            
        Returns:
            Dictionary with pivot indices and prices or None
        """
        if i < self.lookback * 5:
            return None
        
        # Find recent swing highs and lows
        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)
        
        # Get valid swing points in range
        highs = []
        lows = []
        
        for j in range(i - self.lookback, i + 1):
            if pd.notna(swing_highs.iloc[j]):
                highs.append((j, self._safe_float(swing_highs.iloc[j])))
            if pd.notna(swing_lows.iloc[j]):
                lows.append((j, self._safe_float(swing_lows.iloc[j])))
        
        # Need at least 3 highs and 3 lows for potential pattern
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        # Try to identify bullish Gartley (X=Low, A=High, B=Low, C=High, D=Low)
        bullish_pivots = self._identify_bullish_pivots(highs, lows, i)
        
        # Try to identify bearish Gartley (X=High, A=Low, B=High, C=Low, D=High)
        bearish_pivots = self._identify_bearish_pivots(highs, lows, i)
        
        if bullish_pivots:
            return {
                'direction': 'bullish',
                'X': bullish_pivots[0],
                'A': bullish_pivots[1],
                'B': bullish_pivots[2],
                'C': bullish_pivots[3],
                'D': bullish_pivots[4]
            }
        elif bearish_pivots:
            return {
                'direction': 'bearish',
                'X': bearish_pivots[0],
                'A': bearish_pivots[1],
                'B': bearish_pivots[2],
                'C': bearish_pivots[3],
                'D': bearish_pivots[4]
            }
        
        return None
    
    def _identify_bullish_pivots(
        self,
        highs: List[Tuple[int, float]],
        lows: List[Tuple[int, float]],
        current_idx: int
    ) -> Optional[List[Tuple[int, float]]]:
        """
        Identify bullish Gartley pivots: X=Low, A=High, B=Low, C=High, D=Low
        
        Args:
            highs: List of (index, price) for swing highs
            lows: List of (index, price) for swing lows
            current_idx: Current bar index
            
        Returns:
            List of (X, A, B, C, D) pivots or None
        """
        # Sort by index (chronological order)
        highs = sorted(highs, key=lambda x: x[0])
        lows = sorted(lows, key=lambda x: x[0])
        
        # For bullish Gartley, we need: Low(X) -> High(A) -> Low(B) -> High(C) -> Low(D)
        # Where D is the most recent low
        
        if len(lows) < 3 or len(highs) < 2:
            return None
        
        # D should be the most recent low
        D = lows[-1]
        
        # Find C (high before D)
        C_candidates = [(idx, price) for idx, price in highs if idx < D[0]]
        if not C_candidates:
            return None
        C = C_candidates[-1]  # Most recent high before D
        
        # Find B (low before C)
        B_candidates = [(idx, price) for idx, price in lows if idx < C[0]]
        if not B_candidates:
            return None
        B = B_candidates[-1]  # Most recent low before C
        
        # Find A (high before B)
        A_candidates = [(idx, price) for idx, price in highs if idx < B[0]]
        if not A_candidates:
            return None
        A = A_candidates[-1]  # Most recent high before B
        
        # Find X (low before A)
        X_candidates = [(idx, price) for idx, price in lows if idx < A[0]]
        if not X_candidates:
            return None
        X = X_candidates[-1]  # Most recent low before A
        
        # Validate Fibonacci ratios
        if not self._validate_fib_ratios(X[1], A[1], B[1], C[1], D[1]):
            return None
        
        return [X, A, B, C, D]
    
    def _identify_bearish_pivots(
        self,
        highs: List[Tuple[int, float]],
        lows: List[Tuple[int, float]],
        current_idx: int
    ) -> Optional[List[Tuple[int, float]]]:
        """
        Identify bearish Gartley pivots: X=High, A=Low, B=High, C=Low, D=High
        """
        highs = sorted(highs, key=lambda x: x[0])
        lows = sorted(lows, key=lambda x: x[0])
        
        if len(highs) < 3 or len(lows) < 2:
            return None
        
        # D should be the most recent high
        D = highs[-1]
        
        # Find C (low before D)
        C_candidates = [(idx, price) for idx, price in lows if idx < D[0]]
        if not C_candidates:
            return None
        C = C_candidates[-1]
        
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
        
        # Find X (high before A)
        X_candidates = [(idx, price) for idx, price in highs if idx < A[0]]
        if not X_candidates:
            return None
        X = X_candidates[-1]
        
        # Validate Fibonacci ratios (inverted for bearish)
        if not self._validate_fib_ratios(X[1], A[1], B[1], C[1], D[1], bearish=True):
            return None
        
        return [X, A, B, C, D]
    
    def _validate_fib_ratios(
        self,
        X: float,
        A: float,
        B: float,
        C: float,
        D: float,
        bearish: bool = False
    ) -> bool:
        """
        Validate Fibonacci ratios for Gartley pattern.
        
        Args:
            X, A, B, C, D: Pivot prices
            bearish: Whether this is a bearish pattern
            
        Returns:
            True if ratios are valid
        """
        # Calculate retracements and extensions
        ab_ratio = calculate_ab_retracement(X, A, B)
        bc_ratio = calculate_bc_retracement(A, B, C)
        cd_ratio = calculate_cd_extension(B, C, D)
        xd_ratio = calculate_xd_retracement(X, A, D)
        
        # Validate AB = 0.618 XA
        if not is_fib_ratio_match(ab_ratio, self.AB_RETRACEMENT_TARGET, self.fib_tolerance):
            return False
        
        # Validate BC = 0.382 to 0.886 AB
        if not (self.BC_RETRACEMENT_MIN - self.fib_tolerance <= bc_ratio <= self.BC_RETRACEMENT_MAX + self.fib_tolerance):
            return False
        
        # Validate CD = 1.27 to 1.618 BC
        if not (self.CD_EXTENSION_MIN - self.fib_tolerance <= cd_ratio <= self.CD_EXTENSION_MAX + self.fib_tolerance):
            return False
        
        # Validate XD = 0.786 XA
        if not is_fib_ratio_match(xd_ratio, self.XD_RETRACEMENT_TARGET, self.fib_tolerance):
            return False
        
        return True
    
    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Gartley pattern at bar index i.
        
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
        
        # Find pivots
        pivots = self._find_pivots(df, i)
        
        if pivots is None:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        X = pivots['X']
        A = pivots['A']
        B = pivots['B']
        C = pivots['C']
        D = pivots['D']
        direction = pivots['direction']
        
        # Check for confirmation bar if required
        if self.require_confirmation:
            # Need at least one bar after D
            if D[0] >= i:
                return PatternResult(
                    detected=False,
                    pattern_name=self.name,
                    pattern_type=self.pattern_type,
                    pivot_points={'X': X[1], 'A': A[1], 'B': B[1], 'C': C[1], 'D': D[1]}
                )
        
        # Generate signal
        signal = self._generate_signal(df, i, X, A, B, C, D, direction)
        
        # Calculate Fibonacci ratios for metadata
        ab_ratio = calculate_ab_retracement(X[1], A[1], B[1])
        bc_ratio = calculate_bc_retracement(A[1], B[1], C[1])
        cd_ratio = calculate_cd_extension(B[1], C[1], D[1])
        xd_ratio = calculate_xd_retracement(X[1], A[1], D[1])
        
        return PatternResult(
            detected=True,
            pattern_name=f"{self.name} ({direction.capitalize()})",
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                'X_idx': X[0], 'X': X[1],
                'A_idx': A[0], 'A': A[1],
                'B_idx': B[0], 'B': B[1],
                'C_idx': C[0], 'C': C[1],
                'D_idx': D[0], 'D': D[1],
                'direction': direction,
                'ab_ratio': ab_ratio,
                'bc_ratio': bc_ratio,
                'cd_ratio': cd_ratio,
                'xd_ratio': xd_ratio
            },
            bars_since_detection=0,
            start_index=X[0],
            end_index=D[0]
        )
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal (delegates to pattern detection)."""
        result = self.detect(df, i)
        return result.signal if result.detected else None
    
    def _generate_signal(
        self,
        df: pd.DataFrame,
        i: int,
        X: Tuple[int, float],
        A: Tuple[int, float],
        B: Tuple[int, float],
        C: Tuple[int, float],
        D: Tuple[int, float],
        direction: str
    ) -> Optional[TradeSignal]:
        """Generate trade signal for Gartley pattern."""
        
        # Calculate AD range for targets
        ad_range = abs(A[1] - D[1])
        
        if direction == 'bullish':
            # Long entry above confirmation bar high
            if self.require_confirmation and D[0] + 1 <= i:
                entry_bar_high = self._safe_float(df.iloc[D[0] + 1]['High'])
            else:
                entry_bar_high = D[1]  # Use D point as entry reference
            
            entry_price = entry_bar_high + self.entry_offset
            stop_loss = D[1] - self.stop_offset  # Below D point (PRZ)
            
            # Targets
            take_profit_1 = A[1]  # Target at A level
            take_profit_2 = D[1] + (ad_range * 1.27)  # 127% of AD
            take_profit_3 = D[1] + (ad_range * 1.618)  # 162% of AD
            
            signal_direction = SignalDirection.LONG
        
        else:  # bearish
            # Short entry below confirmation bar low
            if self.require_confirmation and D[0] + 1 <= i:
                entry_bar_low = self._safe_float(df.iloc[D[0] + 1]['Low'])
            else:
                entry_bar_low = D[1]
            
            entry_price = entry_bar_low - self.entry_offset
            stop_loss = D[1] + self.stop_offset  # Above D point (PRZ)
            
            # Targets
            take_profit_1 = A[1]  # Target at A level
            take_profit_2 = D[1] - (ad_range * 1.27)  # 127% of AD
            take_profit_3 = D[1] - (ad_range * 1.618)  # 162% of AD
            
            signal_direction = SignalDirection.SHORT
        
        # Volume confirmation
        volume_confirmed = True
        if self.volume_filter:
            vol_sma = volume_sma(df['Volume'], 20)
            check_idx = D[0] + 1 if D[0] + 1 < len(vol_sma) else D[0]
            if check_idx < len(vol_sma):
                current_vol = self._safe_float(df.iloc[check_idx]['Volume'])
                avg_vol = self._safe_float(vol_sma.iloc[check_idx])
                volume_confirmed = current_vol > avg_vol
        
        # Confidence based on Fibonacci ratio precision
        ab_ratio = calculate_ab_retracement(X[1], A[1], B[1])
        xd_ratio = calculate_xd_retracement(X[1], A[1], D[1])
        
        confidence = 0.6
        if abs(ab_ratio - 0.618) < 0.02:  # Very close to ideal
            confidence += 0.1
        if abs(xd_ratio - 0.786) < 0.02:
            confidence += 0.1
        if volume_confirmed:
            confidence += 0.05
        
        return TradeSignal(
            pattern_name=f"{self.name} ({direction.capitalize()})",
            direction=signal_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=min(confidence, 1.0),
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], 'name') else None,
            metadata={
                'direction': direction,
                'X': X[1], 'A': A[1], 'B': B[1], 'C': C[1], 'D': D[1],
                'ab_ratio': ab_ratio,
                'bc_ratio': calculate_bc_retracement(A[1], B[1], C[1]),
                'cd_ratio': calculate_cd_extension(B[1], C[1], D[1]),
                'xd_ratio': xd_ratio,
                'ad_range': ad_range,
                'volume_confirmed': volume_confirmed,
                'entry_type': 'buy_stop' if direction == 'bullish' else 'sell_stop'
            }
        )


class ButterflyPattern(BasePattern):
    """
    Butterfly Pattern Detector
    
    Similar to Gartley but with different Fibonacci ratios.
    The Butterfly is an extension pattern where D point extends beyond X.
    
    Fibonacci ratios:
    - AB = 0.786 XA
    - BC = 0.382 to 0.886 AB
    - CD = 1.618 to 2.618 BC
    - XD = 1.27 to 1.618 XA (D extends beyond X)
    """
    
    AB_RETRACEMENT_TARGET = 0.786
    BC_RETRACEMENT_MIN = 0.382
    BC_RETRACEMENT_MAX = 0.886
    CD_EXTENSION_MIN = 1.618
    CD_EXTENSION_MAX = 2.618
    XD_RETRACEMENT_MIN = 1.27
    XD_RETRACEMENT_MAX = 1.618
    
    def __init__(
        self,
        lookback: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        fib_tolerance: float = 0.05,
        require_confirmation: bool = True,
        volume_filter: bool = False
    ):
        super().__init__(
            name="Butterfly Pattern",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=lookback * 6
        )
        self.lookback = lookback
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.fib_tolerance = fib_tolerance
        self.require_confirmation = require_confirmation
        self.volume_filter = volume_filter
    
    def _find_pivots(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        """Find potential XABCD pivots for Butterfly pattern."""
        if i < self.lookback * 5:
            return None
        
        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)
        
        highs = []
        lows = []
        
        for j in range(i - self.lookback, i + 1):
            if pd.notna(swing_highs.iloc[j]):
                highs.append((j, self._safe_float(swing_highs.iloc[j])))
            if pd.notna(swing_lows.iloc[j]):
                lows.append((j, self._safe_float(swing_lows.iloc[j])))
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        # Try bullish Butterfly (X=Low, A=High, B=Low, C=High, D=Low where D < X)
        bullish_pivots = self._identify_bullish_pivots(highs, lows, i)
        
        # Try bearish Butterfly (X=High, A=Low, B=High, C=Low, D=High where D > X)
        bearish_pivots = self._identify_bearish_pivots(highs, lows, i)
        
        if bullish_pivots:
            return {
                'direction': 'bullish',
                'X': bullish_pivots[0], 'A': bullish_pivots[1],
                'B': bullish_pivots[2], 'C': bullish_pivots[3],
                'D': bullish_pivots[4]
            }
        elif bearish_pivots:
            return {
                'direction': 'bearish',
                'X': bearish_pivots[0], 'A': bearish_pivots[1],
                'B': bearish_pivots[2], 'C': bearish_pivots[3],
                'D': bearish_pivots[4]
            }
        
        return None
    
    def _identify_bullish_pivots(self, highs, lows, current_idx):
        """Identify bullish Butterfly pivots: X=Low, A=High, B=Low, C=High, D=Low where D < X"""
        highs = sorted(highs, key=lambda x: x[0])
        lows = sorted(lows, key=lambda x: x[0])
        
        if len(lows) < 3 or len(highs) < 2:
            return None
        
        D = lows[-1]
        C_candidates = [(idx, price) for idx, price in highs if idx < D[0]]
        if not C_candidates:
            return None
        C = C_candidates[-1]
        
        B_candidates = [(idx, price) for idx, price in lows if idx < C[0]]
        if not B_candidates:
            return None
        B = B_candidates[-1]
        
        A_candidates = [(idx, price) for idx, price in highs if idx < B[0]]
        if not A_candidates:
            return None
        A = A_candidates[-1]
        
        X_candidates = [(idx, price) for idx, price in lows if idx < A[0]]
        if not X_candidates:
            return None
        X = X_candidates[-1]
        
        if not self._validate_fib_ratios(X[1], A[1], B[1], C[1], D[1]):
            return None
        
        return [X, A, B, C, D]
    
    def _identify_bearish_pivots(self, highs, lows, current_idx):
        """Identify bearish Butterfly pivots: X=High, A=Low, B=High, C=Low, D=High where D > X"""
        highs = sorted(highs, key=lambda x: x[0])
        lows = sorted(lows, key=lambda x: x[0])
        
        if len(highs) < 3 or len(lows) < 2:
            return None
        
        D = highs[-1]
        C_candidates = [(idx, price) for idx, price in lows if idx < D[0]]
        if not C_candidates:
            return None
        C = C_candidates[-1]
        
        B_candidates = [(idx, price) for idx, price in highs if idx < C[0]]
        if not B_candidates:
            return None
        B = B_candidates[-1]
        
        A_candidates = [(idx, price) for idx, price in lows if idx < B[0]]
        if not A_candidates:
            return None
        A = A_candidates[-1]
        
        X_candidates = [(idx, price) for idx, price in highs if idx < A[0]]
        if not X_candidates:
            return None
        X = X_candidates[-1]
        
        if not self._validate_fib_ratios(X[1], A[1], B[1], C[1], D[1], bearish=True):
            return None
        
        return [X, A, B, C, D]
    
    def _validate_fib_ratios(self, X, A, B, C, D, bearish=False):
        """Validate Butterfly Fibonacci ratios."""
        ab_ratio = calculate_ab_retracement(X, A, B)
        bc_ratio = calculate_bc_retracement(A, B, C)
        cd_ratio = calculate_cd_extension(B, C, D)
        xd_ratio = calculate_xd_retracement(X, A, D)
        
        # AB = 0.786 XA
        if not is_fib_ratio_match(ab_ratio, self.AB_RETRACEMENT_TARGET, self.fib_tolerance):
            return False
        
        # BC = 0.382-0.886 AB
        if not (self.BC_RETRACEMENT_MIN - self.fib_tolerance <= bc_ratio <= self.BC_RETRACEMENT_MAX + self.fib_tolerance):
            return False
        
        # CD = 1.618-2.618 BC
        if not (self.CD_EXTENSION_MIN - self.fib_tolerance <= cd_ratio <= self.CD_EXTENSION_MAX + self.fib_tolerance):
            return False
        
        # XD = 1.27-1.618 XA (D extends beyond X)
        if not (self.XD_RETRACEMENT_MIN - self.fib_tolerance <= xd_ratio <= self.XD_RETRACEMENT_MAX + self.fib_tolerance):
            return False
        
        return True
    
    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect Butterfly pattern at bar index i."""
        if not self._validate_data(df, i, window_start):
            return PatternResult(detected=False, pattern_name=self.name, pattern_type=self.pattern_type)
        
        pivots = self._find_pivots(df, i)
        if pivots is None:
            return PatternResult(detected=False, pattern_name=self.name, pattern_type=self.pattern_type)
        
        X, A, B, C, D = pivots['X'], pivots['A'], pivots['B'], pivots['C'], pivots['D']
        direction = pivots['direction']
        
        if self.require_confirmation and D[0] >= i:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type,
                pivot_points={'X': X[1], 'A': A[1], 'B': B[1], 'C': C[1], 'D': D[1]}
            )
        
        signal = self._generate_signal(df, i, X, A, B, C, D, direction)
        
        return PatternResult(
            detected=True,
            pattern_name=f"{self.name} ({direction.capitalize()})",
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                'X_idx': X[0], 'X': X[1], 'A_idx': A[0], 'A': A[1],
                'B_idx': B[0], 'B': B[1], 'C_idx': C[0], 'C': C[1],
                'D_idx': D[0], 'D': D[1], 'direction': direction,
                'ab_ratio': calculate_ab_retracement(X[1], A[1], B[1]),
                'bc_ratio': calculate_bc_retracement(A[1], B[1], C[1]),
                'cd_ratio': calculate_cd_extension(B[1], C[1], D[1]),
                'xd_ratio': calculate_xd_retracement(X[1], A[1], D[1])
            },
            bars_since_detection=0,
            start_index=X[0],
            end_index=D[0]
        )
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None
    
    def _generate_signal(self, df, i, X, A, B, C, D, direction):
        """Generate Butterfly trade signal."""
        ad_range = abs(A[1] - D[1])
        
        if direction == 'bullish':
            entry_bar_high = self._safe_float(df.iloc[D[0] + 1]['High']) if self.require_confirmation and D[0] + 1 <= i else D[1]
            entry_price = entry_bar_high + self.entry_offset
            stop_loss = D[1] - self.stop_offset
            take_profit_1 = A[1]
            take_profit_2 = D[1] + (ad_range * 1.27)
            take_profit_3 = D[1] + (ad_range * 1.618)
            signal_direction = SignalDirection.LONG
        else:
            entry_bar_low = self._safe_float(df.iloc[D[0] + 1]['Low']) if self.require_confirmation and D[0] + 1 <= i else D[1]
            entry_price = entry_bar_low - self.entry_offset
            stop_loss = D[1] + self.stop_offset
            take_profit_1 = A[1]
            take_profit_2 = D[1] - (ad_range * 1.27)
            take_profit_3 = D[1] - (ad_range * 1.618)
            signal_direction = SignalDirection.SHORT
        
        confidence = 0.6
        ab_ratio = calculate_ab_retracement(X[1], A[1], B[1])
        if abs(ab_ratio - 0.786) < 0.02:
            confidence += 0.1
        
        return TradeSignal(
            pattern_name=f"{self.name} ({direction.capitalize()})",
            direction=signal_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=min(confidence, 1.0),
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], 'name') else None,
            metadata={
                'direction': direction,
                'X': X[1], 'A': A[1], 'B': B[1], 'C': C[1], 'D': D[1],
                'ad_range': ad_range,
                'entry_type': 'buy_stop' if direction == 'bullish' else 'sell_stop'
            }
        )


class BatPattern(BasePattern):
    """
    Bat Pattern Detector
    
    Fibonacci ratios:
    - AB = 0.382 to 0.5 XA
    - BC = 0.382 to 0.886 AB
    - CD = 1.618 to 2.618 BC
    - XD = 0.886 XA
    
    The Bat pattern is known for its high accuracy and tight stop losses.
    """
    
    AB_RETRACEMENT_MIN = 0.382
    AB_RETRACEMENT_MAX = 0.5
    BC_RETRACEMENT_MIN = 0.382
    BC_RETRACEMENT_MAX = 0.886
    CD_EXTENSION_MIN = 1.618
    CD_EXTENSION_MAX = 2.618
    XD_RETRACEMENT_TARGET = 0.886
    
    def __init__(
        self,
        lookback: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        fib_tolerance: float = 0.05,
        require_confirmation: bool = True,
        volume_filter: bool = False
    ):
        super().__init__(
            name="Bat Pattern",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=lookback * 6
        )
        self.lookback = lookback
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.fib_tolerance = fib_tolerance
        self.require_confirmation = require_confirmation
        self.volume_filter = volume_filter
    
    def _find_pivots(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        """Find potential XABCD pivots for Bat pattern."""
        if i < self.lookback * 5:
            return None
        
        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)
        
        highs = []
        lows = []
        
        for j in range(i - self.lookback, i + 1):
            if pd.notna(swing_highs.iloc[j]):
                highs.append((j, self._safe_float(swing_highs.iloc[j])))
            if pd.notna(swing_lows.iloc[j]):
                lows.append((j, self._safe_float(swing_lows.iloc[j])))
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        bullish_pivots = self._identify_bullish_pivots(highs, lows, i)
        bearish_pivots = self._identify_bearish_pivots(highs, lows, i)
        
        if bullish_pivots:
            return {
                'direction': 'bullish',
                'X': bullish_pivots[0], 'A': bullish_pivots[1],
                'B': bullish_pivots[2], 'C': bullish_pivots[3],
                'D': bullish_pivots[4]
            }
        elif bearish_pivots:
            return {
                'direction': 'bearish',
                'X': bearish_pivots[0], 'A': bearish_pivots[1],
                'B': bearish_pivots[2], 'C': bearish_pivots[3],
                'D': bearish_pivots[4]
            }
        
        return None
    
    def _identify_bullish_pivots(self, highs, lows, current_idx):
        """Identify bullish Bat pivots: X=Low, A=High, B=Low, C=High, D=Low"""
        highs = sorted(highs, key=lambda x: x[0])
        lows = sorted(lows, key=lambda x: x[0])
        
        if len(lows) < 3 or len(highs) < 2:
            return None
        
        D = lows[-1]
        C_candidates = [(idx, price) for idx, price in highs if idx < D[0]]
        if not C_candidates:
            return None
        C = C_candidates[-1]
        
        B_candidates = [(idx, price) for idx, price in lows if idx < C[0]]
        if not B_candidates:
            return None
        B = B_candidates[-1]
        
        A_candidates = [(idx, price) for idx, price in highs if idx < B[0]]
        if not A_candidates:
            return None
        A = A_candidates[-1]
        
        X_candidates = [(idx, price) for idx, price in lows if idx < A[0]]
        if not X_candidates:
            return None
        X = X_candidates[-1]
        
        if not self._validate_fib_ratios(X[1], A[1], B[1], C[1], D[1]):
            return None
        
        return [X, A, B, C, D]
    
    def _identify_bearish_pivots(self, highs, lows, current_idx):
        """Identify bearish Bat pivots: X=High, A=Low, B=High, C=Low, D=High"""
        highs = sorted(highs, key=lambda x: x[0])
        lows = sorted(lows, key=lambda x: x[0])
        
        if len(highs) < 3 or len(lows) < 2:
            return None
        
        D = highs[-1]
        C_candidates = [(idx, price) for idx, price in lows if idx < D[0]]
        if not C_candidates:
            return None
        C = C_candidates[-1]
        
        B_candidates = [(idx, price) for idx, price in highs if idx < C[0]]
        if not B_candidates:
            return None
        B = B_candidates[-1]
        
        A_candidates = [(idx, price) for idx, price in lows if idx < B[0]]
        if not A_candidates:
            return None
        A = A_candidates[-1]
        
        X_candidates = [(idx, price) for idx, price in highs if idx < A[0]]
        if not X_candidates:
            return None
        X = X_candidates[-1]
        
        if not self._validate_fib_ratios(X[1], A[1], B[1], C[1], D[1], bearish=True):
            return None
        
        return [X, A, B, C, D]
    
    def _validate_fib_ratios(self, X, A, B, C, D, bearish=False):
        """Validate Bat Fibonacci ratios."""
        ab_ratio = calculate_ab_retracement(X, A, B)
        bc_ratio = calculate_bc_retracement(A, B, C)
        cd_ratio = calculate_cd_extension(B, C, D)
        xd_ratio = calculate_xd_retracement(X, A, D)
        
        # AB = 0.382-0.5 XA
        if not (self.AB_RETRACEMENT_MIN - self.fib_tolerance <= ab_ratio <= self.AB_RETRACEMENT_MAX + self.fib_tolerance):
            return False
        
        # BC = 0.382-0.886 AB
        if not (self.BC_RETRACEMENT_MIN - self.fib_tolerance <= bc_ratio <= self.BC_RETRACEMENT_MAX + self.fib_tolerance):
            return False
        
        # CD = 1.618-2.618 BC
        if not (self.CD_EXTENSION_MIN - self.fib_tolerance <= cd_ratio <= self.CD_EXTENSION_MAX + self.fib_tolerance):
            return False
        
        # XD = 0.886 XA
        if not is_fib_ratio_match(xd_ratio, self.XD_RETRACEMENT_TARGET, self.fib_tolerance):
            return False
        
        return True
    
    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect Bat pattern at bar index i."""
        if not self._validate_data(df, i, window_start):
            return PatternResult(detected=False, pattern_name=self.name, pattern_type=self.pattern_type)
        
        pivots = self._find_pivots(df, i)
        if pivots is None:
            return PatternResult(detected=False, pattern_name=self.name, pattern_type=self.pattern_type)
        
        X, A, B, C, D = pivots['X'], pivots['A'], pivots['B'], pivots['C'], pivots['D']
        direction = pivots['direction']
        
        if self.require_confirmation and D[0] >= i:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type,
                pivot_points={'X': X[1], 'A': A[1], 'B': B[1], 'C': C[1], 'D': D[1]}
            )
        
        signal = self._generate_signal(df, i, X, A, B, C, D, direction)
        
        return PatternResult(
            detected=True,
            pattern_name=f"{self.name} ({direction.capitalize()})",
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                'X_idx': X[0], 'X': X[1], 'A_idx': A[0], 'A': A[1],
                'B_idx': B[0], 'B': B[1], 'C_idx': C[0], 'C': C[1],
                'D_idx': D[0], 'D': D[1], 'direction': direction,
                'ab_ratio': calculate_ab_retracement(X[1], A[1], B[1]),
                'bc_ratio': calculate_bc_retracement(A[1], B[1], C[1]),
                'cd_ratio': calculate_cd_extension(B[1], C[1], D[1]),
                'xd_ratio': calculate_xd_retracement(X[1], A[1], D[1])
            },
            bars_since_detection=0,
            start_index=X[0],
            end_index=D[0]
        )
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None
    
    def _generate_signal(self, df, i, X, A, B, C, D, direction):
        """Generate Bat trade signal."""
        ad_range = abs(A[1] - D[1])
        
        if direction == 'bullish':
            entry_bar_high = self._safe_float(df.iloc[D[0] + 1]['High']) if self.require_confirmation and D[0] + 1 <= i else D[1]
            entry_price = entry_bar_high + self.entry_offset
            stop_loss = D[1] - self.stop_offset
            take_profit_1 = A[1]
            take_profit_2 = D[1] + (ad_range * 1.27)
            take_profit_3 = D[1] + (ad_range * 1.618)
            signal_direction = SignalDirection.LONG
        else:
            entry_bar_low = self._safe_float(df.iloc[D[0] + 1]['Low']) if self.require_confirmation and D[0] + 1 <= i else D[1]
            entry_price = entry_bar_low - self.entry_offset
            stop_loss = D[1] + self.stop_offset
            take_profit_1 = A[1]
            take_profit_2 = D[1] - (ad_range * 1.27)
            take_profit_3 = D[1] - (ad_range * 1.618)
            signal_direction = SignalDirection.SHORT
        
        confidence = 0.6
        xd_ratio = calculate_xd_retracement(X[1], A[1], D[1])
        if abs(xd_ratio - 0.886) < 0.02:
            confidence += 0.1
        
        return TradeSignal(
            pattern_name=f"{self.name} ({direction.capitalize()})",
            direction=signal_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=min(confidence, 1.0),
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], 'name') else None,
            metadata={
                'direction': direction,
                'X': X[1], 'A': A[1], 'B': B[1], 'C': C[1], 'D': D[1],
                'ad_range': ad_range,
                'entry_type': 'buy_stop' if direction == 'bullish' else 'sell_stop'
            }
        )
