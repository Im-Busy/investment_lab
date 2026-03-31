# Candlestick Patterns Implementation Plan (Phase 3)

## Overview

This document outlines the implementation plan for **Phase 3: Candlestick Patterns** from the new trading patterns specification. These patterns are lower priority but provide valuable reversal signals when combined with other confirmation factors.

**Status**: Phase 7 (Classic) ✅ COMPLETED | Phase 8 (Continuation/Breakout) ✅ COMPLETED | **Phase 9 (Candlestick) - THIS PLAN**

---

## Current Pattern Count

| Category | Count | Patterns |
|----------|-------|----------|
| **Basic** | 6 | MSL, Matching Lows, NR7ID, N-Bar Decline, Floor Pivot, Two-Bar Reversal |
| **Harmonic** | 5 | Gartley, ABC, Symmetric Triangle, Donchian, Bollinger |
| **Complex** | 5 | Cup and Handle, Head and Shoulders, Spike and Ledge, Three Hills, Parabolic Arc |
| **Classic** | 10 | Double Top, Double Bottom, 2B, Triple Top, Triple Bottom, Ascending Triangle, Descending Triangle, Rectangle, Wedge, Dead Cat Bounce |
| **Continuation** | 2 | Flag, Pennant |
| **Breakout** | 1 | Gap |
| **Candlestick** | 0 | *TO BE IMPLEMENTED* |
| **TOTAL** | 29 | → 34 after Phase 3 |

---

## Candlestick Patterns to Implement

### Pattern Summary

| Pattern | Type | Signal | Reliability | Key Feature |
|---------|------|--------|-------------|-------------|
| **Doji** | Indecision | Reversal Warning | Low | Open ≈ Close |
| **Harami** | Reversal | Continuation/Reversal | Medium | Small body inside large body |
| **Hammer/Hanging Man** | Reversal | Bullish/Bearish | Medium | Small body, long lower shadow |
| **Engulfing** | Reversal | Bullish/Bearish | High | Body completely engulfs prior |
| **Dark Cloud/Piercing** | Reversal | Bearish/Bullish | Medium | Penetrates prior body |

### Important Notes

1. **Low Standalone Reliability**: Candlestick patterns should NOT be traded alone
2. **Require Confirmation**: Must be combined with:
   - Next candle close confirmation
   - Volume spike
   - Support/resistance confluence
   - Other pattern signals (confluence)
3. **Trend Context**: Pattern interpretation depends on prior trend direction
4. **Timeframe Sensitivity**: More reliable on higher timeframes (4H, Daily, Weekly)

---

## Directory Structure

```
src/patterns/
├── candlestick/               # NEW DIRECTORY
│   ├── __init__.py           # NEW - Module exports
│   ├── doji.py               # NEW - Doji pattern
│   ├── harami.py             # NEW - Harami pattern
│   ├── hammer.py             # NEW - Hammer/Hanging Man pattern
│   ├── engulfing.py          # NEW - Engulfing pattern
│   └── dark_cloud.py         # NEW - Dark Cloud Cover/Piercing Line
└── ...
```

---

## Pattern Specifications

### 1. Doji Pattern (`candlestick/doji.py`)

**Structure**: Single candle where |Open - Close| < threshold% of (High - Low)

**Types**:
- **Standard Doji**: Open equals Close
- **Long-legged Doji**: Long upper and lower shadows
- **Dragonfly Doji**: Open=Close=High, long lower shadow
- **Gravestone Doji**: Open=Close=Low, long upper shadow

**Detection Logic**:
```python
def is_doji(open_price, close, high, low, threshold=0.1):
    """
    Detect Doji candle.
    
    Args:
        open_price: Opening price
        close: Closing price
        high: High price
        low: Low price
        threshold: Maximum body/range ratio (default 10%)
    
    Returns:
        tuple: (is_doji, doji_type)
    """
    body = abs(close - open_price)
    total_range = high - low
    
    if total_range == 0:
        return False, None
    
    body_ratio = body / total_range
    
    if body_ratio > threshold:
        return False, None
    
    # Determine doji type
    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low
    
    if upper_shadow < 0.1 * total_range and lower_shadow > 0.6 * total_range:
        return True, 'dragonfly'
    elif lower_shadow < 0.1 * total_range and upper_shadow > 0.6 * total_range:
        return True, 'gravestone'
    elif upper_shadow > 0.3 * total_range and lower_shadow > 0.3 * total_range:
        return True, 'long_legged'
    else:
        return True, 'standard'
```

**Signal Generation**:
- **DO NOT generate signal on Doji alone**
- Flag for watchlist
- Generate signal ONLY if:
  - Next candle breaks Doji high/low with volume confirmation
  - At key support/resistance level
  - Confluence with other patterns

**Confidence**: 0.3 (low - requires confirmation)

---

### 2. Harami Pattern (`candlestick/harami.py`)

**Structure**: Two candles
- Candle 1: Large body (bullish or bearish)
- Candle 2: Small body of opposite color, fully contained within Candle 1's body

**Detection Logic**:
```python
def detect_harami(candle1, candle2, body_threshold=0.6):
    """
    Detect Harami pattern.
    
    Args:
        candle1: Prior candle dict with 'open', 'high', 'low', 'close'
        candle2: Current candle dict
        body_threshold: Minimum body/range ratio for candle1
    
    Returns:
        tuple: (is_harami, harami_type, direction)
    """
    # Calculate bodies
    body1 = abs(candle1['close'] - candle1['open'])
    body2 = abs(candle2['close'] - candle2['open'])
    range1 = candle1['high'] - candle1['low']
    
    # Candle 1 must have large body
    if range1 == 0 or body1 / range1 < body_threshold:
        return False, None, None
    
    # Candle 2 must have small body
    if body2 >= body1 * 0.7:  # Body2 should be < 70% of Body1
        return False, None, None
    
    # Bodies must be opposite colors
    candle1_bullish = candle1['close'] > candle1['open']
    candle2_bullish = candle2['close'] > candle2['open']
    
    if candle1_bullish == candle2_bullish:
        return False, None, None
    
    # Candle 2 body must be inside Candle 1 body
    body1_high = max(candle1['open'], candle1['close'])
    body1_low = min(candle1['open'], candle1['close'])
    body2_high = max(candle2['open'], candle2['close'])
    body2_low = min(candle2['open'], candle2['close'])
    
    if body2_high > body1_high or body2_low < body1_low:
        return False, None, None
    
    # Determine type
    if candle1_bullish:
        harami_type = 'bearish_harami'  # Bullish candle followed by bearish
        direction = 'SHORT'
    else:
        harami_type = 'bullish_harami'  # Bearish candle followed by bullish
        direction = 'LONG'
    
    return True, harami_type, direction
```

**Signal Generation**:
- Wait for confirmation: next candle breaks candle1 high/low
- Entry on close of confirmation candle
- Stop loss: opposite side of pattern

**Confidence**: 0.5 (medium - requires confirmation)

---

### 3. Hammer/Hanging Man Pattern (`candlestick/hammer.py`)

**Structure**: Single candle
- Small body at top of range
- Long lower shadow (≥ 2x body length)
- Little/no upper shadow (< 10% of range)

**Detection Logic**:
```python
def detect_hammer(open_price, close, high, low, trend_direction, shadow_ratio=2.0, body_threshold=0.3):
    """
    Detect Hammer or Hanging Man pattern.
    
    Args:
        open_price: Opening price
        close: Closing price
        high: High price
        low: Low price
        trend_direction: 'up' or 'down' - prior trend
        shadow_ratio: Lower shadow / body ratio (default 2.0)
        body_threshold: Max body/range ratio (default 30%)
    
    Returns:
        tuple: (is_pattern, pattern_type, direction)
    """
    body = abs(close - open_price)
    total_range = high - low
    
    if total_range == 0:
        return False, None, None
    
    # Body must be small relative to range
    if body / total_range > body_threshold:
        return False, None, None
    
    # Calculate shadows
    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low
    
    # Lower shadow must be at least shadow_ratio times body
    if body == 0:
        return False, None, None
    
    if lower_shadow < shadow_ratio * body:
        return False, None, None
    
    # Upper shadow should be small
    if upper_shadow > 0.1 * total_range:
        return False, None, None
    
    # Determine pattern type based on trend
    if trend_direction == 'down':
        # Hammer - bullish reversal after downtrend
        return True, 'hammer', 'LONG'
    else:
        # Hanging Man - bearish reversal after uptrend
        return True, 'hanging_man', 'SHORT'
```

**Signal Generation**:
- Confirmation required: next candle closes above (hammer) or below (hanging man) pattern
- Volume spike on confirmation increases reliability
- Confluence with support/resistance increases confidence

**Confidence**: 0.5 (medium - requires confirmation)

---

### 4. Engulfing Pattern (`candlestick/engulfing.py`)

**Structure**: Two candles
- Candle 1: Small body (any color)
- Candle 2: Large body that completely engulfs Candle 1's body

**Types**:
- **Bullish Engulfing**: Candle 1 bearish, Candle 2 bullish, Candle 2 body engulfs Candle 1
- **Bearish Engulfing**: Candle 1 bullish, Candle 2 bearish, Candle 2 body engulfs Candle 1

**Detection Logic**:
```python
def detect_engulfing(candle1, candle2, body_threshold=0.3):
    """
    Detect Engulfing pattern.
    
    Args:
        candle1: Prior candle dict
        candle2: Current candle dict
        body_threshold: Min body/range ratio for candle2
    
    Returns:
        tuple: (is_engulfing, engulfing_type, direction)
    """
    # Calculate bodies
    body1 = abs(candle1['close'] - candle1['open'])
    body2 = abs(candle2['close'] - candle2['open'])
    range2 = candle2['high'] - candle2['low']
    
    # Candle 2 must have substantial body
    if range2 == 0 or body2 / range2 < body_threshold:
        return False, None, None
    
    # Candle 2 must be larger than Candle 1
    if body2 <= body1:
        return False, None, None
    
    # Determine colors
    candle1_bullish = candle1['close'] > candle1['open']
    candle2_bullish = candle2['close'] > candle2['open']
    
    # Must be opposite colors
    if candle1_bullish == candle2_bullish:
        return False, None, None
    
    # Check engulfing
    body1_high = max(candle1['open'], candle1['close'])
    body1_low = min(candle1['open'], candle1['close'])
    body2_high = max(candle2['open'], candle2['close'])
    body2_low = min(candle2['open'], candle2['close'])
    
    # Candle 2 must engulf Candle 1 body
    if body2_high <= body1_high or body2_low >= body1_low:
        return False, None, None
    
    # Determine type
    if candle2_bullish:
        return True, 'bullish_engulfing', 'LONG'
    else:
        return True, 'bearish_engulfing', 'SHORT'
```

**Signal Generation**:
- Entry on close of engulfing candle (no confirmation needed)
- Stop loss: opposite side of engulfing candle
- Higher reliability than other candlestick patterns

**Confidence**: 0.65 (higher reliability)

---

### 5. Dark Cloud Cover / Piercing Line Pattern (`candlestick/dark_cloud.py`)

**Structure**: Two candles

**Dark Cloud Cover (Bearish)**:
- Candle 1: Strong bullish candle
- Candle 2: Opens above Candle 1 high, closes below midpoint of Candle 1 body

**Piercing Line (Bullish)**:
- Candle 1: Strong bearish candle
- Candle 2: Opens below Candle 1 low, closes above midpoint of Candle 1 body

**Detection Logic**:
```python
def detect_dark_cloud_piercing(candle1, candle2, penetration_threshold=0.5):
    """
    Detect Dark Cloud Cover or Piercing Line pattern.
    
    Args:
        candle1: Prior candle dict
        candle2: Current candle dict
        penetration_threshold: How far into candle1 body (default 50%)
    
    Returns:
        tuple: (is_pattern, pattern_type, direction)
    """
    # Candle 1 must have substantial body
    body1 = abs(candle1['close'] - candle1['open'])
    range1 = candle1['high'] - candle1['low']
    
    if range1 == 0 or body1 / range1 < 0.6:
        return False, None, None
    
    # Determine candle 1 type
    candle1_bullish = candle1['close'] > candle1['open']
    candle2_bullish = candle2['close'] > candle2['open']
    
    body1_high = max(candle1['open'], candle1['close'])
    body1_low = min(candle1['open'], candle1['close'])
    body1_mid = (body1_high + body1_low) / 2
    
    if candle1_bullish:
        # Check for Dark Cloud Cover
        # Candle 2 should open above candle 1 high
        if candle2['open'] <= candle1['high']:
            return False, None, None
        
        # Candle 2 should be bearish
        if candle2_bullish:
            return False, None, None
        
        # Candle 2 should close below midpoint of candle 1 body
        if candle2['close'] >= body1_mid:
            return False, None, None
        
        return True, 'dark_cloud_cover', 'SHORT'
    
    else:
        # Check for Piercing Line
        # Candle 2 should open below candle 1 low
        if candle2['open'] >= candle1['low']:
            return False, None, None
        
        # Candle 2 should be bullish
        if not candle2_bullish:
            return False, None, None
        
        # Candle 2 should close above midpoint of candle 1 body
        if candle2['close'] <= body1_mid:
            return False, None, None
        
        return True, 'piercing_line', 'LONG'
```

**Signal Generation**:
- Entry on close of second candle
- Stop loss: above/below the pattern
- Medium reliability

**Confidence**: 0.55 (medium reliability)

---

## Base Pattern Implementation

Each candlestick pattern will inherit from [`BasePattern`](src/patterns/base.py) and implement:

```python
class CandlestickPattern(BasePattern):
    """Base class for candlestick patterns with confirmation support."""
    
    def __init__(self, name: str, pattern_type: PatternType, min_bars_required: int = 2):
        super().__init__(name, pattern_type, min_bars_required)
        self.require_confirmation = True
        self.confirmation_bars = 1
    
    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect candlestick pattern at bar i."""
        pass
    
    def get_trend_direction(self, df: pd.DataFrame, i: int, lookback: int = 10) -> str:
        """Determine prior trend direction."""
        if i < lookback:
            return 'unknown'
        
        closes = df['Close'].iloc[i-lookback:i].values
        if closes[-1] > closes[0]:
            return 'up'
        elif closes[-1] < closes[0]:
            return 'down'
        else:
            return 'sideways'
    
    def check_confirmation(self, df: pd.DataFrame, i: int, direction: str) -> bool:
        """Check if pattern is confirmed by next candle."""
        if i >= len(df) - 1:
            return False
        
        next_close = df['Close'].iloc[i + 1]
        current_close = df['Close'].iloc[i]
        
        if direction == 'LONG':
            return next_close > current_close
        else:
            return next_close < current_close
    
    def check_volume_spike(self, df: pd.DataFrame, i: int, threshold: float = 1.5) -> bool:
        """Check if volume is above average."""
        if 'Volume' not in df.columns or i < 20:
            return False
        
        avg_volume = df['Volume'].iloc[i-20:i].mean()
        current_volume = df['Volume'].iloc[i]
        
        return current_volume > avg_volume * threshold
```

---

## Files to Create

### 1. `src/patterns/candlestick/__init__.py`

```python
"""
Candlestick Pattern Module

Contains single-candle and multi-candle reversal patterns.
All patterns require confirmation for signal generation.
"""

from .doji import Doji
from .harami import Harami
from .hammer import Hammer
from .engulfing import Engulfing
from .dark_cloud import DarkCloudCover, PiercingLine

__all__ = [
    "Doji",
    "Harami", 
    "Hammer",
    "Engulfing",
    "DarkCloudCover",
    "PiercingLine",
]
```

### 2. `src/patterns/candlestick/doji.py`
### 3. `src/patterns/candlestick/harami.py`
### 4. `src/patterns/candlestick/hammer.py`
### 5. `src/patterns/candlestick/engulfing.py`
### 6. `src/patterns/candlestick/dark_cloud.py`

---

## Strategy Integration

Update [`src/strategies/backtest_py/multi_pattern_strategy_optimized.py`](src/strategies/backtest_py/multi_pattern_strategy_optimized.py):

```python
# Add imports for candlestick patterns
from src.patterns.candlestick.doji import Doji
from src.patterns.candlestick.harami import Harami
from src.patterns.candlestick.hammer import Hammer
from src.patterns.candlestick.engulfing import Engulfing
from src.patterns.candlestick.dark_cloud import DarkCloudCover, PiercingLine

# In _init_patterns():
# Candlestick patterns (lower confidence, require confirmation)
patterns.extend([
    Doji(),
    Harami(),
    Hammer(),
    Engulfing(),
    DarkCloudCover(),
    PiercingLine(),
])
```

---

## Unit Tests

Create `tests/test_candlestick.py`:

```python
"""Unit tests for candlestick patterns."""

import pytest
import pandas as pd
import numpy as np

from src.patterns.candlestick.doji import Doji
from src.patterns.candlestick.harami import Harami
from src.patterns.candlestick.hammer import Hammer
from src.patterns.candlestick.engulfing import Engulfing
from src.patterns.candlestick.dark_cloud import DarkCloudCover, PiercingLine


class TestDoji:
    """Tests for Doji pattern."""
    
    def test_standard_doji_detection(self):
        """Test detection of standard doji."""
        pass
    
    def test_dragonfly_doji_detection(self):
        """Test detection of dragonfly doji."""
        pass
    
    def test_gravestone_doji_detection(self):
        """Test detection of gravestone doji."""
        pass
    
    def test_no_signal_without_confirmation(self):
        """Test that doji does not generate signal alone."""
        pass


class TestHarami:
    """Tests for Harami pattern."""
    
    def test_bullish_harami_detection(self):
        """Test detection of bullish harami."""
        pass
    
    def test_bearish_harami_detection(self):
        """Test detection of bearish harami."""
        pass
    
    def test_requires_confirmation(self):
        """Test that harami requires confirmation."""
        pass


class TestHammer:
    """Tests for Hammer/Hanging Man pattern."""
    
    def test_hammer_after_downtrend(self):
        """Test hammer detection after downtrend."""
        pass
    
    def test_hanging_man_after_uptrend(self):
        """Test hanging man detection after uptrend."""
        pass
    
    def test_shadow_ratio_requirement(self):
        """Test that lower shadow must be 2x body."""
        pass


class TestEngulfing:
    """Tests for Engulfing pattern."""
    
    def test_bullish_engulfing_detection(self):
        """Test detection of bullish engulfing."""
        pass
    
    def test_bearish_engulfing_detection(self):
        """Test detection of bearish engulfing."""
        pass
    
    def test_must_engulf_body(self):
        """Test that candle must engulf prior body."""
        pass


class TestDarkCloudPiercing:
    """Tests for Dark Cloud Cover and Piercing Line."""
    
    def test_dark_cloud_cover_detection(self):
        """Test detection of dark cloud cover."""
        pass
    
    def test_piercing_line_detection(self):
        """Test detection of piercing line."""
        pass
    
    def test_penetration_threshold(self):
        """Test that second candle penetrates midpoint."""
        pass
```

---

## Implementation Order

1. **Create directory and `__init__.py`** - Set up module structure
2. **Implement Doji** - Simplest pattern, single candle
3. **Implement Engulfing** - Higher reliability, two candles
4. **Implement Harami** - Two candles, requires confirmation
5. **Implement Hammer** - Single candle, trend-dependent
6. **Implement Dark Cloud/Piercing** - Two candles, penetration logic
7. **Update strategy integration** - Add to multi-pattern strategy
8. **Write unit tests** - Comprehensive test coverage
9. **Update documentation** - Progress log and pattern count

---

## Success Criteria

1. **All 5 candlestick patterns implemented** with proper detection logic
2. **Unit tests pass** for all patterns
3. **Strategy integration complete** - patterns work with confluence system
4. **Confirmation filters working** - patterns require confirmation
5. **Confidence levels set appropriately** - reflecting pattern reliability

---

## Pattern Count After Implementation

| Category | Count |
|----------|-------|
| Basic | 6 |
| Harmonic | 5 |
| Complex | 5 |
| Classic | 10 |
| Continuation | 2 |
| Breakout | 1 |
| **Candlestick** | **5** |
| **TOTAL** | **34** |

---

*Plan Version: 1.0 | Created: 2026-03-19 | Phase 3 of New Patterns Implementation*
