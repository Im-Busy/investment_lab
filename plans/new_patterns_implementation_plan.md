# New Trading Patterns Implementation Plan

## Overview

This document outlines the implementation plan for adding new trading patterns from the Fidelity "Identifying Chart Patterns with Technical Analysis" specification to the existing pattern detection system.

## Current State Analysis

### Already Implemented Patterns (20 total)

| Category | Patterns | Location |
|----------|----------|----------|
| **Basic** (5) | MSL, Matching Lows, NR7ID, N-Bar Decline, Floor Pivot | `src/patterns/basic/` |
| **Harmonic** (5) | Gartley, ABC, Symmetric Triangle, Donchian, Bollinger | `src/patterns/harmonic/` |
| **Complex** (5) | Cup and Handle, Head and Shoulders, Spike and Ledge, Three Hills, Parabolic Arc | `src/patterns/complex/` |
| **Classic** (5) | Double Top, Double Bottom, Trader Vic 2B, Triple Top, Dead Cat Bounce | `src/patterns/classic/` |

### Missing Patterns from Specification

Based on the Fidelity specification, the following patterns need to be implemented:

#### High Priority - Core Chart Patterns

| Pattern | Type | Category | Complexity | Rationale |
|---------|------|----------|------------|-----------|
| **Triple Bottom** | Bullish Reversal | classic | Medium | Mirror of existing Triple Top |
| **Ascending Triangle** | Bullish Continuation | classic | Medium | High reliability, commonly used |
| **Descending Triangle** | Bearish Continuation | classic | Medium | High reliability, commonly used |
| **Rectangle** | Continuation/Reversal | classic | Medium | Horizontal channel pattern |
| **Wedge** (Rising/Falling) | Reversal | classic | Medium | Important reversal signal |

#### Medium Priority - Continuation Patterns

| Pattern | Type | Category | Complexity | Rationale |
|---------|------|----------|------------|-----------|
| **Flag/Pennant** | Continuation | continuation | Medium | Short-term continuation patterns |
| **Gap Patterns** | Breakout | breakout | Medium | Explosion gap pivot strategy |
| **Two-Bar Reversal** | Reversal | basic | Low | Simple but effective pattern |

#### Lower Priority - Candlestick Patterns

| Pattern | Type | Category | Complexity | Rationale |
|---------|------|----------|------------|-----------|
| **Doji** | Indecision | candlestick | Low | Needs confirmation filter |
| **Harami** | Reversal | candlestick | Low | Needs confirmation filter |
| **Hammer/Hanging Man** | Reversal | candlestick | Low | Needs confirmation filter |
| **Engulfing** | Reversal | candlestick | Medium | Higher reliability |
| **Dark Cloud/Piercing** | Reversal | candlestick | Medium | Medium reliability |

#### Already Covered by Existing Patterns

| Specification Pattern | Existing Implementation |
|----------------------|------------------------|
| Double Top | `classic/double_top.py` ✅ |
| Double Bottom | `classic/double_bottom.py` ✅ |
| Triple Top | `classic/triple_top.py` ✅ |
| Head and Shoulders | `complex/head_shoulders.py` ✅ |
| Cup and Handle | `complex/cup_handle.py` ✅ |
| Symmetric Triangle | `harmonic/symmetric_triangle.py` ✅ |
| NR4/NR7 | `basic/nr7id.py` (combined with Inside Day) ✅ |
| Inside Bar | `basic/nr7id.py` (combined with NR7) ✅ |

---

## Implementation Architecture

### Directory Structure Update

```
src/patterns/
├── base.py                    # Existing - no changes
├── __init__.py                # Update - add new exports
├── basic/                     # Existing
│   ├── __init__.py            # Update - add TwoBarReversal
│   ├── msl.py
│   ├── matching_lows.py
│   ├── nr7id.py
│   ├── n_bar_decline.py
│   ├── floor_pivot.py
│   └── two_bar_reversal.py    # NEW
├── classic/                   # Existing
│   ├── __init__.py            # Update - add new exports
│   ├── double_top.py
│   ├── double_bottom.py
│   ├── trader_vic_2b.py
│   ├── triple_top.py
│   ├── triple_bottom.py       # NEW
│   ├── ascending_triangle.py  # NEW
│   ├── descending_triangle.py # NEW
│   ├── rectangle.py           # NEW
│   ├── wedge.py               # NEW
│   └── dead_cat_bounce.py
├── continuation/              # NEW DIRECTORY
│   ├── __init__.py            # NEW
│   ├── flag.py                # NEW
│   └── pennant.py             # NEW
├── breakout/                  # NEW DIRECTORY
│   ├── __init__.py            # NEW
│   └── gap.py                 # NEW
├── candlestick/               # NEW DIRECTORY
│   ├── __init__.py            # NEW
│   ├── doji.py                # NEW
│   ├── harami.py              # NEW
│   ├── hammer.py              # NEW
│   ├── engulfing.py           # NEW
│   └── dark_cloud.py          # NEW
├── harmonic/                  # Existing - no changes
└── complex/                   # Existing - no changes
```

---

## Detailed Pattern Specifications

### 1. Triple Bottom (classic/triple_bottom.py)

```python
# Structure
- Three troughs at approximately same price level (support)
- Separated by two intermediate peaks
- Troughs can be pointed or rounded

# Detection Logic
if (trough_3 exists within ±tolerance% of trough_1.price and trough_2.price)
   and (two_peaks exist between them)
   and (price breaks above highest_peak OR resistance_line):
   pattern_activated = True

# Signal: BUY on confirmed breakout
# Target Price: breakout_price + (highest_peak - lowest_trough)
# Entry: On breakout OR on throwback to broken resistance (now support)
# Protective Stop: Below most recent trough low - filter

# Implementation Notes
- Mirror logic of existing TripleTop pattern
- Use find_swing_lows() for trough detection
- Use find_swing_highs() for peak detection between troughs
- Parameters: lookback=5, trough_tolerance=0.03, min_pattern_bars=15, max_pattern_bars=100
```

### 2. Ascending Triangle (classic/ascending_triangle.py)

```python
# Structure
- Upper bound: horizontal resistance line (flat highs)
- Lower bound: upward-sloping support line (higher lows)
- Minimum 2 touches of resistance + 2 touches of support

# Detection Logic
if (horizontal_resistance_confirmed: at least 2 highs within tolerance)
   and (rising_support_confirmed: at least 2 higher lows with positive slope)
   and (price breaks above resistance with confirmation):
   pattern_activated = True  # Upward breakout more common

# Signal: BUY on upside breakout (primary); SELL only on confirmed downside break
# Target: breakout_price + (resistance - lowest_trough_in_pattern)
# Stop Loss: Below the upward-sloping support line or pattern low

# Implementation Notes
- Detect horizontal resistance: find peaks within price tolerance
- Detect rising support: linear regression on swing lows
- Calculate slope of support line, must be positive
- Confirmation filter: close above resistance
```

### 3. Descending Triangle (classic/descending_triangle.py)

```python
# Structure
- Upper bound: downward-sloping resistance line (lower highs)
- Lower bound: horizontal support line (flat lows)
- Minimum 2 touches of resistance + 2 touches of support

# Detection Logic
if (falling_resistance_confirmed: at least 2 lower highs with negative slope)
   and (horizontal_support_confirmed: at least 2 lows within tolerance)
   and (price breaks below support with confirmation):
   pattern_activated = True  # Downward breakout more common

# Signal: SELL on downside breakout (primary); BUY only on confirmed upside break
# Target: breakout_price - (highest_peak_in_pattern - support)
# Stop Loss: Above the downward-sloping resistance line or pattern high

# Implementation Notes
- Mirror logic of Ascending Triangle
- Detect horizontal support: find troughs within price tolerance
- Detect falling resistance: linear regression on swing highs
- Calculate slope of resistance line, must be negative
```

### 4. Rectangle (classic/rectangle.py)

```python
# Structure
- Price oscillates between parallel horizontal support and resistance
- Minimum: 2 touches of support + 2 touches of resistance
- Can be continuation or reversal depending on breakout direction

# Detection Logic
if (price respects horizontal support AND resistance for N periods)
   and (support_level: at least 2 lows within tolerance)
   and (resistance_level: at least 2 highs within tolerance)
   and (breakout confirmed with filter):
   pattern_activated = True

# Signal: BUY on upside breakout / SELL on downside breakout
# Target: 
   - Upside: resistance + (resistance - support)
   - Downside: support - (resistance - support)
# Warning: High false breakout rate → require strong confirmation

# Implementation Notes
- Detect two horizontal levels within tolerance
- Track number of touches on each level
- Apply confirmation filter (percentage or close filter)
- Pattern height = resistance - support
```

### 5. Wedge (classic/wedge.py)

```python
# Structure - Rising Wedge (Bearish Reversal)
- Both trendlines slope UP
- Upper line has steeper slope than lower line
- Converging toward apex

# Structure - Falling Wedge (Bullish Reversal)
- Both trendlines slope DOWN
- Lower line has steeper slope than upper line
- Converging toward apex

# Detection Logic
if (both_trendlines_slope_same_direction)
   and (touch_count >= 5 total, min 3 on one side, 2 on other)
   and (breakout occurs OPPOSITE to wedge slope with confirmation):
   pattern_activated = True

# Signal: 
   - Rising Wedge breakout DOWN → SELL
   - Falling Wedge breakout UP → BUY
# Target:
   - Downward breakout: target = lowest_trough_in_pattern
   - Upward breakout: target = breakout_price + (highest_peak - lowest_trough)
# Warning: High retracement rate → use wider stops

# Implementation Notes
- Use linear regression to calculate trendline slopes
- Both slopes must have same sign (both positive or both negative)
- Minimum 5 total touches across both trendlines
- Rising wedge = bearish, Falling wedge = bullish (counter-intuitive)
```

### 6. Flag/Pennant (continuation/flag.py, continuation/pennant.py)

```python
# Structure - Flag
- Flag Pole: sharp, steep price move (up or down), slope > threshold
- Flag: small parallel channel sloping against trend
- Duration: typically 1-4 weeks

# Structure - Pennant
- Flag Pole: sharp, steep price move
- Pennant: small symmetrical triangle sloping against trend
- Duration: typically 1-3 weeks

# Detection Logic
if (sharp_price_move_detected: slope > threshold over N bars)
   and (consolidation_pattern_detected: flag OR pennant)
   and (consolidation_slopes_opposite_to_pole)
   and (breakout in original trend direction with confirmation):
   pattern_activated = True

# Signal: Trade in direction of flag pole
# Target: breakout_price + flag_pole_height
   where flag_pole_height = pole_end_price - pole_start_price
# Entry: On breakout of flag/pennant boundary

# Implementation Notes
- Detect flag pole: calculate price change rate over lookback period
- Flag: parallel channel with slope opposite to pole
- Pennant: converging trendlines with slope opposite to pole
- Measure pole height for target calculation
```

### 7. Gap Pattern (breakout/gap.py)

```python
# Structure
- Gap Up: open > prior high (gap zone between prior high and current open)
- Gap Down: open < prior low (gap zone between current open and prior low)

# Explosion Gap Pivot Strategy
1. Wait for gap to occur
2. Monitor for throwback/pullback toward gap
3. If price retraces and STOPS (does not fill gap) → Pivot Point identified
   - Pivot Low (for gap up): lowest point of retracement that holds above gap
   - Pivot High (for gap down): highest point of retracement that holds below gap
4. Entry: Buy stop above high of gap candle (for gap up)
5. Protective Stop: Initially at gap low, then move to below pivot low

# Detection Logic
def detect_gap(current_open, current_low, current_high, prior_high, prior_low):
    gap_up = current_open > prior_high
    gap_down = current_open < prior_low
    gap_size = abs(current_open - (prior_high if gap_up else prior_low))
    return gap_up, gap_down, gap_size

# Signal: Trade in gap direction only if pivot confirmation occurs
# Warning: Gaps that fully fill invalidate signal

# Implementation Notes
- Track gap zones and check for partial/full fills
- Detect pivot points during retracements
- Require pivot confirmation before signal
```

### 8. Two-Bar Reversal / Pipe Bottom (basic/two_bar_reversal.py)

```python
# Structure - Pipe Bottom (bullish reversal)
- Occurs after extended downtrend (N bars declining)
- Bar 1: Long bearish candle, closes at or near low
- Bar 2: Long candle (any color), closes in upper 50% of its range
- Both bars have larger range than preceding 3-5 bars

# Structure - Pipe Top (bearish reversal)
- Occurs after extended uptrend
- Bar 1: Long bullish candle, closes at or near high
- Bar 2: Long candle (any color), closes in lower 50% of its range

# Detection Logic
if (prior_trend == DOWN for N periods)
   and (bar1.range > avg_range_recent AND bar1.close ≈ bar1.low)
   and (bar2.range > avg_range_recent AND bar2.close > bar2.midpoint)
   and (price breaks above bar2.high with confirmation):
   pattern_activated = True

# Signal: BUY on breakout above bar2.high
# Target: bar2.high + max(bar1.range, bar2.range)
# Note: More reliable on weekly data → adjust confidence by timeframe

# Implementation Notes
- Detect trend direction over lookback period
- Calculate average range for comparison
- Check bar characteristics (range, close position)
- Require confirmation breakout
```

### 9. Candlestick Patterns Module (candlestick/)

All candlestick patterns share common requirements:
- **Require confirmation** via next candle close OR volume spike OR indicator confluence
- **Alignment with higher-timeframe trend** for reliability
- **Low standalone reliability** - must be combined with other signals

#### 9.1 Doji (candlestick/doji.py)

```python
# Structure: Single candle where |open - close| < threshold% of (high - low)
# Interpretation: Market indecision → potential reversal warning

# Detection
def is_doji(open, close, high, low, threshold=0.1):
    body = abs(open - close)
    range_total = high - low
    return body < threshold * range_total

# DO NOT trade alone → flag for confirmation watchlist
# Confirm if next candle breaks doji high/low with volume
```

#### 9.2 Harami (candlestick/harami.py)

```python
# Structure: Two candles
# - Candle 1: Large body (bullish or bearish)
# - Candle 2: Small body of opposite color, fully contained within candle1's body

# Detection
if (candle1.body_range > threshold)
   and (candle2.color != candle1.color)
   and (candle2.open/close both within candle1.open/close range):
   harami_detected = True
   # Signal: Wait for breakout of candle1 high/low for direction
```

#### 9.3 Hammer/Hanging Man (candlestick/hammer.py)

```python
# Structure: Single candle
- Small body at top of range
- Long lower shadow (≥ 2x body length)
- Little/no upper shadow
- Hanging Man: appears after uptrend (bearish warning)
- Hammer: appears after downtrend (bullish warning)

# Implementation Note: Low standalone performance → only use with:
# - Volume confirmation
# - Support/resistance confluence
# - Next candle confirmation
```

#### 9.4 Engulfing (candlestick/engulfing.py)

```python
# Bullish Engulfing (reversal up):
- Candle 1: Bearish (close < open)
- Candle 2: Bullish (close > open) AND 
            candle2.open < candle1.close AND 
            candle2.close > candle1.open  # fully engulfs body

# Bearish Engulfing (reversal down): Mirror image

# Signal: 
if (engulfing_detected AND in_downtrend AND volume_spike):
   BUY signal with confirmation
# Target: Use nearest resistance/support or ATR-based projection
```

#### 9.5 Dark Cloud Cover / Piercing Line (candlestick/dark_cloud.py)

```python
# Dark Cloud Cover (Bearish):
- Candle 1: Strong bullish
- Candle 2: Opens above candle1.high, closes below candle1.midpoint, bearish

# Piercing Line (Bullish): Mirror image

# Implementation:
if (pattern_detected AND at_resistance/support AND volume_confirms):
   Generate signal with reduced position size (medium reliability)
```

---

## Implementation Priority and Phases

### Phase 1: Classic Chart Patterns (High Priority)

1. **Triple Bottom** - Mirror of existing TripleTop
2. **Ascending Triangle** - Common continuation pattern
3. **Descending Triangle** - Common continuation pattern
4. **Rectangle** - Horizontal channel pattern
5. **Wedge** - Important reversal pattern

### Phase 2: Continuation/Breakout Patterns (Medium Priority)

6. **Flag** - Short-term continuation
7. **Pennant** - Short-term continuation
8. **Gap Pattern** - Explosion gap pivot strategy
9. **Two-Bar Reversal** - Simple reversal pattern

### Phase 3: Candlestick Patterns (Lower Priority)

10. **Doji** - Indecision pattern
11. **Harami** - Reversal pattern
12. **Hammer/Hanging Man** - Reversal pattern
13. **Engulfing** - Higher reliability reversal
14. **Dark Cloud/Piercing** - Medium reliability reversal

---

## Integration with Existing System

### Base Pattern Class

All new patterns must inherit from [`BasePattern`](src/patterns/base.py:134) and implement:

```python
class NewPattern(BasePattern):
    def __init__(self, ...):
        super().__init__(
            name="Pattern Name",
            pattern_type=PatternType.REVERSAL | CONTINUATION | BREAKOUT,
            min_bars_required=N
        )
    
    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect pattern at bar index i."""
        pass
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal if pattern is detected."""
        pass
```

### Strategy Integration

Update [`multi_pattern_strategy_optimized.py`](src/strategies/backtest_py/multi_pattern_strategy_optimized.py:1) to include new patterns:

```python
# Add imports for new patterns
from src.patterns.classic.triple_bottom import TripleBottom
from src.patterns.classic.ascending_triangle import AscendingTriangle
from src.patterns.classic.descending_triangle import DescendingTriangle
from src.patterns.classic.rectangle import Rectangle
from src.patterns.classic.wedge import Wedge
from src.patterns.continuation.flag import Flag
from src.patterns.continuation.pennant import Pennant
from src.patterns.breakout.gap import GapPattern
from src.patterns.basic.two_bar_reversal import TwoBarReversal
# Candlestick patterns
from src.patterns.candlestick.doji import Doji
from src.patterns.candlestick.harami import Harami
from src.patterns.candlestick.hammer import Hammer
from src.patterns.candlestick.engulfing import Engulfing
from src.patterns.candlestick.dark_cloud import DarkCloudCover
```

### Confirmation Filters

Implement universal confirmation filters as per specification:

```python
class ConfirmationFilter:
    """Universal confirmation filters for pattern validation."""
    
    @staticmethod
    def intrabar(price, level, direction):
        """Price must close beyond level within same bar."""
        pass
    
    @staticmethod
    def multiple_closes(df, i, level, n=2, direction='above'):
        """N consecutive closes beyond breakout level."""
        pass
    
    @staticmethod
    def time_filter(df, i, level, periods=3, direction='above'):
        """Breakout must hold for X periods."""
        pass
    
    @staticmethod
    def percentage_filter(price, level, pct=0.5):
        """Breakout must exceed level by Y%."""
        return abs(price - level) / level >= pct / 100
    
    @staticmethod
    def point_filter(price, level, points):
        """Breakout must exceed level by Z price points."""
        return abs(price - level) >= points
```

---

## Testing Strategy

### Unit Tests

Each pattern requires comprehensive unit tests:

1. **Detection Tests**
   - Test with synthetic data containing the pattern
   - Test with edge cases (missing data, invalid inputs)
   - Test with real market data samples

2. **Signal Tests**
   - Verify entry price calculation
   - Verify stop loss placement
   - Verify take profit targets
   - Verify confidence scoring

3. **Integration Tests**
   - Test with multi-pattern strategy
   - Test with backtesting engine
   - Test confluence scoring

### Test File Structure

```
tests/
├── test_patterns.py           # Existing
├── test_new_patterns.py       # NEW - tests for new patterns
└── test_candlestick.py        # NEW - candlestick-specific tests
```

---

## Risk Management Template

Apply to all patterns as specified:

```python
class TradeSetup:
    def calculate_position_size(self, account_risk_pct, entry_price, stop_price):
        risk_per_share = abs(entry_price - stop_price)
        position_size = (account_balance * account_risk_pct) / risk_per_share
        return position_size
    
    def set_protective_stop(self, pattern_type, entry_price, pattern_data):
        if pattern_type in ['Double Top', 'Head and Shoulders Top', 'Triple Top', 
                           'Rising Wedge', 'Descending Triangle']:
            return max(pattern_data['recent_peak_high'], entry_price) * 1.01
        elif pattern_type in ['Double Bottom', 'Inverse H&S', 'Triple Bottom',
                             'Falling Wedge', 'Ascending Triangle']:
            return min(pattern_data['recent_trough_low'], entry_price) * 0.99
        # ... extend for all patterns
    
    def validate_breakout(self, price, breakout_level, filter_type='percentage', filter_value=0.5):
        if filter_type == 'percentage':
            return abs(price - breakout_level) / breakout_level >= filter_value / 100
        elif filter_type == 'points':
            return abs(price - breakout_level) >= filter_value
        # ... implement other filters
```

---

## Estimated Implementation Effort

| Phase | Patterns | Complexity | Dependencies |
|-------|----------|------------|--------------|
| Phase 1 | 5 classic patterns | Medium | Existing base class, pivot detection |
| Phase 2 | 4 continuation/breakout | Medium | Trend detection, gap detection |
| Phase 3 | 5 candlestick patterns | Low-Medium | Candlestick utilities, confirmation filters |

---

## Next Steps

1. **Review and approve this plan** with stakeholder
2. **Create feature branch** for new patterns
3. **Implement Phase 1** patterns first (highest value)
4. **Write unit tests** for each pattern
5. **Integrate with strategy** and run backtests
6. **Proceed to Phase 2 and Phase 3** based on results

---

*Plan Version: 1.0 | Created: 2026-03-18 | Based on Fidelity Chart Patterns Specification*
