# Phase 1 Implementation Plan: Foundation Layer Optimizations

## Overview

This document provides detailed implementation instructions for Phase 1 optimizations from the multi-pattern backtest optimization plan. Each optimization includes specific file changes, code snippets, and implementation notes.

**Expected Total Speedup:** 15-30x combined

---

## Optimization 1: Pre-compute and Cache Common Indicators

**Impact:** 9/10 | **Difficulty:** 2/10 | **Expected Speedup:** 5-10x

### Problem
Each pattern independently calculates:
- Swing highs/lows (via `find_swing_highs`, `find_swing_lows`)
- Moving averages (SMA, EMA)
- Volume SMA
- ATR for stop-loss calculations

### Implementation

#### Step 1.1: Create IndicatorCache Class

**New File:** `src/indicators/indicator_cache.py`

```python
"""
Indicator Cache for Performance Optimization

Pre-computes and caches common indicators to avoid redundant calculations
across multiple pattern detectors.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from functools import lru_cache

from .pivots import find_swing_highs, find_swing_lows
from .technical import sma, ema, atr, volume_sma, true_range


class IndicatorCache:
    """
    Pre-compute indicators once per backtest and cache for reuse.
    
    Usage:
        cache = IndicatorCache(df)
        swing_highs = cache.get_swing_highs(lookback=5)
        sma_20 = cache.get_sma('Close', 20)
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize cache with DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
        """
        self._df = df
        self._cache: Dict[str, any] = {}
        
        # Pre-extract NumPy arrays for faster access
        self._arrays: Dict[str, np.ndarray] = {
            'open': df['Open'].values,
            'high': df['High'].values,
            'low': df['Low'].values,
            'close': df['Close'].values,
            'volume': df['Volume'].values if 'Volume' in df.columns else None
        }
        self._index = df.index
    
    @property
    def arrays(self) -> Dict[str, np.ndarray]:
        """Get pre-extracted NumPy arrays."""
        return self._arrays
    
    @property
    def df(self) -> pd.DataFrame:
        """Get the underlying DataFrame."""
        return self._df
    
    def get_swing_highs(self, lookback: int = 5) -> pd.Series:
        """Get cached swing highs."""
        key = f'swing_highs_{lookback}'
        if key not in self._cache:
            self._cache[key] = find_swing_highs(self._df, lookback)
        return self._cache[key]
    
    def get_swing_lows(self, lookback: int = 5) -> pd.Series:
        """Get cached swing lows."""
        key = f'swing_lows_{lookback}'
        if key not in self._cache:
            self._cache[key] = find_swing_lows(self._df, lookback)
        return self._cache[key]
    
    def get_sma(self, column: str, period: int) -> pd.Series:
        """Get cached Simple Moving Average."""
        key = f'sma_{column}_{period}'
        if key not in self._cache:
            self._cache[key] = sma(self._df[column], period)
        return self._cache[key]
    
    def get_ema(self, column: str, period: int) -> pd.Series:
        """Get cached Exponential Moving Average."""
        key = f'ema_{column}_{period}'
        if key not in self._cache:
            self._cache[key] = ema(self._df[column], period)
        return self._cache[key]
    
    def get_atr(self, period: int = 14) -> pd.Series:
        """Get cached Average True Range."""
        key = f'atr_{period}'
        if key not in self._cache:
            self._cache[key] = atr(self._df, period)
        return self._cache[key]
    
    def get_volume_sma(self, period: int = 20) -> pd.Series:
        """Get cached Volume SMA."""
        key = f'volume_sma_{period}'
        if key not in self._cache:
            if self._arrays['volume'] is not None:
                self._cache[key] = volume_sma(self._df, period)
            else:
                self._cache[key] = pd.Series(np.nan, index=self._index)
        return self._cache[key]
    
    def get_true_range(self) -> pd.Series:
        """Get cached True Range."""
        key = 'true_range'
        if key not in self._cache:
            self._cache[key] = true_range(self._df)
        return self._cache[key]
    
    def pre_compute_common(self) -> None:
        """Pre-compute most commonly used indicators."""
        # Swing points with default lookback
        self.get_swing_highs(5)
        self.get_swing_lows(5)
        
        # Common moving averages
        self.get_sma('Close', 20)
        self.get_sma('Close', 50)
        self.get_ema('Close', 20)
        
        # ATR for stop-loss
        self.get_atr(14)
        
        # Volume
        self.get_volume_sma(20)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
```

#### Step 1.2: Update `src/indicators/__init__.py`

Add the new IndicatorCache to exports:

```python
from .indicator_cache import IndicatorCache
```

#### Step 1.3: Modify `multi_pattern_strategy_optimized.py`

Update the `init()` method to use IndicatorCache:

```python
# In imports section
from src.indicators.indicator_cache import IndicatorCache

# In init() method, after creating DataFrame:
# OPTIMIZATION: Create indicator cache and pre-compute
self.indicator_cache = IndicatorCache(self._df)
self.indicator_cache.pre_compute_common()

# Pass cache to patterns that support it
for pattern in self.patterns:
    if hasattr(pattern, 'set_indicator_cache'):
        pattern.set_indicator_cache(self.indicator_cache)
```

---

## Optimization 2: Avoid DataFrame Slicing in Hot Loop

**Impact:** 5/10 | **Difficulty:** 2/10 | **Expected Speedup:** 2x

### Problem
Current code creates a new DataFrame slice every bar:
```python
window_start = max(0, current_idx - 200)
df_window = self._df.iloc[window_start : current_idx + 1]
window_idx = len(df_window) - 1
```

### Implementation

#### Step 2.1: Update `base.py` - Add window bounds to detect()

Modify the `BasePattern` class:

```python
@abstractmethod
def detect(self, df: pd.DataFrame, i: int, window_start: int = None) -> PatternResult:
    """
    Detect pattern at bar index i.
    
    Args:
        df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
        i: Current bar index to check for pattern
        window_start: Optional start index for window (avoids slicing)
        
    Returns:
        PatternResult containing detection status and any generated signal
    """
    pass
```

#### Step 2.2: Update `multi_pattern_strategy_optimized.py`

Replace DataFrame slicing with index passing:

```python
# In next() method, REPLACE:
# window_start = max(0, current_idx - 200)
# df_window = self._df.iloc[window_start : current_idx + 1]
# window_idx = len(df_window) - 1
# result = pattern.detect(df_window, window_idx)

# WITH:
window_start = max(0, current_idx - 200)
result = pattern.detect(self._df, current_idx, window_start=window_start)
```

#### Step 2.3: Update Pattern Files

Each pattern's `detect()` method needs to handle window bounds. Example for `msl.py`:

```python
def detect(self, df: pd.DataFrame, i: int, window_start: int = None) -> PatternResult:
    """
    Detect MSL pattern at bar index i.
    
    Args:
        df: Full DataFrame with OHLCV data
        i: Current bar index (absolute position in df)
        window_start: Optional window start for bounds checking
    """
    # Use window_start for bounds validation if provided
    min_idx = window_start if window_start is not None else 0
    
    if i < max(min_idx + 3, self.min_bars_required):
        return PatternResult(
            detected=False,
            pattern_name=self.name,
            pattern_type=self.pattern_type
        )
    
    # Access data directly using absolute index
    c_minus_2 = self._safe_float(df.iloc[i-2]['Close'])
    # ... rest of detection logic
```

---

## Optimization 3: Convert to NumPy Arrays

**Impact:** 5/10 | **Difficulty:** 3/10 | **Expected Speedup:** 1.5-3x

### Problem
Pattern detectors use `df.iloc[i]['Close']` which is slow compared to NumPy array access.

### Implementation

#### Step 3.1: Add Array Extraction to BasePattern

Update `src/patterns/base.py`:

```python
class BasePattern(ABC):
    # ... existing code ...
    
    def _extract_arrays(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract NumPy arrays from DataFrame for faster access.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with 'open', 'high', 'low', 'close', 'volume' arrays
        """
        return {
            'open': df['Open'].values,
            'high': df['High'].values,
            'low': df['Low'].values,
            'close': df['Close'].values,
            'volume': df['Volume'].values if 'Volume' in df.columns else None
        }
    
    def _get_bar_fast(self, arrays: Dict[str, np.ndarray], i: int) -> Dict[str, float]:
        """
        Get bar data at index i using NumPy arrays (faster than DataFrame access).
        
        Args:
            arrays: Dictionary of NumPy arrays from _extract_arrays()
            i: Bar index
            
        Returns:
            Dictionary with 'open', 'high', 'low', 'close', 'volume' values
        """
        if i < 0 or i >= len(arrays['close']):
            return None
        
        return {
            'open': arrays['open'][i],
            'high': arrays['high'][i],
            'low': arrays['low'][i],
            'close': arrays['close'][i],
            'volume': arrays['volume'][i] if arrays['volume'] is not None else np.nan
        }
```

#### Step 3.2: Update Pattern Files (Priority Order)

**High Priority (most frequently used):**
1. `src/patterns/basic/msl.py`
2. `src/patterns/basic/matching_lows.py`
3. `src/patterns/basic/nr7id.py`
4. `src/patterns/basic/n_bar_decline.py`

**Example Update for msl.py:**

```python
def detect(self, df: pd.DataFrame, i: int, window_start: int = None) -> PatternResult:
    """Detect MSL pattern using NumPy arrays for speed."""
    if not self._validate_data(df, i):
        return PatternResult(
            detected=False,
            pattern_name=self.name,
            pattern_type=self.pattern_type
        )
    
    if i < 3:
        return PatternResult(
            detected=False,
            pattern_name=self.name,
            pattern_type=self.pattern_type
        )
    
    # OPTIMIZATION: Use NumPy arrays instead of DataFrame access
    if not hasattr(self, '_arrays') or self._arrays_df is not df:
        self._arrays = self._extract_arrays(df)
        self._arrays_df = df
    
    close = self._arrays['close']
    
    # Direct array access - much faster than df.iloc[i]['Close']
    c_minus_2 = close[i-2]
    c_minus_1 = close[i-1]
    c_0 = close[i]
    
    # Condition 1: Down Move
    condition_1 = c_minus_1 < c_minus_2
    
    # Condition 2: Higher Low of Close
    condition_2 = (c_0 > c_minus_1) and (c_0 < c_minus_2)
    
    # ... rest of logic using array access
```

---

## Optimization 4: Pre-allocate Result Arrays

**Impact:** 3/10 | **Difficulty:** 3/10 | **Expected Speedup:** 1.5x

### Problem
Lists grow dynamically in `detect_all()` method.

### Implementation

#### Step 4.1: Update PatternDetector.detect_all() in base.py

```python
def detect_all(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Run pattern detection across all bars with pre-allocated storage.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        List of all detected signals with metadata
    """
    n_bars = len(df)
    n_patterns = len(self.patterns)
    
    # Pre-allocate result storage (worst case: all patterns at all bars)
    max_results = n_bars * n_patterns
    all_signals = [None] * max_results
    result_idx = 0
    
    # Get minimum bars required
    min_bars = max((p.min_bars_required for p in self.patterns), default=5)
    
    for i in range(min_bars, n_bars):
        for pattern in self.patterns:
            try:
                result = pattern.detect(df, i)
                if result.detected and result.signal:
                    signal_dict = result.to_dict()
                    signal_dict['bar_index'] = i
                    all_signals[result_idx] = signal_dict
                    result_idx += 1
            except Exception as e:
                # Log error but continue
                print(f"Error detecting {pattern.name} at bar {i}: {e}")
    
    # Trim unused slots
    return all_signals[:result_idx]
```

---

## Optimization 5: Memoize with @lru_cache

**Impact:** 4/10 | **Difficulty:** 2/10 | **Expected Speedup:** 2-4x

### Problem
Indicator calculations are repeated across patterns.

### Implementation

#### Step 5.1: Add Cached Functions to `technical.py`

```python
from functools import lru_cache
import hashlib

def _make_hashable(arr: np.ndarray) -> tuple:
    """Convert numpy array to hashable tuple for caching."""
    # Use last N bars for hashing (reduces memory, maintains uniqueness)
    return tuple(arr[-100:].tobytes())

@lru_cache(maxsize=128)
def _cached_sma_core(prices_bytes: bytes, period: int, length: int) -> tuple:
    """Core cached SMA calculation."""
    prices = np.frombuffer(prices_bytes)
    result = np.convolve(prices, np.ones(period)/period, mode='valid')
    return tuple(result)

def sma_cached(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
    """
    SMA with caching for repeated calls.
    
    Use this when the same data is queried multiple times.
    """
    if isinstance(data, pd.Series):
        arr = data.values
        index = data.index
    else:
        arr = data
        index = pd.RangeIndex(len(arr))
    
    # Create hashable key from last 100 bars + period
    key_data = arr[-min(100, len(arr)):].tobytes()
    
    result = _cached_sma_core(key_data, period, len(arr))
    
    # Reconstruct full SMA (cached result is partial)
    # Fall back to regular calculation for full data
    return pd.Series(arr, index=index).rolling(window=period).mean()

@lru_cache(maxsize=64)
def _cached_ema_core(prices_bytes: bytes, period: int) -> tuple:
    """Core cached EMA calculation."""
    prices = np.frombuffer(prices_bytes)
    # EMA calculation
    multiplier = 2 / (period + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    return tuple(ema)

def ema_cached(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
    """EMA with caching for repeated calls."""
    if isinstance(data, pd.Series):
        arr = data.values
        index = data.index
    else:
        arr = data
        index = pd.RangeIndex(len(arr))
    
    key_data = arr[-min(100, len(arr)):].tobytes()
    result = np.array(_cached_ema_core(key_data, period))
    
    return pd.Series(result, index=index)
```

#### Step 5.2: Add Cached Functions to `pivots.py`

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def _cached_swing_highs_numba(highs_tuple: tuple, lookback: int) -> tuple:
    """Cached swing high detection using tuple input."""
    highs = np.array(highs_tuple)
    n = len(highs)
    swing_highs = np.full(n, np.nan)
    
    for i in range(lookback, n - lookback):
        is_swing_high = True
        current_high = highs[i]
        
        for j in range(1, lookback + 1):
            if highs[i - j] >= current_high:
                is_swing_high = False
                break
            if highs[i + j] >= current_high:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs[i] = current_high
    
    return tuple(swing_highs)

def find_swing_highs_cached(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Find swing highs with caching."""
    # Use a subset for cache key to limit memory
    highs_tuple = tuple(df['High'].values)
    result = _cached_swing_highs_numba(highs_tuple, lookback)
    return pd.Series(result, index=df.index)
```

---

## Optimization 6: Parallel Pattern Detection

**Impact:** 8/10 | **Difficulty:** 5/10 | **Expected Speedup:** 4-8x on multi-core CPU

### Problem
Pattern detection runs sequentially for each bar.

### Implementation

#### Step 6.1: Add Parallel Detection to `multi_pattern_strategy_optimized.py`

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

class MultiPatternStrategyOptimized(Strategy):
    # ... existing code ...
    
    def init(self):
        """Initialize with parallel detection support."""
        # ... existing initialization ...
        
        # OPTIMIZATION: Parallel pattern detection
        self._max_workers = min(len(self.patterns), multiprocessing.cpu_count())
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
    
    def _detect_pattern(self, pattern, df: pd.DataFrame, idx: int, window_start: int = None):
        """
        Helper for parallel pattern detection.
        
        Args:
            pattern: Pattern detector instance
            df: DataFrame with OHLCV data
            idx: Current bar index
            window_start: Optional window start bound
            
        Returns:
            Signal dict if pattern detected, None otherwise
        """
        try:
            result = pattern.detect(df, idx, window_start=window_start)
            if result.detected and result.signal:
                return {
                    'pattern_name': result.pattern_name,
                    'direction': result.signal.direction.value,
                    'entry_price': result.signal.entry_price,
                    'stop_loss': result.signal.stop_loss,
                    'take_profit_1': result.signal.take_profit_1,
                    'take_profit_2': result.signal.take_profit_2,
                    'take_profit_3': result.signal.take_profit_3,
                    'confidence': result.signal.confidence,
                    'pattern_type': result.pattern_type.value,
                }
        except Exception:
            pass
        return None
    
    def next(self):
        """Execute trading logic with parallel pattern detection."""
        # ... existing checks ...
        
        window_start = max(0, current_idx - 200)
        
        # OPTIMIZATION: Parallel pattern detection
        futures = [
            self._executor.submit(
                self._detect_pattern, 
                pattern, 
                self._df, 
                current_idx, 
                window_start
            )
            for pattern in self.patterns
        ]
        
        signals = []
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                signals.append(result)
        
        # ... rest of trading logic ...
```

**Note:** ThreadPoolExecutor works well for I/O-bound tasks. For CPU-bound pattern detection, consider ProcessPoolExecutor or the `multiprocessing` module for true parallelism.

---

## Optimization 7: Batch Process Bars

**Impact:** 6/10 | **Difficulty:** 4/10 | **Expected Speedup:** 2-4x

### Problem
Pattern detection processes one bar at a time, missing optimization opportunities.

### Implementation

#### Step 7.1: Add Batch Detection to BasePattern

Update `src/patterns/base.py`:

```python
def detect_batch(
    self, 
    df: pd.DataFrame, 
    start_idx: int, 
    end_idx: int,
    window_start: int = None
) -> List[PatternResult]:
    """
    Detect patterns across a range of bars efficiently.
    
    Default implementation calls detect() for each bar.
    Subclasses can override for vectorized detection.
    
    Args:
        df: DataFrame with OHLCV data
        start_idx: Start bar index (inclusive)
        end_idx: End bar index (exclusive)
        window_start: Optional window start bound
        
    Returns:
        List of PatternResult objects for bars with detections
    """
    results = []
    for i in range(start_idx, end_idx):
        result = self.detect(df, i, window_start=window_start)
        if result.detected:
            results.append(result)
    return results
```

#### Step 7.2: Add Batch Detection to PatternDetector

```python
def detect_batch(
    self, 
    df: pd.DataFrame, 
    start_idx: int, 
    end_idx: int
) -> List[Dict[str, Any]]:
    """
    Run all pattern detectors across a batch of bars.
    
    Args:
        df: DataFrame with OHLCV data
        start_idx: Start bar index
        end_idx: End bar index
        
    Returns:
        List of all detected signals
    """
    all_signals = []
    window_start = max(0, start_idx - 200)
    
    for pattern in self.patterns:
        try:
            results = pattern.detect_batch(df, start_idx, end_idx, window_start)
            for result in results:
                if result.signal:
                    signal_dict = result.to_dict()
                    signal_dict['bar_index'] = result.end_index
                    all_signals.append(signal_dict)
        except Exception as e:
            print(f"Error in batch detection for {pattern.name}: {e}")
    
    return all_signals
```

#### Step 7.3: Add Caching Strategy in Strategy

```python
class MultiPatternStrategyOptimized(Strategy):
    def init(self):
        # ... existing code ...
        
        # OPTIMIZATION: Batch processing cache
        self._detection_cache = {}
        self._cache_valid_until = 0
    
    def next(self):
        current_idx = len(self.data.Close) - 1
        
        # Use cached results if still valid
        if current_idx <= self._cache_valid_until:
            signals = self._detection_cache.get(current_idx, [])
        else:
            # Process batch and cache results
            batch_size = 10  # Process 10 bars at once
            end_idx = min(current_idx + batch_size, len(self._df))
            
            window_start = max(0, current_idx - 200)
            signals = []
            
            for pattern in self.patterns:
                try:
                    result = pattern.detect(self._df, current_idx, window_start)
                    if result.detected and result.signal:
                        signals.append(result.to_dict())
                except Exception:
                    continue
            
            # Cache for future bars
            self._detection_cache[current_idx] = signals
        
        # ... rest of trading logic ...
```

---

## File Changes Summary

### New Files to Create
1. `src/indicators/indicator_cache.py` - Indicator caching class

### Files to Modify
1. `src/indicators/__init__.py` - Export IndicatorCache
2. `src/indicators/technical.py` - Add cached SMA/EMA functions
3. `src/indicators/pivots.py` - Add cached swing detection
4. `src/patterns/base.py` - Add array extraction, batch detection, window bounds
5. `src/strategies/backtest_py/multi_pattern_strategy_optimized.py` - Integrate all optimizations
6. All pattern files in `src/patterns/basic/`, `src/patterns/classic/`, `src/patterns/complex/`, `src/patterns/harmonic/` - Update detect() signatures

---

## Implementation Order

1. **Optimization 1** (IndicatorCache) - Foundation for other optimizations
2. **Optimization 3** (NumPy Arrays) - Works well with IndicatorCache
3. **Optimization 2** (No DataFrame Slicing) - Requires pattern file updates
4. **Optimization 4** (Pre-allocate Arrays) - Simple change to base.py
5. **Optimization 5** (lru_cache) - Independent, can be done anytime
6. **Optimization 6** (Parallel Detection) - Builds on previous optimizations
7. **Optimization 7** (Batch Processing) - Final optimization layer

---

## Testing Strategy

After each optimization:

1. Run existing tests: `pytest tests/`
2. Compare backtest results with original implementation
3. Measure performance improvement with timing
4. Verify no regression in pattern detection accuracy

```python
# Performance test script
import time
from src.strategies.backtest_py.multi_pattern_strategy_optimized import MultiPatternStrategyOptimized

# Before optimization
start = time.time()
# Run backtest
elapsed_before = time.time() - start

# After optimization
start = time.time()
# Run backtest with optimizations
elapsed_after = time.time() - start

print(f"Speedup: {elapsed_before / elapsed_after:.2f}x")
```

---

## Dependencies

No new external dependencies required. All optimizations use standard library (functools, concurrent.futures, multiprocessing) and existing dependencies (numpy, pandas).
