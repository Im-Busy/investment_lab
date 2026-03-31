# Testing and Debugging Plan

## Overview

This document outlines a comprehensive testing and debugging strategy for the Trading Pattern Detection System, covering both Phase 1 optimizations (IndicatorCache, NumPy arrays, parallel detection) and Phase 2 optimizations (Numba JIT compilation, vectorized detection).

---

## Current State Analysis

### Test Files Present
- [`tests/test_base.py`](tests/test_base.py) - Tests for base pattern classes
- [`tests/test_indicators.py`](tests/test_indicators.py) - Tests for technical indicators
- [`tests/test_strategies.py`](tests/test_strategies.py) - Tests for strategy components
- [`tests/test_numba.py`](tests/test_numba.py) - Tests for Numba-accelerated functions

### Known Issues from Previous Test Run
1. `test_safe_float_conversion` - Expects `None` to return `0.0` but returns `np.nan`
2. `test_sma_period_1` - dtype mismatch (int64 vs float64)
3. `test_get_pattern_config` - Pattern config lookup returns None

### Key Files to Test
1. **Phase 1 Optimizations:**
   - [`src/indicators/indicator_cache.py`](src/indicators/indicator_cache.py) - NEW
   - [`src/patterns/base.py`](src/patterns/base.py) - MODIFIED
   - [`src/strategies/backtest_py/multi_pattern_strategy_optimized.py`](src/strategies/backtest_py/multi_pattern_strategy_optimized.py) - MODIFIED
   - [`src/patterns/basic/msl.py`](src/patterns/basic/msl.py) - MODIFIED

2. **Phase 2 Optimizations:**
   - [`src/indicators/pivots_numba.py`](src/indicators/pivots_numba.py) - NEW
   - [`src/indicators/technical_numba.py`](src/indicators/technical_numba.py) - NEW
   - [`src/patterns/basic/floor_pivot.py`](src/patterns/basic/floor_pivot.py) - MODIFIED with Numba

---

## Testing Strategy

### Phase 1: Unit Tests

#### 1.1 Test IndicatorCache

```python
# tests/test_indicator_cache.py

class TestIndicatorCache:
    """Tests for IndicatorCache class."""
    
    def test_cache_initialization(self):
        """Test IndicatorCache initializes correctly."""
        
    def test_get_swing_highs_caching(self):
        """Test swing highs are cached properly."""
        
    def test_get_swing_lows_caching(self):
        """Test swing lows are cached properly."""
        
    def test_get_sma_caching(self):
        """Test SMA is cached properly."""
        
    def test_get_ema_caching(self):
        """Test EMA is cached properly."""
        
    def test_get_atr_caching(self):
        """Test ATR is cached properly."""
        
    def test_pre_compute_common(self):
        """Test pre_compute_common populates cache."""
        
    def test_arrays_extraction(self):
        """Test NumPy arrays are extracted correctly."""
        
    def test_get_bar_fast(self):
        """Test fast bar access returns correct values."""
        
    def test_get_bars_range(self):
        """Test range extraction returns correct slice."""
```

#### 1.2 Test BasePattern Optimizations

```python
# tests/test_base_optimizations.py

class TestBasePatternOptimizations:
    """Tests for BasePattern optimization methods."""
    
    def test_extract_arrays(self):
        """Test _extract_arrays returns correct arrays."""
        
    def test_extract_arrays_caching(self):
        """Test arrays are cached by id()."""
        
    def test_get_bar_fast_valid_index(self):
        """Test _get_bar_fast with valid index."""
        
    def test_get_bar_fast_invalid_index(self):
        """Test _get_bar_fast with out-of-bounds index."""
        
    def test_detect_with_window_start(self):
        """Test detect() with window_start parameter."""
        
    def test_detect_batch(self):
        """Test detect_batch() returns correct results."""
        
    def test_set_indicator_cache(self):
        """Test set_indicator_cache() stores reference."""
```

#### 1.3 Test PatternDetector Optimizations

```python
class TestPatternDetectorOptimizations:
    """Tests for PatternDetector optimization methods."""
    
    def test_detect_all_preallocated(self):
        """Test detect_all uses pre-allocated storage."""
        
    def test_detect_batch(self):
        """Test detect_batch processes range correctly."""
        
    def test_set_indicator_cache_propagates(self):
        """Test indicator cache is propagated to patterns."""
```

### Phase 2: Numba Integration Tests

#### 2.1 Test pivots_numba.py

```python
# tests/test_numba.py - Extended

class TestPivotsNumbaExtended:
    """Extended tests for Numba-accelerated pivot detection."""
    
    def test_find_swing_highs_numba_matches_pandas(self):
        """Test Numba version matches original pandas version."""
        
    def test_find_swing_lows_numba_matches_pandas(self):
        """Test Numba version matches original pandas version."""
        
    def test_find_local_extrema_numba(self):
        """Test local extrema detection."""
        
    def test_numba_fallback_when_unavailable(self):
        """Test fallback works when Numba not installed."""
        
    def test_numba_performance(self):
        """Benchmark Numba vs pure Python."""
```

#### 2.2 Test technical_numba.py

```python
class TestTechnicalNumbaExtended:
    """Extended tests for Numba-accelerated technical indicators."""
    
    def test_sma_numba_matches_pandas(self):
        """Test Numba SMA matches pandas version."""
        
    def test_ema_numba_matches_pandas(self):
        """Test Numba EMA matches pandas version."""
        
    def test_atr_numba_matches_pandas(self):
        """Test Numba ATR matches pandas version."""
        
    def test_rsi_numba_matches_pandas(self):
        """Test Numba RSI matches pandas version."""
        
    def test_true_range_numba(self):
        """Test Numba true range calculation."""
        
    def test_donchian_channel_numba(self):
        """Test Numba Donchian channel calculation."""
```

### Phase 3: Integration Tests

#### 3.1 Test Strategy Integration

```python
# tests/test_strategy_integration.py

class TestMultiPatternStrategyOptimized:
    """Integration tests for optimized strategy."""
    
    def test_strategy_initialization(self):
        """Test strategy initializes with all optimizations."""
        
    def test_indicator_cache_created(self):
        """Test IndicatorCache is created and populated."""
        
    def test_patterns_have_indicator_cache(self):
        """Test all patterns receive indicator cache."""
        
    def test_detect_patterns_sequential(self):
        """Test sequential pattern detection works."""
        
    def test_detect_patterns_parallel(self):
        """Test parallel pattern detection works."""
        
    def test_window_bounds_passed_correctly(self):
        """Test window_start is passed to pattern detectors."""
        
    def test_numpy_arrays_used(self):
        """Test NumPy arrays are used for data access."""
```

#### 3.2 Test Backtest Runner

```python
# tests/test_backtest_runner.py

class TestBacktestPyRunner:
    """Tests for backtesting.py runner."""
    
    def test_runner_initialization(self):
        """Test runner initializes correctly."""
        
    def test_data_preparation(self):
        """Test data is prepared correctly for backtesting."""
        
    def test_run_backtest(self):
        """Test backtest runs without errors."""
        
    def test_backtest_with_optimized_strategy(self):
        """Test backtest with optimized strategy."""
```

### Phase 4: Performance Benchmarks

#### 4.1 Create Benchmark Script

```python
# scripts/benchmark_phase2.py

"""
Performance Benchmark for Phase 1 and Phase 2 Optimizations

Compares:
1. Baseline (original implementation)
2. Phase 1 (IndicatorCache, NumPy, parallel)
3. Phase 2 (Numba JIT, vectorized)
"""

import time
import numpy as np
import pandas as pd

def benchmark_indicator_cache():
    """Benchmark IndicatorCache vs repeated calculations."""
    
def benchmark_numpy_vs_dataframe():
    """Benchmark NumPy array access vs DataFrame access."""
    
def benchmark_numba_vs_python():
    """Benchmark Numba JIT vs pure Python."""
    
def benchmark_pattern_detection():
    """Benchmark pattern detection with different optimizations."""
    
def benchmark_full_backtest():
    """Benchmark complete backtest run."""
    
def run_all_benchmarks():
    """Run all benchmarks and print results."""
```

---

## Debugging Checklist

### Known Issues to Fix

1. **test_safe_float_conversion**
   - Location: `tests/test_base.py:297`
   - Issue: Test expects `None` to return `0.0`, but `_safe_float()` returns `np.nan`
   - Fix: Either update test or update `_safe_float()` to return `0.0` for `None`

2. **test_sma_period_1**
   - Location: `tests/test_indicators.py:58`
   - Issue: dtype mismatch (float64 vs int64)
   - Fix: Ensure SMA returns float64 consistently

3. **test_get_pattern_config**
   - Location: `tests/test_strategies.py:444`
   - Issue: Pattern config lookup returns None
   - Fix: Investigate pattern config initialization

### Phase 1 Integration Issues to Check

1. **IndicatorCache Integration**
   - [ ] Verify cache is created in strategy init()
   - [ ] Verify cache is passed to all patterns
   - [ ] Verify pre_compute_common() is called
   - [ ] Verify cached values are used in pattern detection

2. **NumPy Array Access**
   - [ ] Verify _extract_arrays() works in all patterns
   - [ ] Verify array bounds checking
   - [ ] Verify NaN handling in arrays

3. **Window Bounds**
   - [ ] Verify window_start is passed correctly
   - [ ] Verify patterns handle window_start parameter
   - [ ] Verify no DataFrame slicing occurs

4. **Parallel Detection**
   - [ ] Verify ThreadPoolExecutor is created lazily
   - [ ] Verify parallel mode can be toggled
   - [ ] Verify thread safety

### Phase 2 Numba Issues to Check

1. **Numba Availability**
   - [ ] Verify NUMBA_AVAILABLE flag works
   - [ ] Verify fallback functions work without Numba
   - [ ] Verify JIT compilation cache works

2. **Numba Function Correctness**
   - [ ] Verify Numba functions match original implementations
   - [ ] Verify edge cases (NaN, empty arrays, etc.)
   - [ ] Verify numerical precision

---

## Test Execution Plan

### Step 1: Run Existing Tests
```bash
uv run pytest tests/ -v --tb=short
```

### Step 2: Create Missing Test Files
```bash
# Create new test files
touch tests/test_indicator_cache.py
touch tests/test_base_optimizations.py
touch tests/test_strategy_integration.py
touch tests/test_backtest_runner.py
```

### Step 3: Run Tests by Category
```bash
# Unit tests
uv run pytest tests/test_base.py tests/test_indicators.py -v

# Numba tests
uv run pytest tests/test_numba.py -v

# Strategy tests
uv run pytest tests/test_strategies.py -v
```

### Step 4: Run Benchmarks
```bash
uv run python scripts/benchmark_phase2.py
```

### Step 5: Run Integration Test with Real Data
```bash
uv run python scripts/test_spy_backtest.py
```

---

## Expected Test Results

### After Fixes
- All unit tests pass (69/69)
- All Numba tests pass
- All integration tests pass
- Benchmark shows expected speedups:
  - Phase 1: 5-10x improvement
  - Phase 2: 50-300x improvement

---

## Files to Create/Modify

### New Files
1. `tests/test_indicator_cache.py` - Tests for IndicatorCache
2. `tests/test_base_optimizations.py` - Tests for BasePattern optimizations
3. `tests/test_strategy_integration.py` - Integration tests
4. `tests/test_backtest_runner.py` - Runner tests
5. `scripts/benchmark_phase2.py` - Performance benchmarks

### Files to Modify
1. `tests/test_base.py` - Fix `test_safe_float_conversion`
2. `tests/test_indicators.py` - Fix `test_sma_period_1`
3. `tests/test_strategies.py` - Fix `test_get_pattern_config`
4. `src/patterns/base.py` - Ensure `_safe_float()` behavior is consistent

---

## Success Criteria

1. **All tests pass** (no failures)
2. **No runtime errors** when running backtests
3. **Performance improvements verified** via benchmarks
4. **Code coverage** maintained or improved
5. **Documentation updated** for new features
