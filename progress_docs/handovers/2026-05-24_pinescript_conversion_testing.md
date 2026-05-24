# Handover: PineScript→Python Indicator Conversion — Comprehensive Testing (2026-05-24)

## Context

10 PineScript v5/v6 indicators from `useful_resources/pinescript_from_TradingView/` were manually converted to Python. They live in `src/indicators/` (9 modules) and `src/strategies/` (1 strategy). All modules import correctly and pass basic smoke tests on synthetic data — but **zero real-data validation** has been done.

**Your task: systematically test each module for correctness, edge cases, and whether it "does what it says it does."**

---

## Phase 0 — Read State

Start by reading these in order:
1. `MEMORY.md`
2. `progress_docs/plans/full.md`
3. `progress_docs/current.md`
4. `.kilo/project-loop.md`

Then load this handover file as your primary task.

---

## Phase 1 — Understand What Each Module Should Do

Before testing, read each module's PineScript source (in `useful_resources/pinescript_from_TradingView/indicators/`) and compare with the Python implementation. Here's the map:

### Indicators (src/indicators/)

| # | Python Module | PineScript Source | Author | Complexity |
|---|--------------|-------------------|--------|-----------|
| 1 | `fvg_profile.py` | `Fair Value Gap Profile + Rolling POC [BigBeluga]/` | BigBeluga | Simple |
| 2 | `autotune_filter.py` | `TASC 2026.05 The AutoTune Filter/` | John Ehlers | Medium |
| 3 | `asian_sweep_composite.py` | `AsianSweep_MSS_IFVG_HTF_Bias.txt` | TV community | Medium |
| 4 | `knn_pivots.py` | `Machine Learning Pivot Points (KNN) [SS]/` | Steversteves | Medium |
| 5 | `swing_forecast.py` | `Swing Structure Forecast [BOSWaves]/` | BOSWaves | Medium |
| 6 | `whale_profile.py` | `Whale Liquidity and Absorption Profile [AlgoAlpha]/` | AlgoAlpha | Medium |
| 7 | `microstructure.py` | `Market Microstructure Analytics/` | EdgeTools | Complex |
| 8 | `nwo.py` | `Neural Weight Oscillator (Zeiierman)/` | Zeiierman | Complex |
| 9 | `structure_volume.py` | `Market Structure Volume Profiles [Kioseff Trading]/` | KioseffTrading | Medium |

### Strategy (src/strategies/)

| # | Python Module | PineScript Source |
|---|--------------|-------------------|
| 10 | `asian_sweep_strategy.py` | `AsianLiquiditySweepTradingSystem.txt` |

---

## Phase 2 — Real-Data Smoke Tests

Run each module on real market data (not synthetic). Use yfinance or existing data files.

```bash
# Fetch SPY daily data for testing
uv run python -c "
import yfinance as yf
df = yf.download('SPY', '2020-01-01', '2024-12-31', auto_adjust=False)
df.to_csv('outputs/spy_test_data.csv')
print(f'{len(df)} bars loaded')
"
```

Then test each module on this data and verify:

### Test Checklist Per Module

#### 1. FVG Profile (`fvg_profile.py`)
- [ ] `compute_fvg_profile()` returns non-zero gaps on real data
- [ ] POC price is within the price range of the lookback window
- [ ] Bull/bear gap counts are sensible (not all zeros, not all one-sided)
- [ ] Strength percentages are in 0-100 range
- [ ] `compute_rolling_poc()` produces a non-NaN series
- [ ] Rolling POC prices are reasonable (within min/max of data)

```bash
uv run python -c "
import pandas as pd, numpy as np
from src.indicators.fvg_profile import compute_fvg_profile, compute_fvg_profile_df, compute_rolling_poc
df = pd.read_csv('outputs/spy_test_data.csv', index_col=0, parse_dates=True)
r = compute_fvg_profile_df(df, period=200, bin_count=30)
print(f'Gaps: {r.total_gap_count} (bull={r.bull_counts.sum()} bear={r.bear_counts.sum()})')
print(f'POC: \${r.poc_price:.2f} (bin {r.poc_index})')
print(f'Price range: \${r.bin_edges[0]:.2f}-\${r.bin_edges[-1]:.2f}')
print(f'Strength range: {r.strength_pct.min():.1f}%-{r.strength_pct.max():.1f}%')
# Verify POC is within valid range
assert r.bin_edges[0] <= r.poc_price <= r.bin_edges[-1], 'POC outside price range!'
print('PASS: POC within range')
"
```

#### 2. AutoTune Filter (`autotune_filter.py`)
- [ ] `auto_tune_filter()` completes without array size errors
- [ ] Dominant cycle is in sensible range (2-200)
- [ ] HP filter and band-pass outputs are not all NaN
- [ ] Cycle is rate-limited (doesn't jump more than ±2 per bar)
- [ ] Band-pass filter output has fewer zero-crossings than raw price

```bash
uv run python -c "
import pandas as pd, numpy as np
from src.indicators.autotune_filter import auto_tune_filter, DisplayMode, compute_display_series
df = pd.read_csv('outputs/spy_test_data.csv', index_col=0, parse_dates=True)
close = df['Close'].to_numpy(dtype=np.float64)
hp, mc, dc, bp = auto_tune_filter(close, window=20, bw=0.25)
print(f'Dominant cycle: min={np.nanmin(dc):.0f} max={np.nanmax(dc):.0f} last={dc[-1]:.0f}')
print(f'HP filter NaN count: {np.isnan(hp).sum()} of {len(hp)}')
print(f'BP filter NaN count: {np.isnan(bp).sum()} of {len(bp)}')
# Verify cycle rate limit
diffs = np.abs(np.diff(dc[~np.isnan(dc)]))
max_diff = np.nanmax(diffs) if len(diffs) > 0 else float('inf')
print(f'Max cycle jump: {max_diff:.1f} (should be <= 2)')
assert max_diff <= 2.1, 'Cycle jump exceeds rate limit!'
print('PASS: Max cycle jump within rate limit')
"
```

#### 3. Asian Sweep Composite (`asian_sweep_composite.py`)
- Attention: **This module uses artificial hour-based session detection** (simplified from the original PineScript's session time logic). Verify this matches the intended behavior.
- [ ] `detect_asian_sweep_setup()` runs on intraday data
- [ ] Produces signals only on real data (not random)
- [ ] Sweep detection logic is correct (sweep is detected after session close)

```bash
uv run python -c "
import pandas as pd, numpy as np
from src.indicators.asian_sweep_composite import detect_asian_sweep_setup
# Need intraday data — use 5-min or hourly
dates = pd.date_range('2024-01-02', periods=2000, freq='5min')
df = pd.DataFrame({
    'High': np.cumsum(np.random.randn(2000)*0.02)+100,
    'Low': np.cumsum(np.random.randn(2000)*0.02)+99.5,
    'Close': np.cumsum(np.random.randn(2000)*0.02)+99.8,
    'Volume': np.ones(2000)*1000
}, index=dates)
df['High'] = df[['High','Low','Close']].max(axis=1)
df['Low'] = df[['High','Low','Close']].min(axis=1)
sigs = detect_asian_sweep_setup(df, pivot_len=5)
print(f'Signals detected: {len(sigs)}')
for s in sigs[:5]:
    print(f'  {s.entry_signal} @ {s.entry_price:.2f} SL={s.stop_loss:.2f} TP={s.take_profit:.2f} conf={s.confidence:.1f}')
# Verify signal properties
for s in sigs:
    assert s.entry_signal in ('LONG', 'SHORT'), f'Invalid signal: {s.entry_signal}'
    assert s.stop_loss < s.entry_price if s.entry_signal == 'LONG' else s.stop_loss > s.entry_price, 'SL on wrong side!'
print('PASS: Signal properties valid')
"
```

#### 4. KNN Pivots (`knn_pivots.py`)
- [ ] `compute_knn_pivots()` finds transition signals on trending data
- [ ] Classification is one of three valid strings
- [ ] Confidence is 0-100
- [ ] Distance metrics are non-negative
- [ ] `compute_backtest_stats()` returns valid dict

```bash
uv run python -c "
import pandas as pd, numpy as np
from src.indicators.knn_pivots import compute_knn_pivots, compute_backtest_stats
df = pd.read_csv('outputs/spy_test_data.csv', index_col=0, parse_dates=True)
high = df['High'].to_numpy(dtype=np.float64)
low = df['Low'].to_numpy(dtype=np.float64)
close = df['Close'].to_numpy(dtype=np.float64)
results = compute_knn_pivots(high, low, close)
classes = set(r.classification for r in results)
transitions = [r for r in results if r.transition]
print(f'Unique classifications: {classes}')
print(f'Transitions: {len(transitions)}')
for r in results[-10:]:
    print(f'  {r.classification} conf={r.confidence:.1f}% dist_hi={r.distance_to_high:.4f} dist_lo={r.distance_to_low:.4f}')
# Verify no NaN in valid results
for r in results[20:]:
    assert not np.isnan(r.confidence), 'NaN confidence!'
    assert 0 <= r.confidence <= 100, f'Confidence {r.confidence} out of range!'
print('PASS: All confidences valid')
"
```

#### 5. Swing Structure Forecast (`swing_forecast.py`)
- [ ] `detect_swings()` returns non-empty pcts and durs on real data
- [ ] `forecast_next_swing()` returns None on insufficient data
- [ ] When returning a forecast, all fields are valid
- [ ] `compute_fib_levels()` produces correct Fibonacci sequence

```bash
uv run python -c "
import pandas as pd, numpy as np
from src.indicators.swing_forecast import detect_swings, forecast_next_swing, compute_fib_levels
df = pd.read_csv('outputs/spy_test_data.csv', index_col=0, parse_dates=True)
high = df['High'].to_numpy(dtype=np.float64)
low = df['Low'].to_numpy(dtype=np.float64)
close = df['Close'].to_numpy(dtype=np.float64)
swings, pcts, durs = detect_swings(high, low, swing_length=16)
print(f'Swing percentages: {len(pcts)} values, range [{min(pcts):.2f}%, {max(pcts):.2f}%]' if pcts else 'No swings')
print(f'Swing durations: {len(durs)} values, range [{min(durs):.0f}, {max(durs):.0f}] bars' if durs else '')
# Test all 3 methods
for method in ['Weighted', 'Average', 'Median']:
    f = forecast_next_swing(high, low, close, method=method)
    if f:
        print(f'{method}: target=\${f.target_price:.2f} pct={f.projected_pct:.2f}% std={f.std_dev:.3f}')
fibs = compute_fib_levels(100, 110)
print(f'Fib levels from 100→110: {[round(x,2) for x in fibs]}')
print('PASS' if pcts else 'PASS (no swings detected — needs more data)')
"
```

#### 6. Whale Profile (`whale_profile.py`)
- [ ] `compute_whale_profile()` completes on real data
- [ ] POC index is valid (0 to profile_bins-1)
- [ ] Strong+weak volume sum is approximately total volume
- [ ] Delta is buy_vol - sell_vol (verify sign consistency)
- [ ] Value area bins sum to a reasonable subset (not all True, not all False)

```bash
uv run python -c "
import pandas as pd, numpy as np
from src.indicators.whale_profile import compute_whale_profile
df = pd.read_csv('outputs/spy_test_data.csv', index_col=0, parse_dates=True)
high = df['High'].to_numpy(dtype=np.float64)
low = df['Low'].to_numpy(dtype=np.float64)
close = df['Close'].to_numpy(dtype=np.float64)
open_ = df['Open'].to_numpy(dtype=np.float64)
vol = df['Volume'].to_numpy(dtype=np.float64)
r = compute_whale_profile(high, low, close, vol, open_, lookback=200, profile_bins=35, detect_value_area=True)
total_split = r.strong_bull + r.weak_bull + r.weak_bear + r.strong_bear
total_from_parts = np.sum(total_split)
total_direct = np.sum(r.total_volume)
print(f'POC bin: {r.poc_index} vol={r.poc_volume:.0f}')
print(f'Total vol (parts): {total_from_parts:.0f} vs (direct): {total_direct:.0f}')
print(f'Delta sign: pos={np.sum(r.delta > 0)} neg={np.sum(r.delta < 0)}')
print(f'Value area bins: {np.sum(r.value_area_bins)} of {len(r.value_area_bins)}')
assert 0 <= r.poc_index < 35, 'POC index out of range!'
assert np.sum(r.value_area_bins) > 0, 'Value area is empty!'
print('PASS')
"
```

#### 7. Market Microstructure Analytics (`microstructure.py`)
- [ ] All 4 spread estimators produce non-NaN results
- [ ] Composite spread is bounded (not exploding)
- [ ] LSI (Liquidity Stress Index) is in reasonable range (-5 to +5 typically)
- [ ] LSI scaled is 0-100
- [ ] Regime strings are one of: NORMAL, ELEVATED, COMPRESSED, STRESS
- [ ] Weights sum to approximately 1.0

```bash
uv run python -c "
import pandas as pd, numpy as np
from src.indicators.microstructure import compute_microstructure
df = pd.read_csv('outputs/spy_test_data.csv', index_col=0, parse_dates=True)
high = df['High'].to_numpy(dtype=np.float64)
low = df['Low'].to_numpy(dtype=np.float64)
close = df['Close'].to_numpy(dtype=np.float64)
open_ = df['Open'].to_numpy(dtype=np.float64)
vol = df['Volume'].to_numpy(dtype=np.float64)
ms = compute_microstructure(high, low, close, vol, open_, lookback=20)
# Check spreads
for name, arr in [('Roll', ms.spread_roll), ('CS', ms.spread_cs), ('AR', ms.spread_ar), ('Comp', ms.spread_comp)]:
    valid = arr[~np.isnan(arr)]
    print(f'{name}: min={np.min(valid):.4f}% max={np.max(valid):.4f}% last={arr[-1]:.4f}%')
print(f'LSI: z={ms.lsi[-1]:.2f} scaled={ms.lsi_scaled[-1]:.1f}/100')
print(f'Regime: {ms.regime_strs[-1]}')
print(f'Weights: Roll={ms.weights[0]:.2f} CS={ms.weights[1]:.2f} AR={ms.weights[2]:.2f} sum={sum(ms.weights):.2f}')
# Verify
assert 0 <= ms.lsi_scaled[-1] <= 100, 'LSI scaled out of 0-100!'
assert ms.regime_strs[-1] in ('NORMAL', 'ELEVATED', 'COMPRESSED', 'STRESS'), f'Invalid regime: {ms.regime_strs[-1]}'
assert abs(sum(ms.weights) - 1.0) < 0.1, f'Weights do not sum to 1: {sum(ms.weights)}'
print('PASS')
"
```

#### 8. Neural Weight Oscillator (`nwo.py`)
- [ ] `compute_nwo()` returns 4 arrays of correct length
- [ ] Oscillator is 0-100
- [ ] Signal line is 0-100
- [ ] Trend direction is -1, 0, or +1
- [ ] With training on, oscillator adapts over time (not constant)
- [ ] `compute_nwo_signals()` produces boolean arrays

```bash
uv run python -c "
import pandas as pd, numpy as np
from src.indicators.nwo import compute_nwo, compute_nwo_signals, NWOConfig
df = pd.read_csv('outputs/spy_test_data.csv', index_col=0, parse_dates=True)
high = df['High'].to_numpy(dtype=np.float64)
low = df['Low'].to_numpy(dtype=np.float64)
close = df['Close'].to_numpy(dtype=np.float64)
# Test with training ON
osc, sig, hist, trend = compute_nwo(high, low, close)
print(f'OSC: min={np.min(osc):.1f} max={np.max(osc):.1f} last={osc[-1]:.1f}')
print(f'SIG: min={np.min(sig):.1f} max={np.max(sig):.1f} last={sig[-1]:.1f}')
print(f'Trend signs: +1={np.sum(trend>0)} -1={np.sum(trend<0)} 0={np.sum(trend==0)}')
# Test with training OFF
config = NWOConfig(use_training=False)
osc2, sig2, _, _ = compute_nwo(high, low, close, config)
print(f'Without training: last_osc={osc2[-1]:.1f}')
# Test signals
bull, bear = compute_nwo_signals(osc, sig, high, low, close)
print(f'Bull signals: {np.sum(bull)}, Bear signals: {np.sum(bear)}')
# Verify ranges
assert np.all((osc >= 0) & (osc <= 100)), 'Oscillator out of 0-100!'
assert np.all((sig >= 0) & (sig <= 100)), 'Signal out of 0-100!'
print('PASS: All values in [0,100]')
"
```

#### 9. Structure Volume Profiles (`structure_volume.py`)
- [ ] `compute_structure_cvd()` returns 3 values (cvd, atr, profiles)
- [ ] CVD array has same length as input
- [ ] Profiles have valid POC and volume data
- [ ] Buy/sell volumes are non-negative
- [ ] `_find_swing_highs()` and `_find_swing_lows()` find actual pivots

```bash
uv run python -c "
import pandas as pd, numpy as np
from src.indicators.structure_volume import compute_structure_cvd, _find_swing_highs, _find_swing_lows
df = pd.read_csv('outputs/spy_test_data.csv', index_col=0, parse_dates=True)
high = df['High'].to_numpy(dtype=np.float64)
low = df['Low'].to_numpy(dtype=np.float64)
close = df['Close'].to_numpy(dtype=np.float64)
vol = df['Volume'].to_numpy(dtype=np.float64)
cvd, atr, profiles = compute_structure_cvd(high, low, close, vol)
print(f'CVD length: {len(cvd)} (expected {len(close)})')
print(f'Profiles: {len(profiles)}')
for p in profiles[-3:]:
    print(f'  \${p.lowest:.2f}-\${p.highest:.2f} buy={p.buy_volume:.0f} sell={p.sell_volume:.0f} POC=\${p.poc_price:.2f}')
# Test pivot detection
sw_hi = _find_swing_highs(high, 5)
sw_lo = _find_swing_lows(low, 5)
print(f'Swing highs: {np.sum(sw_hi)}, Swing lows: {np.sum(sw_lo)}')
assert len(cvd) == len(close), 'CVD length mismatch!'
assert len(profiles) > 0, 'No profiles detected!'
print('PASS')
"
```

#### 10. Asian Sweep Strategy (`asian_sweep_strategy.py`)
- [ ] `run_asian_sweep_strategy()` produces signal DataFrame
- [ ] Entry/exit arrays are boolean
- [ ] Stop losses are on correct side of entries
- [ ] Take profits are on correct side of entries
- [ ] Trade count is reasonable (not zero on sufficient data)

```bash
uv run python -c "
import pandas as pd, numpy as np
from src.strategies.asian_sweep_strategy import run_asian_sweep_strategy
dates = pd.date_range('2024-01-02', periods=2000, freq='5min')
df = pd.DataFrame({
    'High': np.cumsum(np.random.randn(2000)*0.02)+100,
    'Low': np.cumsum(np.random.randn(2000)*0.02)+99.5,
    'Close': np.cumsum(np.random.randn(2000)*0.02)+99.8,
    'Volume': np.ones(2000)*1000,
    'Open': np.cumsum(np.random.randn(2000)*0.02)+99.7
}, index=dates)
df['High'] = df[['High','Low','Close','Open']].max(axis=1)
df['Low'] = df[['High','Low','Close','Open']].min(axis=1)
result = run_asian_sweep_strategy(df)
print(f'Entry longs: {result.entry_long.sum()}')
print(f'Entry shorts: {result.entry_short.sum()}')
print(f'Exit longs: {result.exit_long.sum()}')
print(f'Exit shorts: {result.exit_short.sum()}')
print(f'Non-zero signals: {(result.signal != 0).sum()}')
# Verify trade logic consistency
for i in range(len(result)):
    if result.entry_long.iloc[i]:
        assert result.stop_loss.iloc[i] < result.take_profit.iloc[i], f'Long SL >= TP at {i}'
    if result.entry_short.iloc[i]:
        assert result.stop_loss.iloc[i] > result.take_profit.iloc[i], f'Short SL <= TP at {i}'
print('PASS: All SL/TP relationships correct')
"
```

---

## Phase 3 — Logic Correctness Deep-Dive

### Known Issues to Investigate

1. **autotune_filter.py**: The autocorrelation computation in the ACF loop might have boundary issues. Test with various window sizes (5, 10, 50) and verify the dominant cycle makes physical sense (e.g., 10-50 bars for daily data).

2. **asian_sweep_composite.py**: Session detection is simplified to hour-based. The original PineScript uses `input.session("1900-0000")` which is exchange-time based and may include day rollover logic not captured by simple hour checks. Test with actual intraday data spanning midnight UTC.

3. **knn_pivots.py**: The `_find_pivots()` function iterates on EVERY bar for pivots. For 5000 bars with left/right=10, this is O(n^2) — ~250,000 comparisons. Verify performance on full-scale data.

4. **microstructure.py**: The `_rolling_mad()` function computes median/MAD via full-window sort at every bar. This is O(n * window * log(window)). For regime_win=100 on 5000 bars, this could be slow. Check performance.

5. **nwo.py**: The online gradient descent is very crude (single-sample update). Compare oscillator values with training ON vs OFF. If the oscillator is nearly identical, the learning layer isn't contributing signal.

6. **whale_profile.py**: The `_estimate_intrabar_data()` function evenly distributes volume across the bar. This is a rough approximation — the original uses `request.security_lower_tf()` for actual intrabar data. The Python version WILL diverge from the PineScript output.

7. **structure_volume.py**: The structure detection is simplified from the original's `import Trading-IQ/ICTlibrary/2 as IQ` library. BOS/CHoCH detection uses basic pivot breaks, not the full ICT library logic. The CVD reset logic is partially implemented.

### Edge Case Tests

```bash
# Test with very short data (edge case: insufficient bars)
uv run python -c "
import numpy as np
from src.indicators.fvg_profile import compute_fvg_profile
r = compute_fvg_profile(np.array([100.0]), np.array([99.0]), np.array([1.0]), period=500)
print(f'Short data: gaps={r.total_gap_count}')  # should be 0
"

# Test with NaN input
uv run python -c "
import numpy as np
from src.indicators.autotune_filter import auto_tune_filter
c = np.array([100.0, np.nan, 101.0, 102.0, 103.0])
hp, mc, dc, bp = auto_tune_filter(c, window=3)
print(f'NaN input: dc={dc}, bp={bp}')
"

# Test with zero volume
uv run python -c "
import numpy as np
from src.indicators.whale_profile import compute_whale_profile
n=50; c=np.ones(n)*100; h=np.ones(n)*100.1; l=np.ones(n)*99.9; o=np.ones(n)*100
v=np.zeros(n); r=compute_whale_profile(h,l,c,v,o,lookback=30)
print(f'Zero volume: max={r.max_bin_volume}')  # should be 0
"

# Test with flat price (all same value)
uv run python -c "
import numpy as np
from src.indicators.microstructure import compute_microstructure
n=100; c=np.ones(n)*100; h=np.ones(n)*100; l=np.ones(n)*100; o=np.ones(n)*100
v=np.ones(n)*1000; ms=compute_microstructure(h,l,c,v,o,lookback=10)
print(f'Flat price: spread_comp range={np.nanmin(ms.spread_comp):.4f}-{np.nanmax(ms.spread_comp):.4f}')
"

# Test all modules with empty input
uv run python -c "
import numpy as np
e = np.array([])
# These should all return gracefully, not crash
from src.indicators.fvg_profile import compute_fvg_profile
r = compute_fvg_profile(e, e, e); print(f'FVG empty: {r.total_gap_count}')
from src.indicators.nwo import compute_nwo
osc, sig, hist, trend = compute_nwo(e, e, e); print(f'NWO empty: {len(osc)}')
from src.indicators.structure_volume import compute_structure_cvd
cvd, atr, profs = compute_structure_cvd(e, e, e, e); print(f'Struct empty: {len(profs)}')
print('Edge cases handled')
"
```

---

## Phase 4 — Cross-Module Integration

Verify new modules don't break existing code:
```bash
# Import all existing strategy/pipeline modules
uv run python -c "
from src.strategies.rules_first_strategy import RulesFirstStrategy
from src.strategies.smc_strategy import SMCStrategy
from src.backtest.engine import BacktestEngine
# ... add other critical imports
print('Existing imports intact')
"
```

---

## Phase 5 — Known Limitations & Architecture Notes

| Module | Limitation |
|--------|-----------|
| `whale_profile.py` | Uses synthetic intrabar estimates, not real LTF data. Will differ from PineScript output. |
| `asian_sweep_composite.py` | Simplifies session logic to hour-based. May not match exact exchange-time behavior. |
| `structure_volume.py` | Does not use the external `Trading-IQ/ICTlibrary` dependency. Structure detection is basic. |
| `microstructure.py` | Rolling MAD uses per-bar sorts — slow on large windows. Consider vectorizing. |
| `nwo.py` | Training is single-sample online SGD. No batch training, no validation. |
| `swing_forecast.py` | `forecast_next_swing()` may return None on short data — handle gracefully in callers. |
| All modules | Use NumPy arrays. Callers must convert DataFrames. No built-in DataFrame wrappers except `fvg_profile.compute_fvg_profile_df()`. |

---

## Deliverable

A markdown test report `reports/pinescript_conversion_tests_2026-05-24.md` with:
1. Per-module test results (PASS/FAIL)
2. Any logic bugs found and their fixes
3. Performance benchmarks (time per module on 5000 bars)
4. Any modules that need refactoring before production use
5. Comparison notes where PineScript behavior is known to differ

After tests, update `MEMORY.md` and `progress_docs/current.md` with findings.
