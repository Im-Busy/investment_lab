# Indicator & Pattern Audit Report

## Status: CRITICAL FIXES APPLIED - Remaining items documented for future work

**Last Updated**: 2026-03-31

### Completed Fixes
- ✅ P1-1: Fixed lookahead bias in MSL pattern
- ✅ P1-2: Fixed IFVG fill logic
- ✅ P1-3: Fixed ATR to use RMA smoothing
- ✅ P1-4: Fixed ADX to use RMA smoothing
- ✅ P2-1: Relaxed Matching Lows epsilon (0.05 → 0.20)
- ✅ P2-2: Strengthened N-Bar Decline reversal bar check
- ✅ P2-3: Increased Double Top/Bottom max_pattern_bars (60 → 120)
- ✅ P2-4: Increased Head & Shoulders max_pattern_bars (120 → 180)
- ✅ Strategy: Verified multi-pattern strategy is compatible with daily timeframes

## Executive Summary

This audit reviews all indicator and pattern implementations in the trading system to identify issues that may cause:
1. Patterns not triggering during backtesting
2. Patterns triggering but producing unprofitable results
3. Logic errors in pattern detection
4. Timeframe incompatibilities

**Key Finding**: The primary issues stem from **parameter calibration for daily timeframes**, **overly strict detection criteria**, and **lookahead bias in several implementations**.

---

## Part 1: Indicators Audit

### 1.1 `asian_range.py` - Asian Session Range Detection

**Status**: ⚠️ **PARTIALLY CORRECT - Timeframe Issue**

**Issues Found**:
1. **Timeframe Incompatibility**: The Asian range detection is designed for 5-minute intraday bars (expects 96 bars for 8-hour session). On **daily data**, this will never work correctly because a daily bar encompasses the entire day.
2. **`between_time` Logic**: Line 125 uses `df.between_time(start_time, end_time)` which filters intraday bars. On daily data with one bar per day, this will likely return empty or a single bar.
3. **Expected Bars Warning**: Line 144-149 warns if `bar_count != 96`, which will always trigger on daily data.

**Impact**: This indicator is **non-functional on daily data**. It's only useful for intraday (5m, 15m, 1h) timeframes.

**Recommendation**:
- For daily data: Either skip Asian range detection entirely, or use a simplified version that tracks overnight ranges using previous day's close vs current day's open.
- Add a timeframe check at the start of the function.

---

### 1.2 `fibonacci.py` - Fibonacci Calculations

**Status**: ✅ **MOSTLY CORRECT**

**Issues Found**:
1. **Minor**: The `fibonacci_extension` calculation (lines 96-102) uses `(ratio - 1.0)` which produces correct extension levels, but the formula could be clearer. Standard extension is `high + diff * ratio` not `high + diff * (ratio - 1)`.
2. **No Issue**: The retracement, BC retracement, CD extension, and XD retracement calculations are mathematically correct.

**Impact**: Minimal. The extension formula is mathematically equivalent, just expressed differently.

**Recommendation**: No changes needed.

---

### 1.3 `ifvg.py` - Inverse Fair Value Gap Detection

**Status**: ❌ **SIGNIFICANT LOGIC ISSUES**

**Issues Found**:
1. **IFVG Definition Error** (Lines 172-191, 193-212): The current implementation checks for gaps between bar 1 and bar 3 (`bar3['Low'] - bar1['High']` for bullish). However, the **standard IFVG definition** uses a 3-bar pattern where:
   - Bullish IFVG: `low[bar3] > high[bar1]` creates a gap zone between `high[bar1]` and `low[bar3]`
   - The current code has the zone boundaries **correct** but the **detection logic is checking the wrong condition**.

2. **Gap Direction Confusion** (Lines 174, 195):
   - Bullish IFVG should form when price **gaps up** and the zone acts as **support**
   - Current code: `bullish_gap = bar3['Low'] - bar1['High']` - This is correct for detecting a gap
   - But the zone is stored as `high=bar3['Low'], low=bar1['High']` which is also correct

3. **Fill Logic Reversed** (Lines 337-349):
   - Bullish IFVG is considered "filled" when `bar['Low'] <= ifvg.low` (line 340)
   - This is **backwards**: A bullish IFVG (support zone) should be filled when price **drops into it**, not below it
   - Should be: filled when price **enters** the zone, not exits it

4. **Missing Middle Bar Check**: Standard IFVG detection also checks that bar 2 (the middle bar) doesn't overlap with the gap. Current implementation ignores bar 2 entirely.

**Impact**: IFVG detection may produce false positives, and fill tracking is incorrect.

**Recommendation**:
- Fix the fill logic to detect when price **enters** the zone
- Add middle bar validation
- Consider lowering `atr_mult` from 1.2 to 0.5-0.8 for daily timeframes where gaps are rarer

---

### 1.4 `liquidity_sweep.py` - Liquidity Sweep Detection

**Status**: ⚠️ **DEPENDS ON ASIAN RANGE - Timeframe Issue**

**Issues Found**:
1. **Dependency on Asian Range**: The sweep detection requires `asia_high` and `asia_low` as inputs (line 63-64). If Asian range is not computed correctly (see 1.1), sweeps cannot be detected.
2. **Time-Based Filtering** (Line 147-151): Uses `df.index.time >= start_time_obj` to filter post-session data. This doesn't work on daily data where each bar represents a full day.
3. **Sweep Detection Logic** (Lines 169, 194): The sweep logic itself is correct - it checks if price breaks beyond the level by a buffer.

**Impact**: Non-functional on daily data due to Asian range dependency and time-based filtering.

**Recommendation**:
- For daily data: Use previous day's high/low as sweep targets instead of Asian range
- Remove or adapt time-based filtering for daily timeframes

---

### 1.5 `mss.py` - Market Structure Shift Detection

**Status**: ⚠️ **LOGIC ISSUES - Lookahead Bias**

**Issues Found**:
1. **Lookahead Bias** (Lines 262-307, 310-355): The `detect_mss` function checks if price **after** the pivot breaks the pivot level. However, it searches from `pivot_high.bar_index + 1` to `end_bar`. If `end_bar` is the current bar, this means the function is checking if **future** bars broke **past** pivots. This creates lookahead bias because in real trading, you wouldn't know if a pivot will be broken until after it happens.

2. **Pivot Search Logic** (Lines 110, 164): The pivot search uses `range(bar_index - lookback, lookback, -1)` which searches backwards from the current position. This is correct, but the `lookback` parameter defaults to 3, which is very short for daily data.

3. **Missing Confirmation**: MSS detection returns immediately on first break (line 268-286). In real trading, MSS confirmation typically requires a **close** beyond the pivot level, not just a wick. The `require_close` parameter helps, but the default behavior should be more conservative.

4. **Pivot Confirmation Issue** (Lines 115-127, 169-181): The pivot check requires `lookback` bars on **both sides** to confirm a pivot. With `lookback=3`, this means you need 7 bars total to confirm one pivot. On daily data, this is 7 trading days, which may be reasonable, but the parameter should be higher (5-10) for daily timeframes.

**Impact**: MSS may trigger too frequently with `lookback=3` on daily data, and there's potential lookahead bias in how breaks are detected.

**Recommendation**:
- Increase default `lookback` to 5-10 for daily data
- Add a "confirmation delay" parameter to simulate real-world MSS confirmation
- Consider requiring multiple consecutive closes beyond the pivot for stronger signals

---

### 1.6 `pivots.py` and `pivots_numba.py` - Pivot Point Detection

**Status**: ✅ **MOSTLY CORRECT**

**Issues Found**:
1. **Lookback Parameter Default**: Default `lookback=5` is reasonable for daily data but may be too sensitive. For daily timeframes, 5-10 is typical.
2. **Numba Implementation**: The Numba-accelerated versions are mathematically identical to the pure Python versions, which is correct.
3. **Cache Implementation**: The caching logic in `find_swing_highs_cached` and `find_swing_lows_cached` has a potential issue with the hash key generation (line 454-460). Using `tobytes()` on large arrays can be memory-intensive.

**Impact**: Minimal. The core logic is correct.

**Recommendation**:
- Consider making `lookback` configurable based on timeframe
- The cache key generation could use a hash of the bytes instead of the full bytes

---

### 1.7 `regime.py` - Market Regime Detector

**Status**: ⚠️ **PARAMETER CALIBRATION ISSUES**

**Issues Found**:
1. **ADX Calculation** (Lines 124-160): The ADX calculation uses a simplified smoothing method (`rolling().mean()`) instead of the standard Wilder smoothing (RMA). This produces slightly different values than standard ADX implementations.
2. **Threshold Calibration** (Lines 91-92): Default `adx_trend_threshold=25` and `adx_strong_threshold=30` are standard values, but on daily data with longer trends, these may need adjustment.
3. **Trend Detection Logic** (Lines 193-225): Requires **both** DI crossover AND price relative to SMA. This is a stricter condition than typical implementations, which may cause the detector to classify more markets as "sideways" than expected.
4. **SMA Period** (Line 97): Default `sma_period=50` is standard, but for daily data on assets with longer cycles (like SPY), 200-day SMA is often used for trend determination.

**Impact**: May over-classify markets as "sideways" due to strict dual-condition requirement.

**Recommendation**:
- Fix ADX to use RMA smoothing
- Make trend detection configurable (DI only, SMA only, or both)
- Consider adding a 200-day SMA option for long-term trend filtering

---

### 1.8 `technical.py` - Technical Indicators

**Status**: ✅ **CORRECT**

**Issues Found**:
1. **ATR Calculation** (Lines 125-148): Uses simple moving average of True Range (`tr.rolling(window=period).mean()`). The standard ATR uses **Wilder smoothing** (RMA), which gives more weight to recent values. This is a minor deviation.
2. **ADX Calculation** (Lines 203-249): Same issue as regime.py - uses simple smoothing instead of RMA.
3. **All other indicators** (SMA, EMA, RSI, Bollinger Bands, Donchian Channel): Correctly implemented.

**Impact**: ATR and ADX values will differ slightly from standard implementations (e.g., TradingView, ThinkOrSwim).

**Recommendation**:
- Switch ATR and ADX to use RMA (Wilder) smoothing for consistency with industry standards
- This is especially important if strategy parameters were calibrated against standard indicator values

---

## Part 2: Patterns Audit

### 2.1 Basic Patterns

#### 2.1.1 `floor_pivot.py` - Floor Pivot Breakout

**Status**: ⚠️ **TIMEFRAME ISSUE**

**Issues Found**:
1. **Pivot Calculation**: Uses previous bar's H/L/C to calculate pivot points. On **daily data**, this means pivots are recalculated every day from the previous day's data, which is correct for daily pivot trading.
2. **Entry Offset** (Line 147): Default `entry_offset=0.01` is a **fixed dollar amount**. For high-priced assets (like SPY at $500+), this is negligible. For low-priced assets, this may be too large. Should be percentage-based or ATR-based.
3. **Vectorized Detection** (Lines 62-105): The Numba implementation is correct and matches the bar-by-bar logic.
4. **Trend Alignment Check** (Lines 244-268): Checks if previous close was above/below the pivot point. This is a reasonable trend filter.

**Impact**: The `$0.01` offset is problematic for assets with different price levels. On daily data, pivot breakouts are less common than intraday.

**Recommendation**:
- Make `entry_offset` ATR-based or percentage-based
- For daily data, consider using the previous day's range as the offset

---

#### 2.1.2 `matching_lows.py` - Matching Lows/Highs

**Status**: ⚠️ **PARAMETER TOO STRICT**

**Issues Found**:
1. **Epsilon Multiplier** (Line 282): Default `epsilon_atr_multiplier=0.05` is **extremely tight**. This means lows must be within 5% of ATR of each other. For an asset with ATR of $5, this means lows must be within $0.25 of each other. This is very restrictive.
2. **Min Test Bars** (Line 283): Default `min_test_bars=3` requires at least 3 bars testing the support level. On daily data, finding 3 days with lows within such a tight range is rare.
3. **Breakout Condition** (Lines 439-442): Requires `current_high > prev_high` for breakout. This is a reasonable condition but combined with the tight epsilon, makes the pattern very rare.
4. **Vectorized Implementation** (Lines 55-118): The Numba implementation has a subtle issue - it finds the **minimum low** in the lookback window and then counts bars near that minimum. This means it's looking for bars near the **absolute low**, not necessarily bars that **test the same level** multiple times.

**Impact**: The pattern will trigger very rarely on daily data due to the tight epsilon. The vectorized implementation may miss valid patterns.

**Recommendation**:
- Increase `epsilon_atr_multiplier` to 0.15-0.25 for daily data
- Reduce `min_test_bars` to 2 for daily data
- Fix the vectorized implementation to look for repeated tests of a **specific level**, not just proximity to the minimum

---

#### 2.1.3 `msl.py` - Market Structure Low

**Status**: ⚠️ **LOOKAHEAD BIAS**

**Issues Found**:
1. **Lookahead Bias** (Lines 186-216): The pattern checks `c_plus_1 = float(arrays["close"][i + 1])` (line 212) to confirm the pattern. This means the pattern at bar `i` requires knowledge of bar `i+1`, which is **future data**. In a backtest, this creates lookahead bias because the signal is generated with knowledge of the next bar's close.
2. **Confirmation Logic**: The pattern requires `Close[i+1] > Max(C[-2], C[-1], C[0])` for confirmation. This means the signal should only be generated **after** bar `i+1` closes, not at bar `i`.
3. **Entry Validity Window** (Line 124): `confirmation_bars=5` suggests the entry is valid for 5 bars after confirmation, but the implementation doesn't track this window.

**Impact**: **Critical lookahead bias** - signals are generated with future information, inflating backtest results.

**Recommendation**:
- Move confirmation check to bar `i+1` and generate signal at `i+1`, not at `i`
- Track the entry validity window properly

---

#### 2.1.4 `n_bar_decline.py` - n-Bar Decline

**Status**: ⚠️ **PARAMETER CALIBRATION ISSUE**

**Issues Found**:
1. **Lookback Period** (Line 165): Default `lookback_period=21` is reasonable for daily data (about one trading month).
2. **Min Successive** (Line 164): Default `min_successive=3` requires 3 consecutive lower lows. This is reasonable but may be too common in a downtrend, generating false reversal signals.
3. **Reversal Bar Check** (Lines 279-298): Only checks if `Close > Open` (bullish candle). This is a weak reversal signal - a stronger signal would require the reversal bar to close in the upper half of its range or above the previous bar's close.
4. **New Lookback Low Check** (Lines 258-277): Correctly checks if current bar makes a new 21-bar low.

**Impact**: The pattern may trigger too frequently in strong downtrends, generating counter-trend signals that fail.

**Recommendation**:
- Strengthen the reversal bar requirement (e.g., close in upper 60% of range, or close above previous bar's high)
- Add a minimum decline magnitude requirement
- Increase `min_successive` to 4-5 for daily data

---

#### 2.1.5 `nr7id.py` - NR7 + Inside Day

**Status**: ✅ **CORRECT BUT STRICT**

**Issues Found**:
1. **NR7 Logic** (Lines 189-214): Correctly checks if current bar's range is the narrowest of the last 7 bars.
2. **Inside Day Logic** (Lines 216-240): Correctly checks if current bar is inside the previous bar's range.
3. **Combined Condition**: Both NR7 AND Inside Day must be true simultaneously. This is a **very rare** occurrence on daily data.
4. **Entry Offset** (Line 142): Default `entry_offset=0.01` has the same issue as Floor Pivot - fixed dollar amount doesn't scale with price level.

**Impact**: The combined NR7+ID condition is extremely rare. On daily data, this pattern may trigger only a handful of times per year per asset.

**Recommendation**:
- Consider allowing NR7 **OR** Inside Day as separate patterns
- Or relax the condition to NR4+ID (Narrow Range 4)
- Make entry offset ATR-based

---

#### 2.1.6 `two_bar_reversal.py` - Two-Bar Reversal (Pipe)

**Status**: ⚠️ **LOGIC ISSUES**

**Issues Found**:
1. **Trend Detection** (Lines 96-126): Uses a simplified trend check that requires 60% of bars to be declining AND overall decline. This is reasonable but the `trend_bars=5` default may be too short for daily data.
2. **Range Comparison** (Lines 150-158): Compares bar ranges to the average of **preceding 3 bars**. On daily data, 3 bars may not be enough context.
3. **Breakout Confirmation** (Lines 183-184, 210-211): Uses `confirmation_filter=0.005` (0.5%) which means the current close must be 0.5% beyond the bar 2 high/low. This is reasonable.
4. **Close Position Calculation** (Lines 174, 177, 202, 205): Correctly calculates where the close is within the bar's range.

**Impact**: The pattern logic is sound, but the trend detection may be too lenient with only 5 bars.

**Recommendation**:
- Increase `trend_bars` to 10-15 for daily data
- Increase the preceding range lookback to 5-10 bars
- Add a minimum bar size requirement relative to ATR

---

### 2.2 Candlestick Patterns

#### 2.2.1 `engulfing.py` - Engulfing Pattern

**Status**: ✅ **MOSTLY CORRECT**

**Issues Found**:
1. **Engulfing Definition** (Lines 153-158): Checks if candle2's body **completely contains** candle1's body. This is the standard definition.
2. **Body Threshold** (Line 41): `body_threshold=0.3` requires the second candle's body to be at least 30% of its range. This is reasonable.
3. **Trend Detection** (Lines 99-121): Uses linear regression slope normalized by price. The threshold of 0.001 (0.1% per bar) is reasonable for daily data.
4. **Entry at Close** (Lines 291-292): Enters at the close of the engulfing candle. This is correct for candlestick patterns.

**Impact**: Minimal issues. The implementation is sound.

**Recommendation**: No major changes needed.

---

#### 2.2.2 `hammer.py` - Hammer / Hanging Man

**Status**: ✅ **CORRECT**

**Issues Found**:
1. **Shadow Ratio** (Line 40): `shadow_ratio=2.0` requires lower shadow to be at least 2x the body. This is the standard definition.
2. **Body Threshold** (Line 41): `body_threshold=0.35` means body must be less than 35% of range. Reasonable.
3. **Upper Shadow** (Line 42): `upper_shadow_threshold=0.1` means upper shadow must be less than 10% of range. This is standard.
4. **Confirmation** (Lines 176-220): Requires confirmation candle for signal generation. This is correct practice.

**Impact**: No major issues. Implementation is sound.

**Recommendation**: No changes needed.

---

#### 2.2.3 `doji.py` - Doji Pattern

**Status**: ✅ **CORRECT**

**Issues Found**:
1. **Body Threshold** (Line 43): `body_threshold=0.1` means body must be less than 10% of range. Standard definition.
2. **Classification Logic** (Lines 80-130): Correctly classifies Dragonfly, Gravestone, Long-legged, and Standard Doji types.
3. **Low Confidence** (Line 288): Base confidence of 0.30 is appropriate for Doji patterns.
4. **Confirmation Required** (Line 46): `require_confirmation=True` by default, which is correct practice.

**Impact**: No issues. Implementation is correct.

**Recommendation**: No changes needed.

---

### 2.3 Classic Patterns

#### 2.3.1 `double_top.py` - Double Top

**Status**: ⚠️ **PARAMETER CALIBRATION ISSUE**

**Issues Found**:
1. **Peak Tolerance** (Line 47): `peak_tolerance=0.05` (5%) is reasonable for daily data.
2. **Min Pattern Bars** (Line 48): `min_pattern_bars=20` is reasonable for daily data (about one month).
3. **Max Pattern Bars** (Line 49): `max_pattern_bars=60` may be too restrictive. Double tops can form over 3-6 months on daily data.
4. **Volume Filter** (Line 52): `volume_filter=False` by default. The volume dissipation check (line 206) is correct but not enabled by default.
5. **Breakdown Check** (Line 213): Checks if `current_close < neckline_level`. This is correct.

**Impact**: The `max_pattern_bars=60` may filter out valid double top patterns that take longer to form.

**Recommendation**:
- Increase `max_pattern_bars` to 100-120 for daily data
- Enable volume filter by default

---

#### 2.3.2 `double_bottom.py` - Double Bottom

**Status**: ⚠️ **SAME ISSUES AS DOUBLE TOP**

**Issues Found**: Same as double_top.py - `max_pattern_bars=60` is too restrictive.

**Recommendation**: Same as double_top.py.

---

#### 2.3.3 `head_shoulders.py` - Head and Shoulders

**Status**: ⚠️ **PARAMETER CALIBRATION ISSUE**

**Issues Found**:
1. **Shoulder Tolerance** (Line 46): `shoulder_tolerance=0.10` (10% of head height) is reasonable.
2. **Min Pattern Bars** (Line 47): `min_pattern_bars=30` is reasonable for daily data.
3. **Max Pattern Bars** (Line 48): `max_pattern_bars=120` is reasonable but may miss longer-forming patterns.
4. **Neckline Calculation** (Lines 183-184): Uses linear interpolation between two trough points. This is correct.
5. **Breakdown Check** (Line 263): Checks if `current_close < neckline_value`. Correct.

**Impact**: Implementation is sound. May miss longer-forming patterns.

**Recommendation**: Consider increasing `max_pattern_bars` to 180 for daily data.

---

### 2.4 Harmonic Patterns

#### 2.4.1 `gartley.py` - Gartley Pattern

**Status**: ⚠️ **PIVOT IDENTIFICATION ISSUES**

**Issues Found**:
1. **Pivot Identification** (Lines 98-155): Uses the **most recent** swing highs and lows to identify XABCD pivots. This approach is **greedy** and may miss valid patterns where the most recent swings don't form a pattern but earlier swings do.
2. **Fibonacci Validation** (Lines 265-306): The Fibonacci ratio validation is correct and matches standard Gartley definitions.
3. **Lookback** (Line 68): `lookback=5` for pivot detection is reasonable for daily data.
4. **Confirmation Requirement** (Lines 343-352): Requires at least one bar after D for confirmation. This is correct.
5. **Butterfly and Bat Patterns** (Lines 486-571): These are **stub implementations** that always return "not detected". They need full implementation.

**Impact**: The greedy pivot selection may miss valid patterns. Butterfly and Bat patterns are non-functional.

**Recommendation**:
- Implement a more comprehensive pivot search that considers multiple candidate pivots
- Complete the Butterfly and Bat pattern implementations

---

### 2.5 Continuation Patterns

#### 2.5.1 `flag.py` - Flag Pattern

**Status**: ⚠️ **POLE SLOPE CALCULATION ISSUE**

**Issues Found**:
1. **Pole Slope Calculation** (Lines 103-107): Calculates slope as `(end_price - start_price) / (end_idx - start_idx)`. This is **price per bar**, not a percentage. The comparison at line 173 uses `pole_pct = pole_height / pole_start_price`, which is correct.
2. **Flag Slope Threshold** (Line 49): `flag_slope_threshold=0.005` is in **price per bar** units, not percentage. This is inconsistent with how it's used.
3. **Flag Duration** (Lines 47-48): `flag_bars_min=5, flag_bars_max=20` is reasonable for daily data.
4. **Breakout Check** (Line 199): Uses `confirmation_filter=0.005` (0.5%) for breakout confirmation. Reasonable.

**Impact**: The flag slope threshold units are inconsistent, which may cause incorrect filtering.

**Recommendation**:
- Standardize slope calculations to use percentage-based units
- Clarify the flag slope threshold documentation

---

### 2.6 Complex Patterns

#### 2.6.1 `cup_handle.py` - Cup and Handle

**Note**: Not read in detail during this audit. Should be reviewed separately.

#### 2.6.2 `spike_ledge.py` - Spike and Ledge

**Note**: Not read in detail during this audit. Should be reviewed separately.

#### 2.6.3 `three_hills.py` - Three Hills Mountain

**Note**: Not read in detail during this audit. Should be reviewed separately.

#### 2.6.4 `parabolic_arc.py` - Parabolic Arc

**Note**: Not read in detail during this audit. Should be reviewed separately.

---

## Part 3: Prioritized Fix Plan

### Priority 1: Critical Issues (Fix Immediately)

| # | File | Issue | Impact | Effort |
|---|------|-------|--------|--------|
| 1 | `msl.py` | Lookahead bias in confirmation check | Inflated backtest results | Medium |
| 2 | `ifvg.py` | Fill logic reversed | Incorrect signal tracking | Low |
| 3 | `technical.py` | ATR/ADX use SMA instead of RMA | Inconsistent with industry standards | Low |
| 4 | `regime.py` | ADX uses SMA instead of RMA | Inconsistent regime detection | Low |

### Priority 2: High Impact Issues (Fix Soon)

| # | File | Issue | Impact | Effort |
|---|------|-------|--------|--------|
| 5 | `matching_lows.py` | Epsilon too strict (0.05 ATR) | Pattern rarely triggers | Low |
| 6 | `n_bar_decline.py` | Weak reversal bar check | False reversal signals | Low |
| 7 | `gartley.py` | Greedy pivot selection | Misses valid patterns | High |
| 8 | `flag.py` | Inconsistent slope units | Incorrect filtering | Low |
| 9 | `head_shoulders.py` | Max pattern bars too restrictive | Misses valid patterns | Low |
| 10 | `double_top.py/bottom.py` | Max pattern bars too restrictive | Misses valid patterns | Low |

### Priority 3: Timeframe Adaptations (Plan for Multi-Timeframe Support)

| # | File | Issue | Impact | Effort |
|---|------|-------|--------|--------|
| 11 | `asian_range.py` | Not compatible with daily data | Non-functional on daily | Medium |
| 12 | `liquidity_sweep.py` | Depends on Asian range | Non-functional on daily | Medium |
| 13 | `mss.py` | Lookback too short for daily | Too many false signals | Low |
| 14 | All patterns | Fixed dollar offsets | Doesn't scale with price | Medium |

### Priority 4: Completing Missing Implementations

| # | File | Issue | Impact | Effort |
|---|------|-------|--------|--------|
| 15 | `gartley.py` | Butterfly stub | Pattern non-functional | High |
| 16 | `gartley.py` | Bat stub | Pattern non-functional | High |
| 17 | Complex patterns | Not audited | Unknown | Medium |

---

## Part 4: Recommended Parameter Adjustments for Daily Timeframes

| Parameter | Current | Recommended for Daily |
|-----------|---------|----------------------|
| MSS Lookback | 3 | 5-10 |
| Matching Lows Epsilon | 0.05 ATR | 0.15-0.25 ATR |
| Matching Lows Min Bars | 3 | 2 |
| N-Bar Decline Min Successive | 3 | 4-5 |
| Double Top/Bottom Max Bars | 60 | 100-120 |
| Head & Shoulders Max Bars | 120 | 180 |
| Gartley Lookback | 5 | 7-10 |
| Entry/Stop Offsets | $0.01 | 0.1% of price or ATR-based |
| Flag Pole Min Slope | 2% | 3-5% |

---

## Part 5: Questions for User

Before proceeding with fixes, please clarify:

1. **Gartley Pattern**: The standard Gartley pattern has specific Fibonacci ratios. Are you using the classic Gartley (AB=0.618 XA, XD=0.786 XA) or a variant?

2. **SMC/ICT Patterns**: The IFVG, Liquidity Sweep, and MSS are SMC/ICT concepts. Do you have specific source references for how these should be implemented? Different ICT mentors define them slightly differently.

3. **Two-Bar Reversal**: Is this meant to be the "Pipe Bottom/Pipe Top" from Larry Williams' work, or a different two-bar reversal definition?

4. **Timeframe Priority**: Are you primarily using daily data, or do you also trade intraday (1h, 5m) timeframes? This affects whether we need to support both intraday-specific indicators (Asian Range) and daily-compatible versions.

5. **Pattern Entry Logic**: Many patterns use Buy Stop/Sell Stop orders. In your actual trading, do you use market orders at the close, limit orders, or stop orders?
