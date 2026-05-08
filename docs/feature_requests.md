# Feature Requests: Integration from Repositories

**Date:** 2026-04-20
**Based On:** je-suis-tm/quant-trading Analysis
**Priority:** Implementation Order

---

## Phase 1: Time-of-Day Breakout Strategies (2-3 hours)

### FR-001: London Breakout Strategy
**Source:** `London Breakout backtest.py` from je-suis-tm
**Description:** Use Tokyo trading hour (2:00 EST) to predict London open (3:00 EST), trade first 30 min of London session with risk management

**Implementation Steps:**
1. Create `src/strategies/london_breakout.py`
2. Implement `tokyo_hour_price_collector()` function
3. Implement `london_open_signal_generator()` with thresholds
4. Add stop loss on entry logic (`risky_stop` parameter)
5. Implement reverse position detection (short → long crossover)
6. Add intraday position clearing at London close (12:00 EST)
7. Wrap for `src/backtest/engine.py` compatibility
8. Add tests in `tests/strategies/test_london_breakout.py`
9. Run backtest validation against GBP/USD dataset
10. Document signal generation timing

**Technical Details:**
- Data requirement: Minute-frequency OHLCV (FX pairs)
- Time zones: Tokyo (EST+9), London (GMT), New York (EST-5)
- Risk parameters: `risky_stop=0.01`, `open_minutes=30`
- Signal windows: Tokyo hour collection → first 30 min London
- Position management: Clear all at London close

**Validation Criteria:**
- Signal generates only during London trading hour (3:00-12:00 EST)
- Thresholds set at London open based on Tokyo hour max/min
- Stop loss filters weak signals before execution
- Cumulative sum prevents duplicate signals
- Backtest produces 50+ trades on 2+ year FX data

**Expected Benefit:** 15-25% Sharpe improvement on FX pairs, time-of-day alpha capture

---

### FR-002: Dual Thrust Strategy
**Source:** `Dual Thrust backtest.py` from je-suis-tm
**Description:** Opening range breakout strategy, uses Tokyo open to set upper/lower thresholds, trades at London open with range-based dynamic thresholds

**Implementation Steps:**
1. Create `src/strategies/dual_thrust.py`
2. Implement `intraday_range_calculator()` function
3. Convert minute data to daily OHLCV with ranges
4. Implement `range_based_threshold_generator()` at Tokyo open
5. Add signal generation at London open (3:00 EST)
6. Implement reverse position detection (crossing bounds)
7. Add intraday position clearing at London close
8. Wrap for `src/backtest/engine.py` compatibility
9. Add tests in `tests/strategies/test_dual_thrust.py`
10. Run backtest validation against futures/indices dataset

**Technical Details:**
- Data requirement: Minute-frequency OHLCV
- Range calculation: `max(high,5) - min(close,5)` and `max(close,5) - min(low,5)`
- Parameters: `rg=5` (range window), `param=0.5` (trigger multiplier)
- Time zones: Tokyo open (3:00 EST), London close (12:00 EST)
- Dynamic thresholds: `upper = param * range + price`, `lower = -(1-param) * range + price`
- Position management: Clear all at London close

**Validation Criteria:**
- Thresholds set at Tokyo open based on 5-day ranges
- Signals generate when London open price exceeds thresholds
- Reverse detection when price crosses from below upper to above lower
- Cumulative sum enforces position limits
- Backtest produces 30+ trades on 3+ year data

**Expected Benefit:** 10-20% Sharpe improvement on futures/indices, range-based breakout

---

## Phase 2: Pattern Recognition Enhancement (4-6 hours)

### FR-003: Bollinger Bands Geometric Pattern Recognition
**Source:** `Bollinger Bands Pattern Recognition backtest.py` from je-suis-tm
**Description:** Add Bottom-W pattern recognition with 5-node structure (l,k,j,m,i) using geometric conditions and bandwidth contraction exit

**Implementation Steps:**
1. Enhance `src/indicators/bollinger_bands.py` with pattern recognition
2. Create `src/patterns/bollinger_w_bottom.py` pattern detector
3. Implement `five_node_pattern_validator()` function
4. Add condition checks:
   - Condition 1: Node k at lower band
   - Condition 2: Node j at mid band (alpha tolerance)
   - Condition 3: Node m at lower band, lower than node k
   - Condition 4: Node i breaks upper band
5. Add bandwidth contraction exit (std < beta)
6. Implement coordinate storage for visualization
7. Add tests in `tests/patterns/test_bollinger_w_bottom.py`
8. Run backtest validation on equity indices

**Technical Details:**
- Pattern horizon: 75 days (3 months per Investopedia)
- Alpha tolerance: `0.0001` (price proximity to bands)
- Beta tolerance: `0.0001` (bandwidth contraction threshold)
- Node structure: l (top), k (bottom1), j (mid), m (bottom2), i (breakout)
- Signal generation: Upper band breakout after W pattern confirmation
- Exit condition: Bandwidth contraction (std < beta)

**Validation Criteria:**
- All 5 nodes identified in correct temporal order
- Pattern recognition only on upper band breakout
- Contraction clears all positions
- Pattern visualization coordinates stored
- Backtest produces 20-30 patterns on 5+ year data

**Expected Benefit:** 8-15% Sharpe improvement, reduced false signals via geometric validation

---

### FR-004: Awesome Oscillator with Saucer Patterns
**Source:** `Awesome Oscillator backtest.py` from je-suis-tm
**Description:** Implement Awesome Oscillator using (High+Low)/2 midpoint, SMA(5) vs SMA(34), plus saucer pattern detection for early signals

**Implementation Steps:**
1. Create `src/indicators/awesome_oscillator.py`
2. Implement `midpoint_oscillator()` function using (High+Low)/2
3. Add `bearish_saucer_detector()` (AO < 0, 2 green → red)
4. Add `bullish_saucer_detector()` (AO > 0, 2 red → green)
5. Implement standard MA crossover signals
6. Add cumulative sum to prevent saucer → MA conflict
7. Wrap for `src/backtest/engine.py` compatibility
8. Add tests in `tests/indicators/test_awesome_oscillator.py`
9. Run comparative backtest vs MACD oscillator

**Technical Details:**
- Midpoint: `(High + Low) / 2` instead of Close
- Oscillator: `SMA(midpoint, 5) - SMA(midpoint, 34)`
- Bearish Saucer: AO < 0, green-green → red (momentum reversal)
- Bullish Saucer: AO > 0, red-red → green (momentum reversal)
- Signal priority: Saucer > MA crossover (early signal)
- Position management: Cumulative sum prevents duplicates

**Validation Criteria:**
- Saucer patterns generate signals before MA crossover
- Saucer only when AO < 0 (bearish) or AO > 0 (bullish)
- MA crossover ignored if saucer already triggered
- Cumulative sum enforces single position per crossover
- Backtest produces 30-50 trades on 3+ year data

**Expected Benefit:** 5-12% Sharpe improvement, earlier signal entry vs MA-only

---

### FR-005: Heikin-Ashi Marubozu Patterns
**Source:** `Heikin-Ashi backtest.py` from je-suis-tm
**Description:** Add Marubozu-style reversal signals to existing Heikin-Ashi transformation, improve candlestick-based trend following

**Implementation Steps:**
1. Enhance `src/indicators/ichimoku_cloud.py` with marubozu patterns
2. Create `src/patterns/heikin_marubozu.py` pattern detector
3. Implement `ha_candle_transformer()` function (if not exists)
4. Add marubozu long signal:
   - HA open > HA close AND HA open == HA high
   - Abs(HA open - HA close) > Abs(prev HA open - prev HA close)
   - Prev HA open > prev HA close
5. Add marubozu exit signal:
   - HA open < HA close AND HA open == HA low
   - Prev HA open < prev HA close
6. Add stop loss limit (max long positions)
7. Wrap for `src/backtest/engine.py` compatibility
8. Add tests in `tests/patterns/test_heikin_marubozu.py`
9. Run backtest validation

**Technical Details:**
- HA transformation: `(Open + High + Low + Close) / 4`
- Marubozu requires: Trend filter + body size comparison + previous confirmation
- Stop loss: `stls` parameter (max long positions)
- Exit signal: Clear all or exit current position
- Position management: Cumulative sum tracking

**Validation Criteria:**
- All 4 marubozu conditions met for long signal
- Exit conditions met for position clear
- Stop loss prevents overtrading
- HA candles computed correctly from OHLC
- Backtest produces 40-60 trades on 3+ year data

**Expected Benefit:** 10-18% Sharpe improvement, reduced whipsaws via HA filtering

---

## Phase 3: Advanced Risk Metrics (1-2 hours)

### FR-006: Omega Ratio
**Source:** `Heikin-Ashi backtest.py` from je-suis-tm
**Description:**
Variation of Sharpe ratio using threshold instead of risk-free return, integrates over returns above/below threshold using Student's t-distribution

**Implementation Steps:**
1. Create `src/risk/metrics.py` (if doesn't exist)
2. Implement `omega_ratio()` function:
   ```python
   def omega_ratio(returns, risk_free=0.0, threshold=0.0, dof=3):
       y = quad(lambda g: 1 - t.cdf(g, dof), risk_free, threshold)
       x = quad(lambda g: t.cdf(g, dof), threshold, risk_free)
       return y[0] / x[0]
   ```
3. Add to `src/backtest/engine.py` metrics calculation
4. Add tests in `tests/risk/test_metrics.py`
5. Validate against equity backtests

**Technical Details:**
- Formula: `∫(1 - CDF(x)) / ∫CDF(x)` over specified range
- Distribution: Student's t-distribution (degrees of freedom parameter)
- Threshold: User-specified (default 0.0 or benchmark return)
- Use case: Better than Sharpe for non-normal return distributions

**Validation Criteria:**
- Returns positive value
- Integrates correctly over returns
-` Handles edge cases (all returns above/below threshold)
- Unit tests for known distributions

**Expected Benefit:** More accurate risk-adjusted return measure for fat-tailed distributions

---

### FR-007: Sortino Ratio
**Source:** `Heikin-Ashi backtest.py` from je-suis-tm
**Description:**
Sharpe variation using negative returns std instead of all returns, measures impact of downside risk

**Implementation Steps:**
1. Add to `src/risk/metrics.py`
2. Implement `sortino_ratio()` function:
   ```python
   def sortino_ratio(returns, risk_free=0.0, growth_rate, dof=3):
       v = sqrt(quad(lambda g: ((risk_free - g)**2) * t.pdf(g, dof), risk_free, min))
       return (growth_rate - risk_free) / v[0]
   ```
3. Add to `src/backtest/engine.py` metrics calculation
4. Add tests in `tests/risk/test_metrics.py`
5. Validate against equity backtests

**Technical Details:**
- Formula: `(return - risk_free) / std(negative_returns)`
- Distribution: Student's t-distribution for tail weight
- Use case: Penalizes strategies with high downside volatility
- Focus: Asymmetric risk (worst losses matter more than symmetric volatility)

**Validation Criteria:**
- Returns positive value for strategies outperforming risk-free
- Negative returns correctly isolated
- Higher than Sharpe for strategies with limited downside

**Expected Benefit:** Better risk measure for strategies with asymmetric loss profiles

---

### FR-008: Calmar Ratio
**Source:** `Heikin-Ashi backtest.py` from je-suis-tm
**Description:**
Sharpe variation using max drawdown instead of std, measures return after worst-case adjustment

**Implementation Steps:**
1. Add to `src/risk/metrics.py`
2. Implement `calmar_ratio()` function:
   ```python
   def calmar_ratio(returns, max_drawdown):
       # max_drawdown from existing mdd() function
       growth_rate = geometric_mean(returns) - 1
       return growth_rate / max_drawdown
   ```
3. Add to `src/backtest/engine.py` metrics calculation
4. Add tests in `tests/risk/test_metrics.py`
5. Validate against equity backtests

**Technical Details:**
- Formula: `return / max_drawdown`
- Growth rate: Geometric mean of returns (not arithmetic)
- Use case: Emphasizes recovery after drawdowns
- Focus: Worst-case scenario adjustment

**Validation Criteria:**
- Returns positive value for profitable strategies
- Uses existing max_drawdown calculation
- Higher than Sharpe for strategies with deep drawdowns but quick recovery

**Expected Benefit:** Better measure for strategies with high recovery capability

---

## Phase 4: Code Modernization (2-3 hours)

### FR-009: Fix Hardcoded Paths
**Source:** Multiple strategies from je-suis-tm
**Description:** Replace `os.chdir('d:/')` and hardcoded paths with configurable parameters or environment variables

**Implementation Steps:**
1. Audit all strategy files for hardcoded paths
2. Create `config/settings.py` with data directory configuration
3. Update strategies to use `config.DATA_DIR` instead of hardcoded paths
4. Add environment variable support (`INVESTMENT_DATA_DIR`)
5. Add tests in `tests/config/test_settings.py`
6. Document configuration options in README.md

**Affected Files:**
- All strategies using `os.chdir()`
- Data loading functions
- CSV/excel file paths

**Expected Benefit:** Portability, CI/CD compatibility, easier deployment

---

### FR-010: Update Pandas Syntax
**Source:** Multiple strategies from je-suis-tm
**Description:** Update deprecated pandas methods (e.g., `inplace=True`, `reset_index()`) to modern pandas syntax

**Implementation Steps:**
1. Audit all pandas operations
2. Replace `inplace=True` with chain assignment
3. Replace `reset_index(inplace=True)` with `df = df.reset_index()`
4. Update `.ix`/`.iloc` deprecation warnings
5. Add tests to verify data integrity
6. Add pandas version compatibility check

**Affected Patterns:**
- `df.reset_index(inplace=True)` → `df = df.reset_index()`
- `df.drop(..., inplace=True)` → `df = df.drop(...)`
- `df.set_index(..., inplace=True)` → `df = df.set_index(...)`

**Expected Benefit:** Future-proof code, cleaner pandas style, fewer warnings

---

## Implementation Summary

### Total Effort Estimate
- Phase 1: 2-3 hours
- Phase 2: 4-6 hours
- Phase 3: 1-2 hours
- Phase 4: 2-3 hours
- **Total: 9-14 hours**

### Expected Impact
- **New Strategies:** 2 (London Breakout, Dual Thrust)
- **Enhanced Patterns:** 3 (Bollinger W, Awesome Saucer, Heikin Marubozu)
- **New Metrics:** 3 (Omega, Sortino, Calmar)
- **Code Quality:** 2 (Path config, Pandas modernization)
- **Total Features:** 10

### Prioritized Order
1. FR-001: London Breakout (HIGH)
2. FR-002: Dual Thrust (HIGH)
3. FR-003: Bollinger W Pattern (HIGH)
4. FR-004: Awesome Saucer (MEDIUM)
5. FR-006: Omega Ratio (MEDIUM)
6. FR-007: Sortino Ratio (MEDIUM)
6. FR-008: Calmar Ratio (MEDIUM)
6. FR-005: Heikin Marubozu (LOW)
9. FR-009: Fix Paths (LOW)
10. FR-010: Pandas Syntax (LOW)

---

## Next Steps

### Immediate (This Session)
1. Implement FR-001 (London Breakout)
2. Implement FR-002 (Dual Thrust)

### Short-Term (Next 1-2 Sessions)
3. Implement FR-003 (Bollinger W Pattern)
4. Implement FR-004 (Awesome Saucer)

### Medium-Term (Next 2-3 Sessions)
5. Implement FR-005 (Heikin Marubozu)
6. Implement FR-006, FR-007, FR-008 (Advanced Metrics)

### Long-Term (Next 3-4 Sessions)
7. Implement FR-009 (Fix Paths)
8. Implement FR-010 (Pandas Syntax)

---

## Validation Plan

After each feature implementation:
1. Run `uv run pytest` on new tests
2. Run backtest on validation dataset
3. Compare metrics vs baseline
4. Document results in docs/validation_results.md
5. Update feature request status in this document

---

**Created:** 2026-04-20
**Status:** Ready for Implementation
**Next Review:** Machine-Learning-for-Algorithmic-Trading-Second-Edition analysis
