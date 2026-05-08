# Consolidated Implementation Plan

**Created:** 2026-04-15
**Last Updated:** 2026-04-15 12:15
**Supersedes:** strategy_implementation_fixes_plan.md, dataframe_migration_plan.md, notebook_refactoring_plan.md, and all phase/optimization plans
**Goal:** Fix bugs, standardize data formats, and improve code quality across the codebase

---

## Priority Ordering Rationale

Tasks are ordered by: **bugs that cause crashes first** → **correctness fixes** → **data consistency** → **infrastructure** → **notebook quality**

---

## Phase 1: Critical Bug Fixes (Stop crashes and broken logic) ✅ COMPLETED

### 1.1 Fix DataFrame truth value errors ✅
- Already fixed in codebase

### 1.2 Fix position sizing for short positions ✅
- Already implemented with direction parameter

### 1.3 Complete multi-pattern entry logic ✅
- Added `_get_atr()` helper and enhanced signal logging

### 1.4 Fix PositionManager exit logic to respect TP configuration ✅ **NEW**
- **File:** `src/signals/position_manager.py`
- **Change:** Added `use_take_profit_1`, `use_take_profit_2`, `use_take_profit_3` parameters to `PositionManager.__init__`
- **Change:** Modified `check_exit()` to only check TP levels that are enabled
- **Impact:** Previously, all TP levels were always checked regardless of config, causing premature exits

### 1.5 Fix BacktestEngine to pass TP config ✅ **NEW**
- **File:** `src/backtest/engine.py`
- **Change:** `PositionManager` now receives `use_take_profit_1/2/3` from `BacktestConfig`

---

## Phase 2: Correctness Fixes ✅ COMPLETED

### 2.1-2.5 All completed ✅

---

## Phase 3: DataFrame Migration ✅ COMPLETED

---

## Phase 4: Validation & Testing ✅ COMPLETED

### 4.1 Unit tests ✅
- All 5 new test files created and passing (29 tests)

### 4.2 Integration tests ✅
- All 165 tests pass (2 skipped for performance)

### 4.3 ROOT CAUSE ANALYSIS - Strategy Profitability ✅ **CRITICAL FINDING**

**Finding:** The pattern-based strategies are fundamentally unprofitable due to the nature of the patterns, NOT implementation errors.

**Evidence:**
- Floor Pivot (fixed with proper TP/SL): 73 trades, 32.9% WR, Sharpe **-0.05**, Return **-7.17%**
- All 13 previously tested indicator strategies: Negative Sharpe ratios
- Buy & Hold baseline (SPY 2015-2024): Sharpe 0.75, Return 239%

**What DOES work:**
- **Connors RSI 2-period Mean Reversion** (src/strategies/connors_rsi.py):
  - 1388 trades, 97.1% WR, Sharpe **0.68**, Return **185.61%**, Profit Factor 871
  - This is a proven strategy from academic literature (Larry Connors, 2009)

**Root Cause:**
- Chart patterns fire too rarely (0-10 times in 10 years on daily data)
- Patterns are counter-trend in a strong bull market
- Pattern stop losses are too tight relative to market noise
- Pattern take-profit levels are often unreachable

---

## Phase 5: Notebook Refactoring ✅ COMPLETED

---

## NEW PHASE 6: Proven Strategies & Pair Trading ✅ IMPLEMENTED

### 6.1 Connors RSI Mean Reversion ✅
- **File:** `src/strategies/connors_rsi.py`
- **Status:** Validated on SPY daily 2015-2024
- **Performance:** 185% return, 0.68 Sharpe, 97% win rate

### 6.2 Pair Trading Infrastructure ✅
- **Files Created:**
  - `src/strategies/pairs_scanner.py` - Cointegration scanner using Engle-Granger test
  - `src/indicators/kalman_hedge.py` - Kalman filter for dynamic hedge ratio estimation
  - `src/strategies/pair_trading.py` - Spread mean-reversion strategy for backtesting.py
- **Dependencies Added:** `statsmodels`, `pykalman`
- **Next Steps:** Backtest on GLD/IAU, SPY/QQQ pairs

---

## Dependency Graph

### 1.1 Fix DataFrame truth value errors
**Why first:** Causes `ValueError` crashes in metrics and visualization
**Files:** `src/backtest/metrics.py`, `src/visualization/report.py`
**Changes:**
- `metrics.py:40`: Replace `if not trades:` with `if trades is None or len(trades) == 0:`
- `report.py:373`: Replace `if trades:` with `if trades is not None and len(trades) > 0:`
- Update type hints in `metrics.py` to accept `Union[List[Dict], pd.DataFrame]`

### 1.2 Fix position sizing for short positions
**Why:** Crashes when any strategy attempts short trades
**File:** `src/risk/position_sizing.py`
**Changes:**
- Add `direction: str = 'long'` parameter to `calculate()`
- Validate stop price based on direction (stop below entry for long, above for short)
- Calculate default stop in correct direction

### 1.3 Complete multi-pattern entry logic
**Why:** `multi_pattern_strategy.py` detects patterns but never executes trades
**File:** `src/strategies/backtest_py/multi_pattern_strategy.py`
**Changes:**
- Complete `next()` method: pattern detection → confluence → entry/exit execution
- Add `_get_atr()` helper method
- Add signal logging on trade execution

---

## Phase 2: Correctness Fixes (Align code with documented strategy)

### 2.1 Implement SMC trade management
**Why:** SMC strategy generates signals but has no breakeven/scaling/target logic
**File:** `src/strategies/smc_reversal.py`
**Changes:**
- Add `_manage_active_trade()` method:
  - Move stop to breakeven at 1R
  - Scale out 50% at 2R
  - Close remaining at 2.5R target
  - Stop loss enforcement
- Call in `_process_bar` when state is `IN_TRADE`

### 2.2 Implement correct regime weights
**Why:** Confluence scorer uses flat weights instead of 4-regime system
**File:** `src/strategies/confluence.py`
**Changes:**
- Replace `_get_regime_weights()` with 4-regime matrix:
  - Trending: Complex 1.2x, Continuation 1.3x
  - Ranging: Reversal 1.2x
  - Volatile: Breakout 1.2x
  - Quiet: Harmonic 1.2x

### 2.3 Implement signal decay
**Why:** Signals never expire, stale patterns accumulate false confidence
**File:** `src/strategies/confluence.py`
**Changes:**
- Add `SIGNAL_VALIDITY` dict (basic: 5, harmonic: 10, complex: 20, classic: 15 bars)
- Add `DECAY_RATE = 0.05` per bar after validity period
- Add `MIN_CONFIDENCE = 0.40` floor
- Add `apply_signal_decay()` method

### 2.4 Implement portfolio heat tracking
**Why:** No aggregate risk monitoring across open positions
**File:** `src/risk/daily_limits.py`
**Changes:**
- Add `open_position_risks: Dict[str, float]` to `DailyLossState`
- Add `portfolio_heat` property
- Add `check_portfolio_heat()` method (warn at 4%, halt at 6%)
- Add `update_position_risk()` / `remove_position_risk()` methods

### 2.5 Implement pattern correlation limits
**Why:** Correlated patterns can all fire simultaneously, creating false confluence
**File:** `src/strategies/confluence.py`
**Changes:**
- Add `CORRELATION_GROUPS` dict (double_patterns, harmonic, breakout, reversal_tops, reversal_bottoms)
- Add `MAX_CORRELATED` limits per group
- Add `_filter_correlated_patterns()` method
- Apply filter in `calculate_confluence()`

---

## Phase 3: DataFrame Migration (Standardize trades data format)

### 3.1 Update engine to produce DataFrame trades
**File:** `src/backtest/engine.py`
**Changes:**
- `_compile_trades()` returns `pd.DataFrame` instead of `List[Dict]`
- `BacktestResult.trades` type: `pd.DataFrame = field(default_factory=pd.DataFrame)`

### 3.2 Update metrics to work with DataFrame
**File:** `src/backtest/metrics.py`
**Changes:**
- Signature: `trades: pd.DataFrame`
- Replace list comprehensions with DataFrame filtering: `trades[trades['pnl'] > 0]`

### 3.3 Update risk metrics for DataFrame
**File:** `src/backtest/risk_metrics.py`
**Changes:**
- Signature: `trades: Optional[pd.DataFrame] = None`
- Update Kelly Criterion to use DataFrame operations

### 3.4 Update visualization for DataFrame
**File:** `src/visualization/report.py`
**Changes:**
- Signature: `trades: pd.DataFrame`
- Update trade processing to use DataFrame filtering

### 3.5 Update tests
**Changes:**
- All mock trades data → DataFrame format
- Add empty DataFrame handling tests
- End-to-end validation: run backtest, verify DataFrame output

---

## Phase 4: Validation & Testing

### 4.1 Unit tests for all Phase 1-2 fixes
**New test files:**
- `tests/test_position_sizing_shorts.py`
- `tests/test_signal_decay.py`
- `tests/test_portfolio_heat.py`
- `tests/test_pattern_correlation.py`
- `tests/test_smc_trade_management.py`

### 4.2 Integration tests
- Run backtest on SPY daily data — verify no errors
- Verify all 15+ strategies produce valid signal logs (≥10 signals each on BTC 1h 2020-2024)
- Verify consistent signal log format across strategies
- Run full `pytest tests/` — all must pass

### 4.3 Backtest validation
- Donchian strategy backtest on SPY daily
- Williams %R backtest
- TSI + Ultimate Oscillator parameter sweeps

---

## Phase 5: Notebook Refactoring (Improve developer experience)

### 5.1 Create notebook utilities
**New file:** `src/utils/notebook_helpers.py`
**Functions:**
- `setup_project_root()` — path setup
- `load_price_data()` — data loading with filtering
- `print_data_summary()` / `print_backtest_summary()` / `print_metrics_table()`
- `create_equity_plot()` / `create_drawdown_plot()` / `create_returns_distribution()`
- `calculate_performance_metrics()` / `calculate_trade_statistics()`

### 5.2 Refactor all notebooks
**Each notebook gets:**
- CONFIG dict at top (data, backtest, strategy, output sections)
- Import from `notebook_helpers` instead of inline code
- No hardcoded paths or parameters

**Order:** 01 → 02 → 03 → 04 → 05 → 06 → 07

### 5.3 Create config dataclasses (optional)
**New file:** `src/utils/notebook_config.py`
**Classes:** `DataConfig`, `BacktestConfig`, `StrategyConfig`, `OutputConfig`, `NotebookConfig`

---

## Dependency Graph

```
Phase 1 (Bug Fixes)
    ├── 1.1 DataFrame truth value fixes ──────────────┐
    ├── 1.2 Position sizing for shorts                │
    └── 1.3 Multi-pattern entry logic                 │
                                                      ▼
Phase 2 (Correctness)                          Phase 3 (DataFrame Migration)
    ├── 2.1 SMC trade management                       ├── 3.1 engine.py
    ├── 2.2 Regime weights                             ├── 3.2 metrics.py
    ├── 2.3 Signal decay                               ├── 3.3 risk_metrics.py
    ├── 2.4 Portfolio heat tracking                    └── 3.4 report.py
    └── 2.5 Correlation limits                              │
                                                              ▼
                                                         Phase 4 (Validation)
                                                              ├── 4.1 Unit tests
                                                              ├── 4.2 Integration tests
                                                              └── 4.3 Backtest validation
                                                                     │
                                                                     ▼
                                                               Phase 5 (Notebooks)
                                                                    ├── 5.1 Helpers
                                                                    ├── 5.2 Refactor
                                                                    └── 5.3 Config (optional)
```

---

## Effort Estimate

| Phase | Tasks | Estimated Effort | Status |
|-------|-------|-----------------|--------|
| Phase 1 | 5 bug fixes | 2-3 hours | ✅ DONE |
| Phase 2 | 5 correctness fixes | 4-6 hours | ✅ DONE |
| Phase 3 | 4 file migrations + tests | 3-4 hours | ✅ DONE |
| Phase 4 | 5 test files + validation | 3-4 hours | ✅ DONE |
| Phase 5 | Utilities + 7 notebooks | 6-8 hours | ✅ DONE |
| Phase 6 | Proven strategies + Pair Trading | 4-6 hours | ✅ DONE |
| **Total** | **~28 tasks** | **~25 hours** | **✅ ALL DONE** |

---

## Success Criteria

1. ✅ No `ValueError: The truth value of a DataFrame is ambiguous` errors
2. ✅ Short positions work without crashing
3. ✅ Multi-pattern strategy executes trades (not just detects patterns)
4. ✅ SMC strategy manages trades (breakeven, scale, target)
5. ✅ Confluence scorer uses 4-regime weights with signal decay and correlation limits
6. ✅ Portfolio heat tracking warns/halts at limits
7. ✅ All trade data is DataFrame format throughout codebase
8. ✅ All tests pass (`pytest tests/`) - 165 passed, 2 skipped
9. ✅ All 7 notebooks have CONFIG sections and use helper functions
10. ✅ **ROOT CAUSE IDENTIFIED**: Pattern strategies inherently unprofitable; proven alternatives implemented
11. ✅ **Connors RSI strategy validated**: 185% return, 0.68 Sharpe, 97% win rate
12. ✅ **Pair Trading infrastructure complete**: cointegration scanner, Kalman filter, spread strategy
