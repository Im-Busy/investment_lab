# Comprehensive Phased Implementation Plan

## investment_trying — Rule-Based Strategy Completion to Paper Trading Readiness

**Created:** 2026-04-09
**Last Updated:** 2026-04-09
**Status:** Phase 1 In Progress
**Architecture:** Event-Driven Signal Pipeline

---

## Progress Summary

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Strategy Completion | ✅ COMPLETE | 100% |
| Phase 2: Pair Trading | ✅ COMPLETE | 100% |
| Phase 3: Parameter Optimization | ✅ COMPLETE | 100% |
| Phase 4: Regime Detection | ✅ COMPLETE | 100% |
| Phase 5: ML Enhancement | Active Development | 0% |
| Phase 6: Paper Trading | Not Started (Optional) | 0% |

### Completed Milestones
- [x] 22 strategies backtested on SPY daily (2015-2024)
- [x] SMC Reversal adapted for backtesting.py
- [x] PairsScanner + PairTradingStrategy + Kalman filter implemented
- [x] vectorbt dependency added, RSI/MACD optimization scripts created
- [x] RegimeDetector + AdaptiveRouter implemented and tested
- [x] 209 tests pass, 2 skipped

---

## Phase 1: Strategy Completion & Gap Closure

**Duration:** 1–2 weeks | **Priority:** Critical | **Status:** 75% Complete

### Objectives
- [x] Complete all pending strategy implementations
- [x] Finalize Donchian Channel breakout strategy
- [ ] Integrate TradingView-sourced strategies not yet in codebase

### Technical Dependencies
- Existing: `src/strategies/`, `src/patterns/breakout/donchian.py`, `src/backtest/engine.py`
- New: None

### Key Deliverables
| Item | Status | File Location |
|------|--------|---------------|
| Donchian Channel strategy | ✅ DONE | `src/strategies/donchian_breakout.py` |
| Donchian pattern detector | ✅ DONE (8/8 tests pass) | `src/patterns/breakout/donchian.py` |
| Donchian backtest runner | ✅ DONE (BTC 1h: 615 trades, 28% WR) | `scripts/backtest_all_strategies.py` |
| Williams %R optimization | Implemented, needs backtest | `src/strategies/williams_r_reversal.py` |
| TSI strategy tuning | Implemented, needs parameter sweep | `src/strategies/tsi_strategy.py` |
| Ultimate Oscillator tuning | Implemented, needs parameter sweep | `src/strategies/ultimate_oscillator.py` |

### Completed Actions
1. ✅ Created `DonchianChannelStrategy` class in `src/strategies/donchian_breakout.py` (backtesting.py compatible)
2. ✅ Added Donchian to `backtest_all_strategies.py` runner
3. ✅ Ran backtest on BTC/USD 1h: 615 trades, 28.0% WR, Sharpe -3.02, MaxDD -80.1%
4. ⏳ Verify all 15+ strategies produce valid signal logs — pending

### Remaining Actions
1. Run backtest on SPY daily data
2. Verify all 15+ strategies produce valid signal logs
3. Run full test suite (`pytest tests/`)

### Deployment Readiness Checkpoint
- [x] Donchian strategy passes backtest on BTC 1h data
- [ ] All strategies pass `pytest tests/` without errors
- [ ] Each strategy produces ≥10 signals on BTC 1h data (2020–2024)
- [ ] Signal log format is consistent across all strategies

---

## Phase 2: Pair Trading Integration

**Duration:** 2–3 weeks | **Priority:** High

### Objectives
- Implement cointegration-based pairs scanner
- Build spread-trading strategy with Kalman filter hedge ratios
- Backtest on existing ETF pairs (GLD/IAU, SPY/QQQ)

### Technical Dependencies
- Existing: `src/data_ingestion/fetch_data.py`, `src/backtest/engine.py`
- New: `statsmodels` (coint), `pykalman` (Kalman filter)

### Key Deliverables
| Item | File Location |
|------|---------------|
| Pairs scanner (`coint`-based) | `src/strategies/pairs_scanner.py` |
| Spread trading strategy | `src/strategies/pair_trading.py` |
| Kalman filter hedge ratio module | `src/indicators/kalman_hedge.py` |
| Pairs backtest notebook | `notebooks/10_pair_trading_backtest.ipynb` |

### Actionable Next Steps
1. `uv add statsmodels pykalman`
2. Implement `PairsScanner` class: takes universe of symbols, returns cointegrated pairs with p-value < 0.05
3. Implement `PairTradingStrategy`: computes spread = price_A - hedge_ratio × price_B, generates signals when spread crosses ±2σ Bollinger Bands
4. Add Kalman filter for dynamic hedge ratio updates (reference: `useful_resources/pairtrading/Pair_Trading_V2.ipynb`)
5. Backtest on GLD/IAU (2020–2024 daily) and SPY/QQQ (2020–2024 daily)
6. Document results in `reports/pair_trading_report.md`

### Risk Management Protocols
- Max spread position: 10% of equity per pair
- Stop-loss: spread exceeds ±3σ for 5+ consecutive bars
- Hedge ratio recalibration: every 20 bars (rolling window)

### Deployment Readiness Checkpoint
- [ ] Scanner identifies ≥3 cointegrated pairs from 20-symbol universe
- [ ] Spread strategy produces positive Sharpe ratio on out-of-sample data
- [ ] Kalman filter hedge ratio converges within 50 bars

---

## Phase 3: Parameter Optimization via vectorbt

**Duration:** 1–2 weeks | **Priority:** High

### Objectives
- Install vectorbt alongside existing framework for research-only parameter sweeps
- Optimize RSI, MACD, Stoch RSI, and CCI thresholds across all assets
- Document optimal parameter ranges per asset class

### Technical Dependencies
- Existing: All strategy implementations in `src/strategies/`
- New: `vectorbt` (research only — does not replace custom backtest engine)

### Key Deliverables
| Item | File Location |
|------|---------------|
| vectorbt parameter sweep scripts | `scripts/optimize_*.py` |
| Optimal parameter report | `reports/parameter_optimization.csv` |
| Parameter robustness analysis | `notebooks/11_parameter_robustness.ipynb` |

### Actionable Next Steps
1. `uv add vectorbt`
2. Create `scripts/optimize_rsi.py`: sweep RSI window (10–30) and thresholds (25–35 / 65–75) across BTC, SPY, GLD, EURUSD
3. Create `scripts/optimize_macd.py`: sweep fast/slow/signal combinations
4. Run all sweeps; aggregate results into `reports/parameter_optimization.csv`
5. Identify stable parameter ranges (performance varies <10% across adjacent parameter values)
6. Update `src/config.py` with optimized defaults

### Framework Selection Rationale
| Tool | Role | Why |
|------|------|-----|
| **Custom engine** (`src/backtest/`) | Production backtesting | Event-driven, supports multi-pattern confluence, signal logging, ablation analysis — no off-the-shelf tool provides this |
| **vectorbt** | Research & parameter sweeps | NumPy/Numba vectorized, excellent for grid searches across 10K+ parameter combos |
| **backtrader** | Reference only | Study for live trading hooks if going live later |
| **zipline / backtesting.py** | Not adopted | zipline = daily factor investing; backtesting.py = too minimal for multi-strategy pipeline |

### Deployment Readiness Checkpoint
- [ ] Each strategy has ≥1 stable parameter set per asset class
- [ ] Optimized parameters improve Sharpe ratio by ≥15% vs. defaults on out-of-sample data
- [ ] No overfitting detected (in-sample vs. out-of-sample performance gap <20%)

---

## Phase 4: Regime Detection & Adaptive Strategy Selection

**Duration:** 2–3 weeks | **Priority:** Medium-High

### Objectives
- Build regime classifier (trending / ranging / volatile) using ADX + ATR
- Implement adaptive strategy selection: enable/disable strategies based on detected regime
- Integrate with existing `src/analysis/pattern_selector.py`

### Technical Dependencies
- Existing: `src/indicators/technical.py` (ADX, ATR), `src/analysis/pattern_selector.py`
- New: None

### Key Deliverables
| Item | File Location |
|------|---------------|
| Regime classifier | `src/indicators/regime_detector.py` |
| Adaptive strategy router | `src/strategies/adaptive_router.py` |
| Regime-aware backtest | `notebooks/12_regime_aware_backtest.ipynb` |
| Regime performance report | `reports/regime_performance.md` |

### Actionable Next Steps
1. Implement `RegimeDetector` class:
   - **Trending**: ADX(14) > 25
   - **Ranging**: ADX(14) < 20 AND ATR(14) < 20-period median
   - **Volatile**: ATR(14) > 80th percentile of trailing 100-bar ATR
   - **Transition**: ADX(14) between 20–25 (default to previous regime)
2. Map strategies to regimes:

   | Regime | Enabled Strategies | Disabled Strategies |
   |--------|-------------------|---------------------|
   | Trending | EMA Ribbon, SMA Crossover, ADX, Parabolic SAR, TSI | RSI Divergence, Williams %R, Stoch RSI |
   | Ranging | RSI Divergence, Williams %R, Stoch RSI, CCI, MFI | EMA Ribbon, SMA Crossover, ADX |
   | Volatile | Chandelier Exit, Bollinger-based, VWAP Bounce | All trend-following |

3. Integrate regime filter into `src/backtest/engine.py` as a pre-signal gate
4. Backtest regime-aware vs. static strategy selection; measure improvement in Sharpe and max drawdown

### Risk Management Protocols
- Regime misclassification penalty: if regime changes mid-trade, apply 1.5× stop-loss
- Volatility circuit breaker: if ATR spikes >3× median, flatten all positions for 5 bars

### Deployment Readiness Checkpoint
- [ ] Regime detector classifies ≥80% of bars correctly (validated against manual labeling on 100 random segments)
- [ ] Regime-aware backtest improves Sharpe by ≥10% vs. static selection
- [ ] Max drawdown reduced by ≥15% in volatile regimes

---

## Phase 5: ML Enhancement (Active Development)

**Duration:** 4–8 weeks | **Priority:** High (no longer deferred)

### Objectives
- Build ML-enhanced regime detector (replace or augment rule-based ADX/ATR classifier)
- Evaluate ML for signal confidence scoring
- Implement feature engineering pipeline for trading signals

### Technical Dependencies
- Existing: All prior phases completed
- Available: `scikit-learn`, `xgboost` (already in pyproject.toml dependencies)

### Key Deliverables
| Item | File Location |
|------|---------------|
| ML feature engineering module | `src/ml/features.py` |
| ML regime classifier | `src/ml/regime_model.py` |
| ML signal scorer | `src/ml/signal_scorer.py` |
| ML validation notebook | `notebooks/13_ml_validation.ipynb` |

### Implementation Path
| Week | Topic | Deliverable |
|------|-------|-------------|
| 1–2 | scikit-learn basics: logistic regression, random forest, cross-validation | Feature importance analysis on existing signals |
| 3–4 | Financial feature engineering: lagged returns, rolling stats, volatility regimes | Feature pipeline integrated with `src/indicators/` |
| 5–6 | Walk-forward validation, purged cross-validation, embargo periods | Validation framework in `notebooks/13_ml_validation.ipynb` |
| 7–8 | ML-enhanced regime detection + signal confidence scoring | Hybrid system (ML for regime, rules for execution) |

### Why Active Now
- Rule-based regime detector already implemented (Phase 4 complete)
- scikit-learn and xgboost already available in dependencies
- ML can augment what works: regime detection accuracy, signal scoring
- Feature engineering leverage: 34 patterns provide rich input features

---

## Phase 6: Paper Trading & Live Readiness (Optional)

**Duration:** 2–4 weeks | **Priority:** Low (Optional)

### Objectives
- Deploy paper trading pipeline using existing backtest engine with live data feed
- Monitor signal quality, execution latency, and slippage for 2+ weeks
- Establish go/no-go criteria for live trading

### Technical Dependencies
- Existing: `src/backtest/engine.py`, `src/signals/signal_generator.py`, `src/risk/position_sizing.py`
- New: `yfinance` live data polling (or broker API: Alpaca, Interactive Brokers)

### Key Deliverables
| Item | File Location |
|------|---------------|
| Paper trading runner | `scripts/paper_trade.py` |
| Live signal log | `logs/paper_trade_signals.log` |
| Paper trading performance report | `reports/paper_trading_report.md` |
| Go/No-Go decision document | `reports/live_readiness.md` |

### Actionable Next Steps
1. Create `scripts/paper_trade.py`:
   - Polls live data every 1h (or 5m for intraday strategies)
   - Runs signal generation pipeline
   - Logs signals to `logs/paper_trade_signals.log` with timestamp, asset, strategy, confidence, suggested position
   - Simulates fills with realistic slippage (0.05% for liquid assets, 0.2% for illiquid)
2. Run paper trading for minimum 14 calendar days
3. Compare paper trading signals vs. backtest predictions; flag discrepancies >5%
4. Calculate live Sharpe, win rate, profit factor; compare to backtest benchmarks
5. Write `reports/live_readiness.md` with go/no-go recommendation

### Risk Management Protocols
- **Daily loss limit**: 3% of simulated equity → halt trading for the day
- **Weekly loss limit**: 6% → halt trading for the week, review signals
- **Monthly loss limit**: 10% → full system review, parameter recalibration required
- **Position sizing**: Kelly fraction capped at 2% risk per trade, max 5 open positions
- **Circuit breaker**: If 3 consecutive losses >2% each, pause system for 24h review

### Deployment Readiness Checkpoint (Go/No-Go Criteria)
| Metric | Threshold | Status |
|--------|-----------|--------|
| Paper trading duration | ≥14 calendar days | ☐ |
| Signal match rate (vs. backtest) | ≥90% | ☐ |
| Live Sharpe ratio | ≥0.5 | ☐ |
| Live win rate | ≥45% | ☐ |
| Live profit factor | ≥1.2 | ☐ |
| Max drawdown | ≤15% | ☐ |
| Slippage within tolerance | ≤0.2% average | ☐ |
| Zero critical bugs | Yes | ☐ |

**All criteria must pass for go-live decision.**
**This phase is optional** — focus on Phase 5 (ML Enhancement) first.

---

## Infrastructure & Risk Management Summary

### Backtesting Infrastructure
```
src/backtest/
├── engine.py          # Event-driven backtest engine (keep, don't replace)
├── metrics.py         # Sharpe, Sortino, Calmar, max drawdown, win rate
├── risk_metrics.py    # VaR, CVaR, tail risk
├── benchmark.py       # Buy-and-hold benchmark, strategy vs. benchmark
└── adapter.py         # Data adapter (CSV → OHLCV DataFrame)
```

### Risk Management Protocols (All Phases)
| Protocol | Parameter | Enforced In |
|----------|-----------|-------------|
| Risk per trade | 2% of equity | `src/risk/position_sizing.py` |
| Max position size | 20% of equity | `src/risk/position_sizing.py` |
| Max open positions | 5 | `src/signals/position_manager.py` |
| Daily loss limit | 3% | `src/risk/daily_limits.py` |
| Weekly loss limit | 6% | `src/risk/daily_limits.py` |
| Monthly loss limit | 10% | `src/risk/daily_limits.py` |
| Slippage model | 0.05% (liquid), 0.2% (illiquid) | `src/backtest/engine.py` |
| Commission | 0.1% per trade | `src/config.py` |

### Data Pipeline
```
data/raw/
    → src/data_ingestion/fetch_data.py
    → src/backtest/adapter.py
    → src/signals/signal_generator.py
    → src/backtest/engine.py
    → reports/
```

### Execution Order Summary
```
Phase 1 (Strategy Completion) ✅
    → Phase 2 (Pair Trading) ✅
    → Phase 3 (Parameter Optimization) ✅
    → Phase 4 (Regime Detection) ✅
    → Phase 5 (ML Enhancement) — Active Development
    → Phase 6 (Paper Trading) — Optional
```

Each phase's deliverables become inputs to the next phase. No phase should begin until the prior phase's deployment readiness checkpoints are fully satisfied.
