# Comprehensive Phased Implementation Plan

## investment_trying — Rule-Based Strategy Completion to Paper Trading Readiness

**Created:** 2026-04-09
**Last Updated:** 2026-04-19
**Status:** Phase 5 In Progress
**Architecture:** Event-Driven Signal Pipeline

---

## Progress Summary

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Strategy Completion | ✅ COMPLETE | 100% |
| Phase 2: Pair Trading | ✅ COMPLETE | 100% |
| Phase 3: Parameter Optimization | ✅ COMPLETE | 100% |
| Phase 4: Regime Detection | ✅ COMPLETE | 100% |
| Phase 5: ML Enhancement | 95% Complete | B7 integration pending |
| Phase 6: Research-Based Enhancements | Not Started | 0% |
| Phase 7: Paper Trading | Not Started (Optional) | 0% |

### Completed Milestones
- [x] 22 strategies backtested on SPY daily (2015-2024)
- [x] SMC Reversal adapted for backtesting.py
- [x] PairsScanner + PairTradingStrategy + Kalman filter implemented
- [x] vectorbt dependency added, RSI/MACD optimization scripts created
- [x] RegimeDetector + AdaptiveRouter implemented and tested
- [x] 209 tests pass, 2 skipped
- [x] ML feature engineering, regime classification, signal scoring implemented
- [x] Phase B5 (ML Enhancement) 90% complete

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
3.4. Implement `PairTradingStrategy`: computes spread = price_A - hedge_ratio × price_B, generates signals when spread crosses ±2σ Bollinger Bands
5. Add Kalman filter for dynamic hedge ratio updates (reference: `useful_resources/pairtrading/Pair_Trading_V2.ipynb`)
6. Backtest on GLD/IAU (2020–2024 daily) and SPY/QQQ (2020–2024 daily)
7. Document results in `reports/pair_trading_report.md`

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
| Item | Status | File Location |
|------|--------|---------------|
| ML feature engineering module | ✅ DONE | `src/ml/features.py` |
| ML regime classifier | ✅ DONE | `src/ml/regime_model.py` |
| ML signal scorer | ✅ DONE | `src/ml/signal_scorer.py` |
| ML validation notebook | ✅ DONE | `notebooks/13_ml_validation.ipynb` |
| Cross-asset features module | ✅ DONE | `src/ml/cross_asset_features.py` |
| Feature selector module | ✅ DONE | `src/ml/feature_selector.py` |
| Ensemble regime detection | ✅ DONE | `src/ml/ensemble_regime.py` |
| Enhanced validation script | ✅ DONE | `scripts/ml_validation_enhanced.py` |
| ML-Enhanced backtest scripts | 🔄 IN PROGRESS | `scripts/phase_b7_ml_backtest.py`, `scripts/phase_b7_custom_backtest.py` |

### Implementation Path
| Week | Topic | Deliverable |
|------|-------|-------------|
| 1–2 | scikit-learn basics: logistic regression, random forest, cross-validation | Feature importance analysis on existing signals |
| 3–4 | Financial feature engineering: lagged returns, rolling stats, volatility regimes | Feature pipeline integrated with `src/indicators/` |
| 5–6 | Walk-forward validation, purged cross-validation, embargo periods | Validation framework in `notebooks/13_ml_validation.ipynb` |
| 7–8 | ML-enhanced regime detection + signal confidence scoring | Hybrid system (ML for regime, rules for execution) |
| 9 | Full ML-Enhanced Backtest (B7) | End-to-end comparison of ML vs baseline on SPY |

### Phase B7 Tasks (Integration Completion)
| Task | Status | Notes |
|------|--------|-------|
| Fix AggregatedSignal interface | ⏳ Pending | Position manager expects `pattern_name` attribute |
| Complete ML-Enhanced backtest | ⏳ Pending | Run full baseline vs ML comparison on SPY 2015-2024 |
| Verify integration stability | ⏳ Pending | Test both backtesting.py and custom engine approaches |

### Why Active Now
- Rule-based regime detector already implemented (Phase 4 complete)
- scxikit-learn and xgboost already available in dependencies
- ML can augment what works: regime detection accuracy, signal scoring
- Feature engineering leverage: 34 patterns provide rich input features

---

## Phase 6: Research-Based Enhancements

**Duration:** 4–6 weeks | **Priority:** High
**Source:** 13 academic papers synthesized in `useful_resources/papers_md/research_synthesis_report.md`

### Objectives
- Implement Tier 1 research insights (high impact, high feasibility, low complexity)
- Implement Tier 2 insights (high impact, medium feasibility, medium complexity)
- Phase 7 (Tier 3) is research-only, deferred to future

### Research Source: Key Findings
1. **Friction-aware strategy design** (OOM-RL paper): Turnover penalty, dynamic rebalancing frequency
2. **Position-level risk modeling** (Jorion event-driven funds): Per-position success/failure probabilities
3. **No universal strategy** (Against Universal Trading paper): Regime detection, failure-set awareness
4. **Event-type granularity** (Event-Based Trading paper): Granular signals beat aggregation
5. **Crash factor + timing** (Fang et al.): Take-profit beats RSI exits

### Key Deliverables
| Item | Priority | File Location |
|------|----------|---------------|
| Turnover penalty constraint | Tier 1 🔴 | `src/risk/turnover_penalty.py` |
| Per-position risk modeling | Tier 1 🔴 | `src/risk/position_probability.py` |
| Portfolio circuit breakers | Tier 1 🔴 | `src/risk/circuit_breakers.py` |
| Regime declaration per strategy | Tier 1 🟠 | `src/patterns/base.py` enhancements |
| Dynamic rebalancing frequency | Tier 1 🟠 | `src/backtest/engine.py` modifications |
| Event-type weighted signals | Tier 2 🟠 | `src/signals/event_weighting.py` |
| Diversity score for portfolio | Tier 2 🟠 | `src/risk/diversity_score.py` |
| Epistemic Autopsy module | Tier 2 🟠 | `src/back/epistemic_autopsy.py` |
| Failure-set analyzers | Tier 2 🟠 | `src/strategies/failure_analysis.py` |
| Friction-adjusted scoring | Tier 2 🟠 | `src/backtest/engine.py` enhancements |

### Actionable Next Steps (Tier 1 - Implement First)

**R1: Turnover Penalty as Hard Constraint**
```python
# src/risk/turnover_penalty.py
class TurnoverPenalty:
    """Penalizes strategies with excessive annualized turnover."""
    def calculate_penalty(self, trades: int, holding_period_days: float) -> float:
        """Returns 0.0-1.0 penalty scaling factor."""
        # 6700% turnover destroyed alpha in OOM-RL study
        annualized_turnover = (365.0 / holding_period_days) * trades
        max_allowed_turnover = 2000  # Conservative threshold
        if annualized_turnover <= max_allowed_turnover:
            return 0.0
        else:
            excess = (annualized_turnover - max_allowed_turnover) / max_allowed_turnover
            return min(excess, 1.0)
```
1. Create `src/risk/turnover_penalty.py` module
2. Integrate with `src/backtest/engine.py`: apply penalty to signal confidence
3. Set threshold at 2000% annualized turnover (conservative)
4. Test on high-frequency strategies (e.g., daily rebalancing)

**R2: Per-Position Success/Failure Probability**
```python
# src/risk/position_probability.py
class PositionRiskModel:
    """Binomial outcome model per position (Jorion BET approach)."""
    def estimate_success_prob(self, signal_confidence: float, regime: RegimeState) -> float:
        """Returns probability of successful outcome."""
        # Base probability from signal confidence
        base_prob = signal_confidence
        # Adjust for regime (volatile regimes = lower success)
        if regime == RegimeState.VOLATILE:
            return base_prob * 0.7
        elif regime == RegimeState.TRENDING:
            return base_prob * 1.2
        else:
            return base_prob
```
1. Create `src/risk/position_probability.py` module
2. Integrate with position sizing in `src/risk/position_sizing.py`
3. Use for capital allocation: size ∝ success_probability
4. Validate with historical win rates per regime

**R3: Portfolio-Level Circuit Breakers**
```python
# src/risk/circuit_breakers.py
class CircuitBreaker:
    """Portfolio-wide drawdown halt mechanism."""
    def __init__(self, max_drawdown_pct: float = 20.0, cooldown_bars: int = 20):
        self.max_dd = max_drawdown_pct
        self.cooldown = cooldown_bars
        self.halt_active = False
        self.halt_counter = 0

    def check_circuit(self, current_drawdown: float, equity_curve: pd.Series) -> bool:
        """Returns True if trading should halt."""
        if current_drawdown >= self.max_dd:
            self.halt_active = True
            self.halt_counter = self.cooldown
            return True  # Halt all new positions
        elif self.halt_active:
            self.halt_counter -= 1
            if self.halt_counter <= 0:
                self.halt_active = False
            return True  # Still in cooldown
        return False
```
1. Create `src/risk/circuit_breakers.py` module
2. Add to `BacktestConfig` in `src/backtest/engine.py`
3. Default: 20% max DD, 20-bar cooldown
4. Test on crash scenarios (2020 COVID, 2008 crisis)

**R4: Regime Declaration Per Strategy**
```python
# src/patterns/base.py
@dataclass
class BasePattern(ABC):
    """Enhanced with regime compatibility declaration."""
    name: str
    min_bars_required: int = 20
    preferred_regimes: List[RegimeState] = field(default_factory=list)
    incompatible_regimes: List[RegimeState] = field(default_factory=list)

    @abstractmethod
    def detect(self, df: pd.DataFrame, i: int) -> PatternResult:
        pass
```
1. Update `BasePattern` in `src/patterns/base.py` with regime fields
2. Add regime validation in `src/signals/signal_generator.py`
3. Tag each pattern with preferred regimes (e.g., RSI Divergence prefers ranging)
4. Backtest with regime-aware signal gating

**R5: Dynamic Rebalancing Frequency**
```python
# src/backtest/engine.py (enhancement)
class BacktestEngine:
    def estimate_optimal_frequency(self, signal_decay_rate: float, tx_cost_pct: float) -> str:
        """Returns 'daily', 'weekly', or 'monthly' based on decay/cost ratio."""
        # OOM-RL: signal decay vs. transaction cost tradeoff
        # Daily rebalancing destroyed 6700% turnover alpha
        ratio = signal_decay_rate / tx_cost_pct
        if ratio > 10:
            return "daily"
        elif ratio > 2:
            return "weekly"
        else:
            return "monthly"
```
1. Add frequency estimation to `BacktestEngine`
2. Calculate signal decay rate from historical signal persistence
3. Default to weekly rebalancing (saferer than daily)
4. Test: compare daily vs. weekly on 10 strategies

### Actionable Next Steps (Tier 2 - Implement Second)

**R6: Event-Type Weighted Signal Aggregation**
1. Build event taxonomy in `src/signals/event_taxonomy.py`
2. Map pattern types to event types (e.g., breakout → "trend_initiation")
3. Weight signals by event-type informativeness (from research)
4. Replace equal-weighted confluence scoring

**R7: Diversity Score for Portfolio Sizing**
1. Implement BET diversity score: `D = N / (1 + 2*sum(rho_ij))`
2. Calculate correlation matrix of open positions
3. Use effective number of independent bets for sizing
4. Replace simple position count with diversity-adjusted limit

**R8: Epistemic Autopsy Module**
1. Create `src/backtest/epistemic_autopsy.py`
2. On drawdown > threshold, generate JSON report:
   ```json
   {
     "drawdown_percent": -15.2,
     "root_cause": "VWAP Bounce pattern on volatile regime",
     "contributing_patterns": ["VWAP Bounce", "EMA Ribbon"],
     "remediation": "Disable VWAP in volatile regime"
   }
   ```
3. Integrate with circuit breaker trigger
4. Review reports for systematic failures

**R9: Failure-Set Analyzers**
1. Build test suite per strategy:
   - Time-reversal test
   - Persistent counter-trend test
   - Fat-tail test
2. Document failure modes per strategy
3. Add to strategy documentation
4. Use for go/no-go decision on live deployment

**R10: Friction-Adjusted Backtest Scoring**
1. Add transaction cost model to `BacktestEngine`
2. Include slippage: 0.05% liquid, 0.2% illiquid
3. Calculate TCA (Total Cost Analysis) per strategy
4. Report friction-adjusted Sharpe alongside raw Sharpe

### Deployment Readiness Checkpoint (Tier 1)
| Metric | Threshold | Status |
|--------|-----------|--------|
| Turnover penalty implemented | Yes | ☐ |
| Per-position risk model | Yes | ☐ |
| Portfolio circuit breaker | Yes | ☐ |
| Regime declarations added | ≥90% of patterns | ☐ |
| Dynamic rebalancing | Yes | ☐ |
| Tier 1 backtests pass | 100% | ☐ |

### Deployment Readiness Checkpoint (Tier 2)
| Metric | Threshold | Status |
|--------|-----------|--------|
| Event taxonomy built | Yes | ☐ |
| Diversity score | Yes | ☐ |
| Epistemic autopsy | Yes | ☐ |
| Failure-set analyzers | ≥5 strategies | ☐ |
| Friction-adjusted scoring | Yes | ☐ |
| Tier 2 backtests pass | 100% | ☐ |

---

## Phase 7: Paper Trading & Live Readiness (Optional)

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
- **Daily loss limit**: 3% of simulated equity → halt trading for day
- **Weekly loss limit**: 6% → halt trading for week, review signals
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
    → Phase 6 (Research-Based Enhancements) — Not Started
    → Phase 7 (Paper Trading) — Optional
```

Each phase's deliverables become inputs to the next phase. No phase should begin until prior phase's deployment readiness checkpoints are fully satisfied.
