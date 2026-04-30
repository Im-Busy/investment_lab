# Insight Registry

**Last Updated:** 2026-04-25  
**Total Insights:** 42  
**Papers Analyzed:** 13

---

## Tag Legend

### Topics
- **Risk**: Risk management, position sizing, drawdown control
- **Friction**: Transaction costs, turnover, slippage
- **Regime**: Market regime detection, adaptive strategies
- **Signal.Quality**: Signal aggregation, confluence scoring, filtering
- **Event.Type**: Event-type extraction, sentiment analysis
- **Position.Sizing**: Capital allocation, Kelly criterion
- **Portfolio**: Portfolio construction, diversity, correlation
- **Strategy.Lifecycle**: Strategy decay, retirement, validation

### Impact Levels
- 🔴 **Critical**: Must implement; alpha-destroying if ignored
- 🟠 **High**: Significant performance impact
- 🟡 **Medium**: Nice to have; incremental improvement
- 🟢 **Low**: Marginal benefit

### Status
- ✅ **Implemented**: Code exists in `src/`
- 🔄 **In-Progress**: Currently being implemented
- ⏳ **Deferred**: Backlogged; not yet started
- ❌ **Rejected**: Explicitly decided against

---

## Insights by Paper

### P1: OOM-RL (arXiv:2604.11477)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I1.1 | Turnover penalty as hard constraint | Friction | 🔴 Critical | ✅ | `src/backtest/friction_scoring.py:461-510` |
| I1.2 | Dynamic rebalancing frequency (daily→weekly) | Signal.Decay | 🟠 High | ✅ | Phase 11 validation |
| I1.3 | Structured drawdown diagnostics ("Epistemic Autopsy") | Risk | 🔴 Critical | ⏳ | Backlog #3 |
| I1.4 | Liquidity loading filter for position sizing | Friction | 🟡 Medium | ⏳ | Backlog #8 |
| I1.5 | Immutable test suites for strategy code | Strategy.Lifecycle | 🟡 Medium | ❌ | Rejected (overhead > benefit) |

### P2: Investing Is Compression (arXiv:2604.10758)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I2.1 | Divergence-in-bits metric for strategy comparison | Signal.Quality | 🟠 High | ⏳ | Backlog #11 |
| I2.2 | Winner-fraction position sizing heuristic | Position.Sizing | 🟠 High | ⏳ | Backlog #12 |
| I2.3 | MDL (Minimum Description Length) for feature pruning | Signal.Quality | 🟡 Medium | ⏳ | Backlog #13 |

### P3: Risk Management for Event-Driven Funds (SSRN-1018281)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I3.1 | Per-position success/failure probability (binary outcomes) | Risk | 🔴 Critical | ✅ | `src/risk/position_sizing.py` |
| I3.2 | Diversity score for portfolio construction (BET formula) | Portfolio | 🟠 High | ✅ | `src/risk/diversity_score.py` |
| I3.3 | Market regime → break probability linkage | Regime | 🟡 Medium | ⏳ | Backlog #14 |
| I3.4 | Economic capital sizing at 99.9% VaR | Risk | 🟠 High | ✅ | `src/risk/daily_limits.py` |
| I3.5 | Optimal portfolio size tradeoff (N vs. expertise dilution) | Portfolio | 🟡 Medium | ⏳ | Backlog #15 |

### P4: Against a Universal Trading Strategy (arXiv:2604.13334)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I4.1 | Regime declaration mandatory per strategy | Regime | 🔴 Critical | ✅ | `src/indicators/regime_detector.py` |
| I4.2 | Failure-set analyzers (time-reversal, counter-trend, fat-tail) | Risk | 🟠 High | ⏳ | Backlog #9 |
| I4.3 | Strategy decay monitoring & automatic retirement | Strategy.Lifecycle | 🟠 High | ⏳ | Backlog #16 |
| I4.4 | Cascade risk prevention (portfolio-level circuit breakers) | Risk | 🔴 Critical | ✅ | `src/risk/daily_limits.py:308-411` |
| I4.5 | Undecidability humility → defense in depth | Risk | 🟡 Medium | ✅ | Position limits, stop-losses, drawdown halts |

### P5: Survey of Statistical Arbitrage Pair Trading (WNE_WP485)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I5.1 | Add pairs trading as 8th pattern category | Strategy.Lifecycle | 🟡 Medium | ⏳ | Backlog #17 |
| I5.2 | Cointegration + ML hybrid for pair selection | Signal.Quality | 🟡 Medium | ⏳ | Backlog #18 |
| I5.3 | Volatility-adaptive thresholds (dynamic z-scores) | Signal.Quality | 🟠 High | ⏳ | Backlog #19 |

### P6: Event-Based Trading: IE Tools (SSRN-2907600)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I6.1 | Event-type signals outperform aggregated sentiment | Event.Type | 🟠 High | ⏳ | Backlog #6 |
| I6.2 | Holding-period alignment per event type | Event.Type | 🟡 Medium | ⏳ | Backlog #20 |
| I6.3 | Event-type taxonomy for signal classification | Event.Type | 🟡 Medium | ⏳ | Backlog #21 |

### P7: Multimodal Event-driven LSTM (IEEE Access)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I7.1 | Tensor-based signal fusion (not vector concatenation) | Signal.Quality | 🟡 Medium | ⏳ | Research phase |
| I7.2 | Event-driven memory updates (not fixed time steps) | Signal.Quality | 🟡 Medium | ❌ | Rejected (complexity > benefit) |
| I7.3 | Co-movement signals (sector peer correlations) | Portfolio | 🟡 Medium | ⏳ | Backlog #22 |

### P8: Algorithmic Trading & AI Review (WJAETS-2024-0054)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I8.1 | Portfolio-level circuit breakers essential | Risk | 🔴 Critical | ✅ | `src/risk/daily_limits.py` |
| I8.2 | Adaptive parameter tuning by market conditions | Regime | 🟠 High | ✅ | `src/strategies/adaptive_router.py` |
| I8.3 | Stress scenario testing (flash crash, liquidity drought) | Risk | 🟡 Medium | ⏳ | Backlog #23 |

### P9: Calendar of Events Database (IEEE Access)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I9.1 | Event database as feature store | Event.Type | 🟡 Medium | ⏳ | Backlog #24 |
| I9.2 | Spike detection for signal triggers | Signal.Quality | 🟡 Medium | ⏳ | Backlog #25 |
| I9.3 | Keyword-signal mapping by sector | Event.Type | 🟢 Low | ⏳ | Backlog #26 |

### P10: Emergence of Statistical Financial Factors (arXiv)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I10.1 | Network-derived factor model (alternative to PCA) | Portfolio | 🟡 Medium | ⏳ | Research phase |
| I10.2 | Coupling matrix for stock relationships | Portfolio | 🟢 Low | ⏳ | Research phase |
| I10.3 | Optimal regime for factor emergence | Regime | 🟢 Low | ⏳ | Research phase |

### P13: Crash-Based Quantitative Trading Strategies (Finance Research Letters, 2022)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I13.1 | Crash factor as pre-trade risk filter | Risk | 🟠 High | ⏳ | Backlog #27 |
| I13.2 | Fixed take-profit (3%) > RSI exits | Signal.Quality | 🟠 High | ✅ | `src/signals/position_manager.py` |
| I13.3 | CMRS scoring for signal ranking | Signal.Quality | 🟡 Medium | ⏳ | Backlog #28 |
| I13.4 | Momentum crash survivability via crash-factor adjustment | Regime | 🟠 High | ⏳ | Backlog #29 |

---

## Cross-Cutting Themes

| Theme | Related Insights | Implementation Status |
|-------|------------------|----------------------|
| **No Free Lunch** (every strategy has failure set) | I4.1, I4.2, I4.3, I5.3 | ✅ Partially implemented (regime detector, adaptive router) |
| **Friction > Signal Strength at Scale** | I1.1, I1.2, I8.1, I13.2 | ✅ Implemented (friction_scoring.py) |
| **Event Granularity Beats Aggregation** | I6.1, I6.2, I6.3, I9.1 | ⏳ Deferred (needs event taxonomy) |
| **Position-Level Risk > Portfolio Metrics** | I3.1, I3.2, I3.4, I13.1 | ✅ Implemented (position_sizing.py, daily_limits.py) |
| **Crash Factors + Timing > Risk Filters Alone** | I13.1, I13.3, I13.4 | ⏳ Partially implemented |

---

## Insights by Topic

### Risk (16 insights)
I1.1, I1.3, I3.1, I3.4, I4.2, I4.4, I4.5, I8.1, I8.3, I13.1, I13.2, I3.3, I3.5, I4.3, I13.3, I13.4

### Friction (4 insights)
I1.1, I1.4, I8.1, I13.2

### Regime (7 insights)
I3.3, I4.1, I8.2, I10.3, I13.4, I5.3, I8.2

### Signal.Quality (11 insights)
I2.1, I2.3, I5.2, I5.3, I7.1, I7.2, I9.2, I13.2, I13.3, I6.2, I6.3

### Event.Type (5 insights)
I6.1, I6.2, I6.3, I9.1, I9.3

### Portfolio (6 insights)
I3.2, I3.5, I7.3, I10.1, I10.2, I10.3

### Position.Sizing (3 insights)
I2.2, I3.1, I3.4

### Strategy.Lifecycle (4 insights)
I1.5, I4.3, I5.1, I13.4

---

## Summary Statistics

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Implemented | 15 | 36% |
| 🔄 In-Progress | 0 | 0% |
| ⏳ Deferred/Backlog | 23 | 55% |
| ❌ Rejected | 2 | 5% |
| 📊 Research Phase | 2 | 5% |

| Impact | Count | Percentage |
|--------|-------|------------|
| 🔴 Critical | 7 | 17% |
| 🟠 High | 15 | 36% |
| 🟡 Medium | 17 | 40% |
| 🟢 Low | 3 | 7% |

---

*Registry generated from 13 research papers. Insights extracted from research_synthesis_report.md and SENTIMENT_ANALYSIS_SUMMARY.md.*
