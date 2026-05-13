# Hypothesis Backlog

**Last Updated:** 2026-05-08
**Total Hypotheses:** 14
**Ready for Testing:** 9

---

## Format

Each hypothesis includes:
- **Statement:** Testable claim with metric
- **Source:** Paper/insight that motivated it
- **Experiment:** How to test
- **Metric:** Success criteria
- **Priority:** Based on impact × feasibility
- **Dependencies:** Prerequisites for testing

---

## H1: Event-Type Weighting Improves Signal Quality

- **Statement:** Weighting signals by event-type (vs. equal weight) improves Sharpe by ≥15%
- **Source:** P6 (Event-Based Trading: IE Tools) — "Event-type signals outperform aggregated sentiment"
- **Experiment:**
  1. Tag all 34 patterns with event-type taxonomy (continuation, reversal, breakout, indecision)
  2. Backtest equal-weight vs. event-type-weight aggregation on SPY 2020-2025
  3. Compare Sharpe, win rate, max drawdown
- **Metric:** ΔSharpe ≥ 0.15, ΔWinRate ≥ 5%, p < 0.05
- **Priority:** 🔴 High
- **Status:** ⏳ Backlog #6
- **Dependencies:** Need event taxonomy first

---

## H2: Dynamic Rebalancing Frequency Reduces Friction

- **Statement:** Adapting rebalance frequency to signal decay rate reduces turnover 30%+ without return loss
- **Source:** P1 (OOM-RL) — "Daily→weekly shift was most impactful change"
- **Experiment:** Compare fixed daily vs. adaptive (decay-based) rebalancing on 10 strategies
- **Metric:** Turnover reduction ≥30%, ΔReturn ≤5%, ΔSharpe ≥ -0.1
- **Priority:** 🟠 High
- **Status:** ✅ Validated (Phase 11)
- **Dependencies:** None

---

## H3: Failure-Set Analyzers Prevent Cascade Losses

- **Statement:** Pre-trade failure-set validation (time-reversal, counter-trend, fat-tail) reduces max drawdown by ≥20%
- **Source:** P4 (Against Universal Trading) — "Every strategy has a mapped failure set"
- **Experiment:**
  1. Build failure-set analyzers for top 5 strategies
  2. Backtest with/without validation on 2008, 2020 stress periods
  3. Measure MDD, tail loss, recovery time
- **Metric:** ΔMDD ≤ -20%, ΔTailLoss ≤ -25%
- **Priority:** 🟠 High
- **Status:** ⏳ Backlog #9
- **Dependencies:** Need failure-set taxonomy per strategy

---

## H4: Diversity Score Improves Risk-Adjusted Returns

- **Statement:** Sizing positions by diversity-adjusted capital (BET formula) improves portfolio Sharpe by ≥10%
- **Source:** P3 (Jorion) — "ρ=0.03 correlation → 6x economic capital reduction"
- **Experiment:**
  1. Calculate pattern signal correlation matrix
  2. Compute diversity score using BET formula
  3. Backtest equal-weight vs. diversity-weighted sizing
- **Metric:** ΔSharpe ≥ 0.1, ΔCapitalEfficiency ≥ 20%
- **Priority:** 🟠 High
- **Status:** ✅ Implemented (`src/risk/diversity_score.py`)
- **Dependencies:** None

---

## H5: Crash Factor Filter Reduces Tail Loss

- **Statement:** Filtering trades by crash probability (top 10% excluded) reduces 99th percentile loss by ≥25%
- **Source:** P13 (Crash-Based Strategies) — CMRS quintiles show monotonic risk pattern
- **Experiment:**
  1. Implement crash factor logistic model (10 features from P13)
  2. Backtest strategy with/without crash filter
  3. Compare tail metrics (99th %ile loss, skewness, kurtosis)
- **Metric:** Δ99thLoss ≤ -25%, ΔSkewness ≥ 0.2
- **Priority:** 🟠 High
- **Status:** ⏳ Backlog #27
- **Dependencies:** Crash factor model implementation

---

## H6: Fixed Take-Profit > Technical Exits

- **Statement:** Fixed percentage take-profit (3%) outperforms RSI-based exits in win rate and Sharpe
- **Source:** P13 — "Method 2 (3% take-profit) win rate 56% vs. Method 1 (RSI) 30%"
- **Experiment:** Backtest fixed 3% TP vs. RSI >70 exit on MSL, Cup&Handle, Flag patterns
- **Metric:** ΔWinRate ≥ 10%, ΔSharpe ≥ 0.15
- **Priority:** 🟠 High
- **Status:** ✅ Implemented (`src/signals/position_manager.py`)
- **Dependencies:** None

---

## H7: Friction Scoring Predicts Strategy Failure

- **Statement:** Strategies with friction drag >50 bps annualized underperform by ≥1.0 Sharpe
- **Source:** P1 — "0.08% slippage × 6700% turnover = alpha destruction"
- **Experiment:**
  1. Calculate friction drag for 20 strategies
  2. Sort into friction quartiles
  3. Compare net Sharpe vs. gross Sharpe across quartiles
- **Metric:** High-friction quartile Sharpe < Low-friction quartile by ≥1.0
- **Priority:** 🟡 Medium
- **Status:** ✅ Implemented (`src/backtest/friction_scoring.py`)
- **Dependencies:** None

---

## H8: Regime-Adaptive Strategies Outperform Static

- **Statement:** Regime-filtered strategy selection (ADX-based) improves Sharpe by ≥0.3 vs. always-on
- **Source:** P4, P8 — "Regime declaration mandatory"
- **Experiment:**
  1. Classify bars into Trending/Ranging/Volatile regimes
  2. Enable/disable strategies per regime mapping
  3. Compare vs. always-on baseline
- **Metric:** ΔSharpe ≥ 0.3, ΔMDD ≤ -15%
- **Priority:** 🟠 High
- **Status:** ✅ Implemented (`src/strategies/adaptive_router.py`)
- **Dependencies:** None

---

## H9: Divergence-in-Bits > Sharpe for Strategy Comparison

- **Statement:** Divergence-in-bits metric better predicts OOS performance than Sharpe ratio
- **Source:** P2 (Investing Is Compression) — "D_KL(W*\|W) is unit-independent strategy comparison metric"
- **Experiment:**
  1. Implement KL divergence estimator for strategy weight distributions
  2. Compute divergence-in-bits for 20 strategies
  3. Correlate with OOS Sharpe degradation
- **Metric:** |Corr(divergence, OOS degradation)| ≥ 0.6 vs. |Corr(Sharpe, OOS degradation)| ≤ 0.4
- **Priority:** 🟡 Medium
- **Status:** ⏳ Backlog #11
- **Dependencies:** Need KL divergence implementation

---

## H10: Winner-Fraction Sizing vs. Equal-Weight

- **Statement:** Winner-fraction heuristic allocates capital proportional to historical win probability, improving growth rate by ≥10%
- **Source:** P2 — "Allocate by fraction of universes where each pattern wins"
- **Experiment:** Backtest winner-fraction vs. equal-weight vs. Kelly sizing
- **Metric:** ΔCAGR ≥ 10%, ΔKellyGrowth ≥ 0.05
- **Priority:** 🟡 Medium
- **Status:** ⏳ Backlog #12
- **Dependencies:** Need historical win counting per pattern

---

## H11: Strategy Decay Detection Prevents Drawdown

- **Statement:** Monitoring rolling Sharpe (60-day) and retiring strategies below threshold reduces MDD by ≥15%
- **Source:** P4 — "As capital follows, market becomes adversarial to your logic"
- **Experiment:**
  1. Implement rolling Sharpe, win rate, capacity monitoring
  2. Set retirement thresholds (Sharpe < 0.5 for 60 days)
  3. Backtest with/without retirement logic
- **Metric:** ΔMDD ≤ -15%, ΔTotalReturn ≥ -5%
- **Priority:** 🟡 Medium
- **Status:** ⏳ Backlog #16
- **Dependencies:** Need trade-level performance tracking

---

## H12: Pairs Trading Improves Diversification

- **Statement:** Adding cointegration-based pairs trading (8th pattern category) improves portfolio Sharpe by ≥0.2
- **Source:** P5 — "Cointegration + ML hybrid approaches dominate pure methods"
- **Experiment:**
  1. Implement cointegration pair screening (Engle-Granger, Johansen)
  2. Add ML spread predictor (XGBoost/LSTM)
  3. Backtest as overlay to existing 34 patterns
- **Metric:** ΔSharpe ≥ 0.2, ΔCorr(portfolio) ≤ -0.1
- **Priority:** 🟡 Medium
- **Status:** ⏳ Backlog #17
- **Dependencies:** Need cointegration testing infrastructure

---

## H13: Inefficient Instruments Yield Higher ML Alpha

- **Statement:** ML models trained on low-efficiency instruments (small-caps, low analyst coverage, high Amihud illiquidity) achieve ≥0.3 higher Sharpe than models trained on high-efficiency instruments (SPY, mega-caps), controlling for feature set and model architecture.
- **Source:** DeMiguel et al. (2024), Bartram & Grinblatt (2019), Damodaran market efficiency propositions. Related: I14.1, I16.1, I16.2.
- **Experiment:**
  1. Select 10 "efficient" instruments (SPY, QQQ, AAPL, MSFT, NVDA, GOOGL, AMZN, META, EURUSD=X, BTC-USD) and 10 "inefficient" instruments (mid-caps with <5 analysts, emerging market ETFs, niche sector ETFs)
  2. Train identical ML model (e.g., XGBoost with same feature set) on each instrument separately
  3. Compare OOS Sharpe, profit factor, and prediction R² across the two groups
  4. Statistical test: Mann-Whitney U on group-wise Sharpe distributions
- **Metric:** ΔSharpe(efficient vs. inefficient) ≥ 0.3, p < 0.05
- **Priority:** 🔴 High
- **Status:** 🔬 Research phase — Phase 1 (2026-05-09) FAILED: severe overfitting. Artifacts deleted. See `ml_training_plan.md` Phase 0 for fix plan.
- **Dependencies:** Need to fix overfitting first (IC filtering + regularization + better labels + PurgedKFold)

## H14: Cross-Asset Features Improve SPY Predictability

- **Statement:** Adding cross-asset features (TLT returns, GLD returns, VIX levels, USO returns) to SPY prediction model improves OOS R² by ≥50% and Sharpe by ≥0.2 vs. using SPY-only features.
- **Source:** Cross-asset alpha engine research, TradeFM (2025) multi-asset training, Regime Aware Cross Asset Alpha (The Data Guy, 2026). Related: I16.3.
- **Experiment:**
  1. Train SPY prediction model with SPY-only features (price, volume, technical indicators)
  2. Train same model with SPY features + cross-asset features (TLT, GLD, VIX, USO, EURUSD)
  3. Compare OOS R² and trading strategy Sharpe on SPY
  4. Ablate individual cross-asset features to identify which contribute most
- **Metric:** ΔR² ≥ 50% relative improvement, ΔSharpe ≥ 0.2, p < 0.05 on Diebold-Mariano test
- **Priority:** 🟠 High
- **Status:** 🔬 Research phase
- **Dependencies:** Cross-asset feature pipeline exists in `src/ml/cross_asset_features.py` — needs empirical validation

---

## Hypothesis Priority Matrix

| Hypothesis | Impact | Feasibility | Dependencies | Priority Score |
|------------|--------|-------------|--------------|----------------|
| H1 | 🔴 High | 🟡 Medium | Event taxonomy | 8/10 |
| H3 | 🟠 High | 🟡 Medium | Failure-set taxonomy | 7/10 |
| H5 | 🟠 High | 🟢 High | None | 9/10 |
| H9 | 🟡 Medium | 🟡 Medium | KL divergence impl | 5/10 |
| H11 | 🟡 Medium | 🟢 High | Performance tracking | 6/10 |
| H12 | 🟡 Medium | 🟡 Medium | Cointegration infra | 5/10 |
| H13 | 🔴 High | 🟡 Medium | Data pipeline expansion | 8/10 |
| H14 | 🟠 High | 🟢 High | Cross-asset features exist | 8/10 |

---

## Testing Queue

**Ready Now (no dependencies):**
- H5: Crash factor filter — implement logistic model from P13
- H7: Friction scoring validation — already implemented, just need backtest
- H8: Regime-adaptive strategies — already implemented, validate on real data

**Blocked (need prerequisites):**
- H1: Event-type weighting ← needs event taxonomy
- H3: Failure-set analyzers ← needs failure-set taxonomy
- H9: Divergence-in-bits ← needs KL divergence estimator
- H10: Winner-fraction sizing ← needs historical win counting
- H11: Strategy decay ← needs rolling performance tracker
- H12: Pairs trading ← needs cointegration infrastructure

**New (from market efficiency research):**
- H13: Inefficient instruments yield higher ML alpha ← needs instrument expansion
- H14: Cross-asset features improve SPY predictability ← ready (pipeline exists)

---

*Backlog generated from research_synthesis_report.md and SENTIMENT_ANALYSIS_SUMMARY.md. Priorities based on stated impact in synthesis report.*
