# Post-Retrain Next Steps Plan

**Created:** 2026-05-14 13:17
**Last Updated:** 2026-05-17 — All 15 items complete. 5/15 gates PASS, 10/15 FAIL.
**Status:** ✅ CONCLUDED — 15/15 done
**Context:** Model retrained through 2026-05-14 with all B9-B14 fixes (Stability Selection + CPCV + WF + PBO/DSR + Hold-Out). Sharpe 1.01 on 2020-2026. 1 remaining health trigger (correlation flips). DSR gate failed (0.95 < 1.0).

---

## Priority Tiers

| Tier | Definition | Items |
|------|-----------|-------|
| **P0 — Blocking** | Must answer before any further work | 1 |
| **P1 — High** | Directly improves model quality or confidence | 4 |
| **P2 — Medium** | Architectural/engineering improvements | 5 |
| **P3 — Exploratory** | New signal sources or research directions | 5 |

---

## Track 1: Validation & Signal Quality (P0-P1)

### P0-1: 2022 Bear Market Backtest

**Question:** Does the strategy protect capital when it matters?
**Why first:** The strategy's only value prop over B&H is drawdown protection. SPY did -18.2% in 2022. If the model can't outperform that, the entire approach is invalid regardless of pipeline quality.

```bash
uv run scripts/run_ml_backtest.py SPY \
    --model models/pattern_classifier_v3_SPY_20260514_124612.pkl \
    --entry-threshold 0.45 --trail-stop \
    --start 2022-01-01 --end 2022-12-31
```

**Success:** Sharpe > 0 AND return > -18% AND max drawdown < SPY max drawdown.
**Tools:** Existing `scripts/run_ml_backtest.py`, `backtesting.py`.
**Evidence:** No external tools needed — pure backtest on existing infrastructure.
**Effort:** 2 min. Gates: all subsequent work.

### P1-1: Probability Calibration Audit (Retrained Model)

**Question:** Are the model's probability estimates honest on fresh data?
**Why:** The stale model was overconfident (ECE 0.128 IS, 0.178 OOS). Retraining may or may not fix calibration. If P=0.45 still means P(win)=0.33, the model is lying to us.
**Approach:** Run `scripts/model_calibration.py` on retrained model, generate reliability diagram, compute Brier score + ECE on hold-out (2025-2026). If still overconfident, apply Platt scaling or isotonic regression via `sklearn.calibration.CalibratedClassifierCV` (in-base, no install needed).

```bash
uv run scripts/model_calibration.py SPY \
    --model models/pattern_classifier_v3_SPY_20260514_124612.pkl \
    --start 2015-01-01 --end 2026-05-14
```

**Tools:** `sklearn.calibration.CalibratedClassifierCV` (sigmoid/isotonic methods), `sklearn.calibration.calibration_curve`, Brier score. All in scikit-learn in-base.
**Metric:** ECE (Expected Calibration Error), reliability diagram slope.
**Success:** ECE < 0.10 on hold-out. P=0.45 wins 40-50% (not 33%).
**Effort:** 10 min. Depends on: nothing.

### P1-2: DSR Investigation — Signal vs Sample Size

**Question:** Is DSR=0.946 < 1.0 because the signal is weak or because we have too few trades?
**Approach:** DSR penalizes low sample sizes. 19 trades over 6 years (3.2/year) is below the threshold where DSR is reliable (<30 trades). Run a Monte Carlo simulation: generate synthetic returns with Sharpe=1.01 at 19 trades and see what DSR the same test produces. If DSR is low even for known-good signals at this sample size, the metric is misleading — not the strategy.

```python
# Synthetic test: generate N=19 returns with target Sharpe=1.01,
# compute DSR, repeat 1000x. Check what % pass DSR > 1.0.
```

**Tools:** `src/analysis/deflated_sharpe.py` (already implemented). Scipy/numpy for MC simulation.
**Evidence:** The original DSR paper (Bailey & López de Prado, 2014) notes DSR requires sufficient track record. PSR/MinTRL already implemented in `deflated_sharpe.py`.
**Success:** Understanding of whether DSR failure is real or a sample-size artifact.
**Effort:** 30 min. Depends on: nothing.

### P1-3: Simpler-Than-ML Baseline Comparison

**Question:** Does the CatBoost model beat dead-simple technical rules?
**Why:** If a 10-line RSI(14)<30 buy / RSI>70 sell or 50/200 MA crossover beats our 38-feature CatBoost model, the complexity isn't earning its keep. This is the "do nothing" baseline every ML strategy needs.

Run these on SPY 2020-2026 with the same backtesting.py engine:
1. RSI(14) < 30 buy, RSI(14) > 70 sell
2. 50/200 MA golden cross buy, death cross sell
3. Buy & hold (already have: 144.15%)
4. Equal-weight of RSI + MA signals

**Tools:** `backtesting.py` (installed). RSI and SMA built into the library.
**Evidence:** Standard practice in quant finance — see Pardo (2008) "Design, Testing and Optimization of Trading Systems".
**Success:** ML model Sharpe > best technical rule Sharpe by >0.15.
**Effort:** 20 min. Depends on: nothing.

### P1-4: Regime-Conditional Return Decomposition

**Question:** The pipeline reports "STRONG" IC in all regimes (0.10-0.40, hit rates 60-73%). But what are the actual returns per regime?
**Why:** A metric can have good IC but still lose money if it's long-only in a bear regime. The health dashboard already classifies bars into Trending/Ranging/Transition and Bull/Bear. Decompose returns by regime.

```bash
# Run MLStrategy backtest with regime logging enabled
# Then aggregate: return, Sharpe, win rate per regime label
```

**Tools:** Regime classification already in `src/ml/regime_detector.py` and `src/ml/r2_rd_regime.py`. Classification results logged during pipeline training.
**Evidence:** Regime-switching factor investing literature (MDPI 2020 paper on HMM-based regime rotation).
**Success:** Identify which regimes the model makes/loses money in. If it loses in bear markets, the capital-protection thesis is falsified.
**Effort:** 30 min. Depends on: P0-1 (2022 bear backtest).

---

## Track 2: Feature & Data Quality (P1-P2)

### P2-1: Identify & Analyze 3 Flipped Correlation Features

**Question:** Which 3 of 38 features flipped correlation in 2025-2026, and are they structural or noise?
**Why:** The retrain trigger is active because 3 features have opposite correlation with labels in the monitoring window. We need to know if these are real structural breaks (should drop the features) or statistical noise from a 339-bar small window (false alarm).

```bash
uv run python -c "
from scripts.model_health import ModelHealthMonitor
m = ModelHealthMonitor('SPY', 'models/pattern_classifier_v3_SPY_20260514_124612.pkl')
m.load_model()
flipped = m.get_flipped_features(monitor_start='2025-01-01')
for f in flipped:
    print(f'{f.name}: IS_corr={f.is_corr:.3f}, OOS_corr={f.oos_corr:.3f}, p={f.p_value:.4f}')
"
```

**Tools:** Already implemented in `scripts/model_health.py` — just need to expose the flipped feature details.
**Evidence:** KS test already in project. Bootstrap confidence intervals for correlation difference (Fisher z-transform).
**Success:** Actionable list: drop feature X, keep feature Y (noise), investigate feature Z.
**Effort:** 20 min. Depends on: nothing.

### P2-2: Feature Importance Ranking Comparison

**Question:** Do stability-selected features match SHAP-important features?
**Why:** The pipeline log shows different rankings: stability top = `vol_percentile_60d`, `trend_strength`, `rel_ret_TLT_5d` vs SHAP top = `macd_signal`, `vol_regime`, `dist_to_high_10`. If stability and SHAP disagree, the signal is fragile and depends on which method you trust.

**Approach:** Compute rank correlation (Spearman) between stability scores and SHAP importance. If ρ < 0.3, investigate why — possible causes: multi-collinearity, non-linear interactions, regime-dependence.

**Tools:** SHAP already integrated (`src/ml/shap_utils.py`). Stability scores from `src/ml/stability_selector.py`. Scipy `spearmanr` for correlation.
**Evidence:** Top-k feature importance ranking research (arXiv 2509.15420, 2025) — bootstrap-based stability identifies different features than SHAP in non-linear models. SHAP + stability comparison paper from NCI RL (2025).
**Success:** Clear understanding of whether the model's signal is robust to importance method.
**Effort:** 30 min. Depends on: nothing.

---

## Track 3: Architecture & Production (P2-P3)

### P2-3: Dynamic Ensemble Collapse Investigation

**Question:** Why did 5 CPCV path models EGD-weighted produce Sharpe 0.26 — worse than any individual component?
**Why:** Pipeline results: individual path test AUC ranged 0.57-0.70. The 0.70 path (path 10, gap 0.10) should dominate. Yet the ensemble underperforms. Possible causes: (1) EGD overweighting wrong paths, (2) regime detector misclassifying, (3) ensemble averaging washing out the sparse signals, (4) bug in weight application.

**Approach:**
1. Log per-path weights over time — check if EGD converges or oscillates
2. Run each path model solo on 2020-2026 backtest — compare to ensemble
3. Try simple equal-weight voting instead of EGD
4. Check if ensemble produces more or fewer signals than individual paths

**Tools:** `src/ml/dynamic_ensemble.py` (EGD optimizer). Alternative: scikit-learn `VotingClassifier` (soft voting) or `StackingClassifier` for comparison.
**Evidence:** EGD literature (Kivinen & Warmuth, 1997). Scikit-learn ensemble docs — stacking often outperforms online learning for small ensembles.
**Success:** Either fix the ensemble to beat single-model Sharpe, or confirm ensemble approach is fundamentally wrong for sparse signals.
**Effort:** 1 hr. Depends on: nothing.

### P2-4: Walk-Forward Retraining Cadence Experiment

**Question:** What happens if we retrain every 6 months instead of once? Does it help or hurt?
**Why:** The model was trained once on 2015-2026. In production, we'd periodically retrain. Does fresh training improve performance, or does the model need long history? This answers "how often should we retrain in production?"

**Approach:** Simulate expanding-window retraining at 6mo, 12mo, and 24mo intervals on 2015-2026. Compare out-of-sample performance at each retrain point. Uses walk-forward infrastructure already in pipeline.

**Tools:** Walk-forward validation already in `src/ml/walk_forward.py` and pipeline v3. Just parameterize the retrain interval.
**Evidence:** Walk-forward optimization literature (Pardo 1992, 2008). "Walk Forward Analysis" is the gold standard for strategy validation. See Wikipedia and QuantStart references.
**Success:** Find optimal retrain cadence (likely 12mo given 19 trades/yr).
**Effort:** 45 min. Depends on: nothing.

### P2-5: Entry Threshold Sensitivity Sweep

**Question:** Is entry_threshold=0.45 still optimal for the retrained model, or is there a better sweet spot?
**Why:** The retrained model has different probability distributions than the old one. The old model's best was 0.45 with 66 trades. The retrained model does 19 trades at 0.45 — fewer signals but potentially higher quality. Sweep 0.35/0.40/0.45/0.50/0.55 on 2020-2026.

**Tools:** Existing `scripts/run_ml_backtest.py` with `--entry-threshold` flag. `scripts/sweep_entry_thresholds.py` (already implemented).
**Evidence:** Standard hyperparameter sensitivity in trading (see Pardo 2008).
**Success:** Confirmation that 0.45 is optimal, or find a better threshold.
**Effort:** 10 min. Depends on: nothing.

### P3-1: Forward Paper-Trading Harness

**Question:** Can we accumulate truly unseen OOS evidence day-by-day without deploying capital?
**Why:** The model was trained through 2026-05-14. Every day from 2026-05-15 onward is live unseen data. A daily signal generation script that logs predictions and hypothetical trades creates a growing paper-trading track record.

**Approach:** Create `scripts/paper_trade_daily.py`: loads model, fetches latest bar, recomputes features, generates signal. Logs to `reports/paper_trading/daily/`. Runs via cron/scheduled task. No trade execution.

**Tools:** `yfinance` for daily data fetch. Existing feature engineering pipeline (`src/ml/feature_engineering.py`). Existing model loading code.
**Evidence:** Standard production ML practice — see "AI Model Drift & Retraining Guide" (SmartDev 2025), "Building Production-Grade ML Retraining Workflows".
**Effort:** 1 hr. Depends on: P0-1, P1-1 passing.

### P3-2: Minimum Capital & Kelly Position Sizing

**Question:** How much capital do we need, and what position size maximizes long-term growth?
**Why:** 2% risk per trade × 3 trades/yr = ~6% expected return. On $100k, that's $6k/yr — barely above risk-free rate. What's the Kelly-optimal allocation to this strategy within a broader portfolio?

**Approach:**
1. Compute Kelly fraction f* = (p × b - q) / b where p=win_rate, b=avg_win/avg_loss
2. Compare full-Kelly, half-Kelly, quarter-Kelly on backtest
3. Estimate minimum capital for meaningful absolute returns

**Tools:** Pure math — no install needed. Reference: Kelly Criterion (Kelly 1956), "Fortune's Formula" (Poundstone), Zerodha Varsity implementation, Frontiers 2020 paper on practical Kelly implementation.
**Evidence:** Half-Kelly is standard risk management practice. Kelly weight = Expected Return / Variance for stocks (Wikipedia).
**Success:** Capital allocation recommendation: "invest X% of portfolio in this strategy, expect Y% contribution to total return."
**Effort:** 30 min. Depends on: P1-1 (calibrated win probabilities).

---

## Track 4: New Signal Sources (P3)

### P3-3: Survival Analysis for Time-to-Exit Prediction

**Question:** Can we predict *when* a trade will hit TP or SL, not just *whether*?
**Why:** Current triple-barrier labeling is binary (hit TP before SL in 5 days?). A survival model predicts the S-shaped curve of "probability of still being in the trade after t days" — providing time-varying exit probabilities, not just a binary signal.

**Approach:** Use `scikit-survival` (already installed, skill available):
- Label: time until TP or SL hit (censored if horizon expires)
- Features: same 38 stability-selected features
- Models: Cox Proportional Hazards, Random Survival Forest, Gradient Boosting Survival
- Evaluate: Concordance Index (C-index), time-dependent AUC, integrated Brier score (IBS)

**Tools:** `scikit-survival` library (`sksurv.linear_model.CoxPHSurvivalAnalysis`, `sksurv.ensemble.RandomSurvivalForest`, `sksurv.ensemble.GradientBoostingSurvivalAnalysis`). Metrics: `sksurv.metrics.concordance_index_censored`, `cumulative_dynamic_auc`, `integrated_brier_score`. Already in project — see skill at `.kilo/skills/scikit-survival/`.
**Evidence:** Extensive literature — Cox (1972) proportional hazards is the classic approach. Random Survival Forest (Ishwaran 2008) handles non-linear effects. scikit-survival (Pölsterl 2020, JMLR) provides scikit-learn compatible API.
**Success:** C-index > 0.55 (better than random). Survival curves that differentiate high-probability trades from low-probability ones.
**Effort:** 2 hr. Depends on: nothing (new label type, same features).

### P3-4: Regression Labels — Predict Forward Return Directly

**Question:** Instead of "will it hit TP?", predict the actual 5-day forward return.
**Why:** Binary classification throws away information — a signal barely missing TP by 0.1% is labeled 0, same as a signal that loses 10%. Regression preserves magnitude and produces a continuous ranking that may produce better signals.

**Approach:**
- Label: 5-day forward return (continuous, normalized by ATR)
- Model: CatBoostRegressor with same 38 features
- Evaluate: Rank IC, hit rate (direction accuracy), R²
- Compare backtest: classification-based signals vs regression-based signals

**Tools:** Existing `src/ml/signal_scorer.py` already has `SignalRegressor` class. CatBoost regressor in-base. Same feature pipeline.
**Evidence:** Common practice in quant ML — rank IC is the standard metric for regression-based signals. See Qlib, WorldQuant, etc.
**Success:** Rank IC > 0.05 on hold-out. If backtest Sharpe > classification-based, regression is the better approach.
**Effort:** 1 hr. Depends on: nothing.

### P3-5: Hidden Markov Model Regime Detection

**Question:** Can HMM-based regime detection improve signal timing compared to the current rule-based regime detector?
**Why:** Current regime detector uses ADX/ATR thresholds. HMM detects latent regimes from return/volatility patterns — may capture subtle transitions the rule-based approach misses. Evidence: Renaissance Technologies reportedly uses HMMs. Multiple papers show HMM-based factor rotation improves returns.

**Approach:**
1. Extract daily returns + volatility for SPY
2. Fit Gaussian HMM with n=3 states (via `hmmlearn`)
3. Label states: high-return/low-vol = bull, low-return/high-vol = bear, intermediate = transition
4. Run backtest with regime-conditional entry (only trade in favorable regimes)
5. Compare to current rule-based regime gate

**Tools:** `hmmlearn` (`pip install hmmlearn`, or `uv add hmmlearn`). Pure Python, no GPU needed. Alternative: `statsmodels.tsa.MarkovRegression` for Markov-switching models.
**Evidence:** "Market Regime Detection using Hidden Markov Models in QSTrader" (QuantStart), "Regime-Switching Factor Investing with Hidden Markov Models" (MDPI 2020), "Use Markov models to detect regime changes" (PyQuant News). HMM-based regime rotation improved returns over static factor models in MDPI paper.
**Success:** HMM-gated backtest Sharpe > current rule-based gate Sharpe, OR HMM identifies meaningful regime boundaries the rule-based approach misses.
**Effort:** 2 hr. Depends on: P1-4 (regime-conditional returns).

---

## Implementation Order

```
Phase 12a: Quick Validation (execute first, ~1 hr total)
  ├── P0-1: 2022 Bear Market Backtest (2 min) — gates all subsequent work
  ├── P1-1: Probability Calibration Audit (10 min)
  ├── P1-3: Simpler-Than-ML Baseline (20 min)
  └── P2-5: Entry Threshold Sweep (10 min)

Phase 12b: Signal Understanding (~2 hr)
  ├── P1-2: DSR Investigation (30 min)
  ├── P1-4: Regime-Conditional Returns (30 min)
  ├── P2-1: Flipped Feature Analysis (20 min)
  └── P2-2: Feature Importance Comparison (30 min)

Phase 12c: Architecture Improvements (~3 hr)
  ├── P2-3: Dynamic Ensemble Fix (1 hr)
  ├── P2-4: Walk-Forward Cadence (45 min)
  ├── P3-1: Paper-Trading Harness (1 hr) — gate on Phase 12a passing
  └── P3-2: Kelly Position Sizing (30 min)

Phase 12d: New Signal Sources (~5 hr, parallelizable)
  ├── P3-3: Survival Analysis (2 hr)
  ├── P3-4: Regression Labels (1 hr)
  └── P3-5: HMM Regime Detection (2 hr)
```

---

## Tools & Dependencies Summary

| Tool | Install | Use | Items |
|------|---------|-----|-------|
| scikit-learn in-base | None | Calibration, voting, stacking | P1-1, P2-3 |
| scipy in-base | None | Spearman correlation, KS test | P1-2, P2-1, P2-2 |
| backtesting.py in-base | None | Backtest engine | P0-1, P1-3, P1-4 |
| SHAP in-base | None | Feature importance | P2-2 |
| scikit-survival in-project | Installed | Cox, RSF, GBSA models | P3-3 |
| hmmlearn | `uv add hmmlearn` | Gaussian HMM regime detection | P3-5 |
| statsmodels in-base | None | Markov switching (alternative to hmmlearn) | P3-5 |
| CatBoost in-base | None | Regression, classification | P3-4, all backtests |
| yfinance in-base | None | Daily data fetch for paper trading | P3-1 |

---

## Success Gates

| Gate | Test | Pass if | Actual | Status |
|------|------|---------|--------|--------|
| G1: Bear market | 2022 backtest | Strategy beats SPY -18% and has positive Sharpe | Return -8.68% (beats SPY -11.27%), Sharpe -1.5, DD -11.28% | **FAIL** (Sharpe < 0) |
| G2: Calibration | Reliability diagram | ECE < 0.10 on hold-out | ECE OOS = 0.197, P=0.45 wins 29.7% | **FAIL** (overconfident) |
| G3: Baseline | ML vs RSI/MA | ML Sharpe > technical +0.15 | ML 0.60 < MA 0.70 (1 trade), ML 0.60 > RSI 0.41 (6 trades) | **FAIL** (vs MA, but MA is not comparable) |
| G4: DSR | MC simulation | DSR failure is sample-size artifact, not signal weakness | — | Pending |
