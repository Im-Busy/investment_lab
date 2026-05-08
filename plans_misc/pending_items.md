# Pending Implementation Items

**Created:** 2026-04-26
**Updated:** 2026-04-30 (ML Phase B complete)
**Purpose:** Centralized tracking of all pending work items sourced from plan files.
**Instructions for Future Sessions:**
1. When completing an item, delete it from this file AND update the status in the original plan file
2. Add new items here when creating new plans
3. Keep this file in priority order (P0 = critical, P1 = high, P2 = medium, P3 = low)

---

## P2 — Medium Priority

### 0. Auto-Research Tools Evaluation & Integration
- **Source:** `plans/auto_research_tools_evaluation_round2.md`
- **Status:** Evaluation complete, repos cloned, skills installed, architecture analysis complete
- **Completed:**
  - All 9 repos evaluated (KEEP/READ/SKIP decisions made)
  - Read-only repos cloned: DeepScientist, Idea2Paper, DeepResearchAgent, MLE-Agent
  - OpenScholar cloned (reversed previous skip decision)
  - 17 new skills installed to .kilo/skills/
  - Architecture analysis document created at `useful_resources/useful_repos/ARCHITECTURE_ANALYSIS.md`
  - Key patterns identified: Durable state + iterative refinement + versioned resources
- **To Do Later (requires Linux/Docker — unavailable on current Windows environment):**
  - ~~Set up RD-Agent with Qlib for quant factor evolution~~ — **BLOCKED:** RD-Agent requires Docker + Linux runtime. Defer until WSL2 or Linux environment available.
  - ~~Set up OpenScholar model + retrieval data~~ — **BLOCKED:** Requires Linux environment + 8B model GPU setup. Defer until appropriate hardware/environment available.

### 1. ML Phase A: Code Foundation (Infrastructure)
- **Source:** `plans/ml_reset_plan.md` (Phase A)
- **Status:** COMPLETE — All 6 components verified with passing tests
- **Completed:**
  - A1: `src/ml/experiment_logger.py` — 17 tests passing
  - A2: `src/ml/purged_cv.py` — 19 tests passing
  - A3: `src/ml/feature_store.py` — 16 tests passing
  - A4: `src/ml/metrics.py` — 23 tests passing (3 tests fixed: annualization, empty data, filter threshold)
  - A5: `src/ml/registry.py` — tests passing
  - A6: `src/ml/backtest_bridge.py` — tests passing

### 2. ML Phase B: Enhancement
- **Source:** `plans/ml_reset_plan.md` (Phase B)
- **Status:** COMPLETE (2026-04-30)
- **Completed:**
  - B1: `src/ml/features.py` — FeatureEngineer with IC-based selection, forward returns labels, 53+ alpha factors ✅
  - B2: `src/ml/regime_model.py` — RegimeClassifier upgraded: PurgedKFold from Phase A, SHAP, permutation importance, IC evaluation, ExperimentLogger integration ✅
  - B3: `src/ml/signal_scorer.py` — SignalRegressor (regression-based, predicts forward returns via Rank IC) + legacy SignalScorer preserved ✅
  - B4: `src/ml/feature_selector.py` — SFISelector (Sequential Feature Importance) + FeatureSelector with auto-detection (MI classification vs regression) ✅
  - B5: `src/ml/cnn_regime.py` — Regime1DCNN + CNNRegimeDetector with EarlyStopping, focal loss, PyTorch ✅
  - B6: `src/ml/risk_factors.py` — RiskFactorAutoencoder for unsupervised latent risk factor discovery (3-5 dims) ✅
  - B7: Full ML-Enhanced Backtest (DONE - scripts fixed and verified)

### 3. Phase 2: High-Performance Numba Layer
- **Source:** `plans/phase2_implementation_plan.md`
- **Status:** Not Started
- **Tasks:** Add numba, create Numba-optimized pivots/technical indicators, vectorized detection

### 4. Phase 3: VectorBT Migration
- **Source:** `plans/phase3_implementation_plan.md`
- **Status:** Not Started (depends on Phase 2)
- **Tasks:** vectorbt signal generator, multi-pattern, portfolio runner, benchmarks

### 5. Contribution Analysis System
- **Source:** `plans/contribution_analysis_plan.md`
- **Status:** Files exist (`src/analysis/`), verify completeness

### 6. Pair Trading Backtest
- **Source:** `plans/phased_implementation_plan.md` (Phase 2)
- **Status:** Infrastructure ready, backtest validation pending on GLD/IAU and SPY/QQQ

### 7. Backtesting Framework Migration — Remaining
- **Source:** `plans/backtesting_framework_migration_plan.md`
- **Status:** Adapter created, B7 scripts run. Remaining: notebook updates, optimization examples, deprecation warnings

---

## P3 — Low Priority / Optional

### 8. Phase 7: Paper Trading & Live Readiness
- **Source:** `plans/phased_implementation_plan.md` (Phase 7)
- **Status:** Not Started

### 9. London Breakout Strategy (FR-001)
- **Source:** `docs/implementation_plan_fr001.md`
- **Status:** Not Implemented

### 10. Multi-Pattern Backtest Optimization — Profiling
- **Source:** `plans/multipattern_backtest_optimization_plan.md` (Phase 4)
- **Status:** Not Started

---

## Items Completed (2026-04-26 Session)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | ML-Enhanced Backtest B7 (P0-1) | DONE | Fixed `phase_b7_ml_backtest.py` (logging import, bar index tracker), `phase_b7_custom_backtest.py` (PatternResult import). Both run on SPY 2020-2024. |
| 2 | Multi-Pattern Entry Logic (P0-2) | DONE | Verified `next()`, `_get_atr()`, signal logging in `multi_pattern_strategy.py`. |
| 3 | SMC Trade Management (P1-3) | DONE | `_manage_active_trade()` at `smc_reversal.py:540` — BE@1R, scale@2R, close@2.5R. |
| 4 | Signal Decay (P1-4) | DONE | `SIGNAL_VALIDITY`, `DECAY_RATE`, `apply_signal_decay()` in `confluence.py`. |
| 5 | Portfolio Heat Tracking (P1-5) | DONE | `open_position_risks`, `check_portfolio_heat()` (4% warn, 6% halt) in `daily_limits.py`. |
| 6 | Pattern Correlation Limits (P1-6) | DONE | `CORRELATION_GROUPS`, `_filter_correlated_patterns()` in `confluence.py`. |
| 7 | Research R1-R5 (P1-7) | DONE | TurnoverPenalty, PositionRiskModel, CircuitBreaker, Regime Declaration, DynamicRebalancer — all integrated in `engine.py`. |
| 8 | Research R6-R10 (P1-8) | DONE | R6 EventWeighting, R7 DiversityScore, R9 FailureSetAnalyzer, R10 FrictionScorer already existed. R8 EpistemicAutopsy created at `src/backtest/epistemic_autopsy.py`. R10 FrictionScorer wired into `engine.py`. |
| 9 | Pattern Selection Pipeline (P1-9) | DONE | 7 modules: statistical_filter, correlation_analyzer, contribution_analyzer, walk_forward_validator, signal_quality_filter, pattern_selector orchestrator, `__init__.py`. All imports verified. |
| 10 | Framework Migration Validation (P1-10) | DONE | Both B7 scripts run. Results saved to `reports/ml_backtest_comparison/`. |

## Items Completed (2026-04-30 Session)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 11 | ML Phase B1 — Feature Engineering | DONE | FeatureEngineer with IC-based selection, 53+ alpha factors, forward returns labels. Bug fix: FeatureSelector now auto-detects continuous vs categorical targets for MI. |
| 12 | ML Phase B2 — Regime Classification | DONE | RegimeClassifier upgraded: PurgedKFold from Phase A purged_cv.py, SHAP via TreeExplainer, permutation importance, IC-based evaluation, ExperimentLogger integration, compare_models() static method. |
| 13 | ML Phase B3 — Signal Generation (Regression) | DONE | New SignalRegressor class: predicts forward returns (continuous), Spearman rank IC as metric, PurgedKFold CV, walk-forward with IC tracking, model comparison. Legacy SignalScorer preserved. |
| 14 | ML Phase B4 — Feature Selection (SFI) | DONE | SFISelector already implemented correctly. FeatureSelector fixed: auto-detects label type for mutual_info_classif vs mutual_info_regression. |
| 15 | ML Phase B5 — CNN Regime Detection | DONE | New cnn_regime.py: Regime1DCNN (3-block Conv1D→BN→ReLU→Pool→Dropout), EarlyStopping with restore, CNNRegimeDetector wrapper with data preprocessing, comparison methods. |
| 16 | ML Phase B6 — Autoencoder Risk Factors | DONE | New risk_factors.py: RiskFactorAutoencoder (3-layer encoder/decoder, L1 sparsity, Adam), extract_factors(), add_risk_features(), evaluate_ic_improvement(), get_factor_loadings(). |
