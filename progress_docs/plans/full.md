---
project: investment_trying
last_updated: 2026-05-13 (B6-B7 done. B9-B14 planned — 6 overfitting fixes from research. Phase 10b done — 5 loop hardening tasks.)
summary: |
  Rule-based multi-pattern trading system with 34+ chart pattern detectors,
  ML-enhanced regime detection, Numba-accelerated indicators, and event-driven
  backtesting engine. 8 phases + ML robustness phase + tool evaluation phase +
  autonomous loop phase + overfitting fix phase spanning pattern detection through production deployment.
  21 new friend-suggested ML items added from external OHLCV pipeline review.
phases_total: 12
phases_complete: 7
phases_active: 4
phases_deferred: 1
---

# Master Plan

## Phase Status

| # | Phase | Plan | Log | Status |
|---|-------|------|-----|--------|
| 01 | Pattern Detection & Strategies | [plan](01-patterns.md) | [log](../logs/01-patterns.md) | ✅ Complete |
| 02 | Performance Optimization | [plan](02-optimization.md) | [log](../logs/02-optimization.md) | 🔄 Numba done, vectorbt deferred (Windows) |
| 03 | Regime Detection & Adaptation | [plan](03-regime.md) | [log](../logs/03-regime.md) | ✅ Complete |
| 04 | ML Foundation | [plan](04-ml-foundation.md) | [log](../logs/04-ml-foundation.md) | ✅ Complete |
| 05 | ML Advanced | [plan](05-ml-advanced.md) | [log](../logs/05-ml-advanced.md) | ⏸️ Deferred |
| 06 | Research-Based Enhancements | [plan](06-research.md) | [log](../logs/06-research.md) | ✅ Complete |
| 07 | Paper Trading & Live Readiness | [plan](07-paper-trading.md) | [log](../logs/07-paper-trading.md) | ⏸️ Deferred |
| 08 | Contribution & Attribution | [plan](08-attribution.md) | [log](../logs/08-attribution.md) | ✅ Complete |
| 10 | Tool Evaluation Additions | [plan](enhance-tool-evaluation.md) | [log](../logs/10-tool-evaluation.md) | ✅ Complete |
| 10b | Autonomous Loop Hardening | [plan](enhance-loop-hardening.md) | [log](../logs/10b-loop-hardening.md) | ✅ Complete — 5/5 done |
| 11 | Overfitting Fixes (B9-B14) | [plan](fix-overfitting.md) | [log](../logs/11-overfitting-fixes.md) | 🔴 Planned — 12 tasks ready |

## Enhancements

| Name | Plan | Status |
|------|------|--------|
| **Overfitting Fixes** (Stability Selection, Per-Sector Models, CPCV, DEL, Meta-Labeling, Production Guards) | [plan](fix-overfitting.md) | 🔴 Planned — 12 tasks (6 P0, 4 P1, 2 P2) |
| ML Capability Enhancements (GWO, InterpretML, AutoGluon + wider gaps) | [plan](enhance-ml-capabilities.md) | 🔴 Planned — Tier 1 ready |
| External Review Additions (21 friend-suggested items) | [plan](enhance-ml-capabilities.md#friends-pipeline-recommendations--external-review-additions) | 🔴 Planned — 6 Tier 1, 7 Tier 2, 8 Tier 3 |
| Notebook Audit Fixes (F1-F4, ML1-ML4, S1-S4) | [plan](notebook-audit-fixes.md) | 🔄 In Progress — F1 ✅, F2 ✅ |
| Knowledge Graph Insights (48 papers cross-referenced) | [plan](enhance-knowledge-graph.md) | 🔴 Planned — 12 tasks from paper analysis |
| Paper-to-Module Gap Closure (overfitting, sentiment, RL, events) | [plan](enhance-knowledge-graph.md) | 🔴 Planned — 6 modules targeted |
| **Tool Evaluation Additions** (Optuna, PyPortfolioOpt, Bandit, CCXT, FinGPT, aeon) | [plan](enhance-tool-evaluation.md) | 🔴 Planned — 3 HIGH priority ready |

---

## Pending (Ready to Start)

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| P1 | T1a | ARO feature selection engine | ML Foundation (Phase 04) ✅ | ✅ Done — `src/ml/tuning/aro_selector.py`, 7 tests |
| P1 | T1b | GWO CatBoost hyperparameter tuner | ML Foundation (Phase 04) ✅ | ✅ Done — `src/ml/tuning/gwo_tuner.py`, 7 tests, CLI `scripts/tune_model.py` |
| P1 | T1c | GA unsupervised regime optimizer | Phase 03 (Regime) ✅ | ✅ Done — `src/ml/tuning/ga_tuner.py`, 4 tests, CLI |
| P1 | T1d | WOA pattern threshold tuner | Phase 01 (Patterns) ✅ | ✅ Done — `src/ml/tuning/woa_tuner.py`, 4 tests, CLI |
| P1 | T2 | InterpretML EBM regime classifier | ML Foundation (Phase 04) ✅ | ✅ Done — `src/ml/ebm_classifier.py`, 10 tests, glassbox explainability |
| P1 | T0 | Unify classifier defaults to CatBoost | Phase 04 ✅ | ✅ Done — RegimeClassifier + SignalRegressor + PatternClassifier all default to catboost |
| P1 | T3 | AutoGluon AutoML baseline | ML Foundation (Phase 04) ✅ | ✅ Done — `src/ml/automl.py`, 8 tests, CLI `scripts/run_automl.py` |
| P1 | T4 | Advanced backtest validation (DSR, PSR, FDR) | Phase 04 ✅, Phase 08 ✅ | ✅ Done — `src/analysis/deflated_sharpe.py`, 43 tests |
| P1 | QW1 | Pattern confidence scoring | PatternClassifier exists | ✅ Done — `src/ml/pattern_scorer.py`, 6 tests |

| **MR0** | C1 | Paper trade re-run with fixed drawdown (12 tickers) | PatternClassifier V3 model, paper_trade_v3.py ✅ | ✅ Done — `reports/paper_trading/20260511_044647/` |
| **MR0** | C2 | Walk-forward chronological validation (per-ticker) | C1 | ✅ Done — 15 bugs fixed, 5/12 pass (XLK/QQQ/SPY/KODK/GLD) |
| **MR1** | C3 | Basket vs single-ticker ablation study | C2 | ✅ Done — 5-winner model = worse IC everywhere (-0.107→-0.02). 12-all model = +0.048→+0.112 across winners. Adding diverse tickers IMPROVES generalization. GATE OPEN. |
| **MR1** | C4 | Model prediction correlation decomposition | C2 | ✅ Done — SPY/QQQ/XLK r>0.82 (tech cluster, concentrated). KODK r≈0.27, GLD r≈0.30 vs tech (independent). 3 distinct signals, not 1. |
| **MR1** | C6a | Ticker screening pipeline (auto) | C3, C4 passing | ✅ Done — `scripts/screen_tickers.py` + `.kilo/skills/ticker-screener/`. Screened 75 tickers across 8 sectors (v2 relaxed criteria: MC>$200M, vol>12%, inst>25%). 22 passed walk-forward IC > 0.03. See `docs/ticker-test-log.md`. |
| **MR2** | C6 | Expand basket to 30+ tickers | C6a | ✅ Done (2026-05-11). 33-ticker model beats 12-ticker model on all 3 gate criteria: (1) 91% tickers improved IC, (2) max drop 0.012, (3) 20/21 new tickers pass IC>0.03. Mean IC: 0.038→0.067 (+75%). Model: `models/pattern_classifier_v3_SPY_20260511_164601.pkl`. See `experiments/c6_comparison.csv`. |
| **MR2** | C6b | Edge relaxation analysis (what drives usefulness) | C6 | ✅ Done (2026-05-11). Tested 81 tickers against 33-ticker model. Volatility is the binding constraint (r=−0.44, p<0.0001). Market cap irrelevant (r=+0.02). Volume weakly negative (r=−0.31). Sweet spot: 12–25% ann. vol (mean IC 0.056, 73% pass, 0% negative). See `docs/guide-edge-characterization.md`. |
| **MR2** | C5 | ML + pattern detector integration (`ml_strategy.py`) | C6 ✅ | ✅ Done (2026-05-11). `src/strategies/ml_strategy.py` with backtesting.py integration. No-cross-asset model: `models/pattern_classifier_v3_SPY_20260511_224704.pkl` (57 features, CV AUC 0.546, gap −0.002). All 33 tickers tested. Best: WMT +175% (Sharpe 0.76), UNP +146% (0.66), D +134% (0.62). 19/33 profitable. Mean Sharpe 0.15. Results map to vol-based edge characterization: utilities/REITs/staples strong, energy/miners/EM weak. 2933 total trades. See `reports/ml_backtest/comparison.csv`. |

### Phase 6c: Strategy Refinement (post-C5)

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P1** | **C7** | Strategy execution improvements | ✅ DONE | (1) Volatility gate — implemented, no benefit (Sharpe unchanged). (2) Consecutive confirmation — implemented, reduced trades (Sharpe 0.33). (3) **Trailing stop — WINNER**: Sharpe 0.36→0.47 (+31%). (4) Position scaling by conviction — marginal (Sharpe 0.38). Best config: `--trail-stop` alone. All 4 are toggleable in ml_strategy.py. |
| **P1** | **C8** | Pattern detector confluence (PatternBoostFilter) | C7 ✅ | ✅ DONE (2026-05-12). `src/signals/pattern_boost.py` — 35 pattern detectors with reliability weights from NCFE/Duddella research (H&S=0.87, Gartley=0.85, C&H=0.80, etc.). PatternBoostFilter precomputes vectorized signals, computes directional bull/bear boost (0.0–0.10) at each bar. Integrated into MLStrategy via `use_pattern_boost=True`. |
| **P1** | **B1** | OOS validation on 2025-2026 data | C5 ✅, BESTS.md | ✅ Done — **FAILED**: Sharpe -0.27 vs train +0.73. Model overfit. Does not generalize past 2024. |
| **P1** | **B2** | Fix `backtest_ml_enhanced.py` (broken ML comparison baseline) | — | ✅ Done — 3 bugs fixed: (1) Signal gen now uses entry points, (2) replaced local training with pre-trained PatternClassifier V3, (3) backtesting.py handles pos sizing. Ruff clean. |
| P2 | **B3** | Model probability calibration audit | B1 | ✅ Done (2026-05-13) — Rewrote `scripts/model_calibration.py` with triple-barrier labels. IS ECE=0.128, OOS ECE=0.197. P=0.45 wins 39.5% IS, 33.1% OOS. Model overconfident. Degraded OOS — regime shift, not probability drift. |
| P1 | **B4** | Regime shift root cause investigation | B1 | ✅ Done (2026-05-13) — `scripts/investigate_regime_shift.py`. KS tests: raw ATR doubled (3.97→8.06, KS=0.62 p=10^-109). 18/57 shifted, 4 flipped, 13 weakened. Fix: normalized ATR=ATR/Close in `feature_engineering.py:173-174`. |
| P1 | **B5** | Walk-forward optimization comparison | B4 | ✅ Done (2026-05-13) — `scripts/backtest_wfo.py`. WFO OOS +1.7% vs single-split -1.3%. WFO total Sharpe 1.76. |
| — | **B8** | Agent-centric workflow infrastructure | B3-B5 | ✅ Done (2026-05-13) — `.kilo/agent/{model-doctor,backtest-runner,ml-trainer}.md` + `.kilo/command/{model-diagnose,backtest,train-ml}.md`. AGENTS.md updated. |
| **P0** | **B6** | Retrain basket model with normalized ATR | B5, ATR fix | ✅ Done (2026-05-13) — 2/5 pass. SPY-only model: CV AUC=0.595, WF IC=0.182, OOS Sharpe -0.44 (worse). Basket model: ARO collapsed to 5 features (2 cross-asset → 0 trades). Root cause not fixed — deferred to B9-B14. |
| **P0** | **B7** | Evaluate Qlib concept-drift models (ADARNN/ADD) | B6 | ✅ Done (2026-05-13) — CatBoost trained but 0 positions (weak signal). ADARNN/ADD fail: Alpha360-only architecture incompatible with Alpha158. Qlib IC NaN for single-stock. |
| **P2** | **C9** | Honest walk-forward paper trading | B9 ✅ | Re-run `scripts/paper_trade_v3.py` with stability-selected + per-sector model. Avoids backtesting.py precomputation — recomputes features on expanding windows only. |
| **P2** | **C10** | Portfolio-level backtest | B10 ✅ | Equal-weight portfolio of profitable tickers, monthly rebalancing. Threshold: Sharpe > 0.5, DD < 15%, 50+ total trades. |

### Phase 11: Overfitting Fixes (B9-B14) — NEW (2026-05-13)

**Source:** Web research synthesis of 32 papers and production resources (2026-05-13). Six evidence-backed fixes targeting the three root causes of model overfit.

See [full plan](fix-overfitting.md) for detailed task breakdown, implementation steps, and references.

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P0** | **B9** | Stability Selection (replace ARO) | — | Replace single-run ARO with bootstrapped Stability Selection (Meinshausen & Bühlmann 2010). Runs CatBoost feature importance on N=100 bootstrap samples, keeps features with stability_score ≥ 0.6. Target: 5 → 40+ stable features. Library: `scikit-learn-contrib/stability-selection`. |
| **P0** | **B10** | Per-Sector Models (eliminate cross-asset gap) | B9 | Train 7 sector-specific models (Tech, Financials, Energy, Healthcare, Consumer, Industrials, Utilities/REITs). Intra-sector features only — no cross-asset features → no 0-filled gaps at inference. Evidence: "Less is More" paper (QuantPedia 2024) — group-specific models outperform full cross-section models. |
| **P1** | **B11** | Combinatorial Purged CV (replace PurgedKFold) | B10 | Replace PurgedKFold with CPCV (López de Prado). Generates φ(N,k) backtest paths, each testing against different regime sequences. Add Bagged CPCV variant (ensemble across CPCV paths). Evidence: CPCV has lower PBO + higher DSR than both PurgedKFold and Walk-Forward (ScienceDirect 2024). Library: `github.com/hudson-and-thames/mlfinlab`. |
| **P1** | **B12** | Dynamic Ensemble Learning (regime adaptation) | B11 | Train 5 CatBoost variants (different params + training windows). Weight predictions via exponential gradient descent on recent OOS error. Re-initialize weights every K=60 bars (OneNet, NeurIPS 2023). Evidence: No single model dominates across regimes; dynamic ensembles handle non-stationarity (IJIMAI survey, 223 papers). |
| **P2** | **B13** | Meta-Labeling secondary filter | B10 | Train CatBoost classifier on regime/volatility/signal-clustering features to predict: "will this primary signal be profitable?" Trade only when primary AND meta agree. No cross-asset dependency. Evidence: JFDS 2022 paper + full open-source implementation at `github.com/hudson-and-thames/meta-labeling`. |
| **P2** | **B14** | Production hardening (DSR/PBO/walk-forward gates) | B11, B12, B13 | Enforce strict walk-forward invariant at ALL pipeline stages (preprocessing, feature construction, HP tuning, model selection). PBO < 0.3 and DSR > 1.0 as hard gates. Final untouched hold-out (2025-2026). Feature drift monitor (KS test vs training distribution). Evidence: "Spurious Predictability" (arXiv 2604.15531). |

### Phase 6d: PDF Insight Integration

**Source:** `useful_resources/CHART_PATTERN_KNOWLEDGE_BASE.md` (12 parts, consolidated 2026-05-12 from 5 PDFs: Duddella 366pp, Harmonic Guide, NCFE, Fidelity/Kirkpatrick, 151 Trading Strategies). 8 insights extracted.

| Priority | # | Task | Insight Source | Depends On | Notes |
|----------|---|------|---------------|------------|-------|
| **P1** | **C11** | Volume/OI pattern validation layer | Insight #3 — 4 independent sources converge (NCFE, Fidelity, Duddella, Warrior Trading) on volume/OI rules | C10 | Rules: (1) High volume on breakout = confirm (+0.02 bonus already in C8). (2) Declining volume during formation = normal (no penalty). (3) Volume dissipating on Right Shoulder (H&S) = required validation for signal generation. (4) OI declining at Head (H&S) = required validation. Touches: `src/patterns/complex/head_shoulders.py`, `src/patterns/classic/double_top.py`, `src/patterns/classic/ascending_triangle.py`. |
| **P1** | **C12** | Multi-TP exit logic (partial take-profit) | Insight #5 — Universal practice in harmonic trading guides. 2-3 TP levels per trade. | C10 | Add partial TP to MLStrategy: TP1 at 50% of ATR target (exit 50% position), TP2 at 100% target (exit remainder). Move SL to breakeven after TP1 hit. Mechanical improvement — unchanged signals, only exit logic. Estimated +0.05–0.15 Sharpe from volatility drag reduction. |
| **P1** | **C13** | Gap pattern hierarchy + size filter | Insight #6 — Duddella 4-type gap classification (Common/Breakaway/Continuation/Exhaustion) | C10 | Enhance `src/patterns/breakout/gap.py`: (1) Classify gaps by type. (2) Breakaway → trade direction (almost never fills). (3) Exhaustion → fade (reversal). (4) Gap size > 2.5× ATR(10) → skip bar (noise filter). (5) Common gaps → skip entirely (low reliability). |
| **P2** | **C14** | Missing harmonic pattern detectors | Insight #2 — Butterfly, Bat, Crab, Cypher, Shark have real statistical structure (non-random Fibonacci path constraints) | C8 ✅ | Implement 5 missing harmonic detectors: Butterfly (B=78.6% XA, D=1.272 XA), Bat (B<50% XA, D=88.6% XA), Crab (D=1.618 XA, CD=2.24-3.618 AB), Cypher (C beyond X, highest win rate claim), Shark (5-point O-X-A-B-C, entry at C). Follow `src/patterns/harmonic/gartley.py` structure. Fibonacci tables in knowledge base Parts 2, 9. |
| **P2** | **C15** | Pipe pattern detector | Insight #8 — Simplest exact formula: L=max(pipe1,pipe2), T1=±L, T2=±2L | C8 ✅ | Implement `src/patterns/complex/pipe.py`. Two-bar, zero parameters, non-Fibonacci, purely mechanical — easiest to validate statistically. Entry: beyond extreme of both pipes. Stop: opposite extreme. |
| **P2** | **C16** | Dead Cat Bounce ≥15% threshold fix | Insight #4 — Duddella: event-day move ≥15% required, bounce 50-62% retracement, target = 100% gap range | C8 ✅ | Update `src/patterns/classic/dead_cat_bounce.py`: enforce minimum 15% event-day move. Verify bounce retracement to 50-62% range. Target = full gap range from entry. |
| **P3** | **C17** | Empirical pattern reliability calibration | Insight #1 — NCFE empirical stats: H&S 86-88%, Triangle 75-80%, Flag/Pennant high | C10 | Backtest each pattern detector solo on SPY 20y. Measure actual win rate. Replace literature-default `PATTERN_RELIABILITY` weights in `src/signals/pattern_boost.py` with empirically calibrated values. Uses `src/analysis/ablation_engine.py` solo results. |
| — | **V1** | Design validation: 151 Strategies confirms multi-condition approach | Insight #7 — 18 of 151 strategies are TA-based; 3-MA (3.13) and dual-momentum (4.1.2) use filter-on-filter logic | C3 ✅, C4 ✅ | Validates existing MLStrategy design (vol gate + confirm bars + trail stop). No code changes. Document in `docs/guide-pdf-insights.md`. |

### Insight → Implementation Mapping

| Insight | Plan Item | Priority | Rationale |
|---------|-----------|----------|-----------|
| #1 Pattern reliability rankings (empirical) | C17 | P3 | Already wired in C8 via literature defaults. Calibrate empirically later. |
| #2 Harmonic patterns have real structure | C14 | P2 | 5 missing detectors = uncovered alpha. |
| #3 Volume/OI missing validation | C11 | P1 | 4 independent sources converge. Low-hanging fruit for signal quality. |
| #4 Dead Cat Bounce ≥15% threshold | C16 | P2 | Existing detector may lack threshold. Quick fix. |
| #5 Multi-TP exit = free Sharpe boost | C12 | P1 | Mechanical exit improvement, no signal changes. |
| #6 Gap hierarchy underutilized | C13 | P1 | Breakaway vs Exhaustion = opposite trades. Critical classification. |
| #7 151 Strategies validates design | V1 | Validated | No code. Documentation only. |
| #8 Pipe pattern simplest formula | C15 | P2 | Two-bar, zero-parameter = most testable pattern. |

### Phase 10: Tool Evaluation Additions (NEW)

**Source:** Online research evaluation of 18 open-source tools vs project stack (2026-05-12). 12 already covered, 6 represent real gaps. [Full plan](enhance-tool-evaluation.md).

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P1** | **T10a-1** | Install dependencies (Optuna + PyPortfolioOpt + Bandit) | — | `uv add optuna PyPortfolioOpt bandit` |
| **P1** | **T10a-2** | Optuna hyperparameter tuning for CatBoost/LightGBM | T10a-1 | Replace GWO/GA/WOA with Bayesian (TPE) tuning. Create `src/ml/tuning/optuna_tuner.py`. Integrate into `scripts/tune_model.py --algo optuna`. MLflow logging of trials. |
| **P1** | **T10a-3** | Optuna strategy parameter tuning | T10a-1 | Replace manual `scripts/optimize_rsi.py`/`optimize_macd.py` grid searches with Optuna studies. Backtesting.py eval per trial. |
| **P1** | **T10a-4** | PyPortfolioOpt integration | T10a-1 | Replace `scipy.minimize` in `src/optimizer/portfolio_optimizer.py` with PyPortfolioOpt (EfficientFrontier, HRP, CVaR). Integrate with existing `black_litterman.py`. |
| **P1** | **T10a-5** | Test + validate tool additions | T10a-2,3,4 | Unit tests (≥5 each for Optuna tuner + PyPortfolioOpt). Full Optuna study on CatBoost (SPY, 20 trials). HRP vs equal-weight comparison on 33-ticker basket. |
| **P1** | **T10a-6** | Update documentation | T10a-5 | COMMAND_CHEATSHEET.md + `.useful_commands/` + ML_TRAINING_GUIDE.md Section 7. |
| **P2** | **T10b-1** | Add Bandit to pre-commit hooks | T10a-1 | ✅ Done — `.pre-commit-config.yaml`, 13 issues fixed, 0 medium+ |
| **P3** | **T10c-1** | CCXT crypto exchange integration | T10a-1 | Deferred — gate on crypto trading direction. |
| **P3** | **T10c-2** | FinGPT sentiment signal integration | T10a-1 | Deferred — gate on sentiment alpha proven via KG insights. |
| **P3** | **T10c-3** | aeon time-series ML integration | T10a-1 | Deferred — gate on shapelets beating CatBoost AUC by ≥5%. |

### Phase 10b: Autonomous Loop Hardening (NEW — 2026-05-13)

**Source:** Architecture re-evaluation after autonomous loop implementation (5 phases, 1418 lines). Loop wraps `train_ml_pipeline_v3.py`, `run_ml_backtest.py`, `tune_model.py` with independence clustering, consecutive confirmation, cross-group OOS, and Optuna Bayesian sweep. 5 hardening items identified.

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P0** | **LH-1** | Add checkpointing + resume (`--resume`) | Autonomous loop Phase 4 ✅ | ✅ Done — `save_checkpoint()` + `load_checkpoint()` + `--resume` flag. ~60 lines in `autonomous_train_loop.py`. |
| **P0** | **LH-2** | Multi-objective Pareto optimization (`--pareto`) | Autonomous loop Phase 5 ✅ | ✅ Done — `_optuna_sweep()` multi-objective branch. Pareto frontier from `study.best_trials`. ~55 lines. |
| **P1** | **LH-3** | Auto ETF-component cross-asset exclusion | Autonomous loop Phase 1 ✅, `train_ml_pipeline_v3.py` ✅ | ✅ Done — `build_exclusion_pairs()` + `exclusion_pairs.json` + pipeline column drop. ~30 lines. |
| **P1** | **LH-4** | Next-bar-direction label option (`--label-type next_bar`) | `train_ml_pipeline_v3.py` ✅, `autonomous_train_loop.py` ✅ | ✅ Done — `label_type` param in `run_pipeline()` + `generate_labels()`. `--label-type` CLI flag. ~25 lines. |
| **P2** | **LH-5** | Phase completion tracking (`phase_state.json`) | Autonomous loop all phases ✅ | ✅ Done — `save_phase_state()` after phases 1,3,4,5. Auto-load model from state. ~35 lines. |

### Implementation Order

```
LH-1 (Checkpointing) → LH-2 (Pareto) → LH-3 (ETF exclusion) → LH-4 (Next-bar label) → LH-5 (Phase tracking)
```

All 5 items are additive (~200 lines total, 2 files) — no architectural changes.

 |

| P2 | T5 | Ensemble methods (Stacking, Voting, Blending) | T1b (tuned base models) | Combines CatBoost + LightGBM + RF |
| P2 | T6 | Monte Carlo VaR + CVaR risk modeling | Risk module ✅ | Replace binomial with proper VaR |
| P2 | T8 | SHAP visualization dashboard | SHAP already integrated | Waterfall, beeswarm, force plots |
| P2 | QW2 | Stop-loss optimization | Trade history data | GBDT predicts optimal stop distance |
| P2 | RS1 | EBM shape function alpha research pipeline | T2 (EBM) ✅ | Extract per-feature contribution curves → quantifiable alpha signals. Run on all 34 pattern detectors |
| P2 | RS2 | Dream team stacking ensemble (CatBoost + LightGBM) | T1b (GWO) ✅, T0 (unified) ✅ | Literature-validated: R² 0.815 vs 0.788 single-model. 3-5% accuracy gain |
| P2 | FS1 | Survival Analysis for Time-to-Target | scikit-survival (skill), existing multi-horizon labels | Predicts *when* TP/SL hits via censored regression. Requires FS4 completion. |
| P2 | FS2 | Historical Analog Matching (k-NN) | sklearn.NearestNeighbors or faiss | "When did market look like this before?" Trader-facing confidence tool. |
| P2 | FS3 | MAE/Drawdown as Primary Target | Existing max_drawdown_N computation | Predict worst-case drawdown → dynamic stops instead of fixed ATR. |
| P2 | FS5 | GMM Soft Regime Assignments | sklearn.mixture.GaussianMixture | Swap KMeans→GMM. Feed regime_proba to PatternClassifier. |
| P2 | FS8 | Volatility Forecasting Model | catboost (existing) | Predict realized vol over N bars → dynamic position sizing. |
| P2 | FS11 | Fractional Differentiation | statsmodels (installed) | Apply fracdiff(d≈0.3-0.5) to price features. Replace leaking StandardScaler. |
| P3 | T7 | Black-Litterman portfolio optimization | Portfolio module ✅ | ✅ Done — `src/portfolio/black_litterman.py` |
| P3 | T9 | Signal meta-labeling (kept as broader research task) | Signals module ✅ | López de Prado triple-barrier research task. Implementation now tracked as FS4. |
| P3 | RS3 | TabNet evaluation for regime detection | T2 (EBM) ✅ | Evaluate TabNet on high-dimensional feature sets. Only if >10K samples and GPU available |
| P3 | FS7 | Change-Point Detection for Regimes | ruptures (pure Python) | Real-time regime shift detection. Plugs into RegimeDetectorBase. |
| P3 | FS9 | Volume-Price Profile Clustering | sklearn.cluster | Accumulation vs distribution detection from volume-at-price. |
| P3 | FS10 | HDBSCAN Anomaly Detection | hdbscan | Auto-detect flash crashes, gaps. Feed anomaly_score → circuit breakers. |
| P3 | FS12 | Cross-Symbol Dynamic Clustering | sklearn + CrossAssetFeatures | Lead-lag relationships, correlation regime shifts. |
| P3 | FS13 | Breakout Probability ML | catboost + Donchian detector | ML score on breakouts: true breakout vs false one. |
| **KG1** | KG-H1 | Training-history overfitting detection | 5520 paper (training history), existing ML pipeline | Monitor loss curves for overfit in PurgedKFold. Source: Knowledge Graph Insights #1. |
| **KG1** | KG-H2 | Synthetic OOS comparison framework | Backtest Overfitting paper, `src/ml/` | Comprehensive OOS testing: combinatorial CV + synthetic controls. Source: KG Insights #1. |
| **KG1** | KG-H3 | Sentiment scores as signal weight modifier | 4+ sentiment papers, `src/signals/` | Feed Twitter/news sentiment into `EventWeightedAggregator`. Source: KG Insights #3. |
| **KG1** | KG-H4 | Event-driven pattern category | Building Calendar paper, Event-Based Trading paper, `src/patterns/` | New pattern category for event-based signals. Source: KG Insights #5. |
| **KG2** | KG-M1 | `src/rl/` module with trade execution env | OOM-RL paper, Adaptive RL paper, Deep Portfolio RL paper | New module: RL environment for trade execution + portfolio optimization. Source: KG Insights #2. |
| **KG2** | KG-M2 | Kelly criterion allocator | Investing Is Compression paper, `src/portfolio/` | Entropy/divergence-based position sizing. Source: KG Insights #4. |
| **KG2** | KG-M3 | AutoAlpha factor mining pipeline | AutoAlpha paper, `src/ml/` | Hierarchical evolutionary algorithm for formulaic alpha generation. Source: KG Insights #6. |
| **KG2** | KG-M4 | Circuit-based overfitting detection | Circuit Intrinsic Methods paper, `src/ml/` | Perturb rare patterns through model circuits. Source: KG Insights #1. |
| **KG2** | KG-M5 | Behavioral crash regime detection | Crash-based trading paper, `src/risk/` | Herding/overconfidence indicators for crash timing. Source: KG Insights #7. |
| **KG3** | KG-L1 | Adversarial overfitting detection | advrisk_neurips2019 paper, `src/ml/` | Use adversarial examples to expose overfit boundaries. Source: KG Insights #1. |
| **KG3** | KG-L2 | Financial event calendar database | 2 event papers, `src/data_ingestion/` | Build event DB from price spikes + news. Source: KG Insights #5. |
| **KG3** | KG-L3 | Defensive backtesting with time-reversal | Against Universal Trading paper, `src/backtest/` | Time-reversal heuristic for strategy validation. Source: KG Insights #7. |
| Research | T9 | Signal Meta-Labeling | LGBM default (CatBoost if ≥10% win) | Phase 6b Tier 1. "Should I take this signal?" |
| Research | FS19 | Gap-Fill Prediction | LGBM default | Phase 6b Tier 1. Predict gap fill within N bars. |
| Research | Ablation | Pattern Detector Audit | Backtest engine | Phase 6b Tier 1. Which 34 patterns produce edge? |
| Research | FS16 | Shapelets Discovery | aeon (CPU) | Phase 6b Tier 2. Learn discriminative price subsequences. |
| Research | FS15-lite | VAE Latent Embeddings [CPU] | torch (CPU, 32GB RAM ok) | Phase 6b Tier 2. Small VAE discovers market structure. |
| Research | FS20 | Heikin-Ashi Bars | Custom converter | Phase 6b Tier 2. Test smoothed bars on detectors. |
| Research | FS14 | Volume/Dollar/Tick Bars | Custom bar builder | Phase 6b Tier 3. Gate on Tier 1/2 positive results. |
| Deferred | FS17 | Volume Anomaly Forecasting | catboost | Absorbed into FS10 (HDBSCAN). |
| Deferred | FS18 | Synthetic OHLCV via GANs | torch (GPU) | GANs unstable. Defer indefinitely. |
| Deferred | FS21 | Sparse PCA | sklearn.decomposition | Feature selector more effective. |

## Deferred

| # | Phase/Item | Reason | Since | Revisit When |
|---|-----------|--------|-------|-------------|
| 05 | ML Advanced (full phase) | GPU >=16GB required for CNN training, autoencoder sweeps | 2026-04-30 | WSL2 with GPU passthrough or cloud GPU |
| 02 | vectorbt integration (sub-phase) | C++ compilation fails on Windows | 2026-04-20 | Linux/macOS environment |
| 07 | Paper Trading (full phase) | Optional — depends on Phase 06 completion | 2026-04-19 | Stages 01-06 stable |

## Execution Order

```
Phase 01 (Patterns) ✅
  → Phase 02 (Perf) 🔄 Numba done / vectorbt deferred
  → Phase 03 (Regime) ✅
  → Phase 04 (ML Foundation) ✅
      → Phase 06 (Research Enhancements) ✅
          → Phase 08 (Attribution) ✅
          → Phase 07 (Paper Trading) ⏸️
  Phase 05 (ML Advanced) ⏸️ — wait for GPU

Enhancement Execution Order:
  Enhancement Phase 1: Foundation Fixes (P1) ✅ DONE
    ├── T4 (DSR/PSR/FDR) ✅
    ├── FS4 (Triple Barrier Labeling) ✅
    └── FS6 (Combinatorial Purged CV) ✅
  Enhancement Phase 2: Quick Wins (P2) ✅ DONE
    ├── FS2 (Historical Analog Matching) ✅
    ├── FS3 (MAE/Drawdown Target) ✅
    ├── FS5 (GMM Regimes) ✅
    └── FS11 (Fractional Differentiation) ✅
  Enhancement Phase 3: New Models (P2) ✅ DONE
    ├── FS1 (Survival Analysis) ✅
    ├── FS8 (Volatility Forecasting) ✅
    ├── FS13 (Breakout Probability ML) ✅
    ├── T5 (Ensemble Methods) ✅
    └── T8 (SHAP Dashboard) ✅
  Enhancement Phase 4: Advanced Pipeline (P3) ✅ DONE
    ├── FS7 (Change-Point Detection) ✅
    ├── FS9 (Volume-Price Profile Clustering) ✅
    ├── FS10 (HDBSCAN Anomalies) ✅
    ├── FS12 (Cross-Symbol Clustering) ✅
  Enhancement Phase 4+: Risk, Alpha & Ensemble (T6, QW2, RS1, RS2) ✅ DONE
    ├── T6 (MC VaR + CVaR) ✅
    ├── QW2 (Stop-Loss Optimization) ✅
    ├── RS1 (EBM Alpha Pipeline) ✅
    └── RS2 (Dream Team Ensemble) ✅
   Enhancement Phase 6a: Model Robustness & Production Readiness ← IN PROGRESS
      │  └──  Why first: edge found on 5/12 tickers only. Must understand scope before
      │      building on top of it. Research/audit work is noise until the edge is clear.
      ├── MR0.1 C1 (Paper trade re-run) ✅
      ├── MR0.2 C2 (Walk-forward validation) ✅ — 15 bugs fixed, 5/12 pass
      ├── MR1 C3 (Basket vs single-ticker ablation) ✅ — 12-all > 5-winner. More diversity = better IC.
      ├── MR1 C4 (Prediction correlation decomposition) ✅ — Tech cluster r>0.82, KODK/GLD independent.
      ├── MR1 C6a (Auto ticker screening pipeline) ✅ — 75 screened, 22 passed. docs/ticker-test-log.md
       ├── MR2 C6 (Expand basket to 30+ tickers) ✅ (2026-05-11) — 33-ticker model. Mean IC +75%. All gates pass.
       ├── MR2 C6b (Edge relaxation analysis) ✅ (2026-05-11) — Volatility is the binding constraint. Sweet spot ≤25% ann. vol.
       ├── MR2 C5 (ML + pattern detector integration) ✅ (2026-05-11) — ml_strategy.py, 33-ticker backtest, 19/33 profitable.
       ├── P1  C7 (Strategy execution improvements) ✅ DONE (trail-stop winner)
       ├── P2  C8 (Pattern detector confluence) ✅ DONE (PatternBoostFilter)
      ├── P2  B1 (OOS validation) ✅ — FAILED: Sharpe -0.27 vs +0.73. Model overfit.
      ├── P1  B2 (Fix backtest_ml_enhanced.py) ✅ — 3 bugs fixed.
      ├── P2  B3 (Calibration audit) ✅ — ECE 0.128 IS, 0.197 OOS. Overconfident.
      ├── P1  B4 (Regime shift investigation) ✅ — Root cause: raw ATR scaling.
      ├── P1  B5 (WFO comparison) ✅ — WFO OOS +1.7% vs single-split -1.3%.
       ├── —   B8 (Agent infrastructure) ✅ — 3 agents + 3 commands + AGENTS.md update.
       ├── P0  B6 (Retrain with normalized ATR) ✅ — 2/5 pass. OOS worse. Root cause not fixed.
       ├── P0  B7 (Evaluate Qlib ADARNN/ADD) ✅ — ADARNN/ADD incompatible. CatBoost 0 positions.
       ├── P0  B9 (Stability Selection) ← NEXT
       ├── P0  B10 (Per-Sector Models)
       ├── P1  B11 (CPCV)
       ├── P1  B12 (Dynamic Ensemble Learning)
       ├── P2  B13 (Meta-Labeling)
       ├── P2  B14 (Production Hardening)
       └── P2  C9, C10, C11-C16 (gated on B9-B14)

  Enhancement Phase 6b: Pioneer Research (prove-or-discard) ← DEFERRED until MR passes
    ├── P1.1 T9 (Meta-Labeling) ✅
    ├── P1.2 FS19 (Gap-Fill Prediction) ✅
    ├── P1.3 Ablation (Pattern Detector Audit) ✅
    ├── P2.1 FS16 (Shapelets)
    ├── P2.2 FS15-lite (VAE on CPU)
    ├── P2.3 FS20 (Heikin-Ashi)
    └── P3.1 FS14 (Alt Bars — gate on Tier 1/2)
  Enhancement Phase 7: Notebook Audit Fixes ← DEFERRED until MR passes
    ├── F1 (Fix ablation/synergy metric extraction) ✅
    ├── F2 (Fix PatternSelector pattern discovery) ✅
    ├── F3 (Re-run notebooks 06, 07 to validate)
    ├── ML1 (Fix signal scorer)
    ├── ML2 (Reduce regime classifier overfit)
    └── ML3-4, S1-S4 (Feature, ensemble, meta-labeling improvements)
  Enhancement Phase 5: Deferred / GPU
    └── T7 (Black-Litterman) ✅, FS17, FS18, FS21, Phase 05 (ML Advanced)

   Phase 6c: Strategy Refinement ← ACTIVE
     ├── P1  C7 (Strategy execution improvements) ✅ DONE (trail-stop winner)
     ├── P1  C8 (Pattern detector confluence — PatternBoostFilter) ✅ DONE
    ├── P2  C9 (Honest walk-forward paper trading)
       ├── P2  C10 (Portfolio-level backtest)
       └── Phase 6d: PDF Insight Integration ← NEW
           ├── P1  C11 (Volume/OI validation)
           ├── P1  C12 (Multi-TP exit logic)
           ├── P1  C13 (Gap hierarchy + size filter)
           ├── P2  C14 (Missing harmonic detectors: Butterfly/Bat/Crab/Cypher/Shark)
           ├── P2  C15 (Pipe pattern detector)
           ├── P2  C16 (Dead Cat Bounce threshold fix)
           └── P3  C17 (Empirical reliability calibration)

   Phase 10: Tool Evaluation Additions ← NEW (2026-05-12)
     ├── P1  T10a-1 (Install Optuna + PyPortfolioOpt + Bandit)
     ├── P1  T10a-2 (Optuna HP tuning for CatBoost/LightGBM)
     ├── P1  T10a-3 (Optuna strategy parameter tuning)
     ├── P1  T10a-4 (PyPortfolioOpt integration)
     ├── P1  T10a-5 (Test + validate)
     ├── P1  T10a-6 (Documentation)
     ├── P2  T10b-1 (Bandit pre-commit hook)
     └── P3  T10c-1/2/3 (CCXT, FinGPT, aeon — deferred)

   Phase 10b: Autonomous Loop Hardening ← DONE (2026-05-13)
     ├── P0  LH-1 (Checkpointing + resume) ✅
     ├── P0  LH-2 (Multi-objective Pareto optimization) ✅
     ├── P1  LH-3 (Auto ETF cross-asset exclusion) ✅
     ├── P1  LH-4 (Next-bar-direction label) ✅
     └── P2  LH-5 (Phase completion tracking) ✅

   Phase 11: Overfitting Fixes ← NEW (2026-05-13) — 32 papers synthesized
     ├── P0  B9  (Stability Selection — replace ARO collapse)
     ├── P0  B10 (Per-Sector Models — eliminate cross-asset gap)
     ├── P1  B11 (CPCV — replace PurgedKFold)
     ├── P1  B12 (Dynamic Ensemble Learning — regime adaptation)
     ├── P2  B13 (Meta-Labeling — secondary signal filter)
     └── P2  B14 (Production Hardening — DSR/PBO/walk-forward gates)
```
```
