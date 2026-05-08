---
project: investment_trying
last_updated: 2026-05-07
summary: |
  Rule-based multi-pattern trading system with 34+ chart pattern detectors,
  ML-enhanced regime detection, Numba-accelerated indicators, and event-driven
  backtesting engine. 8 phases spanning pattern detection through paper trading.
  21 new friend-suggested ML items added from external OHLCV pipeline review.
phases_total: 8
phases_complete: 6
phases_active: 1
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

## Enhancements

| Name | Plan | Status |
|------|------|--------|
| ML Capability Enhancements (GWO, InterpretML, AutoGluon + wider gaps) | [plan](enhance-ml-capabilities.md) | 🔴 Planned — Tier 1 ready |
| External Review Additions (21 friend-suggested items) | [plan](enhance-ml-capabilities.md#friends-pipeline-recommendations--external-review-additions) | 🔴 Planned — 6 Tier 1, 7 Tier 2, 8 Tier 3 |

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
| P2 | T5 | Ensemble methods (Stacking, Voting, Blending) | T1b (tuned base models) | Combines CatBoost + LightGBM + RF |
| P2 | T6 | Monte Carlo VaR + CVaR risk modeling | Risk module ✅ | Replace binomial with proper VaR |
| P2 | T8 | SHAP visualization dashboard | SHAP already integrated | Waterfall, beeswarm, force plots |
| P2 | QW2 | Stop-loss optimization | Trade history data | GBDT predicts optimal stop distance |
| P2 | RS1 | EBM shape function alpha research pipeline | T2 (EBM) ✅ | Extract per-feature contribution curves → quantifiable alpha signals. Run on all 34 pattern detectors |
| P2 | RS2 | Dream team stacking ensemble (CatBoost + LightGBM) | T1b (GWO) ✅, T0 (unified) ✅ | Literature-validated: R² 0.815 vs 0.788 single-model. 3-5% accuracy gain |
| P3 | T7 | Black-Litterman portfolio optimization | Portfolio module ✅ | ✅ Done — `src/portfolio/black_litterman.py` |
| P1 | FS4 | Triple Barrier Labeling (elevated from T9) | PositionManager TP/SL levels | ✅ Done — `src/ml/triple_barrier.py`, 34 tests |
| P1 | FS6 | Combinatorial Purged CV | Extend existing `src/ml/purged_cv.py` | ✅ Done — `src/ml/combinatorial_purged_cv.py`, 22 tests |
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
  Enhancement Phase 6b: Pioneer Research (prove-or-discard) ← NEXT
    ├── P1.1 T9 (Meta-Labeling)
    ├── P1.2 FS19 (Gap-Fill Prediction)
    ├── P1.3 Ablation (Pattern Detector Audit)
    ├── P2.1 FS16 (Shapelets)
    ├── P2.2 FS15-lite (VAE on CPU)
    ├── P2.3 FS20 (Heikin-Ashi)
    └── P3.1 FS14 (Alt Bars — gate on Tier 1/2)
  Enhancement Phase 5: Deferred / GPU
    └── T7 (Black-Litterman) ✅, FS17, FS18, FS21, Phase 05 (ML Advanced)
```
```
