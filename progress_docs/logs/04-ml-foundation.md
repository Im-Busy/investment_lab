# Phase 04 Log: ML Foundation

| # | Date | Type | Summary | Files |
|---|------|------|---------|-------|
| 1 | 2026-04-19 | create | A1: Experiment logger with JSONL structured logging | `src/ml/experiment_logger.py` (17 tests) |
| 2 | 2026-04-19 | create | A2: PurgedKFold + Embargo for proper financial CV | `src/ml/purged_cv.py` (19 tests) |
| 3 | 2026-04-19 | create | A3: Feature store with parquet cache | `src/ml/feature_store.py` (16 tests) |
| 4 | 2026-04-19 | create | A4: ML metrics — IC, rank IC, hit rate beyond accuracy | `src/ml/metrics.py` (23 tests) |
| 5 | 2026-04-19 | create | A5: Model registry for version management | `src/ml/registry.py` |
| 6 | 2026-04-19 | create | A6: Backtest bridge — ML predictions to backtest | `src/ml/backtest_bridge.py` |
| 7 | 2026-04-19 | create | B1: Feature engineering — 53+ alpha factors, IC selection | `src/ml/features.py` |
| 8 | 2026-04-19 | create | B2: Regime classifier — PurgedKFold, SHAP, permutation importance | `src/ml/regime_model.py` |
| 9 | 2026-04-19 | create | B3: Signal scorer — regression-based, Spearman rank IC | `src/ml/signal_scorer.py` |
| 10 | 2026-04-19 | create | B4: Feature selector — SFI + auto-detection | `src/ml/feature_selector.py` |
| 11 | 2026-04-19 | create | B5: CNN regime detection — Regime1DCNN, EarlyStopping, focal loss | `src/ml/cnn_regime.py` |
| 12 | 2026-04-19 | create | B6: Risk factor autoencoder — L1 sparsity, 3-5 dims | `src/ml/risk_factors.py` |
| 13 | 2026-04-19 | test | B7: ML-enhanced backtest on SPY 2015 — Sharpe -1.86 (baseline -1.86) | `scripts/phase_b7_ml_backtest.py` |
| 14 | 2026-04-26 | fix | B7 scripts: date format parsing, MLFilteredPattern wrapper | `scripts/phase_b7_ml_backtest.py`, `phase_b7_custom_backtest.py` |
| 15 | 2026-04-30 | fix | B1: FeatureSelector auto-detects continuous vs categorical targets | `src/ml/feature_selector.py` |
| 16 | 2026-04-30 | upgrade | B2: RegimeClassifier with PurgedKFold, SHAP, permutation importance | `src/ml/regime_model.py` |
| 17 | 2026-04-30 | create | B3: SignalRegressor class — predicts forward returns via Rank IC | `src/ml/signal_scorer.py` |
| 18 | 2026-04-30 | fix | B4: FeatureSelector MI classification vs regression auto-detect | `src/ml/feature_selector.py` |
| 19 | 2026-04-30 | create | B5: Regime1DCNN + CNNRegimeDetector — PyTorch, EarlyStopping | `src/ml/cnn_regime.py` |
| 20 | 2026-04-30 | create | B6: RiskFactorAutoencoder — L1 sparsity, Adam optimizer | `src/ml/risk_factors.py` |
| 21 | 2026-04-30 | test | B7 scripts fixed and verified on SPY 2020-2024 | `scripts/phase_b7_ml_backtest.py` |

**Phase 04 Complete** — 2026-04-30
