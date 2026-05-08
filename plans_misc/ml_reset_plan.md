# ML Reset Plan: Study → Learn → Implement with Proper Logging

**Date:** 2026-04-19
**Status:** Phase A COMPLETE | Phase B COMPLETE (2026-04-30)

## Session Handover Instructions

Read this entire plan, then execute **Phase A only** (Code Foundation — no new ML models).

**Phase A steps (in order):**

| Step | File to Create | Reference | Description |
|------|---------------|-----------|-------------|
| A1 | `src/ml/experiment_logger.py` | ML4T Ch6 | JSONL structured logging: metadata, config, per-fold metrics, feature importance, predictions, model artifacts |
| A2 | `src/ml/purged_cv.py` | ML4T Ch6:04 | PurgedKFold + Embargo for proper financial CV |
| A3 | `src/ml/feature_store.py` | ML4T Ch4+Ch12:04 | Pre-computed feature cache as parquet |
| A4 | `src/ml/metrics.py` | ML4T Ch7:06+Ch12:06 | IC, rank IC, hit rate — beyond accuracy |
| A5 | `src/ml/registry.py` | DLQT pattern | Model version management |
| A6 | `src/ml/backtest_bridge.py` | ML4T Ch5:01+Ch12:09 | ML predictions → backtest connection |

**Constraints:** `uv run` for everything, run tests after each step, write tests for new modules, type hints + docstrings. Do NOT start Phase B until Phase A is done.

---

## Background: What Went Wrong

The project drifted from **"study reference repos → extract patterns → implement"** into building ML components ad-hoc from scratch. Key failures:

1. **No PurgedKFold** — Every training fold has information leakage because labels overlap
2. **Wrong metrics** — Classification accuracy instead of Spearman IC, rank IC, information ratio
3. **No logging** — Training results returned as dicts and discarded, no experiment tracking
4. **No feature evaluation** — 81 features generated but zero IC analysis to know which actually predict
5. **No out-of-sample discipline** — Train/test split without proper embargo period
6. **sklearn importances used** — These are biased toward high-cardinality features; MDI/permutation/SHAP needed

The 4 reference repos in `useful_resources/useful_repos/` already solve all of these. We never studied them.

---

## Part 1: Learnings from Reference Repositories

### 1A. Machine-Learning-for-Algorithmic-Trading-Second-Edition (Stefan Jansen / Lopez de Prado methods)

24 chapters. **This is the bible for our use case.**

| Ch | Key Learning | Why It Matters for Us | Priority |
|----|-------------|---------------------|----------|
| **06:04** | **PurgedKFold + Embargo** — Purge training samples where labels overlap with test period. Add embargo buffer after test. | We retrain every week in walk-forward but training labels from overlapping periods leak test information. This inflates performance by 20-30pp. | CRITICAL |
| **06:02** | **Mutual Information as feature selection** — Not just correlation, detect non-linear relationships | We compute MI but don't use it to validate feature quality before training | HIGH |
| **11:07** | **Backtesting with Zipline** — Full pipeline: feature engineering → RF training → signal generation → Alphalens evaluation → backtest | We jump from "train model" to "backtest" skipping signal quality evaluation | HIGH |
| **12:04** | **Model data preparation** — How to properly structure multi-asset data for ML: date × ticker × features, forward returns as labels, point-in-time features only | Our signal scorer trains on 50 SPY trades with 8 features. Standard is 500+ tickers × 1000+ days. | CRITICAL |
| **12:05** | **LightGBM + CatBoost with embargo CV** — Hyperparameter search with TimeSeriesSplit+embargo, evaluate using Spearman rank IC not accuracy | We treat signal scoring as classification. Industry standard uses ranking/regression with IC metrics | CRITICAL |
| **12:07** | **Model interpretation** — MDI vs permutation importance comparison. Partial dependence plots. Feature interaction analysis | We have no feature interpretation at all. We don't know why our model predicts what it predicts. | HIGH |
| **12:08** | **Out-of-sample prediction** — Strict train/validate/test separation: 2010-2016 train, 2015-2016 validate, 2017 test only | Our walk-forward doesn't have a final untouched test period | HIGH |
| **12:09** | **Backtesting with ML signals** — Convert predictions → portfolio weights → backtest. Compare ML vs baseline strategies. | We don't have a clean bridge from ML predictions to backtest execution | MEDIUM |
| **18:05-07** | **CNN for trading** — Convert time series to clustered image format (CIF) → ResNet → prediction | Novel approach that may capture patterns our 34 manual detectors miss | MEDIUM |
| **19** | **LSTM for multi-asset prediction** — Sequence modeling with feature embeddings | Could model temporal dependencies our current approach ignores | LOW (later) |
| **20** | **Conditional autoencoder for risk factors** — Unsupervised latent factor discovery | Risk factors from data, not predefined. Use as features. | LOW (later) |
| **24** | **Alpha Factor Library** — 101 formulaic alphas, factor evaluation pipeline (IC → combine → backtest) | Our 81 features are a subset. Need proper evaluation pipeline. | HIGH |

### Core Workflow from ML4T Chapter 12 (The one we should follow):

```
1. Prep model data (Ch12:04)
   → Multi-asset, point-in-time features, forward returns as labels

2. Train models (Ch12:05)
   → LightGBM + CatBoost with proper CV (TimeSeriesSplit + embargo)
   → Hyperparameter tuning
   → Save all results

3. Evaluate signals (Ch12:06)
   → Spearman rank IC, information ratio, hit rate
   → Compare models by IC not accuracy

4. Interpret model (Ch12:07)
   → MDI + permutation importance
   → Partial dependence plots

5. Out-of-sample (Ch12:08)
   → Predict on never-before-seen data
   → Compare OOS vs in-sample performance

6. Backtest (Ch12:09)
   → ML signals → portfolio weights → backtest
   → Compare ML long-short vs buy-hold
```

### 1B. Deep-Learning-in-Quantitative-Trading (Hao Zhang)

| Ch | Key Learning | Why It Matters |
|----|-------------|---------------|
| **03** | **Standard PyTorch training loop** — train → validate → early stopping → test pattern with metrics logging | We never have a proper DL training loop. This is the template. |
| **04** | **1D CNN for momentum** — Raw OHLCV → CNN learns patterns automatically → portfolio weights | May discover patterns our manual feature engineering misses |
| **05** | **Deep portfolio optimization** — End-to-end learning of portfolio weights | Advanced, for later |

**Reusable utilities from this repo:**

```
Utilis/early_stopper.py  → Patience-based early stopping
Utilis/loss.py           → Custom losses: MSE, Huber, SharpeLoss, SortinoLoss
Utilis/metrics.py        → Trading-specific metric computation
Utilis/torch_data.py     → TimeSeriesDataset with sliding window
```

### 1C. Quant-Developers-Resources
- Production quant development C++ patterns (not directly applicable)
- Risk management framework references (VaR, CVaR needed in our risk layer)

### 1D. Orderbook
- C++ market microstructure (not applicable to our Python ML layer)

---

## Part 2: Industry Standard ML Logging Architecture

### Current State: NOTHING

| What | Current State | Standard |
|------|--------------|----------|
| Model persistence | In memory only | `model.joblib` + metadata per run |
| Training metrics | Printed to console, lost | JSONL per fold: metrics, dates, params |
| Feature importance | Not computed or discarded | MDI + permutation + SHAP saved per run |
| Predictions | Not logged | JSONL: timestamp, features, pred, actual, error |
| Experiments tracked | No | `experiments/runs/<run_id>/` hierarchy |
| Reproducibility | None | Config + seed + data hash saved with every run |
| Overfitting detection | Manual comparison of train/test | Deflated Sharpe Ratio, Combinatorial Purged CV |

### Required Logging Architecture

```
experiments/
├── runs/
│   ├── regime_rf_20260419_001/
│   │   ├── metadata.json        # Description, author, date, goal
│   │   ├── config.json          # hyperparams, data_window, features_used, random_state
│   │   ├── data_hash.json       # SHA256 of training data for reproducibility
│   │   ├── fold_metrics.jsonl   # One line per fold (ML4T Ch6 PurgedKFold format)
│   │   ├── feature_importance.json  # MDI + permutation + SHAP
│   │   ├── predictions.csv      # timestamp, ticker, pred, actual, error
│   │   └── model.joblib         # Serialized model
│   ├── signal_gb_20260419_001/
│   │   └── ...
│   └── cnn_regime_20260419_001/
│       └── ...
├── features/
│   └── feature_store.parquet    # ticker × date × 146 features (precomputed, served to all models)
└── reports/
    └── ml_comparison.md         # Auto-generated: runs compared, best model, verdict
```

### Logging Formats

#### 1. Run Metadata
```json
{
  "run_id": "regime_rf_20260419_001",
  "description": "Random Forest regime classification with PurgedKFold + embargo",
  "date": "2026-04-19T10:00:00",
  "model_type": "random_forest",
  "task": "regime_classification",
  "data": {
    "tickers": ["SPY", "QQQ", ...],
    "start": "2015-01-01",
    "end": "2024-12-31",
    "n_samples": 3340,
    "n_features": 25
  },
  "random_state": 42,
  "data_hash_sha256": "abc123..."
}
```

#### 2. Fold Metrics (Per Fold, ML4T Format)
```jsonl
{"fold": 1, "train_start": "2015-01-01", "train_end": "2018-06-30",
 "test_start": "2018-07-15", "test_end": "2018-12-31",
 "purged_samples": 45, "embargo_days": 10,
 "n_train": 890, "n_test": 112,
 "accuracy": 0.52, "auc_roc": 0.58,
 "spearman_ic": 0.03, "rank_ic": 0.04,
 "train_accuracy": 0.78, "overfit_gap": 0.26}
```

#### 3. Feature Importance (3 Methods)
```json
{
  "mdi": {"vol_regime": 0.107, "atr_50": 0.089, "xle_rs": 0.071, ...},
  "permutation": {"xle_rs": 0.032, "vol_regime": 0.028, "tlt_ief_ratio": 0.019, ...},
  "shap_top10": [{"feature": "vol_regime", "mean_abs_shap": 0.15}, ...]
}
```

#### 4. Prediction Log (Every Single Prediction)
```jsonl
{"timestamp": "2024-01-15", "ticker": "SPY", "regime_pred": "Ranging",
 "regime_proba": {"Trending": 0.12, "Ranging": 0.58, "Volatile": 0.20, "Transition": 0.10},
 "feature_set": "cross_asset_25", "model": "regime_rf_20260419_001"}
```

#### 5. Run Summary Verdict
```json
{
  "run_id": "regime_rf_20260419_001",
  "mean_oos_ic": 0.028,
  "mean_oos_accuracy": 0.51,
  "mean_overfit_gap": 0.27,
  "n_folds": 10,
  "verdict": "OVERFIT — overfit_gap > 0.20, OOS IC < 0.05",
  "recommendation": "Do not use in production. Try feature reduction or different model.",
  "best_fold_ic": 0.045,
  "worst_fold_ic": 0.010
}
```

---

## Part 3: Implementation Plan

### Phase A — Code Foundation (NO NEW ML MODELS)

| Step | Component | What It Does | Reference | Files Created | Acceptance Criteria |
|------|-----------|-------------|-----------|--------------|-------------------|
| **A1** | **ML Experiment Logger** | Structured JSONL logging for all ML runs. Saves metadata, config, per-fold metrics, feature importance, predictions, model artifacts. | ML4T Ch6 workflow | `src/ml/experiment_logger.py` | Every ML run produces a `runs/<id>/` folder with all log files |
| **A2** | **PurgedKFold with Embargo** | CV splitting that purges training samples with overlapping labels + embargo buffer after test period. Prevents look-ahead bias. | ML4T Ch6:04 (PurgedKFold) | `src/ml/purged_cv.py` | Passes unit test: no training sample's label window overlaps with any test sample |
| **A3** | **Feature Store** | Pre-compute features once, store as parquet. ticker × date × features. Includes forward returns as labels. Served to any model without recomputation. | ML4T Ch4 + Ch12:04 | `src/ml/feature_store.py` | Load 50 tickers × 10 years → <5 seconds. Same result every time. |
| **A4** | **Signal Evaluation Metrics** | IC (Pearson correlation between features and forward returns), rank IC (Spearman), IC decay (over horizons), hit rate, information ratio. | ML4T Ch7:06 + Ch12:06 | `src/ml/metrics.py` | Produces IC DataFrame for any feature matrix. Matches ML4T Ch12:06 results |
| **A5** | **Model Registry** | Load/save models with metadata. Version control. Track which model produced which predictions. | DLQT training pattern | `src/ml/registry.py` | Load saved model → produces identical predictions. Metadata queryable |
| **A6** | **Backtest Bridge** | Takes saved model → generates predictions → injects signal scores into confluence scorer → runs backtest. Produces directly comparable results vs baseline. | ML4T Ch5:01 + Ch12:09 | `src/ml/backtest_bridge.py` | Backtest output is identical format to existing runner output |

**Deliverable at end of Phase A:**
- All existing ML components (`regime_model.py`, `signal_scorer.py`, `pipeline.py`) refactored to use the logging infrastructure
- Running `regime_model.train()` now produces a full `experiments/runs/<id>/` folder automatically
- Feature store replaces on-the-fly feature computation (5-10x faster iteration)

### Phase B — ML Enhancement (After A is Complete)

Each step builds on A's logging infrastructure, so every experiment is automatically tracked.

| Step | Component | What We Build | Reference | Replaces | Expected Outcome |
|------|-----------|--------------|-----------|----------|-----------------|
| **B1** | **Proper Feature Engineering** | Build on feature store. Add forward returns (1d, 5d, 20d) as labels. Compute IC per feature. Remove IC < 0.02. Add 101 formulaic alphas subset. | ML4T Ch4 + Ch24 | `src/ml/features.py` | 146 → ~40 high-IC features |
| **B2** | **Regime Classification (Done Right)** | PurgedKFold + embargo. Compare RF, GB, LightGBM. MDI + permutation + SHAP importance. All metrics logged automatically. | ML4T Ch11 + Ch12 | `src/ml/regime_model.py` | Regime model with OOS IC > 0.05, overfit gap < 0.15 |
| **B3** | **Signal Generation (Regression Not Classification)** | Predict forward returns (regression). Spearman rank IC as metric. Proper walk-forward. Cross-asset features. | ML4T Ch12:05 | `src/ml/signal_scorer.py` | Signal scorer with OOS rank IC > 0.03 |
| **B4** | **Feature Selection (SFI)** | Sequential Feature Importance — Lopez de Prado method. Iteratively find best subset. Not simple correlation filter. | ML4T Ch11 + Ch12 | `src/ml/feature_selector.py` | Feature count optimized for max OOS IC |
| **B5** | **CNN Regime Detection** | 1D CNN on OHLCV sequence → regime label. Early stopping. Custom loss. Compare vs rule-based and tree-based. | DLQT Ch4 | New `src/ml/cnn_regime.py` | CNN regime accuracy comparable to RF |
| **B6** | **Autoencoder Risk Factors** | Unsupervised latent factors from 50-ticker returns. 3-5 latent dimensions → added as features. | ML4T Ch20 | New `src/ml/risk_factors.py` | Autoencoder risk factors improve regime IC by 10%+ |
| **B7** | **Full ML-Enhanced Backtest** | Assemble best model → predictions → confluence scoring → backtest vs baseline. Report: return, Sharpe, DD, win rate, profit factor. Compare ML vs rule-based. | ML4T Ch12:09 | `scripts/ml_enhanced_backtest.py` | ML backtest results comparable to baseline, with logging |

---

## Decision Matrix

| Phase | Effort | Risk | Value | Do First? |
|-------|--------|------|-------|-----------|
| **A: Code Foundation** | 3-4 days | LOW (pure engineering) | **HIGH** (enables all future ML to be properly tracked) | **YES** |
| **B1: Feature Engineering** | 2-3 days | LOW | HIGH | After A |
| **B2: Regime (Proper)** | 2-3 days | LOW | HIGH | After B1 |
| **B3: Signal (Regression)** | 3-4 days | MEDIUM | HIGH | After B2 |
| **B4: Feature Selection** | 1-2 days | LOW | MEDIUM | After B3 |
| **B5: CNN** | 3-5 days | MEDIUM | MEDIUM | Optional |
| **B6: Autoencoder** | 3-5 days | HIGH | MEDIUM | Optional/later |
| **B7: Full Backtest** | 2-3 days | LOW | HIGH | After all others |

---

## What to Execute Now

**Start Phase A.** No ML models, no new data downloads. Pure infrastructure:

1. `src/ml/experiment_logger.py` — JSONL-based structured logging
2. `src/ml/purged_cv.py` — PurgedKFold + embargo splitter
3. `src/ml/feature_store.py` — Pre-computed feature cache
4. `src/ml/metrics.py` — IC, rank IC, hit rate
5. `src/ml/registry.py` — Model version management
6. `src/ml/backtest_bridge.py` — ML → backtest connection

Then refactor existing code to use A1-A6, so future ML work produces proper logs automatically.
