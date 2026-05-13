# Phase 2: Pipeline Rebuild — Honest Foundation

**Created:** 2026-05-11
**Status:** Not started
**Estimated effort:** 2-3 sessions
**Depends on:** Phase 1 complete

---

## Goal

Replace the broken Phase B pipeline with a single clean pipeline that:
1. Uses **triple-barrier labels** (not rule-based, not naive thresholded binary)
2. Uses **SignalRegressor** (regression on forward returns, not classification on rules)
3. Applies **ARO feature selection** (top 15 from 82, not all 82)
4. Applies **GWO hyperparameter tuning** (hundreds of combos, not 3 fixed sets)
5. Runs **nested PurgedKFold** with correct `label_span=horizon` and `pct_embargo=0.05`
6. Trains on **cross-asset basket** (JOE + SPY + QQQ + TLT + GLD), not single tickers
7. Evaluates with **rank IC** (Spearman), not accuracy or circular metrics

The correct pipeline already exists in fragment form across `train_ml_model_v2.py` and individual tuning modules. Phase 2 integrates them into one coherent workflow.

---

## Task 2.1: Create Unified Pipeline Script

**New file:** `scripts/train_ml_pipeline_v3.py`
**Effort:** 1 session

### Design

This script replaces both `scripts/phase_b_ml_enhancement.py` AND extends `scripts/train_ml_model_v2.py`. It runs the full 9-stage pipeline for a single ticker (or ticker basket):

```
[1] Load OHLCV data (single ticker or basket)
[2] Feature extraction (FeatureExtractor from feature_engineering.py)
[3] Cross-asset features (cross_asset_features.py — SPY, QQQ, TLT, GLD, IWM)
[4] IC-based filtering (min_abs_ic=0.02)
[5] Triple-barrier labels (atr_mult_tp=1.5, atr_mult_sl=1.0, time_limit=horizon)
[6] ARO feature selection (target_n_features=15, n_rabbits=30)
[7] GWO hyperparameter tuning (n_wolves=5, max_iter=20)
[8] Nested PurgedKFold final training (5 outer × 3 inner, pct_embargo=0.05)
[9] OOS rank IC evaluation + walk-forward backtest
```

### CLI Interface:

```bash
# Single ticker:
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --horizon 5 --top-pct 5.0

# Basket of tickers (train on all, test on each):
uv run scripts/train_ml_pipeline_v3.py --symbol JOE,SPY,QQQ --horizon 5

# Skip expensive stages for fast iterations:
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --skip-aro --skip-gwo --fast

# Compare with rule-based labels (for diagnostic purposes only):
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --diagnostic-regime
```

### Integration of existing components:

| Stage | Existing Implementation | How Integrated |
|-------|------------------------|----------------|
| [1] Load | `train_ml_model_v2.py:58-92` | Direct reuse via `load_pattern_data()` |
| [2] Features | `feature_engineering.py:FeatureExtractor` | Direct import |
| [3] Cross-asset | `cross_asset_features.py:CrossAssetFeatureExtractor` | Direct import, configurable ticker list |
| [4] IC filter | `train_ml_model_v2.py:652-657` (`filter_features_by_ic`) | Extract to shared utility |
| [5] Labels | `triple_barrier.py:TripleBarrierLabeler` | Direct import |
| [6] ARO | `tuning/aro_selector.py:AROFeatureSelector` | Direct import |
| [7] GWO | `tuning/gwo_tuner.py:GWOTuner` | Call with CatBoost param space |
| [8] CV | `purged_cv.py:PurgedKFold` | Nested 5×3 as in train_ml_model_v2.py:250-330 |
| [9] Backtest | `backtest_joe_cross_asset.py` | Extract rank-based backtest to shared utility |

### Experiment Logging:
Use `src/ml/experiment_logger.py` → `ExperimentLogger` for structured JSONL logging:
- Config: all CLI args, feature names, label distribution
- Per-fold: train/test AUC, rank IC, hit rate, overfit gap
- Final: mean/std metrics, SHAP summary, model path

### Output Directory:
```
experiments/v3_{symbol}_{timestamp}/
├── config.json
├── fold_metrics.jsonl
├── feature_importance.csv
├── selected_features.txt
├── best_params.json
├── model.pkl
├── backtest_results.json
└── shap_summary.png
```

---

## Task 2.2: Fix the IC-based Metrics Module

**File:** `src/ml/metrics.py`
**Effort:** 30 minutes

### Current issues in `metrics.py`:

1. **`compute_ic`** calls `shift(-h)` on forward returns (line 154):
   ```python
   fwd_h = forward_returns.rolling(window=h).sum().shift(-h)
   ```
   This is a look-ahead within the metric computation itself. If `forward_returns` is already a shifted series, this double-shifts it.

2. **`ic_summary`** passes `forward_returns` directly (line 59). The metric functions should take predictions and actuals and compute correlation without any shifting.

### Fix:
- Add a `compute_rank_ic` function that simply computes `spearmanr(predictions, actuals)` with no internal shifting.
- Rename existing `shift(-h)` versions to `compute_rolling_ic` and deprecate them.
- All shift operations should happen at label generation time, NEVER at metric computation time.

### Verification:
```python
# Test: IC between random predictions and actuals should be ~0
np.random.seed(42)
fake_preds = pd.Series(np.random.randn(1000))
fake_labels = pd.Series(np.random.randn(1000))
ic = compute_rank_ic(fake_preds, fake_labels)
assert abs(ic) < 0.1, f"IC should be near zero for random data, got {ic}"
```

---

## Task 2.3: Build Cross-Asset Basket Training

**Effort:** 45 minutes

### Rationale:
JOE cross-asset v3 showed the only honest improvement in the repo: Test AUC 0.471 → 0.620, overfit gap 0.378 → 0.164. Cross-asset features nearly halved overfitting. This needs to be the default, not an option.

### Ticker Basket:
Based on the earlier analysis:

| Ticker | Role | Rationale |
|--------|------|-----------|
| **JOE** | Primary target | Showed strongest cross-asset improvement |
| **KODK** | Secondary target | Same criteria (mid-cap, high volume, low analyst coverage) |
| **SPY** | Market baseline | Broad equity market proxy |
| **QQQ** | Growth/tech exposure | Different regime from value |
| **TLT** | Bond/rate proxy | Regime signal when equities diverge from rates |
| **GLD** | Gold/commodity proxy | Inflation/risk-off regime signal |

### Implementation in pipeline script:

```python
def train_basket(
    tickers: list[str],
    horizon: int = 5,
    top_pct: float = 5.0,
) -> dict:
    """Train on all tickers, evaluate OOS on each individually."""
    results = {}

    # Load and feature-engineer each ticker
    dfs = {}
    features = {}
    for ticker in tickers:
        dfs[ticker] = load_data(ticker)
        features[ticker] = extract_features(dfs[ticker], cross_asset=True)

    # Combine all features for training
    X_all = pd.concat(features.values())
    y_all = pd.concat([generate_labels(dfs[t]) for t in tickers])

    # Train on basket
    model, cv_results = train_with_nested_cv(X_all, y_all, horizon)

    # Evaluate on each ticker OOS
    for ticker in tickers:
        X_oos = features[ticker].loc["2021-01-01":]
        y_oos = generate_labels(dfs[ticker]).loc["2021-01-01":]
        results[ticker] = evaluate_oos(model, X_oos, y_oos)

    return results
```

### Success criteria:
- Train on 2015-2020, test on 2021-2024
- Per-ticker OOS rank IC >= 0.02 (honest, not circular)
- At least one ticker shows Sharpe > 0.3 in walk-forward backtest
- Mean overfit gap across tickers < 0.15

---

## Task 2.4: Implement Walk-Forward Validation as Primary OOS Test

**Effort:** 45 minutes

### Problem:
The current `phase_b_ml_enhancement.py` uses PurgedKFold as its only CV method. While correct, PurgedKFold folds are still random splits within the time range. A single 5-fold CV can give optimistic results if the best-performing fold happens during a bull run.

### Solution:
Implement a rolling walk-forward that mimics production: train on a growing window, predict the next step, advance.

### Implementation in `src/ml/walk_forward.py`:

```python
def walk_forward_validation(
    model: SignalRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_years: int = 3,
    step_months: int = 6,
    retrain: bool = True,
) -> pd.DataFrame:
    """
    Rolling walk-forward validation.

    Window: [train_start, train_end] → predict [train_end+1, train_end+step]
    After each step, train window expands by step_months.

    Returns:
        DataFrame with date, predicted, actual, step_number
    """
```

### Metrics per step:
- Rank IC (Spearman between predictions and actuals)
- Hit rate (prediction sign matches actual sign)
- Mean predicted return vs realized return

### CLI:
```bash
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --walk-forward
```

### Success criterion:
- Rank IC across all walk-forward steps >= 0.02 (rolling 12-month average)
- No single step with rank IC < -0.05 (would indicate anti-signal during that period)
- Std of rank IC across steps < 0.10 (stability)

---

## Task 2.5: Run First Honest Experiment

**Effort:** 30 minutes (mostly compute time)

### Command:
```bash
uv run scripts/train_ml_pipeline_v3.py \
    --symbol JOE,SPY,QQQ,TLT,GLD \
    --horizon 5 \
    --top-pct 5.0 \
    --walk-forward \
    --output experiments/v3_honest_basket_20260511
```

### Expected metrics (honest baseline):

| Metric | Expected Value | Worst Acceptable |
|--------|---------------|-----------------|
| OOS rank IC | 0.01-0.03 | > -0.02 |
| OOS AUC | 0.52-0.58 | > 0.50 |
| Walk-forward Sharpe | 0.0-0.3 | > -0.2 |
| Overfit gap (AUC) | 0.05-0.15 | < 0.20 |
| Trade count | 50+ | > 30 |

These are **realistic first-run numbers**. The JOE cross-asset v3 gave Test AUC 0.62 with 37 features and no ARO/GWO. After feature selection and tuning, expect modest improvement but NOT 0.95 IC.

### If results are worse than expected:
1. Check that Task 2.2 (metrics fix) is complete
2. Verify `label_span=horizon` in all PurgedKFold calls
3. Confirm no forward_return columns in feature set
4. Reduce to top 5 features by IC and retry
5. Simplify to LogisticRegression as baseline model

---

## Dependency on Existing Code

| Existing Component | Status | Changes Needed |
|--------------------|--------|---------------|
| `train_ml_model_v2.py` | Correct pipeline, keep | Extract shared functions to `src/ml/pipeline_utils.py` |
| `feature_engineering.py:FeatureExtractor` | Correct, backward-looking | None |
| `cross_asset_features.py:CrossAssetFeatureExtractor` | Works, keep | None |
| `triple_barrier.py:TripleBarrierLabeler` | Correct | None |
| `purged_cv.py:PurgedKFold` | Correct algorithm | None |
| `tuning/aro_selector.py:AROFeatureSelector` | Works | None |
| `tuning/gwo_tuner.py:GWOTuner` | Works | Ensure compatible with SignalRegressor |
| `pattern_classifier.py:PatternClassifier` | Works with CatBoost | Use as-is for binary TP-hit classification |
| `signal_scorer.py:SignalRegressor` | Works, has correct `train_with_purged_cv(label_span=5)` | Use for regression on forward returns |
| `experiment_logger.py:ExperimentLogger` | Works | None |
| `metrics.py` | Has shift(-h) bug in IC computation | Fix as Task 2.2 |
| `features.py:FeatureEngineer` | Has forward-return trap | Deprecate, don't use |

---

## Validation Checklist

- [ ] `train_ml_pipeline_v3.py` exists and runs end-to-end on JOE
- [ ] `compute_rank_ic(random_preds, random_labels)` returns ~0
- [ ] Basket training produces per-ticker OOS metrics
- [ ] Walk-forward produces time-series of rank IC values
- [ ] No `forward_return_` column appears in feature importance
- [ ] Experiment logged with all stages documented
- [ ] `uv run ruff check` passes on all new/modified files

---

## Next: Phase 3
