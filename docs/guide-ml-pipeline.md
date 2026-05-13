# ML Pipeline Reference — Standard Stages for Every ML Task

> **Audience:** AI agents working on this codebase. Read before starting any ML training,
> backtest, or validation task. This is the checklist — do all stages or document why
> you skipped one.

---

## Pipeline Overview

```
[1] Load Data
[2] Feature Engineering
[3] Label Generation (Triple-Barrier)
[4] Feature Selection (ARO)
[5] Hyperparameter Tuning (GWO / GA / WOA)
[6] Final Training (PurgedKFold CV + Calibration)
[7] Backtest Validation (Rank-Based)
[8] Meta-Labeling
[9] SHAP Explainability
```

---

## Stage-by-Stage Reference

### [1] Load Data

| Aspect | Detail |
|--------|--------|
| **Input** | `data/raw/{TICKER}_daily.csv` (or yfinance for ad-hoc) |
| **Index** | `pd.DatetimeIndex`, sorted |
| **Columns** | Open, High, Low, Close, Volume |
| **Warm-up** | Load 2+ years before backtest start for rolling feature windows |
| **Script** | `scripts/train_ml_model_v2.py` uses `load_pattern_data()` at line 58 |

### [2] Feature Engineering

| Aspect | Detail |
|--------|--------|
| **Implementation** | `src/ml/feature_engineering.py` → `FeatureExtractor` class |
| **Call** | `extractor = FeatureExtractor(); features = extractor.extract_all_features(df)` |
| **Output** | ~130 columns (price, momentum, volatility, volume, pattern shape, regime) |
| **Cross-Asset** | `src/ml/cross_asset_features.py` → `CrossAssetFeatureExtractor` |
| **Cross-asset call** | `load_market_data(df)` → `CrossAssetFeatureExtractor(market_data=...)` → `extract(df)` |
| **Important** | All rolling windows use `.shift(1)` — backward-looking only, no look-ahead |
| **Data needed** | Cross-asset requires `data/raw/{SPY,QQQ,TLT,GLD,IWM,XLF,XLK}_daily.csv` |

### [3] Label Generation

| Aspect | Detail |
|--------|--------|
| **Implementation** | `src/ml/triple_barrier.py` → `TripleBarrierLabeler` |
| **Default params** | `atr_mult_tp=1.5`, `atr_mult_sl=1.0`, `horizon=5` (days) |
| **Labels** | +1 = TP hit first, –1 = SL hit first, 0 = time limit expired |
| **Binary target** | `y = (labels == 1).astype(int)` — predict whether TP hits |
| **Fallback** | `(forward_return > threshold).astype(int)` if triple-barrier not available |
| **Script fallback** | `scripts/train_ml_model_v2.py:664` uses triple-barrier by default |

### [4] Feature Selection (ARO)

| Aspect | Detail |
|--------|--------|
| **Implementation** | `src/ml/tuning/aro_selector.py` → `AROFeatureSelector` |
| **What it does** | Binary feature-inclusion vectors, optimized by ARO metaheuristic |
| **Fitness** | PurgedKFold AUC with selected features |
| **Typical params** | `target_n_features=15`, `n_rabbits=30`, `max_iter=100` |
| **When to skip** | Fewer than 30 features total (ARO won't help) |
| **Alternative** | IC-based filtering in `train_ml_model_v2.py:652-657` (weaker, but faster) |

```python
from src.ml.tuning import AROFeatureSelector

selector = AROFeatureSelector(
    feature_names=features.columns.tolist(),
    X=features, y=labels,
    model_class=PatternClassifier,
    target_n_features=15,
    n_rabbits=30,
)
result = selector.select()
selected_features = result.selected_features
```

### [5] Hyperparameter Tuning

**Available metaheuristic optimizers — choose one:**

| Optimizer | File | Best for | CLI flag |
|-----------|------|----------|----------|
| **GWO** (Grey Wolf) | `src/ml/tuning/gwo_tuner.py` | CatBoost/LGBM HP tuning | `scripts/tune_model.py` (default) |
| **GA** (Genetic Algorithm) | `src/ml/tuning/ga_tuner.py` | Regime discovery (KMeans clusters) | `--algo ga` |
| **WOA** (Whale Optimization) | `src/ml/tuning/woa_tuner.py` | Threshold tuning (pattern confidence) | `--algo woa` |

**Standard CatBoost parameter space** (`CATBOOST_PARAM_SPACE` in `gwo_tuner.py:25`):

| Parameter | Type | Range | Scale |
|-----------|------|-------|-------|
| `learning_rate` | float | 0.01–0.3 | log |
| `depth` | int | 3–10 | linear |
| `l2_leaf_reg` | float | 1.0–30.0 | log |
| `random_strength` | float | 0.5–5.0 | linear |
| `bagging_temperature` | float | 0.0–2.0 | linear |
| `border_count` | int | 32–255 | linear |
| `min_data_in_leaf` | int | 5–50 | linear |

**Usage — CLI:**
```bash
# GWO tuning for pattern classifier
uv run scripts/tune_model.py --symbol SPY --wolves 20 --iterations 50

# GA for regime discovery
uv run scripts/tune_model.py --symbol SPY --algo ga --target regime_discovery

# WOA for threshold optimization
uv run scripts/tune_model.py --symbol SPY --algo woa --target pattern_threshold
```

**Usage — Python:**
```python
from src.ml.tuning import GWOTuner, SearchSpace, CATBOOST_PARAM_SPACE

def fitness(params):
    model = PatternClassifier(model_type="catboost", n_estimators=150, **params)
    result = model.train(X, y)
    return result.test_auc

space = SearchSpace(CATBOOST_PARAM_SPACE)
tuner = GWOTuner(space, fitness, n_wolves=20, maximize=True, seed=42)
result = tuner.optimize(max_iter=50, early_stop=10)
# result.best_params → optimized HP dict
# result.best_score  → best AUC
```

**Anti-pattern (what JOE did):** The JOE v3 models used a 3-set fixed grid inside nested PurgedKFold CV (`train_ml_model_v2.py:285-310`). This tests only 3 param combinations. GWO tests hundreds of combinations guided by the metaheuristic.

### [6] Final Training

| Aspect | Detail |
|--------|--------|
| **Implementation** | `scripts/train_ml_model_v2.py` |
| **CV method** | Nested PurgedKFold: 5 outer × 3 inner folds (`N_CV_OUTER=5, N_CV_INNER=3`) |
| **Embargo** | 5% of dataset (`PCT_EMBARGO=0.05`) |
| **Model** | CatBoost: `n_estimators=100`, `learning_rate=0.03` + GWO-optimized params from Stage 5 |
| **Calibration** | Platt scaling (sigmoid) — stored in `.pkl` as `calibration_slope_`, `calibration_intercept_` |
| **Save format** | `models/pattern_classifier_{run_id}.pkl` + `_metadata.json` |
| **Overfit check** | `eval_set` monitoring: stop if train AUC diverges from validation AUC by > 0.15 |

### [7] Backtest Validation

| Aspect | Detail |
|--------|--------|
| **Method** | **Rank-based** — take top N% of bars by raw CatBoost probability |
| **Why rank-based** | Models often lack Platt scaling → probabilities are compressed but ranking works |
| **Script** | `scripts/backtest_joe_cross_asset.py` (generic, works for any ticker/model) |
| **Entry** | Long at signal bar close |
| **Exit** | TP (entry + ATR_MULT_TP × ATR), SL (entry – ATR_MULT_SL × ATR), or horizon bars |
| **Position sizing** | Fixed fraction of capital (2% risk per trade) ÷ ATR distance |
| **Metrics** | Total return, Sharpe, max drawdown, win rate, profit factor, trade count |

```bash
uv run scripts/backtest_joe_cross_asset.py --top-pct 5.0 --horizon 5
```

**Minimum viability thresholds:**
- 30+ trades across backtest
- Sharpe > 0.5, profit factor > 1.2
- Win rate consistent across signal fractions (not just one lucky fraction)

### [8] Meta-Labeling

| Aspect | Detail |
|--------|--------|
| **What it does** | Trains a secondary classifier on trade outcomes to filter primary model signals |
| **Library** | `src/ml/meta_labeler.py` → `MetaLabeler` class |
| **Notebook** | `notebooks/17_meta_labeler.py` — configurable ticker/model |
| **Features** | Same features as the primary model (37 for baseline, 82 for CA) at entry bar |
| **Label** | 1 = TP hit, 0 = SL/Time |
| **Model** | Auto-selects LightGBM vs CatBoost (CatBoost must beat LGBM by ≥ 10% AUC) |
| **Threshold** | Youden's J statistic for optimal probability threshold |
| **Output** | Filtered trade list + comparison table (unfiltered vs filtered) |

**Warning:** Requires 50+ trades for reliable training. JOE at top-5% only gave 35 trades.
Increase `TOP_PCT` to 10–15% if needed, or combine multiple tickers.

### [9] SHAP Explainability

| Aspect | Detail |
|--------|--------|
| **Library** | `src/ml/shap_dashboard.py` |
| **What it does** | SHAP waterfall, beeswarm, bar plots for feature importance and interaction |
| **Why run it** | Explanations have been reported to help understand regime-specific behavior |
| **Anti-pattern** | Skip this and you won't know WHY a model performs well or poorly |

---

## State of Each Pipeline Stage

Use this table to track what has been done for any given model/ticker:

| Stage | Implemented | File |
|-------|------------|------|
| [1] Load Data | ✓ | `data/raw/{TICKER}_daily.csv` |
| [2] Feature Engineering | ✓ | `src/ml/feature_engineering.py` |
| [2b] Cross-Asset Features | ✓ | `src/ml/cross_asset_features.py` |
| [3] Triple-Barrier Labels | ✓ | `src/ml/triple_barrier.py` |
| [4] ARO Feature Selection | ✓ | `src/ml/tuning/aro_selector.py` |
| [5] GWO HP Tuning | ✓ | `src/ml/tuning/gwo_tuner.py` |
| [5] GA Regime Tuning | ✓ | `src/ml/tuning/ga_tuner.py` |
| [5] WOA Threshold Tuning | ✓ | `src/ml/tuning/woa_tuner.py` |
| [6] PurgedKFold Training | ✓ | `scripts/train_ml_model_v2.py` |
| [7] Rank-Based Backtest | ✓ | `scripts/backtest_joe_cross_asset.py` |
| [8] Meta-Labeling | ✓ | `notebooks/17_meta_labeler.py` |
| [9] SHAP Explainability | ✓ | `src/ml/shap_dashboard.py` |

**What JOE v3 models were missing (as of 2026-05-09):**

| Stage | Status | Impact |
|-------|--------|--------|
| [4] ARO | Skipped | 37 features used instead of ~15 optimal ones |
| [5] GWO | Skipped (fixed grid) | Only 3 param sets tested; GWO tests 1000+ |
| [8] Meta-Labeling | Skipped | No post-filter on signals; all top-N trades taken |
| [9] SHAP | Skipped | No explanation for why cross-asset features hurt OOS |

---

## Minimum Pipeline for a New Ticker

When training a new ticker from scratch, do at minimum:

```bash
# Stage 4-6 in one go (ARO not auto-integrated yet — do manually):
uv run scripts/train_ml_model_v2.py --symbol data/raw/FOO_daily.csv --horizon 5

# Stage 5 (separate — better HPs than the fixed grid):
uv run scripts/tune_model.py --symbol FOO --wolves 20 --iterations 50

# Stage 7:
uv run scripts/backtest_joe_cross_asset.py --top-pct 5.0 --horizon 5

# Stage 8 (open notebook and change TICKER + MODEL_PATH):
# notebooks/17_meta_labeler.py
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/ml/feature_engineering.py` | Feature extraction (130+ columns) |
| `src/ml/cross_asset_features.py` | Cross-asset features (25+ columns) |
| `src/ml/triple_barrier.py` | Label generation for training |
| `src/ml/tuning/aro_selector.py` | ARO feature selection |
| `src/ml/tuning/gwo_tuner.py` | GWO hyperparameter tuning |
| `src/ml/tuning/ga_tuner.py` | GA regime discovery tuning |
| `src/ml/tuning/woa_tuner.py` | WOA threshold tuning |
| `src/ml/tuning/base.py` | Shared optimizer abstractions |
| `src/ml/purged_cv.py` | PurgedKFold cross-validation |
| `src/ml/pattern_classifier.py` | CatBoost model wrapper |
| `src/ml/meta_labeler.py` | Meta-labeler class |
| `src/ml/shap_dashboard.py` | SHAP explainability |
| `scripts/train_ml_model_v2.py` | Full training pipeline CLI |
| `scripts/tune_model.py` | Metaheuristic tuning CLI |
| `scripts/backtest_joe_cross_asset.py` | Rank-based backtest (any ticker) |
| `notebooks/17_meta_labeler.py` | Configurable meta-labeling notebook |
