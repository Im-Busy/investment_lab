# Phase 1: Bug Fixes & Cleanup

**Created:** 2026-05-11
**Status:** Not started
**Estimated effort:** 1 session
**Depends on:** Nothing

---

## Root Cause Summary

Three independent bugs in the Phase B pipeline (`scripts/phase_b_ml_enhancement.py`) conspired to produce fake impressive numbers:

| Bug | File | Line(s) | Severity | Impact |
|-----|------|---------|----------|--------|
| **Rule-based labels** | `src/ml/regime_model.py` → `phase_b_ml_enhancement.py:63-67` | Trains CatBoost to memorize ADX/ATR thresholds | **Critical** | All 14 Phase B regime experiments are invalid |
| **Circular IC metric** | `scripts/phase_b_ml_enhancement.py:215-216` | `corr(score × return, return)` instead of `corr(score, return)` | **Critical** | IS IC reads 0.95+ when true IC ≈ 0.00 |
| **label_span mismatch** | `scripts/phase_b_ml_enhancement.py:200` | PurgedKFold default `label_span=1` with 5-day horizon labels | **High** | Under-purging leaks test-period info into training |

The correct pipeline already **exists** in `scripts/train_ml_model_v2.py` — it uses triple-barrier labels, `label_span=horizon`, nested PurgedKFold, and IC-based feature filtering. Phase 1 fixes the broken script so both pipelines use the same standards.

---

## Task 1.1: Fix Circular IC Metric

**File:** `scripts/phase_b_ml_enhancement.py`
**Lines:** 215-216
**Effort:** 5 minutes

### Current (broken):
```python
pred_returns = y_pred * y_test_continuous
rank_ic = pd.Series(pred_returns).corr(pd.Series(y_test_continuous), method="spearman")
```

### Problem:
Both sides of the correlation contain `y_test_continuous`. If the model predicts mostly >= 0.5 (which it does by default), `pred_returns ≈ y_test_continuous` and `corr ≈ 1.0`.

### Fix:
```python
rank_ic = pd.Series(y_pred).corr(pd.Series(y_test_continuous), method="spearman")
pearson_ic = pd.Series(y_pred).corr(pd.Series(y_test_continuous), method="pearson")
```

### Also fix lines 218-219:
```python
# Remove the separate pearson_ic computation and just use one:
ranks_ic_val = pd.Series(y_pred).corr(pd.Series(y_test_continuous), method="spearman")
pearson_ic_val = pd.Series(y_pred).corr(pd.Series(y_test_continuous), method="pearson")
rank_ic = float(rank_ic_val) if not pd.isna(rank_ic_val) else 0.0
pearson_ic = float(pearson_ic_val) if not pd.isna(pearson_ic_val) else 0.0
```

### Verification:
After fix, IS IC should drop from 0.95+ to < 0.10. OOS IC should stay near 0.00 (which is honest).

---

## Task 1.2: Fix PurgedKFold label_span Mismatch

**File:** `scripts/phase_b_ml_enhancement.py`
**Lines:** 200, 111
**Effort:** 2 minutes

### Current (broken):
```python
# Line 111 (regime classification — also broken, but moot since Bug 1 makes it irrelevant)
purged_cv = PurgedKFold(n_splits=5, pct_embargo=0.02)

# Line 200 (signal scorer)
purged_cv = PurgedKFold(n_splits=5, pct_embargo=0.02)
```

Both use default `label_span=1`. The signal scorer labels are 5-day forward returns (line 79: `generate_labels(df, horizon=5)`).

### Fix:
```python
# Line 111
purged_cv = PurgedKFold(n_splits=5, pct_embargo=0.02, label_span=5)

# Line 200
purged_cv = PurgedKFold(n_splits=5, pct_embargo=0.02, label_span=5)
```

### Rationale:
PurgedKFold purges training samples where `i + label_span >= test_start`. With `label_span=1`, samples at t-4 through t-1 are retained in training despite their 5-day labels overlapping the test period. Setting `label_span=5` correctly purges all overlapping training samples.

### Additional hardening:
Increase `pct_embargo=0.02` to `pct_embargo=0.05` (5% of test span) to match `train_ml_model_v2.py`'s `PCT_EMBARGO=0.05` constant.

---

## Task 1.3: Deprecate RegimeClassifier as Prediction Target

**Effort:** 15 minutes

### Problem:
`RegimeClassifier` (`src/ml/regime_model.py`) trains on `RegimeDetector` output (ADX/ATR rule-based labels). The model is trying to memorize a deterministic 4-line formula. The project's own validation report concluded: *"ML cannot reliably improve on the rule-based regime detector."*

### Actions:

**a) Add deprecation warning to `src/ml/regime_model.py`:**
At the top of `RegimeClassifier.__init__` (around line 59):
```python
import warnings
warnings.warn(
    "RegimeClassifier.train() predicts rule-based labels from RegimeDetector "
    "(ADX/ATR thresholds). This is NOT a prediction problem — the model memorizes "
    "a formula. Use the RegimeDetector output as a FEATURE, not a label. "
    "For genuine ML targets, use triple-barrier labels via PatternClassifier or "
    "SignalRegressor. This class will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
```

**b) Remove RegimeClassifier usage from Phase B script:**
In `scripts/phase_b_ml_enhancement.py`, remove or comment out lines 63-67, 101-175, and 298-299 (the `phase_b_b2_regime_classification` call). Add a comment explaining why.

**c) Archive invalid experiments:**
Move all 14 `experiments/phase_b_*` directories to `experiments/_archived_invalid/`:
```bash
mkdir experiments/_archived_invalid
mv experiments/phase_b_* experiments/_archived_invalid/
```

### What stays:
- `src/indicators/regime_detector.py` — `RegimeDetector` is **useful as a feature generator**
- The ADX/ATR regime output (Trending/Ranging/Volatile/Transition) can be added as a categorical feature column during training

---

## Task 1.4: Verify Lineage of feature_engineering.py Leak Source

**Effort:** 10 minutes (read-only audit)

### Context:
`src/ml/features.py:309-328` has `add_forward_returns()` which generates `forward_return_{1,5,20}d` columns using `close.pct_change(h).shift(-h)`. These are forward-looking. Similarly `src/ml/feature_engineering.py:364-378` generates `forward_return_X` and `forward_binary_X` columns.

### Actions:

**a) Verify no existing code path feeds forward returns as features:**
```bash
grep -rn "generate_features_with_labels\|add_forward_returns" src/ scripts/ --include="*.py"
```

Only `features.py:330` defines `generate_features_with_labels`. No callers found in earlier audit.

**b) Verify no code passes forward_return columns as X features:**
```bash
grep -rn "forward_return_" scripts/ --include="*.py" | grep -v "label\|target\|y_train\|y_test\|\.shift"
```

**c) Add safety guard to `generate_features_with_labels`:**
Add a docstring warning and rename the method or make it harder to misuse accidentally. Current naming `generate_features_with_labels` could be misread as "generate features AND labels" (correct) vs "generate features WITH labels mixed in as features" (dangerous).

Decision: Either rename to `generate_features_and_forward_labels()` or add an assertion that forward_return columns are never passed to model.fit().

### Note:
The correct pipeline `train_ml_model_v2.py` uses `FeatureExtractor` from `src/ml/feature_engineering.py`, not `FeatureEngineer` from `src/ml/features.py`. The two feature engineering modules exist in parallel which is confusing. The `FeatureExtractor` in `feature_engineering.py` does NOT inject forward returns into its output by default.

---

## Task 1.5: Standardize Feature Engineering

**Effort:** 15 minutes

### Problem:
Two parallel feature engineering modules exist:
- `src/ml/features.py` → `FeatureEngineer` (126 features, used by Phase B scripts, has `add_forward_returns` trap)
- `src/ml/feature_engineering.py` → `FeatureExtractor` (used by `train_ml_model_v2.py`, doesn't inject forward returns)

### Action:
In `src/ml/features.py`, add a module-level docstring:
```python
"""
⚠️ DEPRECATION WARNING: This module is being phased out in favor of
src/ml/feature_engineering.py → FeatureExtractor.

FeatureEngineer.generate_features_with_labels() generates forward-return columns
that can accidentally be passed as features, causing look-ahead leakage.

Prefer FeatureExtractor from feature_engineering.py for all new pipelines.
"""
```

---

## Validation Checklist

After completing all Phase 1 tasks:

- [ ] `grep -rn "pred_returns = y_pred \* y_test" scripts/` returns empty — circular IC is gone
- [ ] All PurgedKFold instantiations in `phase_b_ml_enhancement.py` include `label_span=5`
- [ ] `RegimeClassifier.__init__` emits DeprecationWarning
- [ ] `experiments/_archived_invalid/` contains all 14 `phase_b_*` runs
- [ ] `phase_b_b2_regime_classification` is removed or gated behind a flag
- [ ] `generate_features_with_labels` callers confirmed safe
- [ ] `uv run ruff check scripts/phase_b_ml_enhancement.py` passes

---

## Next: Phase 2


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


---

## Goal

Take the honest baseline from Phase 2 and productionize it:

1. **Meta-labeling** — second-stage filter that rejects trades likely to fail
2. **Expand ticker universe** — scale from 5 to 30+ liquid tickers
3. **Regime-conditional evaluation** — does the model work differently in bull vs bear?
4. **SHAP explainability** — can we explain every top feature in plain English?
5. **Live paper trading** — same model, no retuning, forward-walk on latest 6 months

The key principle: **no retuning after Phase 2.** Phase 3 validates that the signal works on unseen data. Any retuning resets the OOS clock.

---

## Task 3.1: Meta-Labeling Pipeline

**Implementation:** `src/ml/meta_labeler.py` (already exists)
**Effort:** 1 session

### How it works:
1. Train primary model (PatternClassifier or SignalRegressor) on triple-barrier labels
2. For each signal generated by the primary model, extract features at the entry bar
3. Train a secondary binary classifier (LightGBM/CatBoost) on: did this trade hit TP? (yes=1, no=0)
4. At prediction time: primary model generates signal → meta-labeler scores it → only take trades above Youden's J threshold

### Current state:
- `meta_labeler.py` is implemented and tested on JOE/KODK
- It auto-selects LightGBM vs CatBoost (CatBoost must beat by 10% AUC)
- Uses PurgedKFold with `pct_embargo=0.05`
- Requires 50+ trades for reliable training

### Phase 3 actions:

**a) Integrate meta-labeling into the unified pipeline:**
`train_ml_pipeline_v3.py` should output a `meta_labeler.pkl` alongside `model.pkl`.

**b) Add meta-labeling to backtest:**
`backtest_joe_cross_asset.py` should accept `--meta-model path/to/meta_labeler.pkl` and filter signals.

**c) Report meta-labeling metrics:**
| Metric | Meaning |
|--------|---------|
| Signal reduction % | What fraction of signals does it filter out? |
| Win rate before/after | Does it improve win rate? |
| Sharpe before/after | Does it improve risk-adjusted returns? |
| False positive rate | How many good trades does it incorrectly reject? |

### Success criterion:
- Meta-labeler reduces signals by 30-60% (reasonable filter, not a crippling one)
- Post-filter Sharpe >= pre-filter Sharpe + 0.1
- Post-filter win rate >= pre-filter win rate + 5%

---

## Task 3.2: Expand Ticker Universe

**Effort:** 1 session

### Rationale:
Training on 5 tickers is better than 1, but 30+ is the standard for statistical ML in finance. More tickers = more independent samples = less overfitting. The earlier AI's advice about mid-cap, high-volume, low-analyst-coverage stocks was directionally correct for finding inefficient markets.

### Tier 1 — Liquid ETFs (regime anchors):
| Ticker | Type | Why |
|--------|------|-----|
| SPY | Equity | Broad market |
| QQQ | Tech/growth | Growth regime proxy |
| IWM | Small-cap | Risk-on/off regime |
| TLT | Long bonds | Interest rate regime |
| GLD | Gold | Inflation/risk-off |
| USO | Oil | Commodity cycle |
| UUP | USD | Currency regime |

### Tier 2 — Sector ETFs (rotation signals):
| Ticker | Sector |
|--------|--------|
| XLF | Financials |
| XLK | Technology |
| XLE | Energy |
| XLV | Healthcare |
| XLI | Industrials |
| XLY | Consumer Discretionary |

### Tier 3 — Individual stocks (alpha targets):
Criteria: market cap $500M-$5B, avg daily volume > 500K shares, fewer than 5 analyst ratings

| Ticker | Notes |
|--------|-------|
| JOE | Already in training set |
| KODK | Already identified |
| + 5-10 more | Use yfinance screener to find candidates |

### Implementation:
```python
def build_ticker_universe() -> dict[str, list[str]]:
    """Build tiered ticker universe for training."""
    return {
        "regime_anchors": ["SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "UUP"],
        "sectors": ["XLF", "XLK", "XLE", "XLV", "XLI", "XLY"],
        "targets": find_target_tickers(min_mcap=500e6, max_mcap=5e9, min_volume=500e3, max_analysts=5),
    }
```

### Training strategy:
- **Stage 1 features**: Cross-asset features from regime anchors + sectors (always included)
- **Stage 2 training**: Pool all target tickers together
- **Stage 3 evaluation**: Walk-forward on each target individually (no pooling in OOS)

### Success criterion:
- At least one target ticker shows OOS rank IC >= 0.03
- No ticker shows OOS rank IC <= -0.05
- Expanded universe produces >= 100 trades in walk-forward

---

## Task 3.3: Regime-Conditional Performance Analysis

**Effort:** 45 minutes

### Why:
A model with rank IC = 0.02 overall might be IC = 0.08 in trending markets and IC = -0.03 in ranging markets. Understanding this is critical for live deployment.

### Analysis dimensions:
1. **By ADX regime**: Trending (ADX > 25) vs Ranging (ADX < 20) vs Transition
2. **By volatility regime**: High vol (VIX > 25) vs Normal (15-25) vs Low (< 15)
3. **By market direction**: SPY 20d return > 0 vs < 0
4. **By year**: Does performance degrade over time? (alpha decay)

### Implementation:
```python
def regime_conditional_analysis(
    predictions: pd.Series,
    actuals: pd.Series,
    regime_labels: pd.Series,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute rank IC within each regime bucket.

    Returns:
        DataFrame with regime, n_samples, rank_ic, hit_rate, mean_return
    """
```

### Output table:

| Regime | N | Rank IC | Hit Rate | Mean Return |
|--------|---|---------|----------|-------------|
| Trending (ADX > 25) | 450 | 0.045 | 0.56 | +0.3% |
| Ranging (ADX < 20) | 320 | -0.012 | 0.48 | -0.1% |
| Volatile (ATR > p80) | 180 | 0.031 | 0.53 | +0.5% |
| High VIX (> 25) | 210 | 0.022 | 0.51 | +0.2% |
| SPY bull (20d > 0) | 580 | 0.038 | 0.55 | +0.4% |
| SPY bear (20d < 0) | 390 | -0.005 | 0.47 | -0.3% |

### Action based on results:
- If IC is negative in ranging markets: add a regime gate that disables signals when ADX < 20
- If IC decays in later years: retrain with shorter lookback or add more recent data
- If IC varies wildly by year: the signal is unstable, not production-ready

---

## Task 3.4: SHAP Explainability Audit

**Effort:** 1 session

### Why:
The earlier AI noted: *"If the top feature is some exotic 126th alpha that no human can justify, be skeptical."* Every feature in the top 10 should have a plain-English explanation.

### Implementation:
Use `src/ml/shap_dashboard.py` (already exists) to generate:

1. **Global feature importance** (bar plot, top 20)
2. **SHAP waterfall** for 5 representative predictions (2 correct, 2 wrong, 1 edge case)
3. **SHAP dependence plots** for top 5 features (does SHAP value make economic sense?)
4. **Feature interaction** between top 2 features

### Audit criteria for each top-10 feature:

| Criterion | Pass condition |
|-----------|---------------|
| **Name** | Descriptive, not `alpha_98` or `feat_47` |
| **Direction** | Positive SHAP = higher prediction, makes economic sense |
| **Monotonicity** | SHAP values change smoothly, not chaotically |
| **Stability** | Similar importance rank across folds |
| **Explainability** | Can state in 1 sentence why it matters |

### Red flags that require investigation:
- Feature rank jumps > 5 positions between folds
- SHAP values are bimodal (two clusters of prediction behavior)
- Feature has zero SHAP for 90% of samples but huge SHAP for 10%
- Feature correlation with top feature > 0.8 (redundant — remove one)

### Success criterion:
- Top 10 features all pass the explainability audit
- At least 7 of 10 have intuitive economic meaning
- No feature shows pathological SHAP patterns

---

## Task 3.5: Live Paper Trading Simulation

**Effort:** 1 session

### What it is:
Take the final model from Phase 2 (trained on 2015-2020), run it forward day-by-day on 2021-2024, generating signals and tracking P&L as if trading live. No retuning. This is the ultimate OOS test.

### Implementation:
```python
def paper_trade_simulation(
    model: SignalRegressor,
    meta_model: MetaLabeler | None,
    tickers: list[str],
    start: str = "2021-01-01",
    end: str = "2024-12-31",
    capital: float = 100_000,
    risk_per_trade: float = 0.02,
) -> pd.DataFrame:
    """
    Day-by-day forward simulation.

    Each day:
    1. Generate features up to today only
    2. Model predicts expected return
    3. Meta-labeler filters signals
    4. Execute trade at next bar open
    5. Exit at TP/SL or horizon
    6. Track P&L

    Returns:
        DataFrame with daily portfolio value
    """
```

### Key constraint:
This must use **point-in-time** features only. No feature can use data beyond the current bar. This means:
- No rolling windows that peek forward
- No `shift(-k)` anywhere in the feature pipeline
- Cross-asset features must be computed from data available at that date
- No "market regime" features that use future data to define the regime

### Metrics:
| Metric | Target |
|--------|--------|
| Total return (vs SPY) | Within 5% of SPY or better |
| Sharpe | > 0.3 |
| Max drawdown | < 30% |
| Win rate | > 45% |
| Profit factor | > 1.1 |
| Trade count | > 100 |

### Output:
```
reports/paper_trading/
├── equity_curve.png
├── monthly_returns.csv
├── trade_log.csv
├── drawdown_chart.png
└── summary.json
```

### If paper trading fails:
This is the hardest test and may fail on first attempt. That's honest. Debug steps:
1. Check regime-conditional analysis — disable trading in bad regimes
2. Increase meta-labeler threshold (fewer trades, higher quality)
3. Add volatility-adjusted position sizing
4. Accept that the signal may be too weak for live trading and go back to research

---

## Task 3.6: Integration with Backtesting Framework

**Effort:** 45 minutes

### Current state:
The project has a custom event-driven backtest engine at `src/backtest/engine.py` and strategy wrappers for `backtesting.py` in `src/strategies/`. These are designed for pattern-detector-based strategies, not ML-driven signals.

### Action:
Create `src/backtest/ml_strategy.py`:

```python
class MLSignalStrategy:
    """
    Strategy that generates signals from an ML model.

    Compatible with the custom event-driven backtest engine.

    On each bar:
    1. Extract features
    2. Run model prediction
    3. Apply meta-labeler filter
    4. Generate signal (BUY/SELL/HOLD)
    5. Risk manager sizes position
    """
```

### Integration points:
- `src/risk/` — Position sizing, loss limits, circuit breakers
- `src/backtest/engine.py` — Custom event-driven engine
- `src/signals/` — Signal aggregation (combine ML signals with pattern signals)

### Success criterion:
- ML strategy can be plugged into existing backtesting framework
- ML signals and pattern-detector signals can coexist in the same backtest
- Confluence scoring (pattern + ML agreement) can be tested

---

## Task 3.7: Documentation & Handover

**Effort:** 30 minutes

### Actions:

**a) Update COMMAND_CHEATSHEET.md:**
Add the new pipeline commands:
```bash
# Full 9-stage pipeline
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --horizon 5

# Fast iteration (skip ARO + GWO)
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --fast

# Paper trading simulation
uv run scripts/paper_trade.py --model models/v3_honest_basket.pkl --start 2021-01-01
```

**b) Update docs/guide-ml-pipeline.md:**
- Add Phase 2 pipeline as the standard reference
- Mark `phase_b_ml_enhancement.py` as deprecated
- Document the three bugs and why they matter

**c) Update AGENTS.md:**
Add to the Backtesting Standards section:
```
- Rank IC is the primary ML evaluation metric. Accuracy on rule-based targets is meaningless.
- Every PurgedKFold call must use label_span=horizon where horizon matches the label window.
- Never compute IC as corr(score*return, return). Always corr(score, return).
```

---

## Validation Checklist

- [ ] Meta-labeling integrated into pipeline and backtest
- [ ] Ticker universe >= 20 instruments
- [ ] Regime-conditional analysis shows IC breakdown by market environment
- [ ] Top 10 SHAP features all pass explainability audit
- [ ] Paper trading simulation completes without look-ahead violations
- [ ] ML strategy integrates with existing backtest engine
- [ ] All command cheatsheets and docs updated

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Phase 2 baseline produces IC < 0.0 | Medium | Accept as honest result; revisit feature engineering |
| Meta-labeler reduces trades to < 30 | Medium | Lower meta-labeler threshold; skip if insufficient trades |
| Paper trading loses money | High | Expected for first attempt; iterate on position sizing and regime gates |
| Expanded universe introduces data quality issues | Low | yfinance data is reliable for liquid tickers |
| Model degrades in 2023-2024 (alpha decay) | Medium | Investigate retraining frequency; compare 1yr vs 2yr vs 4yr lookback |
