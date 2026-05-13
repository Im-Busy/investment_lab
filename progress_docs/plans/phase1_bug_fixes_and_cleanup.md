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
