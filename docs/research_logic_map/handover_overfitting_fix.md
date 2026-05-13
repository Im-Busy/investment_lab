# Handover: Fix ML Overfitting Before Retraining

**Created:** 2026-05-09
**Status:** Ready for next session
**Prerequisite to:** H13 test (inefficient instrument ML alpha), H14 test (cross-asset features)

---

## What Happened

Trained CatBoost on 6 instruments (SPY, QQQ, CRVL, KODK, HIFS, JOE) using the default `scripts/train_ml_model.py` pipeline. Every model produced train AUC ~1.0 with test AUC ~0.48-0.56. The models memorized training data perfectly and are useless. Artifacts were deleted.

## What Needs to Happen

Modify `scripts/train_ml_model.py` (or fork it) to wire in 4 anti-overfitting tools that already exist in the project. Then retrain all 6 instruments and compare.

---

## Task: Fix `train_ml_model.py` Overfitting

### Step 1: Add IC-Based Feature Filtering

**Existing code to use:** `src/ml/features.py` lines 348-422

In `train_ml_model.py`, `_run_training()` at ~line 491, after `extract_features()` at line 509:

```python
# After: features, feature_names = extract_features(df)
# Add:
from src.ml.features import FeatureEngineer
engineer = FeatureEngineer()
features_with_labels = engineer.generate_features_with_labels(df)
filtered = engineer.filter_low_ic_features(
    features_with_labels,
    forward_returns="forward_return_5d",
    min_abs_ic=0.02,
    min_abs_rank_ic=0.02,
)
logger.info(f"IC filtering: {len(features_with_labels.columns)} -> {len(filtered.columns)} features")
# Then use filtered features instead of `features` for training
```

### Step 2: Reduce Model Complexity

In `train_ml_model.py`, function `train_final_model()` at line 267-276, change the hardcoded parameters:

```python
classifier = PatternClassifier(
    model_type=model_type,
    n_estimators=100,      # Was 200
    max_depth=3,           # Was 6
    learning_rate=0.03,    # Was 0.05
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    l2_leaf_reg=10.0,      # Was default 3.0 — stronger L2 penalty
    random_strength=3.0,   # Was default 1.0 — more noise in splits
    min_data_in_leaf=50,   # Was default 20 — larger leaves = more generalization
)
```

Also update the walk-forward classifier at ~line 504 (same params).

### Step 3: Replace Binary Label with Triple-Barrier

**Existing code to use:** `src/ml/triple_barrier.py` lines 60-373

In `_run_training()`, replace the label generation. Current code (~line 514):

```python
labels = generate_labels(df, horizon=args.horizon)
```

Replace with:

```python
from src.ml.triple_barrier import TripleBarrierLabeler

labeler = TripleBarrierLabeler()
atr = ...  # compute ATR from df['High'], df['Low'], df['Close']

labels = labeler.fit(
    close=df["Close"],
    high=df["High"],
    low=df["Low"],
    take_profit=None,       # use dynamic ATR-based
    stop_loss=None,         # use dynamic ATR-based
    time_limit=10,          # exit after 10 bars if no TP/SL hit
    atr_mult_tp=1.5,        # take profit at +1.5x ATR
    atr_mult_sl=1.0,        # stop loss at -1.0x ATR
    atr_series=atr,
)
```

**If triple-barrier is too complex for this session:** Do the one-line fallback in `generate_labels()` at line 98:

```python
# Change line 119 from:
labels = (future_return > 0).astype(int)
# To:
labels = (future_return > 0.02).astype(int)  # Only count 2%+ moves as "up"
```

This alone reduces label noise significantly.

### Step 4: Replace 70/30 Split with Purged K-Fold CV

**Existing code to use:** `src/ml/purged_cv.py` lines 32-194

In `train_final_model()` or `_run_training()`, replace the single split:

```python
from src.ml.purged_cv import PurgedKFold

cv = PurgedKFold(n_splits=5, pct_embargo=0.05, label_span=args.horizon)
fold_aucs = []
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    classifier = PatternClassifier(...)
    result = classifier.train(X_train, y_train, calibration_data=(X_test, y_test))
    fold_aucs.append(result.test_auc)
    logger.info(f"Fold {fold_idx} AUC: {result.test_auc:.4f}")

logger.info(f"PurgedKFold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
```

### Step 5: Retrain All 6 Instruments

```bash
uv run scripts/train_ml_model.py --symbol data/raw/SPY_daily.csv --horizon 5 --suffix v2_overfit_fix
uv run scripts/train_ml_model.py --symbol data/raw/QQQ_daily.csv --horizon 5 --suffix v2_overfit_fix
uv run scripts/train_ml_model.py --symbol data/raw/CRVL_daily.csv --horizon 5 --suffix v2_overfit_fix
uv run scripts/train_ml_model.py --symbol data/raw/KODK_daily.csv --horizon 5 --suffix v2_overfit_fix
uv run scripts/train_ml_model.py --symbol data/raw/HIFS_daily.csv --horizon 5 --suffix v2_overfit_fix
uv run scripts/train_ml_model.py --symbol data/raw/JOE_daily.csv --horizon 5 --suffix v2_overfit_fix
```

---

## Success Criteria

| Metric | V1 (Failed) | V2 Target |
|--------|------------|-----------|
| Train AUC | 0.999 | 0.60-0.75 |
| Overfit gap | 0.44-0.52 | < 0.15 |
| Calibration error | 0.24 | < 0.15 |
| PurgedKFold AUC std | N/A (single split) | < 0.05 |

If these thresholds are met, the signal is real enough to retest H13.

If train AUC drops to ~0.55 and test AUC drops to ~0.50 even with these fixes, then the 2015-2026 price data doesn't contain a learnable 5-day directional signal — we need a different approach (cross-asset features, different horizons, or meta-labeling).

---

## Files to Modify

| File | What to change | Lines |
|------|---------------|-------|
| `scripts/train_ml_model.py` | Add IC filtering after feature extraction | ~509 |
| `scripts/train_ml_model.py` | Change CatBoost params in `train_final_model()` | 267-276 |
| `scripts/train_ml_model.py` | Replace `generate_labels()` with triple-barrier (or thresholded binary) | ~514 |
| `scripts/train_ml_model.py` | Replace 70/30 split with PurgedKFold in `train_final_model()` | 259-263 |
| `scripts/train_ml_model.py` | Same CatBoost param change in walk-forward `_run_training()` | ~504 |

Alternatively: fork `train_ml_model.py` → `train_ml_model_v2.py` and make all changes there. Then the original is preserved for reference.

---

## Related Documents

- `docs/research_logic_map/ml_training_plan.md` — full training plan (Phases 0-4)
- `docs/research_logic_map/market_efficiency_and_instrument_selection.md` — instrument selection criteria, current universe assessment
- `docs/research_logic_map/hypothesis_backlog.md` — H13 (inefficient instruments yield higher ML alpha), H14 (cross-asset features improve SPY)
- `docs/research_logic_map/insight_registry.md` — I14-I17 (market efficiency and alpha decay insights)
