# Next Session: Fix ML Overfitting

## Context

Trained CatBoost on 6 instruments (SPY, QQQ, CRVL, KODK, HIFS, JOE) via `scripts/train_ml_model.py`. Every model produced train AUC ~1.0, test AUC ~0.5 — completely overfit. All artifacts deleted.

The training script hardcodes depth=6, trees=200, lr=0.05 with zero regularization, uses 132 features with a noisy binary 5-day label, and a single 70/30 chronological split.

Four fixes needed, using code that ALREADY EXISTS in the project (no new dependencies):

1. **IC feature filtering** → `src/ml/features.py:381` `filter_low_ic_features()`
2. **Regulate CatBoost** → `src/ml/pattern_classifier.py:62-77` (l2_leaf_reg=10, random_strength=3, max_depth=3, min_data_in_leaf=50)
3. **Better labels** → `src/ml/triple_barrier.py:60` `TripleBarrierLabeler` (or simple threshold: `return > 0.02` instead of `return > 0`)
4. **Purged K-Fold CV** → `src/ml/purged_cv.py:32` `PurgedKFold(n_splits=5, pct_embargo=0.05, label_span=5)`

## Task

Modify `scripts/train_ml_model.py` to wire in all 4 fixes. Fork to `train_ml_model_v2.py` if preferred (preserve original).

### Specific changes in order:

**After `extract_features()` (~line 509):**
```python
from src.ml.features import FeatureEngineer
engineer = FeatureEngineer()
features_with_labels = engineer.generate_features_with_labels(df)
filtered = engineer.filter_low_ic_features(
    features_with_labels, forward_returns="forward_return_5d",
    min_abs_ic=0.02, min_abs_rank_ic=0.02,
)
```

**In `train_final_model()` (~line 267):**
Change `PatternClassifier(...)` to use: `n_estimators=100, max_depth=3, learning_rate=0.03, l2_leaf_reg=10.0, random_strength=3.0, min_data_in_leaf=50`

**Replace `generate_labels()` (~line 514):**
Fast option: change `(future_return > 0)` to `(future_return > 0.02)` in line 119.
Full option: use `TripleBarrierLabeler` with `atr_mult_tp=1.5, atr_mult_sl=1.0, time_limit=10`.

**Replace 70/30 split (~line 259):**
Use `PurgedKFold(n_splits=5, pct_embargo=0.05, label_span=5)` — report mean ± std AUC across folds.

### Then retrain:
```bash
uv run scripts/train_ml_model_v2.py --symbol data/raw/SPY_daily.csv --horizon 5 --suffix v2
uv run scripts/train_ml_model_v2.py --symbol data/raw/QQQ_daily.csv --horizon 5 --suffix v2
uv run scripts/train_ml_model_v2.py --symbol data/raw/CRVL_daily.csv --horizon 5 --suffix v2
uv run scripts/train_ml_model_v2.py --symbol data/raw/KODK_daily.csv --horizon 5 --suffix v2
uv run scripts/train_ml_model_v2.py --symbol data/raw/HIFS_daily.csv --horizon 5 --suffix v2
uv run scripts/train_ml_model_v2.py --symbol data/raw/JOE_daily.csv --horizon 5 --suffix v2
```

## Success Criteria

- Train AUC: 0.60-0.75 (down from 0.999)
- Overfit gap: < 0.15 (down from 0.44+)
- Calibration error: < 0.15 (down from 0.24)
- PurgedKFold AUC std: < 0.05

If met → proceed to H13 test (compare inefficient vs efficient AUC).
If train AUC drops to ~0.55 and test ~0.50 → signal doesn't exist in price data alone; need cross-asset features or meta-labeling.

## Reference Docs

- `docs/research_logic_map/ml_training_plan.md` — full plan with Phase 1 results
- `docs/research_logic_map/handover_overfitting_fix.md` — detailed handover with line numbers
- `docs/research_logic_map/hypothesis_backlog.md` — H13/H14 status
- `data/raw/` — CRVL_daily.csv, KODK_daily.csv, HIFS_daily.csv, JOE_daily.csv already downloaded
