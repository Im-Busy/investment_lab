# ML Training Plan: Instrument Efficiency & Alpha Discovery

**Created:** 2026-05-09
**Status:** Planning — execution pending
**Tests Hypotheses:** H13 (inefficient instruments yield higher ML alpha), H14 (cross-asset features improve SPY)
**Related:** `market_efficiency_and_instrument_selection.md`, `insight_registry.md` (I14-I17)

---

## What You Should Train

### The Core Insight

You have two groups of instruments:

| Group | Instruments | Efficiency | Expected ML Difficulty |
|-------|------------|------------|----------------------|
| **Efficient (control)** | SPY, QQQ | Highest | Hard — alpha decays fast |
| **Inefficient (test)** | CRVL, KODK, HIFS, JOE | Low | Easier — alpha persists |

The DeMiguel (2024) research says ML should find **stronger predictability** on the inefficient group. This plan tests that directly.

### What "Training" Means Here

The existing pipeline (`scripts/train_ml_model.py`) does this:

1. Loads OHLCV data from CSV (or downloads via yfinance)
2. Extracts ~130 features (price, momentum, volatility, volume, pattern shapes, regime)
3. Creates binary label: "will the price be higher 5 days from now?"
4. Splits chronologically: first 70% train, last 30% test (no look-ahead)
5. Trains a **CatBoost** classifier (gradient boosting optimized for tabular data)
6. Validates with walk-forward expanding window
7. Reports: AUC, accuracy, calibration error, feature importance

The output is a model that answers: **"Given what I know about this instrument's recent price behavior, what's the probability it goes up in the next 5 days?"**

The AUC (Area Under ROC Curve) is the key metric:
- **0.50** = coin flip (no predictive power)
- **0.55** = weak edge
- **0.60** = meaningful edge
- **0.65+** = strong predictive signal

---

## Phase 1: Single-Instrument Baselines (Direct H13 Test)

**Goal:** Prove inefficient instruments are more predictable than efficient ones.

**Hypothesis:** CRVL, KODK, HIFS, JOE will produce AUC >= 0.55 with statistical significance. SPY and QQQ will produce AUC ~0.50-0.52 (barely above random).

### Commands

```bash
# Train on each new instrument (daily data, 2015-2026)
uv run scripts/train_ml_model.py --symbol data/raw/CRVL_daily.csv --horizon 5
uv run scripts/train_ml_model.py --symbol data/raw/KODK_daily.csv --horizon 5
uv run scripts/train_ml_model.py --symbol data/raw/HIFS_daily.csv --horizon 5
uv run scripts/train_ml_model.py --symbol data/raw/JOE_daily.csv --horizon 5

# Train on control instruments for comparison
uv run scripts/train_ml_model.py --symbol data/raw/SPY_daily.csv --horizon 5
uv run scripts/train_ml_model.py --symbol data/raw/QQQ_daily.csv --horizon 5

# Also test with different horizons
uv run scripts/train_ml_model.py --symbol data/raw/CRVL_daily.csv --horizon 10
uv run scripts/train_ml_model.py --symbol data/raw/CRVL_daily.csv --horizon 20
```

### What to Look For

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Test AUC | > 0.55 on inefficient; < 0.53 on SPY | All ~0.50 (features don't work) |
| Overfit gap (train - test AUC) | < 0.10 | > 0.15 (overfitting) |
| Calibration error | < 0.10 | > 0.20 (probabilities unreliable) |
| Top features | Mix of momentum + volume + pattern | Only one feature dominates (fragile) |

### How to Read Feature Importance

After training, the pipeline saves a `feature_importance_*.png` plot. The top features tell you *what the model found*:

- **return_5 / momentum_20 dominating** → Simple trend-following (fragile, likely to decay)
- **volume_ratio / obv / vwap features** → Volume patterns matter (more durable)
- **atr_pct / volatility features** → Volatility regime matters (defensive signal)
- **pattern features (doji, engulfing, higher_high)** → Chart patterns contribute (your original thesis)
- **Alpha factors (alpha_1, alpha_6, etc.)** → WorldQuant formulas work on this instrument

If momentum dominates: the signal may decay as more people find it.
If volume + pattern + volatility dominate: more durable, less likely to be arbitraged.

---

## Phase 2: Cross-Asset Enhancement (H14 Test)

**Goal:** Show that adding macro/cross-asset features improves predictions, especially on efficient instruments.

**Why this matters:** SPY is "too efficient" for pure price patterns, but SPY doesn't exist in a vacuum. When bonds crash, gold spikes, or VIX surges — SPY reacts. These cross-asset signals inject information that isn't in SPY's own price history.

The cross-asset features already exist in `src/ml/cross_asset_features.py`:
- VIX features (level, return, moving averages, percentile)
- Bond features (TLT/IEF ratio, momentum, volatility)
- Safe haven features (gold return, USD momentum)
- Oil features
- Sector ETF dispersion and breadth
- Momentum divergence (SPY vs TLT, SPY vs GLD, SPY vs VIX)

### How to Add Them

The cross-asset feature module generates features from macro instruments. These need to be merged with the instrument's own features before training. The module `src/ml/cross_asset_features.py` has a `CrossAssetFeatures` class that does this.

**You'll need a modified training script** (I can build this) that:
1. Downloads cross-asset data (VIX, TLT, IEF, GLD, USO, UUP, sector ETFs)
2. Computes cross-asset features
3. Joins them to the instrument's own features
4. Trains with the expanded feature set

### What to Compare

| Configuration | Features | Expected AUC (SPY) |
|--------------|----------|-------------------|
| SPY-only | ~130 price features | ~0.50-0.52 |
| SPY + Cross-Asset | 130 + 35 macro features | 0.53-0.56 |
| CRVL-only | ~130 price features | ~0.55-0.60 |
| CRVL + Cross-Asset | 130 + 35 macro features | 0.57-0.62 |

If cross-asset features add >= 0.02 AUC on SPY: H14 confirmed. If they don't: then SPY truly is "priced in" and we focus exclusively on inefficient instruments.

---

## Phase 3: Multi-Instrument Model (Scale & Generalization)

**Goal:** Train one model across ALL instruments to learn universal patterns.

**Why:** TradeFM (2025) showed that training on thousands of diverse assets across different liquidity regimes produces models that generalize better. The model learns features that work *regardless* of instrument — separating instrument-specific noise from universal signal.

**How:**
1. Stack all instrument data (SPY, QQQ, CRVL, KODK, HIFS, JOE, GLD, TLT, BTC, etc.)
2. Add `instrument_id` as a categorical feature
3. Split chronologically per-instrument (no look-ahead across instruments either)
4. Train one CatBoost model
5. The model learns: "when these patterns appear on ANY instrument, what happens?"

**What to look for:**
- If multi-instrument AUC >= best single-instrument AUC: model is generalizing well
- If multi-instrument AUC < best single-instrument: instruments are too different, train separately
- Feature importance should show both universal features AND instrument-specific ones

---

## Phase 4: Transfer Learning (Optional, Advanced)

**Goal:** Pre-train on all instruments, fine-tune on specific ones.

**Why:** The model learns general market dynamics from all instruments, then specializes to the quirks of a specific one. Like a doctor who trains in general medicine before specializing in cardiology.

**How (pseudocode):**
1. Train CatBoost on stacked multi-instrument data
2. Save the model
3. Load model, continue training (with lower learning rate) on single instrument
4. Compare: fine-tuned vs. from-scratch on that instrument

---

## How to Interpret Results

### AUC by Instrument Type

```
Expected pattern if H13 confirmed:

 0.62 |                                    ■ CRVL
 0.60 |                           ■ KODK
 0.58 |                  ■ HIFS
 0.56 |           ■ JOE
 0.54 |      ■ QQQ
 0.52 | ■ SPY
      +-------------------------------------
        Efficient ← → Inefficient
```

If you see this: your instrument selection thesis is validated. ML edge exists where markets are less efficient.

### Warning Signs

| Observation | What It Means | Action |
|-------------|--------------|--------|
| Train AUC >> Test AUC (gap > 0.15) | Overfitting | Reduce model complexity (lower depth, more regularization) |
| All AUC ~0.50 | Features don't capture signal | Try different horizons, labels, or add cross-asset features |
| Test AUC drops over time | Alpha decay in action | Retrain more frequently, monitor rolling AUC |
| Only 1-2 features have high importance | Model is fragile | Add more diverse features, reduce reliance on single signal |
| Walk-forward AUC degrades in recent period | Strategy decay | Consider retirement if below threshold for 60 days |

---

## Execution Order

### Immediate (Today)

```bash
# Phase 1: Train baselines on all 6 instruments
uv run scripts/train_ml_model.py --symbol data/raw/SPY_daily.csv --horizon 5
uv run scripts/train_ml_model.py --symbol data/raw/QQQ_daily.csv --horizon 5
uv run scripts/train_ml_model.py --symbol data/raw/CRVL_daily.csv --horizon 5
uv run scripts/train_ml_model.py --symbol data/raw/KODK_daily.csv --horizon 5
uv run scripts/train_ml_model.py --symbol data/raw/HIFS_daily.csv --horizon 5
uv run scripts/train_ml_model.py --symbol data/raw/JOE_daily.csv --horizon 5
```

### After Reviewing Phase 1 Results

1. Check AUC, overfit gap, calibration per instrument
2. Identify which features are most important per instrument
3. If inefficient instruments show higher AUC: proceed to Phase 2
4. If not: investigate why (check data quality, feature coverage, label construction)

### Phase 2 (After Phase 1 confirms)

Build cross-asset training script (separate task), then:
```bash
uv run scripts/train_ml_cross_asset.py --symbol SPY --horizon 5  # (to be built)
uv run scripts/train_ml_cross_asset.py --symbol CRVL --horizon 5
```

### Phase 3 (After Phase 2)

Build multi-instrument training script (separate task), then:
```bash
uv run scripts/train_ml_multi_instrument.py --symbols SPY,QQQ,CRVL,KODK,HIFS,JOE --horizon 5  # (to be built)
```

---

## Key Concepts (TL;DR for Non-ML Practitioners)

### CatBoost
A type of gradient boosting — builds decision trees sequentially, each tree correcting the mistakes of previous trees. "Ordered boosting" means it uses only past data when making splits, avoiding look-ahead bias. Good default for financial ML because it handles categorical features natively, is robust to noise, and rarely overfits catastrophically.

### AUC (Area Under ROC Curve)
Measures how well the model separates "up" from "down" predictions. 0.50 = random guess. 0.60 = decent edge. 0.70 = strong. Think of it as: "what's the probability the model ranks a random positive example higher than a random negative one?"

### Walk-Forward Validation
Instead of one train/test split, you train on the first N days, test on the next M, then slide the window forward. This simulates real trading: you never know the future, and the model is periodically retrained. If performance degrades in recent windows, your edge may be decaying.

### Overfit Gap
train_AUC - test_AUC. If the model is much better on training data than test data, it memorized the training set rather than learning general patterns. In finance, overfitting is deadly because the future never looks exactly like the past.

### Calibration Error
Measures whether predicted probabilities are reliable. If the model says "80% chance of going up" on 100 trades, did ~80 of them actually go up? Low calibration error = probabilities mean what they say. High error = probabilities are meaningless.

### Feature Importance
Which inputs the model used most to make decisions. If it only uses 1-2 features, the strategy is fragile (one signal disappearing = model breaks). If importance is spread across many features, the strategy is more robust.

---

## What a "Good" Training Result Looks Like

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Test AUC | > 0.55 | Better than random with margin |
| Overfit gap | < 0.10 | Not memorizing noise |
| Calibration error | < 0.10 | Probabilities are trustworthy |
| Walk-forward trend | Stable or improving | Signal not decaying |
| Top feature diversity | Top 5 features < 50% combined importance | Not reliant on single signal |
| Cross-validation consistency | < 0.05 std across folds | Stable across time periods |

If you see all of these on an instrument: you have found a genuine ML edge.
If you see some but not all: the edge exists but needs refinement (different features, labels, or model).
If you see none: the instrument is either too efficient or your features don't capture the relevant patterns.

---

## Next Steps After Training

1. **If inefficient instruments show edge**: Expand the inefficient universe (more small/mid-caps with low coverage)
2. **If cross-asset features help**: Add more macro data sources (credit spreads, currency baskets, commodity indices)
3. **If multi-instrument model generalizes**: Scale to all 85+ instruments referenced in the project
4. **If nothing works**: Revisit feature engineering (different transformations, alternative data sources)
5. **Production path**: Best model → meta-labeler for trade filtering → backtest with costs → paper trade → live

---

*This plan tests H13 and H14 from the hypothesis backlog. Results update `docs/research_logic_map/hypothesis_backlog.md`.
*

---

## Phase 1 Results (2026-05-09) — FAILED: Severe Overfitting

All 6 instruments trained with default CatBoost (depth=6, trees=200, lr=0.05, 132 features, binary 5-day label, 70/30 split).

| Instrument | Group | Test AUC | Accuracy | Calibration Error | Train AUC | Overfit Gap |
|-----------|-------|----------|----------|-------------------|-----------|-------------|
| **SPY** | Efficient | **0.557** | 48.3% | 0.241 | 0.9996 | 0.442 |
| **QQQ** | Efficient | **0.533** | 48.5% | 0.242 | 0.9999 | 0.467 |
| **KODK** | Inefficient | 0.515 | 54.8% | 0.249 | 0.9996 | 0.484 |
| **JOE** | Inefficient | 0.497 | 53.9% | 0.249 | 0.9990 | 0.502 |
| **HIFS** | Inefficient | 0.487 | 47.5% | 0.243 | 0.9994 | 0.512 |
| **CRVL** | Inefficient | 0.484 | 49.2% | 0.250 | 0.9991 | 0.515 |

**Conclusion:** H13 not confirmed. Models are useless — train AUC ~1.0 with test AUC ~0.5 means they memorized noise. All artifacts deleted.

**Root cause:** `scripts/train_ml_model.py:267-276` hardcodes depth=6, trees=200, lr=0.05 with zero regularization. 132 noisy features with a weak binary label give the model enough capacity to memorize the training set perfectly.

**Next:** Fix overfitting first (Phase 0), then retest H13.

---

## Phase 0: Fix Overfitting (Prerequisite Before Retraining)

Four tools exist in the project but `train_ml_model.py` doesn't use them:

### Fix 1: IC-Based Feature Selection
- **Code:** `src/ml/features.py:381` — `FeatureEngineer.filter_low_ic_features()`
- **What:** Computes Pearson + Spearman IC per feature vs forward returns. Drops features with |IC| < 0.02.
- **Expected:** 132 → ~20-40 features. Fewer dimensions = less room to memorize.

### Fix 2: Regulate CatBoost
- **Code:** `src/ml/pattern_classifier.py:62-77` — constructor accepts `l2_leaf_reg`, `random_strength`, `bagging_temperature`, `min_data_in_leaf`
- **Current hardcode** (`train_ml_model.py:267-276`): `n_estimators=200, max_depth=6, learning_rate=0.05, min_child_samples=20, subsample=0.8, colsample_bytree=0.8` — uses CatBoost defaults for l2_leaf_reg=3.0, random_strength=1.0
- **Target params:** `max_depth=3, n_estimators=100, learning_rate=0.03, l2_leaf_reg=10.0, random_strength=3.0, min_data_in_leaf=50`
- **Expected:** Train AUC drops from 0.999 to ~0.65-0.75. If test AUC stays >0.55, signal is real.

### Fix 3: Triple-Barrier Labels
- **Code:** `src/ml/triple_barrier.py:60` — `TripleBarrierLabeler`
- **What:** Instead of "higher in 5 days?" → "hits +1.5×ATR before -1.0×ATR within 10 days?" This mirrors actual trade mechanics and filters out noise (0.1% gain ≠ "win").
- **Fallback if triple-barrier is complex:** Change binary label from `return > 0` to `return > 0.02` (only count 2%+ moves as "up"). Single-line change.
- **Expected:** Higher signal-to-noise ratio. Labels map to real trade outcomes.

### Fix 4: Purged K-Fold Cross-Validation
- **Code:** `src/ml/purged_cv.py:32` — `PurgedKFold(n_splits=5, pct_embargo=0.05, label_span=5)`
- **What:** Standard k-fold leaks data when forward labels overlap test windows. PurgedKFold removes (purges) training samples whose label window touches the test period.
- **Current:** `train_ml_model.py:259-263` uses a single 70/30 chronological split.
- **Expected:** 5-fold mean AUC distribution instead of one noisy point estimate. If AUC 0.55±0.02, signal is stable. If 0.55±0.15, it was luck.

### Combined Impact

| Stage | Overfit Gap |
|-------|------------|
| Current (baseline) | 0.44-0.52 |
| + IC filtering (132→~30 features) | ~0.15-0.25 |
| + Regularization (l2=10, depth=3) | ~0.05-0.15 |
| + Triple-barrier labels | ~0.03-0.10 |
| + PurgedKFold | Single-split noise eliminated |

---

## Handover: Next Session Tasks

See `docs/research_logic_map/handover_overfitting_fix.md` for the detailed handover prompt.
