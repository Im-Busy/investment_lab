# Pioneer Research Phase 6b — Operational Guide

## Overview

Phase 6b systematically explores deferred experimental trading features using a
**prove-or-discard** model. Each task gets a minimal viable implementation,
evaluated against baseline metrics, then either promoted to full integration or
documented as a negative result.

**LGBM-First Principle:** All new ML components default to LightGBM. CatBoost is
only substituted when it demonstrates ≥10% improvement over LGBM on the task's
primary metric (AUC for classification, R² for regression) in PurgedKFold
cross-validation.

---

## Tier 1 Components (Implemented)

| Component | File | Purpose | Model |
|-----------|------|---------|-------|
| P1.1 MetaLabeler | `src/ml/meta_labeler.py` | Filter low-quality signals | LGBMClassifier (default) |
| P1.2 GapFillPredictor | `src/ml/gap_fill_predictor.py` | Predict gap fill probability | LGBMClassifier |
| P1.3 Ablation Study | `scripts/ablate_patterns.py` | Rank 35 pattern detectors | No ML — pure backtest |

---

## P1.3 — Pattern Detector Ablation Study

### Purpose

Systematically measure which of the 35 pattern detectors produce edge by
running each in isolation through a standard backtest. Identifies
underperforming patterns for removal or retuning before building ML features
around them.

### Quick Start

```bash
# Single symbol (SPY, 2019-2024)
uv run scripts/ablate_patterns.py --symbol SPY --start 2019-01-01 --end 2024-12-31

# Bitcoin
uv run scripts/ablate_patterns.py --symbol BTC-USD --start 2020-01-01 --end 2024-12-31

# Both SPY and BTC
uv run scripts/ablate_patterns.py --all --start 2019-01-01 --end 2024-12-31

# Custom output directory
uv run scripts/ablate_patterns.py --symbol SPY --output-dir outputs/pioneer
```

### What It Does

1. Fetches OHLCV data for the specified symbol via yfinance
2. Instantiates all 35 pattern detectors across 7 categories
3. Runs each pattern in isolation through a simple backtest:
   - Entry on signal, exit on TP/SL from the pattern's own signal
   - 10% position sizing, $10K starting capital
   - No regime gating, no confidence filtering, no position probability
4. Computes per-pattern metrics: Sharpe, Sortino, win rate, profit factor,
   max drawdown, trade count, PnL, average bars held
5. Flags patterns with recommendations: **keep**, **retune**, or **cut**

### Flagging Criteria

| Criterion | Recommendation | Meaning |
|-----------|:---:|---------|
| `< 30 trades` | **cut** | Not enough data to evaluate |
| `Sharpe ≤ 0` | **cut** | Negative or zero risk-adjusted return |
| `Profit factor < 1.0` | **cut** | Losses exceed profits |
| `Win rate < 35%` | **retune** | Edge exists but signal quality is low |
| `Max drawdown < -25%` | **retune** | Risk management needs adjustment |
| All pass | **keep** | Pattern produces measurable edge |

### Output Files

| File | Location | Contents |
|------|----------|----------|
| CSV | `reports/pattern_ablation_{SYMBOL}.csv` | Full metrics for all 35 patterns |
| Console | stdout | Top 20 patterns + flagged lists |

### Interpreting Results

**Patterns to keep** (Sharpe > 0, PF > 1.0, ≥ 30 trades): These are the
foundation. Build ML features around them. Use them in meta-labeling training.

**Patterns to retune** (low win rate or high drawdown): Adjust detector
parameters (lookback, threshold, TP/SL multipliers). Re-run ablation to
verify improvement.

**Patterns to cut** (negative Sharpe, PF < 1.0, or too few trades):
Document in `reports/pioneer_results.md`. Consider removing from the
production signal pipeline. Don't waste ML compute on these.

### Example Output

```
================================================================================
  Pattern Ablation Results — SPY
================================================================================
  Total patterns: 35
  Keep:  12 (Sharpe > 0, PF > 1.0, >= 30 trades)
  Retune: 8 (low win rate or high drawdown)
  Cut:  15 (negative Sharpe, PF < 1.0, or too few trades)
================================================================================

  Pattern                        Cat            Trades   Win%    PF   Sharpe    DD%     Rec
  ------------------------------ -------------- ------- ------ ------ -------- ------ --------
  Double Bottom                  Classic           85  48.2%   1.85   0.723  12.3%     keep
  Cup and Handle                 Complex           62  52.1%   1.62   0.654  15.1%     keep
  ...

  Flagged for removal (15):
    ✗ Doji: too few trades (8)
    ✗ Harami: negative/zero Sharpe (-0.234)

  Flagged for retuning (8):
    ~ Donchian Channel Breakout: large max drawdown (-28.5%)
```

---

## P1.1 — MetaLabeler (T9)

### Purpose

Answer "Should I take this signal?" independently from "which direction?".
Trains on historical trade outcomes to predict whether a signal will hit
take-profit before stop-loss within a time limit.

### Architecture

```
Historical Signals + OHLCV data
  → TripleBarrierLabeler (labels: did TP hit before SL?)
  → Build context features at signal time
  → LGBMClassifier / CatBoostClassifier (PurgedKFold CV)
  → MetaLabeler.predict(signal) → {take_trade: bool, probability: float}
```

### Quick Start

```python
from src.ml.meta_labeler import MetaLabeler
import pandas as pd

# Load your OHLCV data
df = pd.read_csv("data/raw/SPY_daily.csv", index_col=0, parse_dates=True)

# Load historical signals from your backtest
signals_df = pd.read_csv("outputs/historical_signals.csv", parse_dates=["timestamp"])

# Train the meta-labeler
ml = MetaLabeler(tp_atr_mult=2.0, sl_atr_mult=1.5, time_limit=20)
result = ml.fit(df, signals_df)

# Check what model was selected and its AUC
print(f"Model: {result.model_used}")
print(f"LGBM AUC: {result.lgbm_auc:.4f}")
print(f"CatBoost AUC: {result.catboost_auc}")
print(f"Baseline win rate: {result.baseline_win_rate:.2%}")
print(f"Top features: {list(result.feature_importance.keys())[:5]}")

# Predict on a new signal
context = {
    "pattern_confidence": 0.7,
    "atr_ratio": 0.02,
    "rsi_value": 45.0,
    "trend_20": 0.03,
    "trend_50": 0.05,
    "volatility_20": 0.015,
    "volume_ratio": 1.2,
    "relative_position": 0.6,
    "day_of_week": 2.0,
    "is_reversal": 1.0,
    "is_breakout": 0.0,
}
prediction = ml.predict(context)
print(f"Take trade: {prediction.take_trade}")
print(f"Probability: {prediction.probability:.4f}")
print(f"Confidence: {prediction.confidence:.4f}")

# Save model for later use
ml.save("models/meta_labeler_v1.pkl")
```

### Context Features (12 total)

| Feature | Description | Source |
|---------|-------------|--------|
| `pattern_confidence` | Original signal confidence (0-1) | Signal metadata |
| `atr_ratio` | ATR(14) / Close price | OHLCV |
| `rsi_value` | 14-bar RSI (0-100) | OHLCV |
| `trend_20` | 20-bar return (%) | OHLCV |
| `trend_50` | 50-bar return (%) | OHLCV |
| `volatility_20` | 20-bar std dev of returns | OHLCV |
| `volume_ratio` | Volume / 20-bar avg volume | OHLCV |
| `relative_position` | (Close - 50-bar Low) / (50-bar High - 50-bar Low) | OHLCV |
| `day_of_week` | 0=Monday, 4=Friday | Timestamp |
| `hour_of_day` | Hour of signal (intraday only) | Timestamp |
| `is_reversal` | 1 if pattern name contains reversal keywords | Pattern name |
| `is_breakout` | 1 if pattern name contains breakout keywords | Pattern name |

### Model Selection Logic

```
1. Train LGBMClassifier with PurgedKFold (5-fold, 5% embargo)
2. Train CatBoostClassifier with PurgedKFold (same splits)
3. If CatBoost mean AUC >= 1.10 × LGBM mean AUC:
     → Use CatBoost
   Else:
     → Use LGBM
4. Find optimal probability threshold via Youden's J statistic
5. Return MetaLabelResult with all metrics
```

### Integration Pattern

Add meta-labeling to your signal pipeline:

```python
from src.ml.meta_labeler import MetaLabeler

# After training
meta_labeler = MetaLabeler()
meta_labeler.fit(historical_df, historical_signals)

# In your signal processing loop
def process_signal(signal, market_data):
    context = build_context_features(signal, market_data)
    prediction = meta_labeler.predict(context)
    if prediction.take_trade:
        execute_trade(signal, meta_confidence=prediction.confidence)
    else:
        log_skipped_signal(signal, reason="meta-labeler rejected")
```

### Success Criteria

| Metric | Threshold | Meaning |
|--------|:---------:|---------|
| AUC | > 0.55 | Better than coin flip |
| P&L improvement | > 0% | Improves over no meta-labeling baseline |
| False signal rejection | Measure | % of losing signals correctly filtered |

### Running Tests

```bash
uv run pytest tests/test_meta_labeler.py -v
```

---

## P1.2 — GapFillPredictor (FS19)

### Purpose

Predict whether an overnight or weekend gap will fill (price retraces to
previous close) within N bars. Useful for gap-fill trading strategies,
especially in crypto where weekend liquidity gaps are common.

### Architecture

```
OHLCV data
  → Gap detection (close[t] vs open[t+1] exceeding threshold)
  → Build context features around each gap
  → LGBMClassifier (PurgedKFold CV)
  → GapFillPredictor.predict(df, bar_index)
      → {fill_probability: float, estimated_bars: int}
```

### Quick Start

```python
from src.ml.gap_fill_predictor import GapFillPredictor
import pandas as pd

# Load data
df = pd.read_csv("data/raw/SPY_daily.csv", index_col=0, parse_dates=True)

# Train the predictor
gfp = GapFillPredictor(look_forward=5, gap_threshold_pct=0.1)
result = gfp.fit(df)

print(f"AUC: {result.auc_roc:.4f}")
print(f"Accuracy: {result.accuracy:.4f}")
print(f"Gaps detected: {result.n_gaps}")
print(f"Filled: {result.n_filled} ({result.baseline_fill_rate:.1%})")
print(f"Top features: {list(result.feature_importance.keys())[:5]}")

# Predict fill for a specific gap
gaps = gfp.detect_gaps(df)
for gap in gaps[-5:]:  # last 5 gaps
    pred = gfp.predict(df, gap.bar_index)
    print(
        f"Gap at bar {gap.bar_index}: {gap.gap_type} {gap.gap_pct:.2f}% "
        f"→ fill prob: {pred.fill_probability:.2%}, est. bars: {pred.estimated_bars}"
    )

# Batch predict all gaps
predictions = gfp.predict_batch(df)
print(predictions.head())

# Save model
gfp.save("models/gap_fill_v1.pkl")
```

### Context Features (11 total)

| Feature | Description |
|---------|-------------|
| `gap_pct_abs` | Absolute gap size (%) |
| `gap_direction` | +1 = gap up, -1 = gap down |
| `pre_trend_5` | 5-bar return before gap |
| `pre_trend_20` | 20-bar return before gap |
| `volatility_20` | 20-bar std dev of returns |
| `volume_ratio` | Volume / 20-bar avg volume at gap bar |
| `atr_ratio` | ATR(14) / Close |
| `relative_position` | (Close - 50-bar Low) / (50-bar High - 50-bar Low) |
| `day_of_week` | 0=Monday through 4=Friday |
| `is_monday` | 1 if Monday gap (weekend gap) |
| `pre_gap_range` | Pre-gap bar range as % of close |

### Gap Detection Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `look_forward` | 5 | Bars to check for fill |
| `gap_threshold_pct` | 0.1% | Minimum gap size to consider |
| `n_splits` | 5 | PurgedKFold splits |
| `pct_embargo` | 0.05 | Embargo fraction |

### Configuring for Different Markets

```python
# Equities (overnight gaps, default parameters work well)
gfp = GapFillPredictor(look_forward=5, gap_threshold_pct=0.1)

# Crypto (24/7, smaller gaps, more false positives)
gfp = GapFillPredictor(look_forward=10, gap_threshold_pct=0.3)

# Forex (tiny gaps, need very low threshold)
gfp = GapFillPredictor(look_forward=3, gap_threshold_pct=0.05)
```

### Success Criteria

| Metric | Threshold | Meaning |
|--------|:---------:|---------|
| AUC | > 0.55 | Better than coin flip |
| Accuracy | > baseline fill rate | Beats naive "always predict fill" |
| F1 | > 0.5 | Balanced precision/recall |

### Running Tests

```bash
uv run pytest tests/test_gap_fill_predictor.py -v
```

---

## Results Tracking — pioneer_results.md

All results are tracked in `reports/pioneer_results.md`. After running any task:

### 1. Ablation Study Results

After running the ablation script, transfer the key findings:

```markdown
### P1.3 — Pattern Detector Ablation

**Status:** Completed
**Date:** 2026-05-08
**Symbol:** SPY (2019-2024)

| Pattern | Category | Trades | Win% | PF | Sharpe | DD% | Rec |
|---------|----------|--------|------|-----|--------|-----|-----|
| Double Bottom | Classic | 85 | 48.2% | 1.85 | 0.723 | 12.3% | keep |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Key Findings:**
- 12 patterns worth keeping
- 8 patterns need retuning
- 15 patterns recommended for removal
- Full CSV: `reports/pattern_ablation_SPY.csv`
```

### 2. MetaLabeler Results

After training and evaluating:

```markdown
### P1.1 — Meta-Labeling (T9)

**Status:** Completed
**Model:** LightGBM
**PurgedKFold AUC:** 0.682
**Verdict:** PROMOTE

| Metric | Value |
|--------|-------|
| LGBM AUC (CV mean) | 0.682 |
| CatBoost AUC (CV mean) | 0.691 |
| Model Selected | LightGBM (CatBoost only +1.3%) |
| Baseline Win Rate | 48.2% |
| Optimal Threshold | 0.52 |
| N Signals | 1,247 |

**Key Findings:**
- Meta-labeler improves P&L by 11.3% vs no filtering
- Top feature: pattern_confidence (unsurprisingly)
- Second top: trend_20 (signals in strong trends are more reliable)
```

### 3. GapFillPredictor Results

After training and evaluating:

```markdown
### P1.2 — Gap-Fill Prediction (FS19)

**Status:** Completed
**Model:** LightGBM
**AUC:** 0.612
**Verdict:** PROMOTE (marginal)

| Metric | Value |
|--------|-------|
| AUC | 0.612 |
| Accuracy | 58.3% |
| F1 Score | 0.56 |
| Baseline Fill Rate | 52.1% |
| N Gaps | 1,892 |

**Key Findings:**
- Small edge over baseline (+6.2% accuracy)
- Monday gaps fill more reliably (73% vs 58% overall)
- Gap size is the strongest predictor
```

### Verdict Guidelines

| Verdict | Criteria | Action |
|---------|----------|--------|
| **PROMOTE** | Meets all success criteria | Integrate into production pipeline, add to Phase 06 |
| **PROMOTE (marginal)** | Meets some criteria, small edge | Optional integration, document caveats |
| **DISCARD** | Fails success criteria | Write Research Note, close task, no further work |

### Research Notes for Discarded Tasks

```markdown
### P1.1 — Meta-Labeling (T9)

**Status:** Completed
**Model:** LightGBM
**PurgedKFold AUC:** 0.52
**Verdict:** DISCARD

**What was tried:**
- Trained on 847 signals from 2019-2024 SPY data
- 12 context features including pattern confidence, RSI, ATR, trend, volume
- PurgedKFold (5-fold, 5% embargo) with both LGBM and CatBoost

**Why it didn't work:**
- AUC of 0.52 is essentially random (coin flip)
- No single feature had IC > 0.02
- CatBoost AUC = 0.53, only +1.9% over LGBM (below 10% threshold)

**Hypothesis for failure:**
- Pattern confidence already encodes most of the signal quality
- Market context at signal time doesn't add predictive power
- TripleBarrierLabeler with fixed ATR multipliers may not capture
  per-signal dynamics accurately

**Recommendation:**
Close task. No further meta-labeling work until (a) pattern confidence
scoring is improved, or (b) dynamic barrier parameters are available.
```

---

## Workflow: Running a Full Tier 1 Evaluation

Complete end-to-end workflow for evaluating Tier 1 on a new symbol:

```bash
# Step 1: Run ablation to identify which patterns to focus on
uv run scripts/ablate_patterns.py --symbol QQQ --start 2019-01-01 --end 2024-12-31

# Step 2: Review results in reports/pattern_ablation_QQQ.csv
# Identify keep/retune/cut patterns

# Step 3: Extract signals from historical backtest for keep patterns only
# (Requires your backtest pipeline — example)
uv run python -c "
import pandas as pd
# Load your historical signals CSV or generate from backtest
signals = pd.read_csv('outputs/qqq_signals_2019_2024.csv')
# Filter to keep patterns from ablation
keep_patterns = ['Double Bottom', 'Cup and Handle', 'Flag', 'Pennant']
signals = signals[signals['pattern_name'].isin(keep_patterns)]
signals.to_csv('outputs/qqq_signals_filtered.csv', index=False)
"

# Step 4: Train meta-labeler on filtered signals
uv run python -c "
from src.ml.meta_labeler import MetaLabeler
import pandas as pd
df = pd.read_csv('data/raw/QQQ_daily.csv', index_col=0, parse_dates=True)
signals = pd.read_csv('outputs/qqq_signals_filtered.csv', parse_dates=['timestamp'])
ml = MetaLabeler()
result = ml.fit(df, signals)
print(f'AUC: {result.lgbm_auc:.4f}, Model: {result.model_used}')
ml.save('models/meta_labeler_qqq.pkl')
"

# Step 5: Train gap-fill predictor
uv run python -c "
from src.ml.gap_fill_predictor import GapFillPredictor
import pandas as pd
df = pd.read_csv('data/raw/QQQ_daily.csv', index_col=0, parse_dates=True)
gfp = GapFillPredictor()
result = gfp.fit(df)
print(f'AUC: {result.auc_roc:.4f}, Gaps: {result.n_gaps}')
gfp.save('models/gap_fill_qqq.pkl')
"

# Step 6: Update pioneer_results.md with findings
# (Manual step — edit reports/pioneer_results.md)
```

---

## File Reference

```
src/ml/
├── meta_labeler.py           # P1.1: MetaLabeler + MetaLabelPrediction + MetaLabelResult
├── gap_fill_predictor.py     # P1.2: GapFillPredictor + GapFillPrediction + GapDetection
├── triple_barrier.py         # TripleBarrierLabeler (used by MetaLabeler)
├── purged_cv.py              # PurgedKFold (used by both)
└── __init__.py               # Exports MetaLabeler, GapFillPredictor, related types

scripts/
└── ablate_patterns.py        # P1.3: Pattern ablation study CLI

tests/
├── test_meta_labeler.py      # 16 tests
└── test_gap_fill_predictor.py # 18 tests

reports/
└── pioneer_results.md        # Aggregated results tracker

docs/
├── guide-pioneer-research.md # THIS FILE
└── COMMAND_CHEATSHEET.md     # Updated with Pioneer Research commands
```

---

## Common Issues

### "No module named 'lightgbm'"

```bash
uv add lightgbm
```

### "No module named 'catboost'"

```bash
uv add catboost
```

### "No module named 'yfinance'"

```bash
uv add yfinance
```

### MetaLabeler returns AUC = 0.5

This means the model is no better than random. Possible causes:
- Too few signals (< 50 labeled samples)
- All labels are the same class (all profitable or all unprofitable)
- Context features don't capture predictive information
- TripleBarrierLabeler parameters don't match your trade mechanics

Debug by checking `result.baseline_win_rate` — if it's 0.0 or 1.0, you have
a labeling problem. Adjust `tp_atr_mult` and `sl_atr_mult`.

### GapFillPredictor detects too few gaps

- Lower `gap_threshold_pct` (try 0.05 or 0.0)
- Ensure your data has actual overnight/weekend gaps
- Intraday data typically has fewer gaps; increase `look_forward`

### Ablation script "Error detecting X at bar Y"

Pattern detectors may fail on certain bars. The script catches errors
and continues. These are logged but don't affect the results for other
patterns. If a pattern consistently errors, flag it for removal.

---

## Next Steps

After Tier 1 evaluation:

1. **If MetaLabeler AUC > 0.55:** Integrate into the production signal
   pipeline. Train on all historical signals and save model.

2. **If GapFillPredictor AUC > 0.55:** Add gap-fill probability as a
   filter before taking gap-related signals.

3. **After ablation:** Remove "cut" patterns from the production pipeline.
   Re-run ablation on "retune" patterns after parameter adjustments.

4. **Proceed to Tier 2** only if Tier 1 produces positive results.
   Tier 2 tasks (Shapelets, VAE, Heikin-Ashi) have higher effort and
   should only be explored if core signal filtering (Tier 1) is working.

5. **Document everything** in `reports/pioneer_results.md` regardless of
   outcome. Negative results are valuable — they prevent repeating
   dead-end research paths.
