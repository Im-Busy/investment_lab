# Meta-Labeling for JOE and KODK — Plan

**Created:** 2026-05-09
**Prerequisite:** Cross-asset feature experiment (completed — JOE +0.15 AUC, KODK +0.07 AUC)
**Estimated effort:** 1-2 AI sessions

---

## 1. Rationale

JOE and KODK are the two instruments where cross-asset features produced meaningful AUC improvement. However, both still have overfit gaps above 0.15 (JOE: 0.164, KODK: 0.264). This means the primary model still memorizes some patterns.

**Meta-labeling** is a technique from Marcos López de Prado's *Advances in Financial Machine Learning*. Instead of trying to fix the primary model, you train a *second* model that predicts: "Given that the primary model says BUY, will this specific trade actually be profitable?"

This turns the problem from "predict market direction" (very hard) to "filter out bad trades from good signals" (more tractable).

## 2. How Meta-Labeling Works

```
Step 1: Primary model generates BUY/SELL signals
Step 2: For each signal, record features at signal time + market context
Step 3: Label each signal: was the resulting trade profitable? (binary)
Step 4: Train a SECOND model to predict "will this signal make money?"
Step 5: Only take trades where BOTH models agree
```

The meta-model acts as a quality filter. It learns patterns like:
- "When RSI is over 70 AND SPY is below its 50-day MA, JOE buy signals usually fail"
- "When volatility is low AND QQQ is trending up, KODK signals are reliable"

## 3. Implementation Steps

### 3.1 Generate Primary Signals
- Load `models/pattern_classifier_*_v3_ca_JOE_catboost.pkl`
- Load `models/pattern_classifier_*_v3_ca_KODK_catboost.pkl`
- Generate BUY/SELL signals on full date range

### 3.2 Build Meta-Label Dataset
For each signal:
```python
{
    "instrument": "JOE",
    "signal_date": "2022-03-15",
    "primary_confidence": 0.72,       # Model's confidence score
    "features_at_signal": {...},       # All 69-82 features at signal time
    "market_context": {                # Cross-asset features at signal time
        "SPY_trend_50d": 0.03,
        "SPY_vol_percentile_60d": 0.45,
        ...
    },
    "label": 1 if trade_was_profitable else 0
}
```

### 3.3 Train Meta-Model
- Model: CatBoost (same architecture, smaller depth)
- Features: primary confidence + instrument features + market context
- Label: binary (trade profitable or not)
- CV: PurgedKFold to avoid leakage
- Target: AUC > 0.55 for trade filtering to be useful

### 3.4 Evaluate
- Run primary model alone → record trades and P&L
- Run primary + meta-model filter → record trades and P&L
- Compare: fewer trades, higher win rate, better Sharpe

## 4. Existing Code

The project already has a `MetaLabeler` class in `src/ml/meta_labeler.py`. Check if it can be reused or needs adaptation for this specific use case.

Check: `uv run pytest tests/test_meta_labeler.py -v`

## 5. Success Criteria

- Meta-model Test AUC > 0.55 (can distinguish good from bad trades)
- Filtered trades have win rate at least 5% higher than unfiltered
- At least 20 trades remain after filtering (enough for statistical validity)
- Sharpe improvement of at least 0.1

## 6. Risks

- **Too few trades:** If the meta-model filters too aggressively, sample size collapses
- **Overfitting on small sample:** Meta-labeling has fewer samples than primary modeling
- **Leakage:** Must ensure the meta-model doesn't use future information about trade outcomes

## 7. Next Step After This

If meta-labeling works → apply to all 6 instruments, build ensemble.
If meta-labeling fails → try simpler approach: confidence threshold tuning (only trade when primary confidence > 0.65).
