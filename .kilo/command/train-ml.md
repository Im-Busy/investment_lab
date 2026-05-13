---
description: Train a CatBoost pattern classifier model using the 9-stage ML pipeline with triple-barrier labels and PurgedKFold CV. Supports single-ticker, basket, fast, and walk-forward modes.
---

# Train ML — Model Training Pipeline

Load the **ml-trainer** agent to train a pattern classifier model with proper overfitting prevention.

## Arguments

```
/train-ml --symbol SPY --fast                        # Single ticker, quick
/train-ml --basket SPY,QQQ,IWM --fast                # Basket, quick
/train-ml --symbol JOE --walk-forward                # Full pipeline + WFO
/train-ml --basket SPY,QQQ,XLK,TLT,GLD --cross-asset # With market context
```

## Quick Examples

```
# Standard basket training (recommended)
/train-ml --basket SPY,QQQ,IWM,XLK,TLT,GLD --fast

# Full pipeline with walk-forward
/train-ml --symbol SPY --walk-forward

# Retrain with cross-asset features
/train-ml --symbol SPY --cross-asset --fast
```

After training, run `/model-diagnose` to validate the model.
