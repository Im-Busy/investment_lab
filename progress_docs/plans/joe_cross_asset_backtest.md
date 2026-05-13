# JOE Cross-Asset Backtest Validation — Plan

**Created:** 2026-05-09
**Prerequisite:** Cross-asset feature experiment (completed — JOE: 0.47 → 0.62 AUC)
**Estimated effort:** 1 AI session

---

## 1. Rationale

JOE showed the largest cross-asset improvement: Test AUC 0.471 → 0.620, Overfit Gap 0.378 → 0.164. This is the strongest signal improvement across all 6 instruments. The question now: does this AUC improvement translate to actual trading profit?

AUC measures ranking ability. A backtest measures P&L. They don't always align — a model can rank well but still lose money if:
- Profitable predictions are concentrated in low-volatility periods that don't move enough
- The threshold-selected trades have poor risk/reward
- The model generates signals too infrequently to overcome transaction costs

## 2. What to Do

Run two backtests on JOE using the trained models:

| Run | Model | Model file | Expected Test AUC |
|-----|-------|-----------|-------------------|
| A | No cross-asset | `pattern_classifier_*_v3_baseline_JOE_catboost.pkl` | 0.47 |
| B | With cross-asset | `pattern_classifier_*_v3_ca_JOE_catboost.pkl` | 0.62 |

### Backtest parameters
- Symbol: JOE
- Date range: 2017-01-01 to 2024-12-31 (skip 2015-2016 for model warm-up)
- Initial capital: $100,000
- Position sizing: Fixed fraction (2% risk per trade)
- Commission: 0.1% per trade

### Metrics to compare
| Metric | What it means |
|--------|--------------|
| Total Return % | Raw profit |
| Sharpe Ratio | Return per unit of risk (> 0.5 acceptable, > 1.0 good) |
| Max Drawdown % | Worst peak-to-trough loss |
| Win Rate % | % of trades that were profitable |
| Profit Factor | Gross profit / gross loss (> 1.2 acceptable, > 1.5 good) |
| Number of Trades | Enough to be statistically meaningful (> 30) |

## 3. How to Do It

Option A: Use `scripts/backtest_ml_enhanced.py` with the JOE model
Option B: Write a standalone validation script that loads the model and generates signals

The key is loading the trained CatBoost model (`.pkl` files in `models/`) and feeding JOE OHLCV data through it to generate trade signals, then running those signals through the backtesting engine.

## 4. Success Criteria

- **Strong support:** Backtest B (cross-asset) shows Sharpe > 0.5 AND outperforms Backtest A by > 0.2 Sharpe
- **Weak support:** Backtest B shows higher total return with similar drawdown
- **Rejection:** Backtest B performs worse than A despite higher AUC
- **Minimum trades:** At least 30 trades in each backtest for statistical validity

## 5. Next Step After This

If cross-asset backtest wins → proceed to meta-labeling for JOE.
If backtest contradicts AUC → investigate why (signal frequency, threshold calibration, market regime).
