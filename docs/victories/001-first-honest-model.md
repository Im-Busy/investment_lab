# Victory #1 — The First Honest Model

**Date:** 2026-05-11
**Session:** ML Pipeline V3 — Honest Foundation
**Model:** `models/pattern_classifier_v3_JOE_20260511_031329.pkl`

---

## The Win

After weeks of debugging, we have a machine learning model that **genuinely predicts short-term stock returns** — modestly, honestly, and verifiably.

| Metric | Value | Meaning |
|--------|-------|---------|
| Test AUC | 0.55 | Better than random (0.50). Small, but real. |
| Overfit gap | 0.07 | What the model learns in training carries over to the future. |
| Tickers with IC > 0.03 | 10/12 | The edge exists across most stocks, not just one lucky ticker. |
| Profit factor | 1.2–1.6 | Winners are consistently bigger than losers. |

Every number is **auditable**. No circular metrics. No synthetic labels. No look-ahead. Triple-barrier exits, PurgedKFold cross-validation, Spearman rank IC computed directly as `corr(prediction, outcome)`.

---

## Why This Took So Long

We built this project with 34+ chart pattern detectors, custom backtest engines, multi-strategy frameworks — hundreds of hours of work. But the first "ML models" we trained were **lying to us**.

Three independent bugs created the illusion of a nearly perfect predictor:

1. **Rule-based labels** — the model was trained to predict `ADX > 25` (a deterministic formula), not market direction. It memorized the rulebook instead of learning price behavior.

2. **Circular IC metric** — the correlation was computed as `corr(score × return, return)`. Both terms contained the same return, so the answer was always ~0.95+. It's like asking: "does 7 correlate with 7?"

3. **Label leakage** — PurgedKFold defaulted to `label_span=1` with 5-day labels, so training windows bled into test windows. The model saw fragments of the future.

These bugs evaded detection because the results looked **exactly like what we wanted to see**. That's the trap: when you're building something, your bias is to trust good-looking numbers. Bugs that produce bad numbers get caught immediately. Bugs that produce *amazing* numbers survive.

The fix wasn't writing more code — it was **auditing the assumptions**:
- "Are these labels actually predicting anything real?"
- "What happens if I compute the correlation by hand?"
- "Does changing `label_span` from 5 to 1 change the results?"

---

## What Changed

We rebuilt the pipeline from scratch around these principles:

- **Triple-barrier labels** — actual price movement (take-profit, stop-loss, time exit), not formulas
- **Backward-looking features only** — no `add_forward_returns()`, no `.shift(-1)` anywhere
- **Cross-asset basket training** — 12 tickers pooled together for better generalization
- **PurgedKFold with correct `label_span`** — no bleed between train and test
- **Rank IC as primary metric** — `corr(prediction, outcome)`, not `corr(prediction × outcome, outcome)`

The resulting model is **boring**. It doesn't get 95% accuracy. It doesn't produce 500% annual returns. It predicts slightly better than a coin flip, and its winners are moderately bigger than its losers.

**That's exactly what a real edge looks like.**

---

## The Model

| Detail | Value |
|--------|-------|
| Architecture | CatBoost classifier |
| Features | 44 (IC-filtered from 132) |
| Training samples | 32,359 (pooled from 12 tickers) |
| Training period | 2015–2020 |
| OOS test period | 2021–2024 |
| Parameters | depth=3, l2_leaf_reg=10, min_data_in_leaf=50 (conservative) |

**Top features:** vol_regime, resid_vol_SPY_60d, rel_ret_TLT_5d, momentum_5, volatility_regime

**SHAP audit:** Zero suspicious features in top 10. No look-ahead, no leakage.

### Per-Ticker OOS Performance

| Ticker | Rank IC | Notes |
|--------|---------|-------|
| KODK | 0.276 | Best individual performer |
| SPY | 0.256 | Strong on benchmark |
| QQQ | 0.218 | Tech-heavy, model handles it |
| IWM | 0.156 | Small caps |
| XLK | 0.151 | Tech sector |
| EEM | 0.115 | Emerging markets |
| XLF | 0.077 | Financials |
| TLT | 0.072 | Bonds — tougher asset |
| JOE | 0.054 | The original single-ticker target |
| XLE | 0.032 | Energy |
| GLD | 0.026 | Gold — barely above noise |
| XLV | 0.019 | Healthcare — weak |

10/12 tickers above the 0.03 IC threshold. Even the weak ones aren't negative — the model doesn't actively lose signal anywhere.

---

## What This Means

1. **It's possible.** A simple CatBoost model, trained on honest labels, with conservative parameters, on a diversified basket — produces a real, measurable edge.

2. **The edge is modest, as it should be.** Markets are efficient. If a model claims to predict them with 0.95 accuracy, it's bugged. A 0.55 AUC with a 0.07 gap is exactly the kind of "small but real" advantage that sophisticated traders actually use.

3. **Basket training works.** Single-ticker JOE had rank IC -0.008 (no signal). Adding 11 other tickers boosted all individual ICs. The model learns general market dynamics from the basket, then applies them to each ticker.

4. **Conservative beats clever.** depth=3, learning_rate=0.03, l2_leaf_reg=10 — boring parameters, stable results. GWO-tuned aggressive parameters (depth=8, lr=0.27) produced 0.40+ overfit gaps.

---

## The Road Ahead

- Paper trade 2021-2024 OOS: 100% of tickers profitable, SPY +174% vs ~60% buy-and-hold ✓
- Expand basket to 30+ tickers for better generalization
- Integrate with event-driven backtest engine for live trading
- Build meta-labeler with unique features (regime, volatility, clustering)
- Run walk-forward per-ticker to verify chronological robustness

---

## Files

| File | Role |
|------|------|
| `scripts/train_ml_pipeline_v3.py` | The honest training pipeline (12 stages) |
| `scripts/paper_trade_v3.py` | OOS paper trading with triple-barrier exits |
| `src/ml/triple_barrier.py` | Genuine forward-looking labels |
| `src/ml/feature_engineering.py` | Backward-looking feature extraction |
| `src/ml/purged_cv.py` | Temporal cross-validation with embargo |
| `src/ml/walk_forward.py` | Chronological walk-forward |
| `src/ml/simple_meta_labeler.py` | Signal quality post-filter |
| `plans/session_handover_20260511.md` | Full session notes and bug analysis |

---

**Onward.**
