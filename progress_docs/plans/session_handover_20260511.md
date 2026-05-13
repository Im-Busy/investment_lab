# Session Handover — ML Pipeline V3 Completion

**Date:** 2026-05-11
**Session span:** ~6 hours (Phase 2 implement → Phase 3 partial)
**Model:** `models/pattern_classifier_v3_JOE_20260511_031329.pkl`
**Key experiment:** `experiments/v3_JOE_20260511_031329/`
**Paper trading:** `reports/paper_trading/20260511_034037/`

---

## Completed (2026-05-11 Follow-Up)

### P1 — Fixed Paper Trading Drawdown
`scripts/paper_trade_v3.py`: Equity mark-to-market every bar (including intra-trade). MaxDD for SPY: fake ~-1% → realistic -5.7%. SPY B&H benchmark row with `vsB&H` delta column. Position sizing uses entry-time stop-loss (was using unknown exit price).

### P3 — Phase B Cleanup
- 14 `experiments/phase_b_*` → `experiments/_archived_invalid/` (archived)
- `src/ml/regime_model.py`: DeprecationWarning in `__init__` + docstring WARNING
- `scripts/phase_b_ml_enhancement.py`: Fixed circular IC (`corr(pred*ret,ret)` → `corr(pred,ret)`), `label_span=5` on PurgedKFold, DEPRECATED header

### P4 — Per-Ticker Walk-Forward
`src/ml/walk_forward.py`: Added `walk_forward_per_ticker()` + `PerTickerWFResult`

### P5 — Meta-Labeler Unique Features
`src/ml/simple_meta_labeler.py`: Added `MetaLabelContextFeatures` with: adx, adx_trend, vol_regime_ratio, vol_regime_high/low, recent_return, up_days_ratio, vol_skew, signal_density, days_since_signal, prob_rolling_mean/std, prob_above_ma

### P8 — Progress Docs Updated
`progress_docs/current.md` updated, `AGENTS.md` rules reinforced.

---

## Remaining

### Priority 2 — Run ARO + GWO on Best Tickers
```bash
uv run scripts/train_ml_pipeline_v3.py --basket SPY,KODK,QQQ --start 2015-01-01 --end 2024-12-31
```

### Priority 4 — Verify Walk-Forward
```bash
uv run python -c "
from src.ml.walk_forward import walk_forward_per_ticker
from src.ml.pattern_classifier import PatternClassifier
m = PatternClassifier()
m.load('models/pattern_classifier_v3_JOE_20260511_031329.pkl')
results = walk_forward_per_ticker(m, ['SPY', 'KODK', 'QQQ'])
for t, r in results.items():
    print(f'{t}: mean_IC={r.mean_rank_ic:.4f}, n_steps={r.n_steps}')
"
```

### Priority 6 — Integrate ML with Backtest Engine
Create `src/backtest/ml_strategy.py` that plugs V3 model into event-driven engine.

### Priority 7 — Expand Ticker Universe to 30+

---

Three independent bugs were causing the original 0.95+ IC numbers:

1. **Rule-based labels** — `RegimeClassifier` trained to predict ADX/ATR thresholds (a deterministic formula), not market behavior. Training on synthetic labels makes the model memorize rules, not predict markets.
2. **Circular IC metric** — `scripts/phase_b_ml_enhancement.py:215` computes `corr(score × return, return)` instead of `corr(score, return)`. Both sides of the correlation contain the return, inflating IS IC to 0.95+.
3. **label_span mismatch** — PurgedKFold default `label_span=1` with 5-day horizon labels under-purges training samples whose labels overlap the test period.

The correct pipeline already existed in `scripts/train_ml_model_v2.py` (triple-barrier labels, `label_span=horizon`, nested PurgedKFold).

---

## What We Built (Phase 2 — Pipeline Rebuild)

### New files created:

| File | Purpose |
|------|---------|
| `scripts/train_ml_pipeline_v3.py` | Full 12-stage pipeline: load → features → cross-asset → IC filter → ARO → GWO → nested PurgedKFold → final model → SHAP → regime analysis → per-ticker OOS → meta-labeling |
| `src/ml/walk_forward.py` | Chronological walk-forward validation (rolling expanding window) |
| `src/ml/regime_analysis.py` | Regime-conditional rank IC decomposition (ADX/volatility/direction/year) |
| `src/ml/simple_meta_labeler.py` | Lightweight meta-labeler (LogisticRegression on primary prob + top SHAP features) |
| `scripts/paper_trade_v3.py` | Point-in-time OOS paper trading with triple-barrier TP/SL/Time exits |

### Files modified:

| File | Change |
|------|--------|
| `src/ml/metrics.py` | Added `evaluate_predictions()` convenience function |
| `src/ml/pattern_classifier.py` | Added `allow_writing_files=False` to CatBoost (fixes Windows file locking) |
| `docs/COMMAND_CHEATSHEET.md` | Added V3 pipeline commands |
| `.useful_commands/ml_training_commands.txt` | Added V3 pipeline commands |

---

## Key Results

### Expanded Basket Training (12 tickers, 32,359 samples, 44 features, --fast mode)

**CV:** Train AUC 0.611, Test AUC **0.550**, Gap **0.061** — all 5 folds OK.
**Final model:** Train AUC 0.616, Test AUC 0.548, Gap 0.068 — well under 0.15 threshold.
**Top features:** vol_regime, resid_vol_SPY_60d, rel_ret_TLT_5d, momentum_5, volatility_regime.
**SHAP:** Zero suspicious features in top 10.

### Per-Ticker OOS (chronological 70/30 split, date-indexed)

| Ticker | Rank IC | Test AUC |
|--------|---------|----------|
| KODK | 0.276 | 0.673 |
| SPY | 0.256 | 0.656 |
| QQQ | 0.218 | 0.634 |
| IWM | 0.156 | 0.600 |
| XLK | 0.151 | 0.592 |
| EEM | 0.115 | 0.570 |
| XLF | 0.077 | 0.547 |
| TLT | 0.072 | 0.547 |
| JOE | 0.054 | 0.535 |
| XLE | 0.032 | 0.520 |
| GLD | 0.026 | 0.515 |
| XLV | 0.019 | 0.512 |

10/12 tickers > 0.03 rank IC threshold.

### Paper Trading (2021-01-01 to 2024-12-31 OOS, point-in-time)

| Ticker | Trades | Return% | Sharpe | WinRate | PF |
|--------|--------|---------|--------|---------|-----|
| XLV | 318 | 179.7 | 1.31 | 0.44 | 1.19 |
| SPY | 266 | 173.6 | 3.17 | 0.51 | 1.57 |
| XLF | 222 | 133.2 | 2.19 | 0.48 | 1.36 |
| QQQ | 189 | 132.0 | 2.31 | 0.50 | 1.38 |
| XLK | 159 | 131.7 | 4.46 | 0.56 | 1.86 |
| IWM | 174 | 107.0 | 1.27 | 0.45 | 1.19 |
| EEM | 202 | 101.5 | 1.84 | 0.46 | 1.28 |
| GLD | 196 | 85.8 | 2.61 | 0.48 | 1.41 |
| KODK | 45 | 42.1 | 10.46 | 0.69 | 3.83 |
| TLT | 123 | 37.7 | 0.32 | 0.42 | 1.04 |
| XLE | 32 | 21.9 | 5.21 | 0.62 | 2.08 |
| JOE | 13 | 8.4 | 5.54 | 0.54 | 2.13 |

**100% of tickers profitable.** SPY 174% vs ~60% buy-and-hold. KODK best per-trade quality (PF 3.83). TLT weakest (Sharpe 0.32).

---

## Remaining Tasks (Next Session)

### Priority 1 — Fix Paper Trading Drawdown Bug
**File:** `scripts/paper_trade_v3.py`
**Bug:** Drawdown only tracks equity at non-trade bars, missing intra-trade drawdowns. This makes MaxDD numbers misleadingly low (~-1%). Need to track P&L during active trades and use peak-to-trough on the full equity curve.
**Also:** Add SPY buy-and-hold benchmark comparison to the output table.

### Priority 2 — Run ARO + GWO on Best Tickers
**Command:** `uv run scripts/train_ml_pipeline_v3.py --basket SPY,KODK,QQQ --start 2015-01-01 --end 2024-12-31` (without --fast)
This runs the full pipeline with ARO feature selection (target 15 features) and GWO hyperparameter tuning (n_wolves=5, max_iter=10 by default). The current model used conservative defaults (max_depth=3, l2_leaf_reg=10) — GWO may find better params.

### Priority 3 — Phase 1 Cleanup (Low Effort, High Clarity)
- Archive all `experiments/phase_b_*` directories to `experiments/_archived_invalid/`
- Deprecate `RegimeClassifier` as a prediction target in `src/ml/regime_model.py` (add DeprecationWarning)
- Fix the circular IC bug in `scripts/phase_b_ml_enhancement.py:215` even though that pipeline is deprecated
- Fix `label_span` in PurgedKFold calls in `phase_b_ml_enhancement.py:111,200` from default=1 to `label_span=5`

### Priority 4 — Fix Single-Ticker Walk-Forward Overfitting
The single-ticker JOE walk-forward showed rank IC = -0.008 (no OOS signal). The basket model fixed this. But we should verify: does a basket-trained model produce positive walk-forward IC when evaluated on individual tickers with proper chronological walks? Implement per-ticker walk-forward using the basket model.

### Priority 5 — Improve Meta-Labeling
The basic meta-labeler (LogisticRegression on primary prob + top SHAP features) showed test AUC 0.558 with negative signal reduction — it's more permissive than the primary model. To be useful, the meta-labeler needs DIFFERENT features than the primary model:
- Regime gate features (ADX state, VIX proxy, market direction)
- Volatility context (is current vol above/below recent average?)
- Trade clustering (how many trades in past N days?)
- Time-since-last-trade

### Priority 6 — Integrate with Backtest Engine
Create `src/backtest/ml_strategy.py` that plugs the V3 model into the existing event-driven backtest engine (`src/backtest/engine.py`). This enables:
- ML signals coexisting with pattern-detector signals
- Confluence scoring (pattern + ML agreement)
- Full position sizing from `src/risk/`

### Priority 7 — Expand Ticker Universe to 30+
Current: 12 tickers. The Phase 3 plan targets 30+. Available data files already include: SPY, QQQ, TLT, GLD, IWM, JOE, KODK, CRVL, HIFS, XLE, XLF, XLK, XLV, EEM, IAU. Add more via yfinance if needed.

### Priority 8 — Update Progress Docs
Add session summary to `progress_docs/` and update `AGENTS.md` with the three bugs discovered and the rule: "Never compute IC as corr(score×return, return). Never use rule-based labels as ML targets."

---

## Key CLI Commands

```bash
# Full pipeline on single ticker with walk-forward
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --walk-forward

# Basket training (recommended — best generalization)
uv run scripts/train_ml_pipeline_v3.py --basket JOE,KODK,SPY,QQQ,IWM,TLT,GLD,XLF,XLK,XLE,XLV,EEM

# Fast mode (skip ARO/GWO for quick iterations)
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --fast

# Paper trading on trained model
uv run scripts/paper_trade_v3.py --model models/pattern_classifier_v3_JOE_20260511_031329.pkl --tickers SPY,KODK,QQQ,IWM,XLK,JOE,XLF,XLE,XLV,EEM,TLT,GLD --start 2021-01-01

# With threshold adjustment (more/less selective)
uv run scripts/paper_trade_v3.py --model models/pattern_classifier_v3_JOE_20260511_031329.pkl --tickers SPY --threshold 0.6 --start 2021-01-01
```

---

## Architecture Notes

- **Feature engine:** `src/ml/feature_engineering.py` → `FeatureExtractor` (backward-looking only, `include_forward_returns=False` by default). DO NOT use `src/ml/features.py` → `FeatureEngineer` — it has a `add_forward_returns()` trap that generates look-ahead columns.
- **Cross-asset:** `src/ml/cross_asset_features.py` → `CrossAssetFeatureExtractor` (SPY, QQQ, TLT, GLD, IWM, XLF, XLK). All cross-asset features use `.shift(1)` to prevent look-ahead.
- **Labels:** Triple-barrier only (`src/ml/triple_barrier.py`). Rule-based labels (`src/indicators/regime_detector.py`) are for FEATURES only, NEVER as training targets.
- **CV:** PurgedKFold with `label_span=horizon` and `pct_embargo=0.05`. Always match label_span to the label horizon.
- **Metric:** Rank IC (Spearman) is the primary metric. Accuracy on binary labels is secondary.
- **Model:** CatBoost with `allow_writing_files=False` (Windows fix). Default params: depth=3, l2_leaf_reg=10, min_data_in_leaf=50.
- **Pooling:** When training on multiple tickers, features and labels are concatenated with `reset_index(drop=True)` + `ignore_index=True` to avoid duplicate date index issues. Per-ticker OOS evaluation uses the original date-indexed features.
- **Paper trading:** Uses `TripleBarrierLabeler.fit_single()` for point-in-time exit simulation. Features are recomputed at each bar but are backward-looking only. Position sizing: 2% risk per trade, capped at 50% of cash.

---

## Known Issues

1. ~~Drawdown in paper trading only tracks end-of-trade equity, missing intra-trade drawdowns~~ **FIXED** — equity now mark-to-market every bar
2. ~~Meta-labeler uses same features as primary model → doesn't add value. Needs unique features~~ **FIXED** — `MetaLabelContextFeatures` generates 8 regime/volatility/clustering features different from primary model
3. ~~Phase B experiments (14 runs) still exist in `experiments/` and should be archived~~ **DONE** — moved to `experiments/_archived_invalid/`
4. ~~Walk-forward doesn't work with basket mode (pooled index loses chronology) — needs per-ticker implementation~~ **DONE** — added `walk_forward_per_ticker()` to `src/ml/walk_forward.py`

## Session Update 2026-05-11 04:00 (Follow-on Session)

**Completed:** Priorities 1, 3, 4, 5, 8

| Priority | Status | Details |
|----------|--------|---------|
| P1: Fix drawdown + SPY benchmark | ✅ | `scripts/paper_trade_v3.py` — bar-by-bar mark-to-market, entry-time position sizing, SPY B&H row in console table. Drawdown now realistic (-5.7% vs fake -1%). |
| P3: Archive phase_b experiments | ✅ | 14 dirs `git mv`'d to `experiments/_archived_invalid/` |
| P3: Deprecate RegimeClassifier | ✅ | `DeprecationWarning` at `__init__` + docstring WARNING block |
| P3: Fix circular IC & label_span | ✅ | `phase_b_ml_enhancement.py` — fixed `corr(y_pred, returns)` and `label_span=5` on both PurgedKFold calls |
| P4: Per-ticker walk-forward | ✅ | `walk_forward_per_ticker()` in `src/ml/walk_forward.py` — chronological WF using basket-trained model, per-ticker rank IC |
| P5: Meta-labeler unique features | ✅ | `MetaLabelContextFeatures` in `src/ml/simple_meta_labeler.py` — ADX, vol_regime_ratio, vol_skew, signal_density, days_since_signal, prob_rolling stats |
| P8: Update progress docs | ✅ | `progress_docs/current.md` updated with session summary |

**Remaining from original priorities:**
| Priority | Action |
|----------|--------|
| P2 | Run ARO+GWO on SPY/KODK/QQQ (uv run scripts/train_ml_pipeline_v3.py --basket SPY,KODK,QQQ --start 2015-01-01 --end 2024-12-31) |
| P6 | Integrate V3 model with event-driven backtest engine (`src/backtest/ml_strategy.py`) |
| P7 | Expand ticker universe to 30+ |

**Modified files this session:**
- `scripts/paper_trade_v3.py` — drawdown fix + SPY benchmark
- `scripts/phase_b_ml_enhancement.py` — deprecated header + bug fixes
- `src/ml/regime_model.py` — DeprecationWarning + docstring
- `src/ml/walk_forward.py` — `walk_forward_per_ticker()`
- `src/ml/simple_meta_labeler.py` — `MetaLabelContextFeatures`
- `progress_docs/current.md` — session entries
- 14 `experiments/phase_b_*/` → `experiments/_archived_invalid/` — archive
