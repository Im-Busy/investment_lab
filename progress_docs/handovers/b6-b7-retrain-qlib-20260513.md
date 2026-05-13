# Handover: B6 (Retrain with Normalized ATR) + B7 (Qlib ADARNN/ADD)

**Source:** Phase 6a — Model Robustness & Production Readiness. B6 is the P0 gate for C9-C17 (strategy refinement, PDF insights, portfolio backtests).

**Context:** Regime shift investigation (B4) found raw ATR values doubled between IS (2015-2024) and OOS (2025-2026): `atr_14` IS=3.97 → OOS=8.06, KS=0.62, p=10^-109. Fix applied at `src/ml/feature_engineering.py:173-174`: ATR is now normalized `ATR/Close`. But the current V3 model was trained on **raw ATR** — it learns scale-dependent relationships that break when volatility regime shifts. B6 retrains on normalized ATR and validates the fix.

---

## B6: Retrain Basket Model with Normalized ATR

**Priority:** P0
**Problem:** V3 model trained on raw ATR features. When volatility doubled in 2025-2026, the model saw "normal" ATR=8 (which it had never seen before), causing catastrophic OOS degradation (-3.1% return, Sharpe -0.27). Normalized ATR (ATR/Close) decouples from absolute price level.
**Impact:** ~5 min training + ~5 min validation. Gates all Phase 6c/d work. Single command.

### Implementation Steps

**Step 1: Verify the ATR fix is active**

```bash
uv run python -c "
from src.ml.feature_engineering import FeatureExtractor
import yfinance as yf
df = yf.download('SPY', start='2024-01-01', end='2024-06-01', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
ext = FeatureExtractor()
fx = ext.extract_all_features(df)
atr_cols = [c for c in fx.columns if 'atr' in c.lower()]
print(fx[atr_cols].describe())
# atr_14 should be in 0.01-0.05 range (ATR/Close), NOT 3-8
"
```

Expected: `atr_14` mean should be 0.01-0.03 (percentage of Close), not 3-8 dollars.

**Step 2: Retrain the basket model**

```bash
# Full 9-stage pipeline on 6-ticker basket with normalized ATR
uv run scripts/train_ml_pipeline_v3.py \
    --basket SPY,QQQ,TLT,GLD,IWM,JOE \
    --horizon 5 \
    --start 2015-01-01 --end 2024-12-31 \
    --walk-forward
```

Expected: ~5 min. Outputs model to `models/pattern_classifier_v3_SPY_YYYYMMDD_HHMMSS.pkl`.
Note the model path from the final log line.

**Step 3: Run WFO comparison with retrained model**

```bash
# This script trains WFO on SPY using FeatureExtractor (which now has normalized ATR)
uv run scripts/backtest_wfo.py
```

Compare against the pre-fix WFO results in `reports/wfo/wfo_comparison.json`:
- Pre-fix: WFO OOS +1.7%, single-split -1.3%, total Sharpe 1.76
- Post-fix expectation: single-split OOS should improve since ATR is now scale-invariant
- If single-split OOS returns go positive (was -1.3%), the ATR fix is validated

**Step 4: OOS backtest on 2025-2026 post-training period**

```bash
uv run scripts/run_ml_backtest.py SPY \
    --model models/pattern_classifier_v3_SPY_YYYYMMDD_HHMMSS.pkl \
    --entry-threshold 0.45 \
    --trail-stop \
    --start 2025-01-01
```

Pre-fix OOS with raw ATR: Sharpe -0.27, return -3.1%, win rate 27.3%, 11 trades.
Post-fix expectation: Sharpe should be positive if the fix works (target > 0.0 on OOS).
Minimum viable: Sharpe > -0.1 with 15+ trades.

**Step 5: Paper trade validation (if OOS backtest improves)**

```bash
uv run scripts/paper_trade_v3.py \
    --model models/pattern_classifier_v3_SPY_YYYYMMDD_HHMMSS.pkl \
    --tickers SPY,QQQ,TLT,GLD,IWM,JOE \
    --start 2025-01-01
```

**Step 6: Update BESTS.md**

If the retrained model beats the pre-fix OOS baseline at any config:
- Add a new "With Normalized ATR (B6)" section
- Record: config string, return, Sharpe, trades, win%, PF, max DD
- Compare against the existing "OOS Validation" section

### Pass/Fail Criteria

| Metric | Pre-Fix (Raw ATR) | Post-Fix Target | Verdict |
|--------|-------------------|-----------------|---------|
| atr_14 mean | 3.97 (IS), 8.06 (OOS) | 0.01-0.03 range | Must be normalized |
| Single-split OOS return | -1.3% | > -0.5% | Improved |
| WFO OOS Sharpe | 1.76 | > 1.5 | Stable |
| OOS backtest Sharpe | -0.27 | > -0.1 | Positive |
| OOS backtest trades | 11 | > 15 | More confident |

If 3/5 pass → B6 is validated, proceed to C9-C17.
If 1-2/5 pass → ATR fix alone insufficient, proceed to B7 (Qlib concept-drift models).
If 0/5 pass → Escalate: need deeper feature engineering review.

---

## B7: Evaluate Qlib Concept-Drift Models (ADARNN/ADD)

**Priority:** P0 (depends on B6)
**Problem:** Even with normalized ATR, CatBoost + PurgedKFold may not be the right tool for non-stationary financial data. ADARNN (Adaptive RNN) and ADD (Adaptive Deep Dynamics) from Microsoft Qlib are purpose-built for data streams where the generating distribution shifts over time — the exact failure mode we diagnosed in B4.
**Impact:** If B6 fails or shows marginal improvement, B7 provides the alternative architecture. If B6 succeeds, B7 is still valuable for comparison.

### Implementation Steps

**Step 1: Clone Qlib**

```bash
git clone https://github.com/microsoft/qlib.git useful_resources/useful_repos/Qlib
cd useful_resources/useful_repos/Qlib
pip install -e .  # or: uv pip install -e .
```

**Step 2: Prepare data in Qlib format**

Qlib requires data in its own `qlib_bin` format. The project's OHLCV CSVs in `data/raw/` need conversion.

```bash
cd useful_resources/useful_repos/Qlib
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
# For US data, use:
# python scripts/dump_bin.py dump_all ...
```

Alternatively, use the Qlib US data pipeline if available. The key ticker for evaluation is SPY (2020-2026 to match our OOS window).

**Step 3: Run ADARNN baseline on SPY**

Qlib provides example configs. Adapt for SPY:

```bash
cd useful_resources/useful_repos/Qlib
# ADARNN example — adapt YAML config for SPY + our feature set
python examples/benchmarks/ADARNN/workflow_config_AdaRNN_Alpha158.yaml
```

Key metrics to extract: IC, ICIR, Rank IC, annualized return on OOS period.
Compare against CatBoost V3 + normalized ATR from B6.

**Step 4: Run ADD baseline**

```bash
python examples/benchmarks/ADD/workflow_config_ADD_Alpha158.yaml
```

**Step 5: Comparison report**

Create `reports/qlib/comparison.json` with:

```json
{
  "catboost_normalized_atr": { "oos_sharpe": ..., "oos_ic": ... },
  "qlib_adarnn": { "oos_sharpe": ..., "oos_ic": ... },
  "qlib_add": { "oos_sharpe": ..., "oos_ic": ... }
}
```

### Decision Gate

| Scenario | Action |
|----------|--------|
| CatBoost normalized ATR > Qlib on OOS | Stick with CatBoost, proceed to C9-C17 |
| Qlib model > CatBoost on OOS (≥5% Sharpe improvement) | Integrate ADARNN/ADD into `src/ml/`, retrain strategy |
| Neither works on OOS | Re-evaluate feature set. Consider Qlib Alpha158 factor suite |

---

## Dependency Order

```
B6.1 (Verify ATR fix) → B6.2 (Retrain basket) → B6.3 (WFO compare)
                                                    ↓
                                              B6.4 (OOS backtest)
                                                    ↓
                                              B6.5 (Paper trade)
                                                    ↓
                          ┌── B6 passes ──→ C9-C17 (strategy refinement)
                          │
              B6 Result ──┤
                          │
                          └── B6 marginal/fails ──→ B7 (Qlib ADARNN/ADD)
                                                         ↓
                                                   Decision gate
```

---

## File Map

| File | Purpose |
|------|---------|
| `src/ml/feature_engineering.py:173-174` | ATR fix already applied — `atr_14 = _compute_atr(df, 14) / close_pos` |
| `scripts/train_ml_pipeline_v3.py` | Retrain on 6-ticker basket with `--basket` |
| `scripts/backtest_wfo.py` | WFO comparison (already configured for SPY, uses FeatureExtractor with normalized ATR) |
| `scripts/run_ml_backtest.py` | OOS backtest on 2025-2026 |
| `scripts/paper_trade_v3.py` | Walk-forward paper trading if OOS improves |
| `BESTS.md` | Update leaderboard with retrained model results |
| `useful_resources/useful_repos/Qlib/` | Clone target for B7 |
| `reports/wfo/wfo_comparison.json` | Pre-fix WFO baseline |
| `reports/calibration/regime_shift_features.png` | B4 evidence: ATR KS=0.62 |

---

## Pre-Flight Checklist

Before implementing, verify the environment:
```bash
# 1. Confirm ATR fix is in place
uv run python -c "from src.ml.feature_engineering import FeatureExtractor; print('FeatureExtractor OK')"

# 2. Confirm last WFO result exists
cat reports/wfo/wfo_comparison.json

# 3. Check available models
ls models/pattern_classifier_v3_*.pkl | tail -5

# 4. Quick pipeline smoke test (30s)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --fast --horizon 5 --start 2020-01-01 --end 2022-01-01 --skip-cross-asset
```
