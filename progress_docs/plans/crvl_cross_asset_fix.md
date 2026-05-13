# CRVL Cross-Asset Fix — Plan

**Created:** 2026-05-09
**Prerequisite:** Cross-asset feature experiment (completed — CRVL: 0.593 → 0.525 AUC, Scenario D)
**Estimated effort:** 0.5-1 AI session

---

## 1. Rationale

CRVL was the only instrument where cross-asset features made things *worse*:
- Test AUC: 0.593 → 0.525 (-0.067)
- Overfit Gap: 0.171 → 0.367 (+0.197)
- Train AUC: 0.764 → 0.893 (more memorization)

The hypothesis: CRVL (a financial analytics company) may be sensitive to market context, but the full 26 cross-asset features overwhelmed the model. The signal-to-noise ratio worsened when we added QQQ beta, TLT relative returns, IWM relative returns, etc.

CRVL is a mid-cap financial services stock. It likely responds to:
- SPY (broad market) — yes
- XLF (financial sector) — yes
- QQQ (tech) — probably not meaningful
- TLT (bonds) — maybe (financial sector interest-rate sensitivity)

## 2. Experiment: Reduced Cross-Asset Set

Train CRVL with progressively smaller cross-asset feature sets:

| Run | Cross-asset features used | Hypothesis |
|-----|--------------------------|------------|
| A | SPY only (regime + beta + corr) | Core market beta is enough |
| B | SPY + XLF (sector rotation) | Financial sector context matters |
| C | SPY only, no beta features | Even simpler |
| D | Full set (baseline, already done) | Control |

### Specific features per run:

**Run A (SPY only):**
- SPY relative returns (5d, 20d)
- SPY beta (20d, 60d)
- SPY residual vol (20d, 60d)
- SPY correlation (20d, 60d)
- SPY market regime (trend 50d/200d, vol percentile, ATR ratio, breadth)
- Total: ~15 features

**Run B (SPY + XLF):**
- All Run A features
- XLF relative 5d
- Total: ~16 features

**Run C (SPY regime only, no beta):**
- SPY relative returns (5d, 20d)
- SPY market regime (trend 50d/200d, vol percentile, ATR ratio, breadth)
- Total: ~11 features

## 3. Implementation

Modify `load_market_data()` call to accept a custom ticker list:

```bash
# Train CRVL with SPY-only cross-asset
uv run scripts/train_ml_model_v2.py --symbol data/raw/CRVL_daily.csv \
    --horizon 5 --suffix v3_ca_spy_only

# (requires modifying load_market_data or adding --market-tickers flag)
```

Alternative: Add `--market-tickers` CLI flag to `train_ml_model_v2.py`:

```python
parser.add_argument("--market-tickers", type=str, default=None,
                    help="Comma-separated list of market tickers (default: SPY,QQQ,TLT,GLD,IWM,XLF,XLK)")
```

Then:

```bash
uv run scripts/train_ml_model_v2.py --symbol data/raw/CRVL_daily.csv \
    --horizon 5 --suffix v3_ca_spy_only --market-tickers SPY

uv run scripts/train_ml_model_v2.py --symbol data/raw/CRVL_daily.csv \
    --horizon 5 --suffix v3_ca_spy_xlf --market-tickers SPY,XLF
```

## 4. Success Criteria

- At least one reduced set beats the NoCA baseline (Test AUC > 0.593)
- Overfit gap < 0.20 (improvement over 0.367)
- Cross-asset features appear in top 10 (confirms model is using them)

## 5. Next Step After This

If SPY-only works → apply the reduced set pattern to other instruments that showed weak results.
If all reduced sets fail → CRVL may genuinely not benefit from market context; accept and move on.
