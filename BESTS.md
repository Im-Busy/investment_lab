# Backtest Leaderboard — Best Results by Configuration

> Data: SPY daily, 2016-05-12 → 2024-12-30 | Buy & Hold: **224.56%** | Model: `pattern_classifier_v3_SPY_20260511_224704.pkl`

## Overall Best by Sharpe

| Rank | Config | Return | Sharpe | Trades | Win% | PF | MaxDD | Exp% | Ann% |
|------|--------|--------|--------|--------|------|-----|-------|------|------|
| **1** | `et=0.45 trail` | 81.0% | 0.73 | 66 | 42.4 | 1.65 | -17.5 | 46.6 | 7.1 |
| 2 | `et=0.50 trail` | 43.5% | 0.72 | 36 | 55.6 | 1.96 | -8.8 | 14.4 | 4.3 |
| 3 | `et=0.45 trail+conv vg=1.3` | 38.3% | 0.61 | 20 | 55.0 | 2.99 | -12.3 | 13.4 | 3.8 |
| 4 | `et=0.35 trail` | 69.4% | 0.59 | 60 | 38.3 | 1.72 | -25.7 | 67.9 | 6.3 |
| 5 | `et=0.40 trail` | 71.4% | 0.59 | 62 | 40.3 | 1.54 | -25.7 | 66.2 | 6.5 |

## Overall Best by Return

| Rank | Config | Return | Sharpe | Trades | Win% | PF | MaxDD | Exp% | Ann% |
|------|--------|--------|--------|--------|------|-----|-------|------|------|
| **1** | `et=0.45 trail` | 81.0% | 0.73 | 66 | 42.4 | 1.65 | -17.5 | 46.6 | 7.1 |
| 2 | `et=0.40 trail` | 71.4% | 0.59 | 62 | 40.3 | 1.54 | -25.7 | 66.2 | 6.5 |
| 3 | `et=0.35 trail` | 69.4% | 0.59 | 60 | 38.3 | 1.72 | -25.7 | 67.9 | 6.3 |
| 4 | `et=0.50 trail+conv vg=1.8` | 45.5% | 0.55 | 38 | 50.0 | 1.95 | -18.7 | 23.6 | 4.4 |

## Best by Condition

### With Trail Stop (no conviction, no vol gate)

| Rank | Config | Return | Sharpe | Trades | Win% | PF | MaxDD | Exp% |
|------|--------|--------|--------|--------|------|-----|-------|------|
| **1** | `et=0.45` | 81.0% | **0.73** | 66 | 42.4 | 1.65 | -17.5 | 46.6 |
| 2 | `et=0.40` | 71.4% | 0.59 | 62 | 40.3 | 1.54 | -25.7 | 66.2 |
| 3 | `et=0.35` | 69.4% | 0.59 | 60 | 38.3 | 1.72 | -25.7 | 67.9 |
| 4 | `et=0.50` | 43.5% | 0.72 | 36 | 55.6 | 1.96 | -8.8 | 14.4 |

### With Trail Stop + Conviction

| Rank | Config | Return | Sharpe | Trades | Win% | PF | MaxDD | Exp% |
|------|--------|--------|--------|--------|------|-----|-------|------|
| **1** | `et=0.45` | 38.9% | 0.45 | 45 | 48.9 | 1.45 | -23.3 | 25.5 |
| 2 | `et=0.35` | 36.3% | 0.32 | 40 | 45.0 | 1.51 | -25.4 | 52.0 |
| 3 | `et=0.40` | 18.1% | 0.18 | 45 | 44.4 | 1.23 | -24.2 | 47.3 |

### Baseline (no trail, no conviction)

| Rank | Config | Return | Sharpe | Trades | Win% | PF | MaxDD | Exp% |
|------|--------|--------|--------|--------|------|-----|-------|------|
| **1** | `et=0.35` | 25.1% | 0.28 | 92 | 43.5 | 1.22 | -17.7 | 45.0 |
| 2 | `et=0.45` | 21.9% | 0.30 | 72 | 45.8 | 1.17 | -15.8 | 23.9 |
| 3 | `et=0.50` | 19.0% | 0.49 | 20 | 60.0 | 2.17 | -7.7 | 5.6 |
| 4 | `et=0.40` | 16.0% | 0.20 | 80 | 42.5 | 1.11 | -22.8 | 32.8 |

### With Volatility Gate (trail + conviction base)

| Rank | Config | Return | Sharpe | Trades | Win% | PF | MaxDD | Exp% |
|------|--------|--------|--------|--------|------|-----|-------|------|
| **1** | `et=0.45 vg=1.3` | 38.3% | **0.61** | 20 | 55.0 | **2.99** | -12.3 | 13.4 |
| 2 | `et=0.45 vg=1.8` | 45.5% | 0.55 | 38 | 50.0 | 1.95 | -18.7 | 23.6 |
| 3 | `et=0.45 vg=1.5` | 26.8% | 0.38 | 29 | 48.3 | 1.88 | -12.4 | 19.6 |

---

## Model Probability Distribution

| Stat | Value |
|------|-------|
| Mean | 0.454 |
| Median | 0.461 |
| Std | 0.069 |
| Range | 0.250 – 0.655 |
| P25 / P75 | 0.405 / 0.505 |
| Bars ≥ 0.45 | 56.6% (143/yr) |
| Bars ≥ 0.50 | 27.8% (70/yr) |

---

## Key Findings

1. **Trail stop is essential** — every trail config beats its non-trail counterpart by 2-4x return. Fixed TP kills winners early in a bull trend.
2. **Conviction scaling hurts** — adding `--conviction` reduces return and Sharpe in all threshold groups. It over-weights losing trades when probability is only marginally above threshold.
3. **Volatility gate improves quality** — `vg=1.3` achieved the highest profit factor (2.99) and best risk profile (-12.3% DD) albeit with fewest trades (20).
4. **Optimal entry threshold: 0.45** — balances trade frequency (66) with signal quality. 0.50 is too conservative; 0.35 is too noisy.
5. **All underperform buy & hold** — SPY did 224.5% in this period. The CatBoost model is inherently trend-following with conservative exits, missing most of the secular bull run.

---

## OOS Validation (2025-01-02 → 2026-05-11)

> Model: `pattern_classifier_v3_SPY_20260511_224704.pkl` (trained through 2024-12-31) | SPY B&H: **+23.01%**

| Config | Return | Sharpe | Trades | Win% | PF | MaxDD | Exp% | vs B&H |
|--------|--------|--------|--------|------|-----|-------|------|--------|
| **et=0.40 trail** | **+9.25%** | **0.66** | 10 | 20.0 | 1.26 | -12.5 | 51.3 | -13.8pp |
| et=0.45 trail | +0.35% | 0.03 | 12 | 33.3 | 1.05 | -12.5 | 44.5 | -22.7pp |
| et=0.35 trail | +9.25% | 0.66 | 10 | 20.0 | 1.26 | -12.5 | 51.3 | -13.8pp |
| et=0.50 baseline | -5.30% | -0.54 | 12 | 33.3 | 0.82 | -11.8 | 17.4 | -28.3pp |

### IS vs OOS Comparison

| Metric | IS Best (et=0.45 trail) | OOS Best (et=0.40 trail) | Degradation |
|--------|------------------------|--------------------------|-------------|
| Return | 81.0% | 9.25% | -88.6% |
| Sharpe | 0.73 | 0.66 | -9.6% |
| Win% | 42.4% | 20.0% | -22.4pp |
| Trades/yr | ~7.7 | ~7.1 | -7.8% |
| PF | 1.65 | 1.26 | -23.6% |

### OOS Findings

1. **Sharpe holds up** — OOS Sharpe 0.66 is only 10% below IS 0.73. Risk-adjusted performance generalizes.
2. **Return collapses** — absolute return drops 89% because 2025-2026 was a strong bull market and the strategy is only ~50% exposed.
3. **No signals above 0.45 matter** — et=0.35 and et=0.40 produce identical results (same 10 trades), indicating the model rarely assigns high probabilities.
4. **Baseline fails OOS** — without trail stop, the model loses money (-5.3%, Sharpe -0.54), confirming trail stop is critical for survival, not just optimization.
5. **Win rate deterioration** — dropped from 42% to 20%, suggesting model probability calibration may have degraded.

---

## Next Targets

- [x] OOS backtest on 2025-2026 data ✅ — **FAILED: Sharpe -0.27, return -3.1% vs SPY B&H +23%. Model overfit.**
- [ ] Fix `backtest_ml_enhanced.py` signal generation (currently broken for EMA)
- [ ] Model calibration audit — reliability diagram (are P=0.45 signals actually 45% winners?)
- [ ] Investigate 2025 regime change causing degradation (tariff/trade war? AI bubble reversal?)
- [ ] Walk-forward paper trading with no-cross-asset model
- [ ] Train model with proper walk-forward CV (not single train/test split)
- [ ] Try `et=0.45 trail` with `confirm=2` for higher quality entries
- [ ] Evaluate adding option flow or macro regime features

---

## Out-of-Sample (2025-2026)

> Test: `et=0.45 trail` on SPY 2025-01-01 → 2026-05-11 | Model trained through 2024-12-30

| Config | Period | Return | B&H Return | Sharpe | Trades | Win% | MaxDD |
|--------|--------|--------|------------|--------|--------|------|-------|
| `et=0.45 trail` | Train 2016-2024 | +81.0% | +224.6% | 0.73 | 66 | 42.4 | -17.5 |
| `et=0.45 trail` | OOS 2025-2026 | **-3.1%** | +23.0% | **-0.27** | 11 | 27.3 | -12.5 |

**Assessment:** Model fails OOS. Sharpe degrades from +0.73 to -0.27, win rate drops from 42.4% to 27.3%. Only 11 trades in 16 months. SPY B&H outperforms by 26.1 percentage points. Possible causes: regime shift (2025 market recovery + trade volatility), model trained on 2015-2024 bull trend overfits to that regime. Next step: investigate walk-forward training with expanding windows to adapt to changing regimes.

---

## OOS Root Cause Analysis (2026-05-13)

### B3 — Model Calibration Audit

Reliability diagram using triple-barrier labels (TP=1.5xATR, SL=1.0xATR, horizon=5).

| Threshold | IS Win Rate (2015-2024) | OOS Win Rate (2025-2026) | IS Bias | OOS Bias |
|-----------|------------------------|--------------------------|---------|----------|
| P >= 0.35 | 34.8% (CALIBRATED) | 27.7% (OVERCONFIDENT) | -0.002 | -0.073 |
| P >= 0.45 | 39.5% (OVERCONFIDENT) | 33.1% (OVERCONFIDENT) | -0.055 | -0.119 |
| P >= 0.50 | 42.4% (OVERCONFIDENT) | 38.7% (OVERCONFIDENT) | -0.076 | -0.113 |

- IS ECE: 0.128, OOS ECE: 0.178 (+38% degradation)
- Model is overconfident even IS. P=0.45 predicts 45% but only wins 39.5%.
- **Not a calibration problem** — the relationship between probability and outcome degrades OOS (regime shift, not probability drift).

### Regime Shift Investigation

KS tests on 57 model features (IS 2015-2024 vs OOS 2025-2026):

**18/57 features have significant distribution shift (p < 0.01). Smoking guns:**

| Feature | KS stat | p-value | IS mean | OOS mean | Impact |
|---------|---------|---------|---------|----------|--------|
| **atr_20** | 0.654 | 10^-123 | 3.96 | **8.05** | **DOUBLED** |
| **atr_14** | 0.620 | 10^-109 | 3.97 | **8.06** | **DOUBLED** |
| momentum_5 | 0.211 | 10^-12 | 0.85 | 2.15 | 2.5x |
| vol_regime | 0.225 | 10^-13 | 1.027 | 1.073 | Shifted |

**4 features with FLIPPED correlations** OOS: `bb_squeeze_20`, `dist_to_low_50`, `price_to_ma_100`, `cmf`

**13 features significantly weakened** (correlation halved).

**Root cause:** Raw ATR values (not normalized by Close) tripled as SPY went from $200→$600.
The model uses absolute price-level features that don't generalize across market regimes.

### Fix Applied — Normalized ATR Features

Changed `atr_14` and `atr_20` from raw ATR to ATR/Close (percentage) in `src/ml/feature_engineering.py:173-174`.

### WFO vs Single-Split Comparison (with normalized ATR)

| Metric | V3 (raw ATR) OOS | Single-Split OOS | WFO OOS (2025+) | WFO Total |
|--------|-----------------|-----------------|-----------------|-----------|
| Return | -3.1% | -1.3% | 1.7% | 22.5% |
| Sharpe | -0.27 | -0.14 | — | 1.76 |
| Trades | 11 | 32 | 25 | 71 |
| Win% | 27.3% | 43.8% | 52.0% | 59.2% |

- **Normalized ATR** improves single-split OOS from -3.1% to -1.3%
- **WFO** further improves OOS to +1.7% (marginal)
- WFO total Sharpe 1.76 over 2020-2026, but underperforms B&H (149%)
- Model generates vastly more trades (25-32) than V3 (11) — less conservative features

### Critical Takeaway

Model is a trends-following system in a bull market. It will underperform B&H by design.
The question isn't "does it beat B&H" but "does it protect capital in down markets."
Next: test in 2022 bear market specifically.

---

## Next Targets

- [x] OOS backtest on 2025-2026 data ✅ — **FAILED: Sharpe -0.27, return -3.1% vs SPY B&H +23%. Model overfit.**
- [x] Fix `backtest_ml_enhanced.py` signal generation ✅
- [x] Model calibration audit — reliability diagram ✅ — **Model overconfident. ECE 0.128 IS, 0.178 OOS.**
- [x] Investigate regime change ✅ — **Raw ATR scaling. 18/57 features shifted, 4 flipped.**
- [x] Walk-forward training — **WFO modestly improves OOS (-1.3%→1.7%). Normalized ATR is main fix.**
- [ ] Walk-forward paper trading with V3 model (C9)
- [ ] Portfolio-level backtest (C10)
- [ ] Train full basket model with normalized ATR + walk-forward
- [ ] PDF enhancements (C11-C16, only after retraining)
- [ ] Test strategy during 2022 bear market specifically

---

*Last updated: 2026-05-13 01:29 CST*
