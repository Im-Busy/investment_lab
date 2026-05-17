# BESTS Insights — What We Know from Our Best Results

> **Purpose:** This document is the distilled knowledge from every backtest in `BESTS.md`. It tells future AI sessions what works, what doesn't, and what to NEVER do. Load this before making any change to strategy code, ML pipeline, or backtest config.
>
> **Last synced with BESTS.md:** 2026-05-17
> **BESTS.md last_updated:** 2026-05-17 17:00
> **Tools added:** Comprehensive 125-instrument backtest + per-instrument bests

---

## I. Core Principles (Immutable)

### 1. Trail Stop Is Non-Negotiable
Every configuration with a trail stop outperforms its no-trail counterpart by 2-4x return. The mechanism: fixed TP kills winners early during bull trends; trail stop lets them ride while protecting downside.

**BESTS.md evidence:** `Best by Condition > With Trail Stop` — all trail configs dominate baseline.
- `et=0.45 trail` → Return 81.0%, Sharpe 0.73
- `et=0.45 baseline` → Return 21.9%, Sharpe 0.30

### 2. Rules-First > ML for OOS Robustness
Pure rule-based pattern detection (34 chart patterns, reliability weights, mr=0.70 filter) produces **+0.76 OOS Sharpe** on SPY 2025-2026. The best ML variant (RegimeRouter no-flipped) achieves only +0.35. **Adding ML to rules degrades** — every combined config underperforms pure rules-first OOS.

**BESTS.md evidence:** `Direction A+B > Combined ML+Rules Convergence`, `Direction A+B > Rules-First OOS`

### 3. Mid-Cap Commodities Are the Real Alpha Source
The biggest OOS winners are NOT SPY/QQQ mega-caps — they're mid-cap commodity-linked equities ($10-50B market cap):
- NEM (Gold miner, $25B): OOS +89%, Sharpe +0.94
- STLD (Steel, $20B): OOS +34%, Sharpe +0.72 (+1.26 with combined ML)
- NUE (Steel, $18B): OOS +15%, Sharpe +0.39

Mechanism: Above liquidity noise floor, below quant competition ceiling, below analyst saturation point. Fewer competing algorithms = slower price discovery = more time to capture pattern signals.

### 4. Calibration Matters (Not Just Model Quality)
Two of the biggest improvements came from fixing representation, not retraining:
- **Isotonic calibration** (replace buggy Platt): ECE 0.128→0.028. et=0.35 trail Return 69.4%→98.9%, Sharpe 0.59→0.85 (+44%)
- **Normalized ATR** (ATR/Close% instead of raw): Fixed V3 model OOS failure (Sharpe -0.27 → -0.14). Raw ATR tripled as SPY went $200→$600

### 5. IS Performance Is NOT Predictive of OOS
9/11 instruments improved from IS to OOS. JNJ went from -0.346 IS to +1.377 OOS (+1.723 Δ). Low-liquidity stocks had 1/8 positive IS but 5/8 positive OOS. **Never discard a strategy based on IS alone.**

---

## II. Factor Impact Rankings

| # | Factor | Direction | Impact Magnitude | Evidence |
|---|--------|-----------|-----------------|----------|
| 1 | **Trail stop** | MUST HAVE | +200-400% return | Every trail config beats baseline |
| 2 | **Normalized features** (ATR/Close%) | MUST HAVE | Prevents OOS collapse | Raw ATR caused V3 -0.27 OOS Sharpe |
| 3 | **Isotonic calibration** | MUST HAVE | +44% Sharpe | ECE 0.128→0.028 fix |
| 4 | **Rules-first over ML** | PREFER | +0.76 vs +0.35 OOS Sharpe | Rules survive regime shifts |
| 5 | **Mid-cap selection** ($10-50B) | PREFER | +0.94 Sharpe (NEM) | Academic + empirical support |
| 6 | **Meta-labeler** | NICE TO HAVE | +2-36% Sharpe | Filters low-quality signals, AUC 0.51-0.63 |
| 7 | **Entry threshold 0.35-0.45** | CONFIGURE | Varies by instrument | 0.50 too conservative, <0.35 noisy |
| 8 | **Volatility gate** | SITUATIONAL | PF 2.99 with vg=1.3 | Best quality but fewest trades |
| 9 | **Stability Selection** | QUALITY GATE | 38/98 features pass | PBO 0.061 PASS |
| 10 | **WFO training** | MARGINAL | +3% OOS improvement | Less impact than ATR normalization |
| 11 | **Conviction scaling** | **AVOID** | **Negative** | Over-weights losing trades |
| 12 | **Adding ML to Rules** | **AVOID** | **Negative** | Every blend underperforms pure rules |
| 13 | **Per-instrument tuning** | **NOT NEEDED** | 57% OOS pos across 125 instruments | Single config generalizes. IS→OOS corr = -0.198 — tuning on IS is dangerous. |

---

## III. Model Evolution Timeline

| Model | Date | Key Config | IS Sharpe | OOS Sharpe | Key Issue |
|-------|------|-----------|-----------|------------|-----------|
| V1-V2 | Before 2026-05-11 | Various | — | — | Early exploration |
| V3 (stale) | 2026-05-11 | et=0.45 trail | 0.73 | -0.27 | Raw ATR scaling bug |
| V3 + isotonic | 2026-05-14 | et=0.35 trail | 0.85 | -1.24 | Better IS, still fails OOS |
| Retrained B9-B14 | 2026-05-14 | Stability+CPCV+WFO | 1.01 (20-26) | 1.27 (holdout) | PBO 0.061 PASS, DSR 0.946 FAIL |
| Retrained + Meta | 2026-05-14 | primary+meta et=0.45 | 1.03 (20-26) | — | AUC 0.633, only 18 trades |

**Current best model:** `pattern_classifier_v3_SPY_20260514_124612.pkl` + `meta_labeler_v2_SPY_20260514_125515.pkl`

---

## IV. Strategy Tier List (OOS Performance)

### Tier S: Production-Ready
| Strategy | OOS Sharpe | OOS Return | Market |
|----------|-----------|------------|--------|
| Rules-First mr=0.70 et=0.55 | +0.76 | +9.2% | SPY |
| NEM Rules-First | +0.94 | +89.0% | Mid-cap Gold |
| STLD Combined ML+Rules | +1.26 | +62.4% | Mid-cap Steel |

### Tier A: Positive but Limited
| Strategy | OOS Sharpe | Notes |
|----------|-----------|-------|
| RegimeRouter no-flipped | +0.35 | Beats single model 90%+ |
| NUE Rules-First | +0.39 | Steel, lower than STLD |
| CVX-XOM Pairs | +0.40 | Only viable pair |

### Tier B: Marginal / Needs Work
| Strategy | OOS Sharpe | Issue |
|----------|-----------|-------|
| Retrained ML primary+meta | +1.03 (20-26) | Only 18 trades, low exposure |
| Signal-conflict combined | +0.41 | Still worse than rules-first |
| ADX Trend Strength (tech) | +0.53 (2015-26) | Only 17 trades in 11 years |
| Rules-First on HK stocks | -0.504 mean | HK-specific patterns fail |
| Rules-First on MicroCaps | -0.795 mean | Spreads + noise destroy edge |

### Tier F: Broken / Do Not Use
| Strategy | OOS Sharpe | Root Cause |
|----------|-----------|------------|
| Single ML model baseline | -1.25 | Regime shift, no adaptation |
| MeanReversion (standalone) | 0.00 | 1 trade, -100% return |
| RulesFirst trend (Phase 16) | -0.05 | 237 trades, noise-driven |
| Pairs (7/9 pairs) | Negative | No cointegration, single-leg |
| Micro-cap rules-first | Negative | Spreads/noise destroy edge |

---

## V. When to Review This Document

**REVIEW TRIGGERS** — Any of these events require re-reading this document and checking if insights still hold:

### Immediate Review (Major Changes)
1. **New row appears in BESTS.md top-3 of any category** with Sharpe > current leader
2. **A strategy moves from Tier F to Tier A or vice versa**
3. **New model is trained** (new .pkl file, retrained after regime shift)
4. **New asset class tested** (crypto, futures, forex, bonds)
5. **BESTS.md "last_updated" timestamp** is > 7 days newer than this document's sync date

### Consider Review
6. **New strategy module added** (new file in `src/strategies/`)
7. **Major parameter sweep completed** (entry threshold, trail stop, lookback)
8. **OOS window extends** (2025-2026 grows by more than 6 months of new data)
9. **Regime shift detected** (health check triggers, KL divergence spike)
10. **SPY down > 15% from peak** (bear market — re-test all strategies)

---

## VI. Anti-Patterns — What We Learned the Hard Way

| Anti-Pattern | Consequence | Lesson |
|-------------|------------|--------|
| Raw ATR features (not normalized) | V3 model OOS Sharpe -0.27 | Always use ATR/Close%, never raw ATR |
| Buggy Platt scaling | ECE 0.128, prob range 0.25-0.66 | Verify calibration math, prefer isotonic |
| Single train/test split | Overfitting to 2016-2024 regime | Use CPCV or WFO |
| Adding ML to working rules system | OOS Sharpe drops from +0.76 to +0.41 | ML is secondary, rules are primary |
| Conviction-based position sizing | Over-weights marginal trades | Fixed risk_pct more robust |
| Assuming IS predicts OOS | 9/11 instruments improved OOS | Always validate OOS |
| Treating micro-caps like large-caps | HIFS -55% OOS, KODK -60% | Micro-caps need 2-5% spread assumptions |
| Single-leg pairs execution | Not hedged, directional risk | Needs dual-leg or avoid pairs entirely |
| Long lookback for pairs (252d) | Smoothes away mean-reversion signal | Use 60-120 day lookback for pairs |

---

## VII. Production Configuration Matrix

### For SPY Large-Cap
```bash
# Primary system
uv run scripts/run_rules_backtest.py SPY --min-reliability 0.70 --entry-threshold 0.55 --trail-stop 3.0 --confluence 0.1

# Secondary (ML validation only, not primary signals)
uv run scripts/run_ml_backtest.py SPY --entry-threshold 0.45 --trail-stop --meta-labeler
```

### For Mid-Cap Commodities (NEM, STLD, NUE, FCX, X, CLF, AA)
```bash
# Rules-first with same config
uv run scripts/run_rules_backtest.py NEM --min-reliability 0.70 --entry-threshold 0.55 --trail-stop 3.0 --confluence 0.1

# ML as secondary filter (signal-conflict mode)
uv run scripts/run_combined_backtest.py NEM --mode signal-conflict --trail-stop
```

### For BTC / Crypto
```bash
# Higher entry threshold (0.65), must have cash >= $100K for position sizing
uv run scripts/run_rules_backtest.py BTC_USD --min-reliability 0.70 --entry-threshold 0.65 --trail-stop 3.0
```

### Regime Sensitivity Table
| Instrument Class | Best Config | Trail Stop | ML Role | OOS Sharpe | Notes |
|-----------------|-------------|------------|---------|------------|-------|
| US Large-Cap Indices (SPY) | Rules-First mr=0.70 | Yes (3.0 ATR) | Validation only | +1.675 | Anchor asset |
| US Sector ETFs (XLK/XLE/XLV) | Rules-First mr=0.70 | Yes (3.0 ATR) | None | +0.42 mean | All 4 positive OOS |
| Emerging Markets (EEM) | Rules-First mr=0.70 | Yes (3.0 ATR) | None | +1.642 | Phoenix — IS loser |
| Energy stocks (MPC/EOG/PSX/SLB) | Rules-First mr=0.70 | Yes (3.0 ATR) | None | +0.74 mean | Strongest stock sector |
| Mid-cap commodities (NEM/STLD/NUE) | Rules-First | Yes (3.0 ATR) | Secondary | +0.51 mean | Sweet spot |
| China A-shares (CN_CATL/Moutai) | Rules-First mr=0.70 | Yes (3.0 ATR) | None | +0.03 mean | 50% OOS positive |
| Consumer staples (WMT/MO/KO/PEP) | Rules-First mr=0.70 | Yes (3.0 ATR) | None | +0.05 mean | 5/8 OOS positive |
| Healthcare (JNJ/MRK/GILD/BIIB) | Rules-First mr=0.70 | Yes (3.0 ATR) | None | +0.09 mean | Phoenix cluster |
| Commodities (GLD/SLV) | Rules-First mr=0.70 | Yes (3.0 ATR) | None | +0.35 mean | 100% OOS-improved |
| Crypto (BTC/ETH) | Rules-First mr=0.70 | Yes (3.0 ATR) | None needed | +1.13 | 50% trades |
| Financials (JPM/BAC/C/GS) | Rules-First mr=0.70 | Yes (3.0 ATR) | None | -0.51 mean | IS→OOS collapse |
| Tech mega-cap (MSFT/NVDA/AAPL) | Rules-First mr=0.70 | Yes (3.0 ATR) | None | -0.14 mean | Death cross zone |
| Hong Kong (HK_AIA/HK_HSBC) | Rules-First mr=0.70 | Yes (3.0 ATR) | None | -0.50 mean | 25% OOS pos |
| Bonds (TLT) | Skip | — | — | -0.73 | Negative |
| Forex (EURUSD) | Skip | — | — | 0 trades | No edge |
| Micro-cap (<$3B) | **AVOID** | — | — | -0.80 mean | 0% OOS pos |

---

## VIII. Open Questions

1. **Bear market performance:** All testing is in bull regimes (2016-2024, 2025-2026 recovery). How does rules-first perform during 2022-style drawdowns? Need dedicated bear market backtest.

2. **Per-sector pattern calibration:** Mid-cap commodities outperform with large-cap calibrated patterns. Would per-sector calibration (different lookbacks, different thresholds) improve IS->OOS consistency?

3. **Multi-pair portfolio:** Can the 3 profitable pairs (CVX-XOM, DUK-SO, JNJ-MRK) be combined into a portfolio that diversifies away single-pair risk?

4. **Meta-labeler headroom:** Meta-labeler AUC 0.633 is marginal. Would different secondary features (macro, cross-sectional) improve the filter?

5. **Walk-forward frequency:** Monthly re-training vs quarterly vs annual — which produces the best OOS Sharpe?

6. **RegimeRouter with mid-caps:** Does RegimeRouter (ADX-based trend/MR routing) improve mid-cap commodity results, or is the simple rules-first approach sufficient?

7. **Energy sector OOS dominance:** MPC/EOG/PSX/SLB all exceeded +0.95 OOS Sharpe with the exact same config. Is this a structural edge (energy stocks amplify macro trends) or a 2025-2026 regime artifact (oil/gas price surge)?

8. **Death cross pattern:** MSFT (+0.59→-0.91), NVDA (+0.47→-0.81), COST (+0.43→-1.31) all had strong IS that collapsed OOS. Are mega-cap tech patterns regime-dependent in a way energy patterns are not?

9. **China/HK data quality:** 7/19 China instruments and 4/12 HK instruments had insufficient/empty data for backtesting. Would proper data coverage show similar patterns as US markets?

10. **Bonds/forex structural failure:** TLT (-0.73 OOS) and EURUSD (0 trades) show Rules-First patterns don't transfer to non-equity assets. Is this a data issue (fewer bars, different market structure) or a fundamental pattern mismatch?

---

## IX. Statistical Validation Toolkit (Added 2026-05-17)

### Bootstrap Sharpe CI
`src/analysis/deflated_sharpe.py` -- `bootstrap_sharpe_ci()`, `permutation_test()`, `format_significance_summary()`.

**What:** Adds empirical resampling-based significance tests to complement the parameterized PSR/DSR:
- Bootstrap 95% confidence interval on Sharpe ratio (iid or block bootstrap for autocorrelated returns)
- Monte Carlo permutation test -- shuffles returns to destroy structure, computes empirical p-value
- Convenience `print_significance_report()` -- runs all tests and prints a formatted report

**When to use:** After every backtest that produces promising results. At minimum: bootstrap CI and PSR. For formal strategy comparison: add DSR and permutation test.

### Factor IC/IR Analysis Pipeline
`scripts/run_factor_ic_analysis.py` -- standalone CLI for cross-sectional feature validation.

**What:** Computes Information Coefficient (IC), Rank IC, IC decay across horizons, rolling IC stability, and Information Ratio (IR) for every feature in the ML pipeline.

**When to use:**
- Before training a new ML model -- validate which features have predictive power
- After adding new features -- check IC vs existing features
- When a model degrades OOS -- identify which features lost predictive power
- Periodically (monthly) -- track IC stability of production feature set

**Expected thresholds:**
- |Rank IC| > 0.05 = weak signal, > 0.10 = moderate, > 0.15 = strong
- IR > 0.5 = usable, IR > 1.0 = good
- p < 0.05 = statistically significant
- Rolling IC std < mean IC = stable signal

### Calmar & Sortino in Batch Reports
Added to `scripts/backtest_rules_batch.py`, `scripts/backtest_rules_first.py`, `scripts/backtest_combined.py`.

**What:** Every batch comparison table now includes Calmar Ratio and Sortino Ratio alongside Sharpe. These complement Sharpe by:
- **Calmar** = CAGR / |MaxDD| -- "return per unit of worst pain," better for strategies with fat tails
- **Sortino** = same as Sharpe but only penalizes downside volatility -- better for asymmetric return distributions

**When to use:** Always present in batch backtest output. Prefer Calmar over Sharpe when max drawdown > 30%; prefer Sortino over Sharpe when win rate > 55%.

---

*Auto-generated from BESTS.md analysis. Review triggers at Section V above. Update this document whenever BESTS.md records a major change.*
