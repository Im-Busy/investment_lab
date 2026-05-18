---
type: phase
phase: "17"
name: "Resource-Driven Enhancements — Factor Models + Strategy Conversion"
status: complete
started: 2026-05-16
completed: 2026-05-16
sub_phases:
  - name: "P0: Immediate (highest impact, zero deps)"
    status: complete
  - name: "P1: Near-term (high impact, light deps)"
    status: complete
  - name: "P2: Medium-term (moderate impact/complexity)"
    status: complete
  - name: "P3: Long-term (high complexity or library dependence)"
    status: complete
source: "华泰多因子系列1, Beyond Fama-French Integrating Factors, FMZ Strategies Repository (5,807 files)"
---

# Phase 17: Resource-Driven Enhancements

## Overview

14 additions extracted from 3 new resources ingested 2026-05-16. Ranked by impact × feasibility. No GPU or paid data gates on any item.

**Resources driving this phase:**
| Resource | Type | Key Contributions |
|----------|------|-------------------|
| 华泰多因子系列1 (Huatai Securities, 2016) | Chinese quant research report | 4-phase pipeline, 12 factor categories (74 factors), factor purification, HP filter forecasting, IR-weighted synthesis |
| Beyond Fama-French (Multilingual AI Survey, 2025) | AI-generated factor model survey | Default Risk + Liquidity factors, Factor Engine library, QRAFTI protocol, LLM+MCTS alpha mining |
| FMZ Strategies Repository (5,807 strategies) | Crowd-sourced strategy code | 6 PineScript→Python conversions (complete), ATR risk standard confirmed, multi-confirmation pattern validated |

**Already completed from this resource batch (2026-05-16):**
- 6 PineScript strategies → Python `BasePattern` detectors (`src/patterns/fmz/`)
- 1 JavaScript HFT signal → Python module (`src/signals/order_flow.py`)
- 17 PineScript functions → NumPy (`src/indicators/pinescript_helpers.py`)
- All 6 detectors registered in RulesFirstStrategy and optimized multi-pattern wrapper

---

## P0: Immediate — Highest Impact, Zero Dependencies

### R1: IR-Weighted Pattern Synthesis ★★★★★
*Source: 华泰 §2.3*
- **What:** Replace hardcoded `PATTERN_RELIABILITY` dict (45 static floats) with rolling Information Ratio weights computed from IS performance per pattern.
- **Why:** Equal-weight reliability doesn't adapt to regime shifts. A pattern that worked in 2020 may fail in 2025. IR = mean(return)/std(return) accounts for both return magnitude AND consistency.
- **Formula:** `weight_p = IR_p / Σ(IR_all)` where `IR_p = mean(ret_p) / std(ret_p)` over rolling window.
- **Architecture fit:** `src/signals/ir_weighting.py` — consumed by `RulesFirstStrategy.sigmoid()` scoring.
- **Lines:** ~80
- **Deps:** 0 new
- **File:** `src/signals/ir_weighting.py`

### R2: Factor Purification Module ★★★★
*Source: 华泰 §1.3*
- **What:** Before evaluating any pattern signal quality, regress out sector membership and market cap. Compute IC/Sharpe/IR on the purified residual signal.
- **Why:** A "good" pattern may just be a tech-sector beta proxy. Purification reveals whether the pattern itself has signal, or is riding sector/market factors. Prevents false confidence.
- **Formula:** `y = β0 + β1·sector_dummies + β2·log_mcap + ε` → purified signal = ε
- **Architecture fit:** `src/signals/factor_purification.py` — takes signal series + sector labels + size, returns purified series + purity_ratio.
- **Lines:** ~120
- **Deps:** 0 (OLS via numpy)
- **File:** `src/signals/factor_purification.py`

### R3: 4-Step Pattern Evaluation Gate ★★★★
*Source: 华泰 §1.3*
- **What:** Formal entry requirement for any new pattern entering the detector catalog. Four sequential tests:
  1. **Single-factor regression** with industry dummies → t-stat on factor return series
  2. **|t|>2 ratio + directional t-test** → classify as return factor (predicts direction) or risk factor (explains variance only)
  3. **IC analysis** — rank correlation of factor exposure at T vs return at T+1, after purification
  4. **Quantile backtest** — sort by signal, hold top/bottom N portfolios, sector-neutral
- **Why:** Prevents the exact FMZ anti-pattern (5,807 strategies, zero validation). Every `BasePattern` subclass must pass this gate before `_init_patterns()` accepts it.
- **Architecture fit:** `scripts/evaluate_pattern.py` — CLI tool + `src/signals/evaluation_gate.py` — reusable gate class.
- **Lines:** ~200
- **Deps:** statsmodels (for regression)
- **Files:** `src/signals/evaluation_gate.py`, `scripts/evaluate_pattern.py`

### R4: HP Filter Return Forecasting ★★★
*Source: 华泰 §2.5*
- **What:** Replace simple historical mean or EWMA with Hodrick-Prescott filter to extract trend from cumulative factor returns.
- **Why:** Huatai validated HP filter > EWMA > ARIMA > historical mean. HP filter eliminates noise while preserving trend. Minimal parameters (just smoothness λ).
- **Formula:** `min_τ Σ(y_t − τ_t)² + λ Σ[(τ_{t+1} − τ_t) − (τ_t − τ_{t-1})]²` — solved as tridiagonal linear system.
- **Architecture fit:** `src/ml/expected_returns.py` — consumed by any return forecasting step.
- **Lines:** ~30
- **Deps:** 0 (numpy tridiagonal solver)
- **File:** `src/ml/expected_returns.py`

---

## P1: Near-Term — High Impact, Light Dependencies

### R5: Collinearity Analysis for Pattern Overlap ★★★★
*Source: 华泰 §2.1-2.2*
- **What:** Compute VIF (Variance Inflation Factor) matrix across all 45 pattern signal series. Flag redundant patterns.
- **Why:** EMA-MACD-HF and Alpha Beast may fire simultaneously because both require trend+momentum. 45 patterns likely have significant overlap. Collinearity identifies which to synthesize and which to discard.
- **Rules:** Within-category correlation → IR-weight synthesize. Across-category correlation → discard weaker (lower IC).
- **Architecture fit:** `src/signals/collinearity.py` — VIF matrix + synthesis/discard recommendations.
- **Lines:** ~100
- **Deps:** statsmodels (for VIF)
- **File:** `src/signals/collinearity.py`

### R6: Liquidity Factor (CEI) ★★★★
*Source: Beyond Fama-French §2*
- **What:** Compute Acharya-Pedersen Capital Efficiency Index: CEI = 12-month change in market equity − 12-month cumulative return. Add as CatBoost feature.
- **Why:** Validated at r=0.9883 against Stata implementation. Proven alpha source beyond standard Fama-French factors. Zero fundamental data needed — pure price × shares outstanding.
- **Architecture fit:** `src/ml/factor_features.py` — alongside existing 14 fundamental factor features.
- **Lines:** ~60
- **Deps:** FMP API (shares outstanding)
- **File:** `src/ml/factor_features.py`

### R7: Multi-Dimensional Signal Scoring ★★★★
*Source: Beyond Fama-French §4, 华泰 §1.3*
- **What:** Replace single `confidence` score with 5-axis scoring: IC (predictive accuracy), IR (stability), turnover impact, diversity (vs other active signals), overfitting risk (IS/OOS ratio).
- **Why:** Single confidence conflates accuracy with stability. A pattern scoring 0.85 confidence with 300% turnover is worse than a 0.65 confidence pattern with 20% turnover.
- **Architecture fit:** `src/signals/scoring.py` — consumed by `SignalGenerator` and `ConfluenceScorer`.
- **Lines:** ~150
- **Deps:** 0
- **File:** `src/signals/scoring.py`

### R8: MAD + Rank Data Standardization Pipeline ★★★
*Source: 华泰 §1.2*
- **What:** Preprocessing pipeline: (1) median-based outlier removal `|x − median| > n × MAD`, (2) raw standardization or rank standardization (non-parametric).
- **Why:** Rank standardization is non-parametric — handles crypto's fat tails better than z-score. 华泰 recommends rank standardization for broader applicability.
- **Architecture fit:** `src/ml/preprocessing.py` — sklearn pipeline wrapper.
- **Lines:** ~100
- **Deps:** 0 (pure numpy)
- **File:** `src/ml/preprocessing.py`

---

## P2: Medium-Term — Moderate Impact/Complexity

### R9: Return Factor vs Risk Factor Classification ★★★
*Source: 华泰 §1.3*
- **What:** Classify each of the 45 patterns as "return factor" (directional, IC significant) or "risk factor" (explains variance, doesn't predict direction).
- **Why:** Return factors → entry signals. Risk factors → position sizing modifiers, not entry triggers. Misclassification means trading noise as signal.
- **Architecture fit:** `src/signals/classification.py` — annotation dict on each detector.
- **Lines:** ~100
- **Deps:** statsmodels (t-test)
- **File:** `src/signals/classification.py`

### R10: Performance Attribution Decomposition ★★★
*Source: 华泰 §4, Beyond Fama-French §3*
- **What:** Decompose strategy returns: `r_P = β_market + β_sector + Σ β_style + α_specific`. Identify genuine alpha vs factor beta.
- **Why:** If rules-first returns are 80% market beta + sector exposure, it's not alpha — it's leverage in disguise.
- **Architecture fit:** `scripts/attribution.py` — post-hoc CLI analysis, not real-time.
- **Lines:** ~180
- **Deps:** statsmodels (OLS)
- **File:** `scripts/attribution.py`

### R11: Default Risk Factor (Merton DtD) ★★★
*Source: Beyond Fama-French §2*
- **What:** Merton Distance-to-Default: DtD = (ln(V/D) + (r − σ²/2)T) / (σ√T). Add as CatBoost feature.
- **Why:** Default risk is a proven Fama-French extension. Captures credit quality signal that pure price patterns miss.
- **Architecture fit:** `src/ml/factor_features.py` — alongside R6 (CEI).
- **Lines:** ~80
- **Deps:** FMP (market cap, total debt, risk-free rate)
- **File:** `src/ml/factor_features.py`

### R12: QRAFTI Standardized Evaluation Protocol ★★
*Source: Beyond Fama-French §5*
- **What:** 14-test diagnostic suite (Novy-Marx/Velikov 2023) covering factor construction, signal quality, implementation feasibility. Standardized audit before production deployment.
- **Why:** Formal governance prevents bad models from reaching production. Complements R3 (4-step gate) with a broader audit.
- **Architecture fit:** `scripts/evaluate_qrafti.py`
- **Lines:** ~250
- **Deps:** 0 (self-contained tests)
- **File:** `scripts/evaluate_qrafti.py`

---

## P3: Long-Term — Higher Complexity or Library Dependence

### R13: Full IR→HP→Risk→QP Optimization Pipeline ★★
*Source: 华泰 §2.5-4*
- **What:** End-to-end optimization chain: IR-weighted synthesis → HP-filtered return forecast → factor covariance risk model → quadratic programming (max return given risk cap, or min risk given return floor).
- **Why:** Complements Phase 10a (Optuna/PyPortfolioOpt) with 华泰's systematic 4-phase approach. The QP formulation is standard (cvxopt/scipy).
- **Architecture fit:** `src/optimization/huatai_pipeline.py`
- **Lines:** ~400
- **Deps:** cvxopt or scipy.optimize
- **File:** `src/optimization/huatai_pipeline.py`

### R14: Factor Engine Library Wrapper ★
*Source: Beyond Fama-French §5*
- **What:** Thin try-install wrapper around the open-source Factor Engine Python library. Polars backend, decorator API. 11 validated factors.
- **Why:** If installable, gives proven factor computation for free. If not, skip — no lock-in.
- **Architecture fit:** `src/ml/factor_engine.py` — try import, use if available.
- **Lines:** ~40
- **Deps:** `factor-engine` package (availability unknown)
- **File:** `src/ml/factor_engine.py`

---

## Priority Order (recommended execution sequence)

```
1. R1 (IR-Weighted Synthesis)        ← Highest impact: rewires the core scoring
2. R2 (Factor Purification)          ← Removes false confidence in signals
3. R3 (4-Step Evaluation Gate)       ← Prevents bad patterns from entering
4. R4 (HP Filter Forecasting)        ← Quick win, 30 lines
5. R6 (Liquidity Factor CEI)         ← Proven alpha source
6. R7 (Multi-Dimensional Scoring)    ← Better signal ranking
7. R8 (MAD+Rank Pipeline)            ← Better preprocessing
8. R5 (Collinearity Analysis)        ← Removes redundant patterns
9. R9 (Return/Risk Classification)   ← Better signal routing
10. R10 (Performance Attribution)    ← Post-hoc alpha verification
11. R11 (Default Risk DtD)           ← Second alpha source
12. R12 (QRAFTI Protocol)            ← Governance
13. R13 (Full Optimization Pipeline) ← Large, complex
14. R14 (Factor Engine Wrapper)      ← Depends on library availability
```

---

## Integration Points with Existing System

```
R1 (IR Weights) ──→ RulesFirstStrategy.sigmoid()
R2 (Purification) ──→ PatternDetector evaluator
R3 (Eval Gate) ──→ _init_patterns() registration
R4 (HP Filter) ──→ Expected return estimator
R5 (Collinearity) ──→ Pattern redundancy analysis
R6 (Liquidity CEI) ──→ CatBoost feature set
R7 (Multi-Dim Scoring) ──→ SignalGenerator / ConfluenceScorer
R8 (MAD+Rank) ──→ ML preprocessing pipeline
R9 (Return/Risk) ──→ Signal routing logic
R10 (Attribution) ──→ Post-backtest CLI
R11 (Default DtD) ──→ CatBoost feature set
R12 (QRAFTI) ──→ Pre-production audit CLI
R13 (Optimization) ──→ Portfolio weight computation
R14 (Factor Engine) ──→ Optional factor feature provider
```
