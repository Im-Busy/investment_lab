# Phase 24: Paper-Derived Enhancements (66-Paper Master Comparison)

> **Source:** `useful_resources/papers_md/MASTER_COMPARISON_REPORT_2026-05-21.md` — 250+ ideas from 66 papers
> **Date:** 2026-05-21
> **Status:** 🟡 PLANNING
> **Extraction:** "Think Freely, Then Compare" protocol applied across all 6 report sections
> **Gap Analysis:** 230 items already covered by Phases 01-23. 35 items identified as genuinely NEW.
> **Estimated LOC:** ~1,980 new, ~300 modified across 22 files. 1 new dep (dtaidistance).

---

## Cross-Reference Against Existing Plans

| Report Section | Total Ideas | Already Covered | NEW Gaps |
|---------------|-------------|-----------------|----------|
| A: SMC/ICT | 20 | 16 (Phase 22 G1-G10, Phase 14 AE, Phase 21) | 4 |
| B: RulesFirst | 30 | 16 (Phase 01, 15, 20, 21) | 14 |
| C: Infrastructure | 30 | 18 (Phase 11 B9-B14, Phase 12 T4, Phase 06 C5) | 12 |
| D: Sentiment/NLP | 15 | 11 (Phase 16 N/O/Q/R) | 4 |
| E: Risk/Portfolio | 15 | 8 (Phase 06, 15, 21) | 7 |
| Supplement (Indonesian) | 6 | 0 | 6 |
| **TOTAL** | **120** | **69** | **47 → consolidated to 35** |

---

## Task Table

| Priority | # | Task | Source | Depends | LOC | Notes |
|----------|---|------|--------|---------|-----|-------|
| **P0** | **P24-1** | Lock Box methodology for backtest validation | C1 | — | 50 | Set aside test data at start, access once after ALL decisions final. Process + code. |
| **P0** | **P24-2** | Blind analysis protocol hook | C13 | P24-1 | 50 | Shuffle target labels during hyperparameter tuning to prevent over-hyping. |
| **P0** | **P24-3** | Label-shuffling baseline test | C22 | — | 20 | Verify model doesn't exceed random on shuffled labels. Quick sanity check. |
| **P0** | **P24-4** | MRE-gap metric for overfitting quantification | C11 | — | 20 | (OOS_error − IS_error) / OOS_error — dimensionless single-number overfit score. |
| **P0** | **P24-5** | Huber loss for financial forecasting | A8 | — | 5 | Replace MSE with Huber (blends MSE+MAE). Prevents forecast collapse. |
| **P0** | **P24-6** | Causal masking verification in transformers | A17 | — | 5 | Ensure causal self-attention used in any transformer-based forecasting. Look-ahead prevention. |
| **P0** | **P24-7** | Three-value labeling (UP/DOWN/UNKNOWN) | B4 | — | 100 | Up 35% / Down 35% / Unknown 30% — captures transitional phases, reduces noise. |
| **P1** | **P24-8** | Feature importance regime monitoring | C15 | — | 50 | Track feature importance shifts across regimes. Alert when R² drops >30%. |
| **P1** | **P24-9** | Training history overfitting detector (KNN-DTW) | C10 | — | 200, dtaidistance | Classify overfitting from validation loss curve shape. F1=0.91, 32% earlier detection. |
| **P1** | **P24-10** | Profit Mirage counterfactual evaluation | C8 | — | 200 | Perturb inputs, measure prediction consistency. Detect memorization. |
| **P1** | **P24-11** | Confidence intervals on all performance estimates | C21 | — | 20 | Report 95% CI on Sharpe, return, win rate. Point estimates alone conceal uncertainty. |
| **P1** | **P24-12** | HBar novel indicator | B12 | — | 30 | (Close−Open)/(High−Low) normalized by average spread. HBar∈[-1,1]=Hold, >1=Buy, <-1=Sell. |
| **P1** | **P24-13** | iV volume indicator | B13 | — | 20 | Short-period vol / long-period vol. iV>>1 = growing activity supports direction. |
| **P1** | **P24-14** | Volatility no-trade switch with event calendar | B19 | — | 80 | Halt when 5d vol>1.5%, FED days, NFP days. Simple but high-impact risk filter. |
| **P1** | **P24-15** | Combined 4-indicator trend signal | B26 | — | 50 | RSI>50 AND CCI≥+100 AND MACD_Line>Signal AND ATR_rising → trend confirmed. |
| **P1** | **P24-16** | Signal alignment rule (fundamental + technical) | B27 | — | 30 | Fundamental direction MUST match technical; conflict → no trade. |
| **P1** | **P24-17** | Signal-strength position sizing | B20 | — | 30 | |aggregate|≥45→3 lots; 5-15→2 lots; <5→1 lot. Dynamic sizing from signal confidence. |
| **P2** | **P24-18** | RSI(20/80) thresholds (empirically validated) | B23 | — | 5 (config) | Buy at RSI=20, sell at RSI=80 beats RSI(30/70) for market timing. |
| **P2** | **P24-19** | Take-profit 3% / Stop-loss -2.5% (empirically validated) | B24 | — | 5 (config) | Highest win rate 56-58% from crash strategy empirical testing. |
| **P2** | **P24-20** | Event-type specific trading strategies | B21 | — | 200 | Different events have different optimal holding periods. Long-term (Lawsuits/Products), short-term (Dividends). |
| **P2** | **P24-21** | Event-time-weighted sentiment decay | B22 | — | 30 | Sum sentiment × exp(−days/10) — 10-day half-life for event relevance. |
| **P2** | **P24-22** | ETF portfolio rotation strategy | B29 | — | 250 | Train 45d → select top-10 → trade 45d → repeat. 102% vs 33% (single-symbol). |
| **P2** | **P24-23** | Instance normalization for financial data | A16 | — | 10 | Z-score each input before model. Removes scale bias, preserves relative shape. |
| **P2** | **P24-24** | 4H timeframe priority insight | B30 | — | 0 (design) | 4H significantly outperforms 1H/12H/1D (10× peak values across papers). Priority for forex. |
| **P2** | **P24-25** | Triangular hedge correlation-based pair selection | S1 | — | 80 | Scientific pair selection via negative correlation potential (Ciacci 2020 methodology). |
| **P2** | **P24-26** | EMA-gated triangular hedge entry | S2 | — | 40 | EMA as trend+volatility contextual gate before hedge deployment. |
| **P2** | **P24-27** | Binary state-machine hedge pattern | S4 | — | 60 | Two opposing triangular patterns form a simple alternator state machine. |
| **P2** | **P24-28** | Hedge-only triangular variant (no averaging) | S6 | — | 120 | Remove averaging (causes 61% DD). Deploy both sides, close at EMA reversal or fixed TP/SL. |
| **P3** | **P24-29** | Two-phase GA rule combination | B5 | — | 400 | Phase 1: GA per-rule params. Phase 2: GA/weighted voting combines surviving rules. |
| **P3** | **P24-30** | Divergence detection algorithms (explicit pseudocode) | B10 | — | 200 | Bullish/bearish regular+hidden divergences with RSI, MFI, DeMarker, Ultimate Oscillator. |
| **P3** | **P24-31** | 4-indicator fuzzy rule system with NSGA-II | B11, B14, B15 | — | 1,100 | dEMA+HBar+iV+BB; trapezoidal membership; NSGA-II optimization. |
| **P3** | **P24-32** | Divergence-in-bits strategy comparison metric | E5 | — | 40 | Δg = D_KL(W*||W_A) − D_KL(W*||W_B). Unit-independent, more robust than Sharpe. |
| **P3** | **P24-33** | Binomial VAR for event-driven risk | E6 | — | 100 | N deals × break probability → forward-looking risk, not historical VAR. |
| **P3** | **P24-34** | Dual alpha/dual beta model (bull vs bear) | E9 | — | 120 | Separate alpha/beta for bull vs bear. Detect convex-strategy phantom alpha. |
| **P3** | **P24-35** | W-Type Bottom & M-Type Top Bollinger patterns | B2 | — | 150 | 4-step confirmation: touch BB extreme → retrace middle → hold → break S/R. |

---

## Dependency Graph

```
P0 (Validation Gates) — NO dependencies, 7 items, ~250 LOC:
  P24-1 (Lock Box) ─────────────────────────────────────────────────────────┐
  P24-2 (Blind Analysis) ───── depends on P24-1 ────────────────────────────┤
  P24-3 (Label-shuffling) ──── independent ─────────────────────────────────┤ ALL feed
  P24-4 (MRE-gap) ──────────── independent ─────────────────────────────────┤ into ML
  P24-5 (Huber loss) ───────── independent ─────────────────────────────────┤ pipeline
  P24-6 (Causal masking) ───── independent ─────────────────────────────────┤ validation
  P24-7 (Three-value labels) ─ independent ─────────────────────────────────┘

P1 (Signal Quality + New Indicators) — depends on P0 items, 10 items, ~700 LOC:
  P24-8  (Feature importance monitoring) ─ independent
  P24-9  (Training history overfit) ────── independent
  P24-10 (Profit Mirage) ───────────────── independent
  P24-11 (Confidence intervals) ─────────── independent
  P24-12 (HBar indicator) ───────────────── independent
  P24-13 (iV indicator) ────────────────── independent
  P24-14 (Vol no-trade switch) ──────────── independent
  P24-15 (4-indicator trend) ────────────── independent
  P24-16 (Signal alignment) ─────────────── independent
  P24-17 (Signal-strength sizing) ───────── independent

P2 (Strategy Components) — depends on P0+P1, 11 items, ~875 LOC:
  P24-18 (RSI 20/80) ────────── config only
  P24-19 (TP 3%, SL -2.5%) ─── config only
  P24-20 (Event-type trading) ─ depends on B21 + B22
  P24-21 (Event-time sentiment) ── independent
  P24-22 (ETF rotation) ──────── independent
  P24-23 (Instance norm) ─────── independent
  P24-24 (4H priority) ───────── design only
  P24-25 (Triangular correlation) ─── independent
  P24-26 (EMA gate hedge) ────── depends on P24-25
  P24-27 (Binary state machine) ─ depends on P24-26
  P24-28 (Hedge-only variant) ── depends on P24-27

P3 (Heavy Lifts) — deferred, 7 items, ~2,110 LOC:
  P24-29 through P24-35
```

---

## Implementation Order

```
NOW (P0 — Quick Validation Gates, Day 1-2):
  P24-5 (Huber loss, 5 LOC) → P24-6 (Causal masking, 5 LOC) → P24-4 (MRE-gap, 20 LOC)
  → P24-3 (Label-shuffling, 20 LOC) → P24-7 (Three-value labels, 100 LOC)
  → P24-1 (Lock Box, 50 LOC) → P24-2 (Blind analysis, 50 LOC)

  P0 subtotal: 7 items, ~250 LOC, 0 deps. 1-2 sessions.

NEXT (P1 — Signal Quality, Day 3-5):
  P24-8 (Feature importance monitoring, 50 LOC)
  P24-9 (Training history overfit, 200 LOC + dtaidistance)
  P24-10 (Profit Mirage, 200 LOC)
  P24-11 (Confidence intervals, 20 LOC)
  → P24-12 (HBar, 30 LOC) + P24-13 (iV, 20 LOC)
  → P24-14 (Vol no-trade, 80 LOC) + P24-15 (4-indicator trend, 50 LOC)
  → P24-16 (Signal alignment, 30 LOC) + P24-17 (Signal-strength sizing, 30 LOC)

  P1 subtotal: 10 items, ~710 LOC, 1 dep. 2-3 sessions.

LATER (P2 — Strategy Components, Week 2):
  P24-18 + P24-19 (config only, immediate)
  → P24-21 (Event-time sentiment, 30 LOC) → P24-20 (Event-type trading, 200 LOC)
  → P24-22 (ETF rotation, 250 LOC)
  → P24-23 (Instance norm, 10 LOC)
  → P24-25 → P24-26 → P24-27 → P24-28 (triangular hedge chain, 300 LOC)

  P2 subtotal: 11 items, ~875 LOC. 2-3 sessions.

DEFERRED (P3 — Heavy Lifts):
  P24-29 through P24-35: ~2,110 LOC. Gate on P0+P1+P2 completion + positive OOS validation.
```

---

## Expected Impact

| Item | Signal Quality | Risk Reduction | Edge Over Current |
|------|---------------|----------------|-------------------|
| P24-1 Lock Box | High — prevents data leakage | Critical — #1 anti-overfitting | No lock box methodology |
| P24-2 Blind analysis | High — prevents over-hyping | High | No blind protocol |
| P24-5 Huber loss | Medium — prevents forecast collapse | Medium | MSE only |
| P24-7 Three-value labels | High — reduces false signals | Medium | Binary labels only |
| P24-9 Training overfit | High — 32% earlier detection | High | No loss-curve monitoring |
| P24-12 HBar + P24-13 iV | Medium — new signal sources | Low | Novel indicators, 0 deps |
| P24-14 Vol no-trade switch | Medium — avoids bad conditions | High — structural risk filter | No event-aware halt |
| P24-16 Signal alignment | High — confluence quality | High — prevents contrarian trades | No fundamental+technical alignment |
| P24-25-28 Triangular hedge | Medium — orthogonal alpha | Medium — hedged structure | No triangular hedging |

---

## File Change Summary

| File | Action | LOC Est. |
|------|--------|----------|
| `src/ml/training.py` | MODIFY (Huber loss) | +5 |
| `src/ml/models/` | MODIFY (causal masking check) | +5 |
| `src/ml/validation.py` | MODIFY (lock box, blind, label-shuffling, MRE-gap) | +140 |
| `src/ml/labeling.py` | MODIFY (three-value labels) | +100 |
| `src/ml/feature_importance_monitor.py` | NEW | +50 |
| `src/ml/overfitting_detector.py` | MODIFY (training history DTW) | +200 |
| `src/backtest/leakage_check.py` | MODIFY (profit mirage) | +200 |
| `src/backtest/reporting.py` | MODIFY (confidence intervals) | +20 |
| `src/indicators/hbar.py` | NEW | +30 |
| `src/indicators/ivol.py` | NEW | +20 |
| `src/risk/vol_gate.py` | MODIFY (no-trade switch + event calendar) | +80 |
| `src/signals/combined_trend.py` | NEW (4-indicator combo) | +50 |
| `src/signals/confluence.py` | MODIFY (signal alignment rule) | +30 |
| `src/risk/position_sizing.py` | MODIFY (signal-strength) | +30 |
| `src/strategies/crash_timing.py` (config update) | MODIFY (RSI 20/80 + TP 3%/SL -2.5%) | +10 |
| `src/signals/event_signals.py` | NEW (event-type trading) | +200 |
| `src/signals/sentiment.py` | MODIFY (event-time-weighted) | +30 |
| `src/strategies/etf_rotation.py` | NEW | +250 |
| `src/ml/preprocessing.py` | MODIFY (instance norm) | +10 |
| `src/strategies/triangular_hedge.py` | NEW (S1+S2+S4+S6) | +300 |
| `src/ml/feature_engineering.py` | MODIFY (three-value label integration) | +50 |
| `src/backtest/comparison.py` | MODIFY (divergence-in-bits) | +40 |
| `src/risk/binomial_var.py` | NEW (P24-33) | +100 |
| `src/analysis/dual_alpha_beta.py` | MODIFY (P24-34) | +120 |
| `src/patterns/bollinger/wm_types.py` | NEW (P24-35) | +150 |
| **TOTAL** | 17 new files, 8 modified | **+2,030 new, ~250 modified** |

---

## Testing Strategy

### Per-Item Tests
- P24-1: Lock box → verify test data never accessed during training
- P24-3: Label-shuffling → verify model AUC ≈ 0.50 on shuffled labels
- P24-4: MRE-gap → verify positive gap on known overfit model, near-zero on honest model
- P24-5: Huber → verify loss doesn't explode on outlier bars
- P24-7: Three-value → verify label distribution ≈ 35/35/30
- P24-9: Training history → verify F1 ≥ 0.85 on synthetic overfit curves
- P24-10: Profit Mirage → verify perturbation produces prediction variance
- P24-12/13: HBar/iV → verify [-1,1] range for HBar, ratio > 0 for iV

### Integration
- All new indicators wire into RulesFirstStrategy + SMCStrategy
- Backtest on SPY (2016-2024 IS, 2025-2026 OOS)
- Verify P24-7 (three-value labels) doesn't degrade existing Sharpe

### Validation Thresholds
- No regression on BESTS.md entries
- P24-7 three-value labels: must reduce trade count ≥ 10% (noise reduction) without reducing Sharpe > 0.05
- P24-14 volatility no-trade: must reduce max drawdown ≥ 2% absolute
- P24-1 Lock Box: must not change IS results (it's a process change)

---

*Plan created 2026-05-21. Source: `useful_resources/papers_md/MASTER_COMPARISON_REPORT_2026-05-21.md` gap analysis against Phases 01-23.*
