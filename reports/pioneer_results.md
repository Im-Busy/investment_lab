# Pioneer Research Results

Aggregated results from the Phase 6b experimental trading features.
Each task follows a **prove-or-discard** model.

---

## Tier 1 Results

### P1.1 — Meta-Labeling (T9)

**Status:** Implemented
**Model:** LGBMClassifier (CatBoost substitution if AUC >= 1.10x)
**Files:** `src/ml/meta_labeler.py`, `tests/test_meta_labeler.py` (16 tests)
**Verdict:** Pending evaluation on real data

| Metric | Value |
|--------|-------|
| LGBM AUC (CV mean) | TBD — requires real historical signals |
| CatBoost AUC (CV mean) | TBD |
| Model Selected | TBD |
| Baseline Win Rate | TBD |
| Threshold | TBD (Youden's J optimal) |
| N Signals | TBD |

**Implementation:** 12 context features, LGBM/CatBoost with PurgedKFold CV,
model selection (1.10x rule), probability threshold via Youden's J,
save/load persistence. TripleBarrierLabeler integration for target generation.

### P1.2 — Gap-Fill Prediction (FS19)

**Status:** Implemented
**Model:** LGBMClassifier
**Files:** `src/ml/gap_fill_predictor.py`, `tests/test_gap_fill_predictor.py` (18 tests)
**Verdict:** Pending evaluation on real data

| Metric | Value |
|--------|-------|
| AUC | TBD — requires real OHLCV gap data |
| Accuracy | TBD |
| F1 Score | TBD |
| Baseline Fill Rate | TBD |
| N Gaps | TBD |

**Implementation:** 12 context features (including consecutive_gaps), gap
detection with fill verification, PurgedKFold CV, save/load persistence.
Configurable per market (equities/crypto/forex).

### P1.3 — Pattern Detector Ablation

**Status:** Implemented
**Files:** `scripts/ablate_patterns.py`
**Verdict:** Pending execution

| Pattern | Category | Trades | Win% | PF | Sharpe | DD% | Rec |
|---------|----------|--------|------|-----|--------|-----|-----|
| — | — | — | — | — | — | — | — |

**To run:** `uv run scripts/ablate_patterns.py --symbol SPY --start 2019-01-01 --end 2024-12-31`
(requires yfinance data access).

---

## Tier 2 Results

### P2.1 — Shapelets Pattern Discovery (FS16)

**Status:** Not started
**Verdict:** TBD

### P2.2 — VAE Latent Embeddings (FS15-lite)

**Status:** Not started
**Verdict:** TBD

### P2.3 — Heikin-Ashi Bar Integration (FS20)

**Status:** Not started
**Verdict:** TBD

---

## Tier 3 Results

### P3.1 — Alternative Bars (FS14)

**Status:** Not started (gated on Tier 1/2 results)
**Verdict:** TBD

---

## Summary

| Task | Status | Effort | Outcome |
|------|--------|--------|---------|
| P1.1 Meta-Labeling | Implemented | 3-4h | Pending eval on live data |
| P1.2 Gap-Fill | Implemented | 2h | Pending eval on live data |
| P1.3 Ablation | Implemented | 1-2h | Pending execution |
| P2.1 Shapelets | Not started | 3-4h | TBD |
| P2.2 VAE | Not started | 4-5h | TBD |
| P2.3 Heikin-Ashi | Not started | 1h | TBD |
| P3.1 Alt Bars | Not started | 8-12h | Gated |
