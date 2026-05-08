---
type: enhancement
name: Pioneer Research — Experimental Trading Features
status: in_progress
created: 2026-05-08
started: 2026-05-08
depends_on:
  - Phase 04 (ML Foundation) ✅
  - Phase 06 (Research-Based Enhancements) ✅
  - Enhancement ML Capabilities Phase 1-4 ✅
blocks: nothing — self-contained research phase, results feed back into Phase 03/04
rationale: |
  Remainder tasks from FS14-FS21 and T9 were deferred not because of GPU
  requirements (only FS15/FS18 are GPU-gated) but because their value is unproven
  or implementation effort is high. This phase systematically explores them in
  CPU-only environment, discarding what doesn't work and promoting what does.
---

# Phase 6b: Pioneer Research

## Overview

Systematically explore deferred experimental features. Each task follows a
**prove-or-discard** model: build a minimal viable implementation, evaluate
against baseline, and either promote to full integration or document the
negative result.

### LGBM-First Principle

ALL new ML components in this phase default to **LightGBM**. CatBoost is only
substituted when it demonstrates ≥10% improvement over LGBM on the task's
primary metric (AUC for classification, R² for regression) in PurgedKFold
cross-validation. This applies to T9, FS19, and any meta-learners.

Rationale:
- LightGBM is 3-8× faster to train on CPU than CatBoost
- GOSS sampling + EFB handles financial data well
- CatBoost's ordered boosting advantage shrinks when using PurgedKFold (proper
  temporal CV already prevents target leakage)
- Keeping a single default reduces maintenance surface

## Execution Order

```
Tier 1 (do first — low effort, quick prove-or-discard):
  ├── P1.1  Meta-Labeling Research (T9)
  ├── P1.2  Gap-Fill Prediction (FS19)
  └── P1.3  Pattern Detector Ablation Study

Tier 2 (medium effort):
  ├── P2.1  Shapelets Pattern Discovery (FS16)
  ├── P2.2  VAE Latent Embeddings [CPU] (FS15-lite)
  └── P2.3  Heikin-Ashi Bar Integration (FS20)

Tier 3 (high effort — only if Tier 1/2 produce positive results):
  └── P3.1  Volume/Dollar/Tick Alternative Bars (FS14)
```

---

## Tier 1 (High Priority)

### P1.1 — T9: Signal Meta-Labeling

| Attribute | Detail |
|-----------|--------|
| **Concept** | "Should I take this signal?" — separate model from "which direction?" |
| **Input** | Signal context: pattern type, confidence, regime, volatility, time, sector |
| **Target** | Binary: did trade hit TP before SL within time limit? (uses TripleBarrierLabeler from FS4) |
| **Model** | LGBMClassifier (default); CatBoostClassifier if AUC≥10% better on PurgedKFold |
| **Output** | `MetaLabeler.predict(signal) → {take_trade: bool, confidence: float}` |
| **Files** | `src/ml/meta_labeler.py` |
| **Tests** | Unit: `tests/test_meta_labeler.py` |
| **Metrics** | AUC, precision@k, % of false signals rejected, P&L improvement vs no meta-labeling |
| **Effort** | Medium — 3-4 hours |

**Why this is valuable:** Currently all signals above a confidence threshold are
traded. Meta-labeling filters out low-quality signals that the confidence score
can't catch — e.g., a high-confidence bullish flag in a bear market that always
fails. This is the most direct way to improve P&L without changing any detector
code.

**Architecture:**
```
TradeHistory + TripleBarrierLabeler
  → labeled dataset (pattern_context, took_it_or_not)
  → LGBMClassifier (PurgedKFold CV)
  → MetaLabeler.take_trade(signal) → bool
```

---

### P1.2 — FS19: Gap-Fill Prediction

| Attribute | Detail |
|-----------|--------|
| **Concept** | Predict whether an overnight/weekend gap will fill within N bars |
| **Input** | Gap size (%), pre-gap trend, volatility, volume profile, day-of-week |
| **Target** | Binary: gap filled within 5 bars? |
| **Model** | LGBMClassifier (default) |
| **Output** | `GapFillPredictor.predict(gap_pct, context) → {fill_prob: float, fill_bars_est: int}` |
| **Files** | `src/ml/gap_fill_predictor.py` |
| **Tests** | Unit: `tests/test_gap_fill_predictor.py` |
| **Metrics** | AUC, accuracy, calibration, backtest P&L of gap-fill-only strategy |
| **Effort** | Low — 2 hours |

**Why this was deferred:** "Niche case, small edge." But worth proving:
- Gaps are more common than assumed in crypto (24/7 trading has weekend
  liquidity gaps)
- If edge exists even at 55% accuracy, it's a mechanically simple trade
- Low effort to implement and evaluate
- If negative: document and close, no further investment needed

---

### P1.3 — Pattern Detector Ablation Study

| Attribute | Detail |
|-----------|--------|
| **Concept** | Systematically measure which of the 34 pattern detectors produce edge |
| **Input** | Each pattern detector individually, run through standard backtest |
| **Output** | Per-pattern metrics: Sharpe, win rate, profit factor, trade count, max DD |
| **Model** | No ML — pure backtest evaluation |
| **Files** | `scripts/ablate_patterns.py`, `reports/pattern_ablation.md` |
| **Effort** | Low — 1-2 hours |

**Approach:**
1. Run each pattern detector in isolation against SPY and BTC 2019-2024
2. Rank by OOS Sharpe, profit factor, trade count
3. Flag patterns with: Sharpe < 0, <30 trades, profit factor < 1.0
4. Report: recommended keep/cut/retune for each pattern

**Why this matters for pioneer research:** No point exploring niche features
(FS16, FS19) if core pattern detectors aren't pulling their weight. This study
identifies which patterns are worth building ML features around.

---

## Tier 2 (Medium Priority)

### P2.1 — FS16: Shapelets Pattern Discovery

| Attribute | Detail |
|-----------|--------|
| **Concept** | Learn discriminative price subsequences (shapelets) that predict future returns |
| **Input** | OHLCV windows (20-100 bars), labeled by forward return > 2× ATR |
| **Output** | Top-k shapelets with: shape visualization, discriminative score, occurrence stats |
| **Model** | `aeon` library — LearningShapeletClassifier, ShapeletTransformClassifier |
| **Files** | `src/ml/shapelet_discovery.py` |
| **Effort** | Medium — 3-4 hours |

**Evaluation:**
- Compare discovered shapelets against existing 34 rule-based patterns
- Check if shapelets find patterns the rules miss
- If shapelets have no incremental edge over rule-based: document and defer
- If shapelets outperform on specific regimes/assets: promote to PatternType

---

### P2.2 — FS15-lite: VAE Latent Embeddings [CPU]

| Attribute | Detail |
|-----------|--------|
| **Concept** | Train a small VAE on OHLCV windows to discover latent market structure |
| **Input** | OHLCV segments (window_size=50, stride=10) normalized per-window |
| **Architecture** | Encoder: 2-3 dense layers (256→128→64→32 latent). Decoder: symmetric. |
| **Output** | 32-dim latent embeddings per time window → cluster into market states |
| **Model** | PyTorch VAE, CPU-only, batch_size=64, early stopping |
| **Files** | `src/ml/vae_embeddings.py` |
| **Effort** | Medium — 4-5 hours |

**CPU feasibility with 32GB RAM:**
- Model size: ~600K params × 4 bytes = ~2.4MB in memory
- Batch processing: 64 windows × 50 bars × 4 features = 12,800 floats = ~100KB
- Even 100K windows: negligible memory footprint
- Training time: ~10-20 min for 50 epochs on CPU
- 32GB RAM is more than sufficient

**Evaluation:**
- Cluster latent embeddings → do clusters map to known regimes?
- Add VAE latent dimensions as features to PatternClassifier → AUC improvement?
- Compare against EBM shape functions (RS1) — which provides better alpha?

---

### P2.3 — FS20: Heikin-Ashi Bar Integration

| Attribute | Detail |
|-----------|--------|
| **Concept** | Generate Heikin-Ashi bars alongside standard OHLCV for pattern detection |
| **Input** | Standard OHLCV bars |
| **Output** | HA-OHLCV bars, pattern signals on HA bars |
| **Model** | Pure transformation — no ML |
| **Files** | `src/data/heikin_ashi.py` |
| **Effort** | Low — 1 hour |

**Research question:** Do any pattern detectors produce better signals on
smoothed HA bars? HA loses OHLC precision (close = average of O,H,L,C) but
reduces noise. This is a cheap experiment: convert bars, re-run top 5 patterns
from ablation study, compare metrics.

---

## Tier 3 (High Effort — Gate on Tier 1/2 Results)

### P3.1 — FS14: Volume/Dollar/Tick Alternative Bars

| Attribute | Detail |
|-----------|--------|
| **Concept** | Replace time-based bars with volume/dollar/tick bars for all 34 detectors |
| **Input** | Tick/volume data → aggregated bars |
| **Output** | Volume bars, dollar bars, tick bars as alternative OHLCV inputs |
| **Files** | `src/data/alternative_bars.py`, `scripts/rebase_patterns.py` |
| **Effort** | High — 8-12 hours |

**⚠️ WARNING — Only proceed if Tier 1 ablation shows significant noise in
time-based bars AND Tier 2 shapelets/VAE embeddings confirm market structure
benefits from alternative sampling.**

**Why deferred:** Requires re-basing all 34 pattern detectors to accept
alternative bar formats. High risk of breaking the fragile pattern system.

**Mitigation:** Start with a single detector (e.g., `DoubleBottom`) on a single
symbol (SPY). If alternative bars show ≥15% Sharpe improvement, expand to all
detectors. If not: kill immediately.

---

## Model Selection Policy

```
For each task requiring a classifier/regressor:

  1. Design feature set, target, and PurgedKFold evaluation
  2. Train LGBM with default hyperparameters
  3. Train CatBoost with matching hyperparameters
  4. If CatBoost primary metric ≥ 1.10× LGBM primary metric:
       → Use CatBoost for this task
     Else:
       → Use LGBM (default)
  5. Record decision + metrics in task's docstring
```

CatBoost is still available when it genuinely wins — the principle is
"prove the win, don't assume it."

---

## Success Criteria per Task

| Criterion | Threshold | Action if not met |
|-----------|:---------:|-------------------|
| P&L improvement over baseline | > 0 | Kill if negative or zero |
| AUC for classification tasks | > 0.55 | Kill if ≤ 0.55 (coin flip) |
| R² for regression tasks | > 0.05 | Kill if ≤ 0.05 (no signal) |
| Sharpe improvement | > 0.10 | Kill if ≤ baseline |
| Net profit factor | > 1.10 | Kill if ≤ 1.0 |

Tasks that fail their criteria get a **Research Note** in `reports/pioneer_results.md`
documenting what was tried and why it didn't work. Tasks that pass get promoted
to full Phase 06 integration.

---

## File Map

```
src/
├── ml/
│   ├── meta_labeler.py          # P1.1: T9 meta-labeling
│   ├── gap_fill_predictor.py    # P1.2: FS19 gap-fill prediction
│   ├── shapelet_discovery.py    # P2.1: FS16 shapelets
│   ├── vae_embeddings.py        # P2.2: FS15-lite VAE
│   └── __init__.py              # Add new exports
├── data/
│   ├── heikin_ashi.py           # P2.3: FS20 HA bar converter
│   └── alternative_bars.py      # P3.1: FS14 volume/dollar/tick bars
tests/
├── test_meta_labeler.py
├── test_gap_fill_predictor.py
├── test_shapelet_discovery.py
├── test_vae_embeddings.py
├── test_heikin_ashi.py
└── test_alternative_bars.py
scripts/
└── ablate_patterns.py           # P1.3: pattern ablation study
reports/
└── pioneer_results.md           # Aggregated results (prove or discard)
