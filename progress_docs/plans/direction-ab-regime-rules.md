# Direction A+B: Regime-Adaptive ML + Rules-First Pattern System — Agent Execution Plan

> **For AI agents inheriting this session**: This is the third track (after C and E).
> Two parallel sub-tracks that converge: (A) regime-adaptive ML to fix OOS failure,
> (B) rules-first pattern system as fallback/diversification layer.
> Gated on: Direction C completion AND Direction E completion (or explicit decision).

---

## Pre-Flight

```
□ Read MEMORY.md — check Direction C + E progress
□ Read direction-c-research-signals.md + direction-e-foundation-models.md
□ Verify regime detectors exist: ls src/ml/*regime* src/indicators/regime*
□ Verify pattern detectors exist: ls src/patterns/*/*.py
□ Verify CHART_PATTERN_KNOWLEDGE_BASE.md exists (727 lines)
□ Verify BESTS.md has latest backtest results
```

---

## The Problem (Why A+B)

From MEMORY.md:
- ML model Sharpe 0.85 IS → **-1.24 OOS** (2025-2026)
- 3/38 features flipped, `vol_regime` KS=0.81 (complete regime shift)
- Model profitable in all regimes individually but fails when regimes change
- Transition regime weakest (-0.02% per signal)
- 8 regime detection methods built, **none deployed** in production pipeline

**Root cause**: Single model for all regimes. When regime shifts, feature relationships invert (KS=0.81), model breaks.

**Solution**: Per-regime models (Sub-track A) + rule-based pattern system as uncorrelated signal source (Sub-track B).

---

## Sub-track A: Regime-Adaptive ML

### Current State of Regime Detection

| Detector | File | Type | Status |
|----------|------|------|--------|
| HMM | `src/ml/hmm_regime.py` | Hidden Markov Model | Built, tested |
| GMM | `src/ml/gmm_regime.py` | Gaussian Mixture Model | Built, tested |
| PCA+KMeans | `src/ml/pca_kmeans_regime.py` | Unsupervised clustering | Built, tested |
| CNN | `src/ml/cnn_regime.py` | Convolutional NN classifier | Built |
| R² | `src/ml/r2_rd_regime.py` | R² regime detection | Built |
| Changepoint | `src/ml/change_point_regime.py` | Statistical changepoint | Built |
| Path Signature | `src/ml/path_signature_regime.py` | Rough path theory | Built |
| Macro | `src/ml/macro_regime.py` | Macro-economic features | Built |

All 8 have `RegimeDetectorBase` interface. **None are wired into ML training or inference pipeline.**

### AGENT TASK A1: Select and Benchmark Regime Detectors

Pick the top 3 detectors by stability and interpretability:

1. **HMM** — probabilistic, interpretable states (Bull/Bear/Ranging), industry standard
2. **GMM** — fast, handles multi-modal distributions, good for vol regime clustering
3. **PCA+KMeans** — simple, fast, good for feature-space regime clustering

**Benchmark script** (`scripts/benchmark_regimes.py`):
```bash
uv run scripts/benchmark_regimes.py SPY --start 2015-01-01 --end 2024-12-31
```

Compare:
- Number of detected regimes (target: 3-5, not too many)
- Regime stability (avg duration, transitions per year)
- Regime-conditional returns (are bear regimes actually bearish?)
- Correlation between detectors (are they redundant?)

**Decision gate:**
```
IF at least 2 detectors produce stable, interpretable regimes (3-5 states, consistent bear/bull returns):
  → A1 PASS. Select the top 2 for per-regime training. Proceed to A2.
ELSE:
  → Reduce to 2 simple binary regimes: Bull (price > 200MA) + Bear (price < 200MA).
  → Also add VolRegime: HighVol (VIX > 25) + LowVol (VIX < 25).
  → These are dead simple but interpretable and backtestable.
```

### AGENT TASK A2: Build RegimeRouter

Create `src/ml/regime_router.py`:

```python
"""Route predictions to per-regime ML models.

Instead of one model for all regimes, maintain N models (one per regime).
At inference, detect current regime → route to appropriate model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)


class RegimeRouter:
    """Route predictions to regime-specific CatBoost models.

    Each regime has its own trained model. The router:
    1. Detects current regime
    2. Routes features to the correct model
    3. Returns probability
    """

    def __init__(
        self,
        regime_detector,           # any RegimeDetectorBase
        model_paths: dict[int, Path],  # regime_id -> model .pkl path
        fallback_model_path: Path,     # used when regime is unknown
    ):
        self.detector = regime_detector
        self.models: dict[int, object] = {}
        self.fallback_model_path = fallback_model_path
        self.fallback_model = None

        # Load models
        for regime_id, path in model_paths.items():
            with open(path, "rb") as f:
                self.models[regime_id] = pickle.load(f)

        if fallback_model_path.exists():
            with open(fallback_model_path, "rb") as f:
                self.fallback_model = pickle.load(f)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict probabilities using regime-specific models.

        Args:
            features: Feature matrix with regime features included

        Returns:
            Series of probabilities [0, 1]
        """
        # Detect regime for each row
        regimes = self.detector.predict(features)

        probabilities = pd.Series(0.5, index=features.index)

        for regime_id in regimes.unique():
            mask = regimes == regime_id
            regime_features = features.loc[mask]

            model = self.models.get(regime_id, self.fallback_model)
            if model is None:
                continue

            try:
                probs = model.predict_proba(regime_features)[:, 1]
                probabilities.loc[mask] = probs
            except Exception as e:
                logger.warning(f"Prediction failed for regime {regime_id}: {e}")

        return probabilities
```

### AGENT TASK A3: Train Per-Regime Models

**Pipeline** (`scripts/train_per_regime_models.py`):

1. Load SPY data 2015-2024
2. Generate features (`feature_engineering.py`)
3. Run regime detector → label each bar with regime
4. For each regime with ≥ 500 bars:
   a. Split chronologically (70/15/15)
   b. Train CatBoost with triple-barrier labels + PurgedKFold
   c. Save model to `models/regime_{id}_SPY_{timestamp}.pkl`
5. Train ensemble model on ALL data as fallback
6. Save RegimeRouter config JSON

```bash
# Train per-regime models
uv run scripts/train_per_regime_models.py SPY --start 2015-01-01 --end 2024-12-31 --regime-detector hmm --n-regimes 4

# Test all 3 detectors
uv run scripts/train_per_regime_models.py SPY --regime-detector gmm --n-regimes 4
uv run scripts/train_per_regime_models.py SPY --regime-detector pca_kmeans --n-regimes 4
```

**Expected**: Per-regime models should have BETTER IS performance (less feature confusion across regimes) and potentially better OOS performance (correct model for current regime).

### AGENT TASK A4: Backtest RegimeRouter on OOS Data

```bash
# Test on 2025-2026 (known failure period)
uv run scripts/run_ml_backtest.py SPY --start 2025-01-01 --end 2026-05-14 --use-regime-router --regime-detector hmm

# Compare with single-model baseline
uv run scripts/run_ml_backtest.py SPY --start 2025-01-01 --end 2026-05-14
```

### AGENT TASK A5: Remove Flipped Features

From MEMORY.md: `vol_regime` KS=0.81 (complete flip), `volatility_regime` KS=0.36, `ema_21_55_spread` KS=0.31.

**Action:**
1. Modify `src/ml/feature_engineering.py` to add a `--exclude-features` flag
2. Retrain per-regime models WITHOUT `vol_regime`, `volatility_regime`, `ema_21_55_spread`
3. Compare OOS performance with/without flipped features

```bash
# Retrain without flipped features
uv run scripts/train_per_regime_models.py SPY --exclude-features vol_regime,volatility_regime,ema_21_55_spread
```

### Success Criteria for Sub-track A

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| RegimeRouter predicts without errors | 0 exceptions | Smoke test |
| Per-regime models train successfully | ≥ 2 regimes with ≥ 500 bars | Training log |
| OOS Sharpe improves vs single model | ΔSharpe > 0 | Compare backtests |
| RegimeRouter + flipped features removed | Better OOS than baseline | Compare |
| Existing tests still pass | 0 failures | `uv run pytest tests/ -x -q` |

### Decision Gate after Sub-track A

```
IF OOS Sharpe with RegimeRouter > 0.0 (profitable) and > single-model baseline:
  → A PASS. Regime-adaptive approach works. Wire into production pipeline.
  → Keep per-regime models as the primary ML signal source.
ELSE IF OOS Sharpe improved but still negative:
  → PARTIAL PASS. Regime routing helps but doesn't solve generalization.
  → Combine with Sub-track B (rules-first) for diversification.
ELSE IF OOS Sharpe unchanged or worse:
  → A FAIL. Regime routing alone doesn't fix the problem.
  → Sub-track B (rules-first) becomes the primary path forward.
  → ML demoted to supplementary signal source only.
```

---

## Sub-track B: Rules-First Pattern System

### What We Already Have

| Resource | Content | Status |
|----------|---------|--------|
| `src/patterns/` | 34 detectors across 7 categories | Built, tested |
| `src/signals/pattern_boost.py` | PatternBoostFilter with reliability weights | Built, integrated |
| `CHART_PATTERN_KNOWLEDGE_BASE.md` | 727 lines of pattern rules (entry/stop/target) | Complete |
| `BESTS.md` | Trailing stop confirmed +31% Sharpe | Proven |
| `src/analysis/ablation_engine.py` | Per-pattern contribution analysis | Tested |
| `src/analysis/contribution_analyzer.py` | Contribution to P&L | Tested |
| `src/analysis/synergy_analyzer.py` | Pattern co-occurrence effects | Tested |

### AGENT TASK B1: Build Rules-First Strategy from Knowledge Base

Create `src/strategies/rules_first_strategy.py`:

```python
"""Rules-first trading strategy using pattern detectors + knowledge base rules.

No ML model. Pure rule-based signals from:
1. Pattern detection (34 detectors, 7 categories)
2. Reliability weights from NCFE/Duddella research
3. Entry/stop/target rules from CHART_PATTERN_KNOWLEDGE_BASE.md
4. Volume/volatility confirmation
5. Trailing stop (proven +31% Sharpe boost)

Signal Formula:
    signal_score = Σ(pattern_detected × reliability_weight × direction)
                   + confluence_bonus (multi-pattern agreement)
                   + volume_confirmation_bonus
                   - counter_signal_penalty (conflicting patterns)

Entry: signal_score > entry_threshold (default 0.5)
Stop: ATR-based trailing stop
Target: Pattern-specific measured move from knowledge base
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Reliability weights from NCFE/Duddella research (CHART_PATTERN_KNOWLEDGE_BASE.md)
PATTERN_RELIABILITY: dict[str, float] = {
    "head_and_shoulders": 0.87,
    "inverse_head_and_shoulders": 0.87,
    "double_top": 0.83,
    "double_bottom": 0.83,
    "triple_top": 0.80,
    "triple_bottom": 0.80,
    "ascending_triangle": 0.78,
    "descending_triangle": 0.78,
    "symmetrical_triangle": 0.75,
    "cup_handle": 0.80,
    "flag": 0.72,
    "pennant": 0.78,
    "wedge": 0.75,
    "rectangle": 0.70,
    "engulfing": 0.72,
    "hammer": 0.68,
    "doji": 0.65,
    "harami": 0.65,
    "dark_cloud": 0.72,
    "gartley": 0.85,
    "abc_correction": 0.75,
    "gap": 0.70,
    "donchian_breakout": 0.70,
    "dead_cat_bounce": 0.78,
    "parabolic_arc": 0.75,
    "spike_ledge": 0.68,
    "three_hills": 0.75,
    "trader_vic_2b": 0.70,
    "floor_pivot": 0.60,
    "msl": 0.65,
    "nr7id": 0.68,
    "two_bar_reversal": 0.72,
    "matching_lows": 0.68,
    "n_bar_decline": 0.65,
}


class RulesFirstStrategy:
    """Rule-based multi-pattern trading strategy.

    Pure rules — no ML model. Uses pattern detectors + reliability weights
    + volume/volatility confirmation + trailing stops.

    Designed as an uncorrelated alternative to the ML strategy.
    Can be run independently or as a diversifying signal source.
    """

    def __init__(
        self,
        patterns: list[str] | None = None,  # subset of pattern names, None = all
        entry_threshold: float = 0.5,
        confluence_bonus: float = 0.10,
        volume_confirm: bool = True,
        trail_stop_atr: float = 2.0,
        min_reliability: float = 0.60,
    ):
        self.pattern_names = patterns or list(PATTERN_RELIABILITY.keys())
        self.entry_threshold = entry_threshold
        self.confluence_bonus = confluence_bonus
        self.volume_confirm = volume_confirm
        self.trail_stop_atr = trail_stop_atr
        self.min_reliability = min_reliability

        # Filter to patterns above reliability threshold
        self.active_patterns = {
            name: weight
            for name, weight in PATTERN_RELIABILITY.items()
            if name in self.pattern_names and weight >= min_reliability
        }

    def compute_signal(
        self,
        pattern_signals: dict[str, pd.Series],  # pattern_name -> Series of (-1, 0, +1)
        volume: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Compute aggregate signal from pattern detectors.

        Args:
            pattern_signals: Dict mapping pattern names to signal series
            volume: Volume series for confirmation
            close: Close prices

        Returns:
            Signal score series [0, 1] where > entry_threshold triggers entry
        """
        n = len(close)
        signal_score = pd.Series(0.0, index=close.index)

        # 1. Weighted pattern signals
        active_count = pd.Series(0, index=close.index)
        for name, weight in self.active_patterns.items():
            if name in pattern_signals:
                sig = pattern_signals[name]
                signal_score += weight * sig
                active_count += (sig != 0).astype(int)

        # 2. Confluence bonus: extra weight when multiple patterns agree
        confluence_mask = active_count >= 2
        signal_score += confluence_mask.astype(float) * self.confluence_bonus * np.sign(signal_score)

        # 3. Volume confirmation: scale signal by relative volume
        if self.volume_confirm and len(volume) > 20:
            rel_volume = volume / volume.rolling(20).mean()
            vol_multiplier = np.clip(rel_volume, 0.5, 2.0)
            signal_score = signal_score * vol_multiplier

        # 4. Normalize to [0, 1]
        max_abs = signal_score.abs().max()
        if max_abs > 0:
            signal_score = signal_score / max_abs
        signal_score = (signal_score + 1) / 2  # map [-1,1] to [0,1]
        signal_score = signal_score.clip(0, 1)

        return signal_score
```

### AGENT TASK B2: Wire RulesFirstStrategy into backtesting.py

Modify `src/strategies/backtest_py/multi_pattern_strategy.py`:
1. Import `RulesFirstStrategy`
2. Add constructor params for `entry_threshold`, `trail_stop_atr`, `confluence_bonus`
3. In `next()`:
   - Call all active pattern detectors
   - Pass results to `RulesFirstStrategy.compute_signal()`
   - Enter on signal > entry_threshold
   - Set trailing stop at `trail_stop_atr * ATR(14)`

### AGENT TASK B3: Backtest Rules-First on SPY

```bash
# Full rules-first backtest (no ML)
uv run scripts/backtest_rules_first.py SPY --start 2016-01-01 --end 2024-12-31 --trail-stop

# Sweep entry thresholds
uv run scripts/backtest_rules_first.py SPY --sweep-entry 0.3,0.4,0.5,0.6,0.7

# Sweep min reliability
uv run scripts/backtest_rules_first.py SPY --sweep-reliability 0.5,0.6,0.7,0.75
```

### AGENT TASK B4: Optimize Pattern Weights from Contribution Analysis

From the ablation study (done in F3 fix, current.md line 87):
- Top contributors: Triple Bottom (ΔSharpe +0.151), Gap Pattern (+0.142), Matching Lows (+0.128)
- 19/34 patterns generate solo trades
- 5 patterns identified as significant contributors

**Action:**
1. Load ablation results from `reports/ablation/solo_results.json`
2. Use per-pattern Sharpe contribution to adjust reliability weights
3. Retrain: empirical weights replace theoretical weights where empirical > theoretical
4. Document which weights changed and why

```bash
# Optimize pattern weights from ablation data
uv run scripts/optimize_pattern_weights.py --ablation-file reports/ablation/solo_results.json
```

### AGENT TASK B5: Backtest Rules-First on OOS and Compare

```bash
# OOS test (2025-2026) — the failure period for ML
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-05-14 --trail-stop

# Compare ML vs Rules-First OOS
# ML: Sharpe -1.24, Return -7.74%, 12 trades
# Rules-First: ??? (target: Sharpe > 0, Return > 0%)
```

### Success Criteria for Sub-track B

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| RulesFirstStrategy runs without errors | 0 exceptions | Backtest completes |
| Generates trades | ≥ 20 trades in 2016-2024 | Backtest stats |
| OOS Sharpe > ML OOS Sharpe | Rules-First Sharpe > -1.24 | Compare |
| Rules-First IS Sharpe competitive | Sharpe > 0.3 | Backtest stats |
| Updated BESTS.md with rules-first entry | New section | Check file |

### Decision Gate after Sub-track B

```
IF Rules-First OOS Sharpe > 0 (profitable):
  → B PASS. Rules-based system works where ML fails.
  → Rules-First becomes the primary production signal source.
  → ML demoted to supplementary (meta-labeling filter only).
  → Proceed to A+B Convergence.
ELSE IF Rules-First OOS Sharpe > -0.5 (less bad than ML):
  → PARTIAL PASS. Rules are more robust than ML in regime shift.
  → Combine with ML as diversification, not replacement.
ELSE IF Rules-First OOS Sharpe < -1.0:
  → B FAIL. Neither ML nor rules generalize.
  → The market itself may be the limiting factor (2025-2026 is anomalous).
  → Consider: waiting strategy, cash as default, only trade high-conviction regimes.
```

---

## Sub-track A+B Convergence: Combined System

### AGENT TASK AB1: Build CombinedStrategy

Create `src/strategies/combined_strategy.py`:

```python
"""Combined strategy: regime-adaptive ML + rules-first patterns.

Signal = α * ML_signal + (1-α) * Rules_signal
where α is dynamic based on:
  - Regime confidence (high confidence → more ML, low → more rules)
  - Recent performance (per-signal-source win rate)
  - Signal agreement (if both agree → higher conviction)
"""

class CombinedStrategy:
    def __init__(
        self,
        ml_weight: float = 0.5,
        dynamic_weight: bool = True,
        min_agreement: float = 0.6,  # both must exceed this to trade
    ):
        ...
```

### AGENT TASK AB2: Backtest Combined System

```bash
# Combined backtest with dynamic weighting
uv run scripts/backtest_combined.py SPY --start 2016-01-01 --end 2024-12-31 --dynamic-weight

# Static weights
uv run scripts/backtest_combined.py SPY --ml-weight 0.3 --trail-stop
uv run scripts/backtest_combined.py SPY --ml-weight 0.5 --trail-stop
uv run scripts/backtest_combined.py SPY --ml-weight 0.7 --trail-stop
```

### AGENT TASK AB3: Portfolio-Level Backtest

```bash
# Run combined system on 10-ticker basket
uv run scripts/portfolio_backtest.py --basket SPY,QQQ,XLK,GLD,TLT,IWM,D,XLF,XLE,SO --strategy combined --start 2020-01-01 --trail-stop
```

### Success Criteria for A+B Convergence

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| Combined Sharpe > max(ML_Sharpe, Rules_Sharpe) | Portfolio benefit | Compare backtests |
| Correlation between ML and Rules signals < 0.5 | True diversification | Compute |
| Portfolio-level DD reduced | DD < max(ML_DD, Rules_DD) | Compare |

---

## Final Decision Framework

After all three tracks complete, evaluate:

```
                         Direction C (Research Signals)
                               │
                               ▼
                         Direction E (Foundation Models)
                               │
                               ▼
                    Direction A+B (Regime + Rules)
                               │
                               ▼
                    ┌──────────────────────┐
                    │  PRODUCTION DECISION  │
                    └──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        Rules-First      Combined ML+Rules   ML+Regime
        (if B passes)    (if both partial)   (if A passes)
              │                │                │
              ▼                ▼                ▼
         Deploy rules    Deploy combined    Deploy regime-
         + ML as filter  with dynamic α     adaptive ML
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                         Paper Trading
                               │
                               ▼
                        Live Deployment
```

---

## Cross-Cutting Rules

Same as Direction C plan — see that file for full list.

---

*Plan generated: 2026-05-14. Gated on Direction C + E completion.*
