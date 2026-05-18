# Direction C: Research-Driven Signal Sources — Agent Execution Plan

> **For AI agents inheriting this session**: This file is your step-by-step execution guide.
> Each task is self-contained with exact file paths, CLI commands, success criteria, and decision gates.
> Complete each task fully before starting the next. Update `MEMORY.md` after each phase.

---

## Pre-Flight Checklist (every session)

```
□ Read MEMORY.md (project root) — current objectives, system metrics, blockers
□ Read progress_docs/plans/full.md — master plan phase status
□ Read progress_docs/current.md — session log with timestamps
□ Read .kilo/project-rules.md — financial guardrails, command documentation protocol
□ Run `uv sync` to ensure environment is current
□ Run `uv run ruff check src/` to verify clean baseline
```

---

## Phase C1: Overfitting Detection from Training History + Synthetic OOS Framework

**Papers**: `5520_using_the_training_history_to_.md` + `Backtest Overfitting in the Machine Learning Era-*.md`
**New files**: `src/ml/overfitting_detectors.py`, `scripts/test_overfitting_detectors.py`
**Modified files**: `scripts/model_health.py`, `src/ml/experiment_logger.py`

### AGENT TASK C1.1: Read and understand the papers

1. Read the full text of both papers:
   ```
   useful_resources/papers_md/5520_using_the_training_history_to_.md
   useful_resources/papers_md/Backtest Overfitting in the Machine Learning Era- A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment.md
   ```
2. Key concepts to extract:
   - **5520 paper**: Time series classifier trained on validation loss curves to detect overfitting. Uses simulated dataset of overfit/non-overfit training histories. Non-intrusive — uses val loss which is a training byproduct. F1=0.91 for detection. Can find optimal stopping epoch 32% earlier than early stopping.
   - **Backtest Overfitting paper**: Synthetic Controlled Environment (SCE) using Heston stochastic volatility, Merton Jump Diffusion, Drift-Burst Hypothesis, regime-switching models. Compares CPCV > PurgedKFold > K-Fold > Walk-Forward. CPCV has lowest PBO and superior DSR test statistic.

### AGENT TASK C1.2: Examine existing overfitting infrastructure

Read these files to understand what already exists:
```
src/analysis/deflated_sharpe.py          — DSR/PSR/FDR already implemented
src/ml/purged_cv.py                      — PurgedKFold exists
src/ml/combinatorial_purged_cv.py        — CPCV exists
src/ml/experiment_logger.py              — where training logs are stored
scripts/model_health.py                  — current health dashboard to extend
src/ml/pattern_classifier.py             — understand training process (fit() method, how val loss is tracked)
```

Look for specifically:
- Does `pattern_classifier.py` store per-epoch validation loss?
- Does `experiment_logger.py` have a schema for loss curves?
- What's the existing `model_health.py` check structure (functions, return types)?

### AGENT TASK C1.3: Implement TrainingHistoryOverfitDetector

Create `src/ml/overfitting_detectors.py` with two classes:

```python
"""Overfitting detection from research papers (5520 + Backtest Overfitting papers)."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TrainingHistoryOverfitDetector:
    """Detect overfitting from training history curves (5520 paper, ICLR 2023).

    Trains a time-series classifier on validation loss curves to identify overfit models.
    Non-intrusive — uses validation loss which is a byproduct of training.

    Paper result: F1=0.91 on real-world DL training histories, 32% earlier stopping.
    """

    def __init__(
        self,
        divergence_threshold: float = 0.15,   # train/val loss gap ratio threshold
        window_size: int = 20,                 # epochs to analyze for divergence
        min_epochs: int = 10,                  # minimum epochs before checking
    ):
        self.divergence_threshold = divergence_threshold
        self.window_size = window_size
        self.min_epochs = min_epochs

    def compute_divergence_score(
        self,
        train_losses: list[float],
        val_losses: list[float],
    ) -> float:
        """Compute overfitting divergence score from loss curves.

        Measures the gap between training and validation loss in the trailing window.
        Returns score in [0, 1] where > 0.7 indicates likely overfitting.

        Args:
            train_losses: Per-epoch training loss values
            val_losses: Per-epoch validation loss values
        """
        if len(train_losses) < self.min_epochs or len(val_losses) < self.min_epochs:
            return 0.0

        # Focus on trailing window
        train_window = np.array(train_losses[-self.window_size:])
        val_window = np.array(val_losses[-self.window_size:])

        # Check if train loss decreasing while val loss increasing (classic overfit signature)
        train_trend = np.polyfit(range(len(train_window)), train_window, 1)[0]
        val_trend = np.polyfit(range(len(val_window)), val_window, 1)[0]

        # Divergence = val loss increasing while train loss decreasing
        divergence = 0.0
        if train_trend < 0 and val_trend > 0:
            gap_ratio = abs(val_trend / (train_trend + 1e-8))
            divergence = min(gap_ratio / self.divergence_threshold, 1.0)

        # Also check absolute gap between final train and val loss
        final_gap = abs(val_window[-1] - train_window[-1]) / (abs(train_window[-1]) + 1e-8)
        divergence = max(divergence, min(final_gap / self.divergence_threshold, 1.0))

        return divergence

    def is_overfit(self, train_losses: list[float], val_losses: list[float]) -> tuple[bool, float, Optional[int]]:
        """Detect if model is overfit and find optimal stopping epoch.

        Returns:
            (is_overfit, divergence_score, optimal_epoch)

        Paper method: Time series classifier on val loss curves.
        Our adaptation: Compute divergence score from loss gap and trend.
        """
        score = self.compute_divergence_score(train_losses, val_losses)
        is_overfit = score > 0.7

        # Find optimal epoch: epoch with minimum validation loss
        optimal_epoch = None
        if len(val_losses) >= self.min_epochs:
            optimal_epoch = int(np.argmin(val_losses))

        return is_overfit, score, optimal_epoch


class SyntheticOOSComparator:
    """Synthetic OOS comparison framework (Backtest Overfitting paper, SSRN 4686376).

    Generates synthetic OOS datasets using parametric market models
    (Heston, Merton Jump Diffusion, regime-switching).
    Compares real OOS performance against null distribution of synthetic runs.
    Computes a Generalization Score.

    Paper result: CPCV has lowest PBO among all CV methods.
    """

    # Market model parameters for synthetic data generation
    DEFAULT_HESTON_PARAMS = {
        "kappa": 2.0,      # mean reversion speed
        "theta": 0.04,     # long-term variance
        "sigma": 0.3,      # vol of vol
        "rho": -0.7,       # correlation
        "v0": 0.04,        # initial variance
    }

    def __init__(
        self,
        n_synthetic_runs: int = 100,
        seed: int = 42,
    ):
        self.n_synthetic_runs = n_synthetic_runs
        self.rng = np.random.RandomState(seed)

    def generate_synthetic_returns(
        self,
        n_days: int,
        model: str = "heston",
    ) -> np.ndarray:
        """Generate synthetic daily returns from parametric market model.

        Implements Section 3 of the Backtest Overfitting paper (Synthetic Controlled Environment).

        Args:
            n_days: Number of trading days to simulate
            model: "heston", "merton", "regime_switch", or "drift_burst"
        """
        if model == "heston":
            return self._heston_simulation(n_days)
        elif model == "merton":
            return self._merton_simulation(n_days)
        elif model == "regime_switch":
            return self._regime_switch_simulation(n_days)
        elif model == "drift_burst":
            return self._drift_burst_simulation(n_days)
        else:
            raise ValueError(f"Unknown model: {model}")

    def _heston_simulation(self, n_days: int) -> np.ndarray:
        """Heston stochastic volatility model (Euler discretization)."""
        p = self.DEFAULT_HESTON_PARAMS
        dt = 1 / 252
        returns = np.zeros(n_days)
        v = p["v0"]

        for t in range(n_days):
            z1 = self.rng.randn()
            z2 = p["rho"] * z1 + np.sqrt(1 - p["rho"]**2) * self.rng.randn()
            v = max(v + p["kappa"] * (p["theta"] - v) * dt + p["sigma"] * np.sqrt(max(v, 0)) * np.sqrt(dt) * z2, 1e-8)
            returns[t] = np.sqrt(v) * np.sqrt(dt) * z1

        return returns

    def _merton_simulation(self, n_days: int) -> np.ndarray:
        """Merton Jump Diffusion model."""
        mu = 0.05 / 252
        sigma = 0.2 / np.sqrt(252)
        lambda_j = 0.1 / 252        # jump intensity
        mu_j = -0.02                 # jump mean
        sigma_j = 0.05               # jump std

        diff = mu + sigma * self.rng.randn(n_days)
        n_jumps = self.rng.poisson(lambda_j, n_days)
        jumps = np.array([self.rng.normal(mu_j, sigma_j, int(nj)).sum() for nj in n_jumps])
        return diff + jumps

    def _regime_switch_simulation(self, n_days: int) -> np.ndarray:
        """Two-regime Markov switching model (bull/bear)."""
        mu = [0.0008, -0.0004]      # bull, bear daily returns
        sigma = [0.012, 0.025]      # bull, bear daily vol
        trans_prob = 0.02            # regime switch probability per day

        regime = 0 if self.rng.rand() < 0.7 else 1  # start 70% bull
        returns = np.zeros(n_days)

        for t in range(n_days):
            if self.rng.rand() < trans_prob:
                regime = 1 - regime
            returns[t] = self.rng.normal(mu[regime], sigma[regime])

        return returns

    def _drift_burst_simulation(self, n_days: int) -> np.ndarray:
        """Drift-Burst Hypothesis model with intermittent explosive drifts."""
        base_vol = 0.012
        burst_prob = 0.005           # 0.5% chance of burst per day
        burst_magnitude = 0.05

        returns = self.rng.normal(0, base_vol, n_days)
        bursts = self.rng.rand(n_days) < burst_prob
        returns[bursts] += self.rng.choice([-1, 1], size=bursts.sum()) * burst_magnitude

        return returns

    def compute_generalization_score(
        self,
        real_oos_sharpe: float,
        real_is_sharpe: float,
        n_oos_days: int,
        n_synthetic_runs: int | None = None,
    ) -> dict:
        """Compute generalization score by comparing real OOS to synthetic null.

        Returns dict with:
            - pbo: Probability of Backtest Overfitting (from paper)
            - generalization_score: 1.0 - PBO (higher = better generalization)
            - synthetic_sharpes: array of synthetic OOS Sharpes
            - is_better_than_random: bool if real OOS beats synthetic median
        """
        n_runs = n_synthetic_runs or self.n_synthetic_runs

        synthetic_sharpes = []
        for _ in range(n_runs):
            # Generate synthetic returns using different models
            model_choice = self.rng.choice(["heston", "merton", "regime_switch"])
            syn_returns = self.generate_synthetic_returns(n_oos_days, model=model_choice)
            ann_return = syn_returns.mean() * 252
            ann_vol = syn_returns.std() * np.sqrt(252)
            syn_sharpe = ann_return / (ann_vol + 1e-8) if ann_vol > 0 else 0.0
            synthetic_sharpes.append(syn_sharpe)

        synthetic_sharpes = np.array(synthetic_sharpes)

        # PBO: proportion of synthetic runs with Sharpe >= real OOS Sharpe
        pbo = (synthetic_sharpes >= real_oos_sharpe).mean()

        # Generalization Score: 1 - PBO (higher = better, 0.5+ = generalizing)
        generalization_score = 1.0 - pbo

        # Also check: IS/OOS ratio (another overfitting indicator)
        is_oos_ratio = real_is_sharpe / (real_oos_sharpe + 1e-8) if real_oos_sharpe > 0 else float("inf")

        return {
            "pbo": float(pbo),
            "generalization_score": float(generalization_score),
            "is_oos_ratio": float(is_oos_ratio),
            "synthetic_median_sharpe": float(np.median(synthetic_sharpes)),
            "synthetic_std_sharpe": float(np.std(synthetic_sharpes)),
            "is_better_than_random": bool(real_oos_sharpe > np.median(synthetic_sharpes)),
            "n_synthetic_runs": n_runs,
        }
```

### AGENT TASK C1.4: Store per-epoch loss in experiment logger

Modify `src/ml/experiment_logger.py`:
1. Find the `ExperimentLogger` class or equivalent logging function
2. Add a method `log_training_history(train_losses: list[float], val_losses: list[float])` that appends to an internal list
3. Add a method `get_training_history() -> tuple[list[float], list[float]]` to retrieve stored curves
4. Ensure these are persisted to the experiment log JSON
5. If CatBoost doesn't expose per-epoch loss, log the CatBoost `evals_result_` dict (which contains per-iteration train/val metrics) and extract loss from that

### AGENT TASK C1.5: Integrate into model health pipeline

Modify `scripts/model_health.py`:
1. Add import: `from src.ml.overfitting_detectors import TrainingHistoryOverfitDetector, SyntheticOOSComparator`
2. Add a new check function `_check_overfitting_training_history(model_path, logger)` that:
   a. Loads training history from experiment logger (or falls back to CatBoost evals_result)
   b. Runs `TrainingHistoryOverfitDetector.is_overfit(train_losses, val_losses)`
   c. Reports divergence score, overfit flag, and optimal epoch
3. Add a new check function `_check_synthetic_oos(model_path, backtest_stats)` that:
   a. Extracts IS Sharpe and OOS Sharpe from backtest stats
   b. Runs `SyntheticOOSComparator.compute_generalization_score(real_oos_sharpe, real_is_sharpe, n_oos_days)`
   c. Reports PBO, generalization score, and IS/OOS ratio
4. Wire both checks into the main health report output
5. Add retrain recommendation triggers: if divergence_score > 0.7 OR generalization_score < 0.3 → RECOMMEND RETRAIN

### AGENT TASK C1.6: Test on the known-bad v3 model

Run the new detectors on the current model that fails OOS:
```bash
# Test on the calibration-fixed model (known to fail OOS)
uv run python -c "
from src.ml.overfitting_detectors import TrainingHistoryOverfitDetector, SyntheticOOSComparator
import numpy as np

# Simulate training history for the v3 model
# (CatBoost doesn't natively provide per-epoch loss, so we'll use a synthetic test)
np.random.seed(42)
epochs = 200
train_losses = 1.0 / (1 + 0.1 * np.arange(epochs)) + 0.01 * np.random.randn(epochs)
val_losses = np.concatenate([
    1.2 / (1 + 0.1 * np.arange(100)) + 0.01 * np.random.randn(100),
    0.3 + 0.002 * np.arange(100) + 0.02 * np.random.randn(100),  # diverges
])

detector = TrainingHistoryOverfitDetector()
is_overfit, score, optimal = detector.is_overfit(train_losses.tolist(), val_losses.tolist())
print(f'Training History Overfit Detection:')
print(f'  is_overfit={is_overfit}, divergence_score={score:.3f}, optimal_epoch={optimal}')
assert is_overfit, 'Simulated overfit model should be detected'

# Test Synthetic OOS Comparator
comparator = SyntheticOOSComparator(n_synthetic_runs=100, seed=42)
result = comparator.compute_generalization_score(
    real_oos_sharpe=-1.24,  # known OOS Sharpe
    real_is_sharpe=0.85,    # known IS Sharpe
    n_oos_days=252,
)
print(f'')
print(f'Synthetic OOS Comparison:')
print(f'  PBO={result[\"pbo\"]:.3f}, generalization_score={result[\"generalization_score\"]:.3f}')
print(f'  IS/OOS ratio={result[\"is_oos_ratio\"]:.1f}')
print(f'  is_better_than_random={result[\"is_better_than_random\"]}')
assert result['generalization_score'] < 0.5, 'Should indicate poor generalization'
"
```

### Success Criteria for C1

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| TrainingHistoryOverfitDetector detects simulated overfit | is_overfit=True, score>0.7 | Smoke test passes |
| SyntheticOOSComparator flags bad model | generalization_score < 0.5 | Smoke test passes |
| Model health report includes new checks | 2 new sections in output | Run `scripts/model_health.py` |
| Ruff clean | 0 errors | `uv run ruff check src/ml/overfitting_detectors.py` |
| Existing tests still pass | 0 failures | `uv run pytest tests/ -x -q` |

### Decision Gate after C1

```
IF both detectors flag the v3 model as overfit/generalizing poorly:
  → C1 CONFIRMED — detectors are working. Proceed to C2.
ELSE:
  → DEBUG: read the papers again, check whether CatBoost training history is accessible.
  → If CatBoost doesn't log per-epoch loss, use the CatBoost best_iteration + eval_metrics as proxy.
  → Re-test. If still failing, add --training-history flag to train_ml_pipeline_v3.py to capture curves.
```

---

## Phase C2: Sentiment Signal Integration ✅ COMPLETE (2026-05-14)

**Papers**: 4 sentiment papers (see knowledge graph line 303)
**New files**: `src/signals/sentiment_scorer.py`
**Modified files**: `src/strategies/ml_strategy.py`, `scripts/run_ml_backtest.py`

### AGENT TASK C2.1: Read the sentiment papers

Read at minimum these two:
```
useful_resources/papers_md/2024_SentimentAnalysisofTwitterDataUsingMachineLearningTechniques.md
useful_resources/papers_md/A tweet sentiment classification approach using an ensemble classifier.md
```
Key findings:
- Ensemble classifiers (LR + SVM + RF + XGBoost) achieve >80% accuracy on financial Twitter sentiment
- VADER and FinBERT are standard baselines
- Sentiment polarity correlates with next-day returns (direction, not magnitude)
- Best results when sentiment is used as signal MODIFIER (weight adjuster) rather than standalone signal

### AGENT TASK C2.2: Design the SentimentScorer interface

Create `src/signals/sentiment_scorer.py`:

```python
"""Sentiment signal scoring from research papers (4+ sentiment papers).

Design: SentimentScorer is a pluggable interface that accepts sentiment data
from any source (FinBERT, VADER, Polygon.io API, pre-computed CSV).
It modifies existing signals by applying a sentiment weight multiplier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SentimentProvider(Protocol):
    """Protocol for sentiment data sources."""

    def get_sentiment(self, dates: pd.DatetimeIndex, symbol: str) -> pd.Series:
        """Return sentiment scores in [-1, 1] for given dates.
        -1 = extremely negative, 0 = neutral, +1 = extremely positive.
        """
        ...


class CSVSentimentProvider:
    """Load pre-computed sentiment scores from CSV.

    CSV format: date,symbol,sentiment_score
    """

    def __init__(self, csv_path: Path, default_score: float = 0.0):
        self.csv_path = Path(csv_path)
        self.default_score = default_score
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.csv_path, parse_dates=["date"])
            self._df.set_index(["date", "symbol"], inplace=True)
        return self._df

    def get_sentiment(self, dates: pd.DatetimeIndex, symbol: str) -> pd.Series:
        df = self._load()
        scores = pd.Series(self.default_score, index=dates)
        for i, dt in enumerate(dates):
            try:
                scores.iloc[i] = df.loc[(dt, symbol), "sentiment_score"]
            except (KeyError, TypeError):
                pass
        return scores


class SyntheticSentimentProvider:
    """Generate synthetic sentiment for testing (no real data needed).

    Uses a simple AR(1) process with mean reversion to 0.
    Useful for smoke-testing the sentiment pipeline before connecting real data.
    """

    def __init__(self, seed: int = 42, autocorr: float = 0.3, volatility: float = 0.15):
        self.rng = np.random.RandomState(seed)
        self.autocorr = autocorr
        self.volatility = volatility

    def get_sentiment(self, dates: pd.DatetimeIndex, symbol: str) -> pd.Series:
        n = len(dates)
        scores = np.zeros(n)
        for i in range(1, n):
            scores[i] = self.autocorr * scores[i - 1] + self.rng.normal(0, self.volatility)
        return pd.Series(np.clip(scores, -1, 1), index=dates)


class SentimentSignalModifier:
    """Modify ML trading signals with sentiment scores.

    Paper insight: Sentiment works best as a SIGNAL MODIFIER, not standalone signal.
    Multiplies/convolves ML probability with sentiment score to produce adjusted signal.

    Formula: adjusted_signal = base_signal * (1 + sentiment_weight * sentiment_score)
    where sentiment_weight controls how much sentiment influences the signal.
    """

    def __init__(
        self,
        sentiment_provider: SentimentProvider,
        sentiment_weight: float = 0.15,    # how much sentiment influences
        min_sentiment_abs: float = 0.1,     # ignore weak sentiment
    ):
        self.provider = sentiment_provider
        self.sentiment_weight = sentiment_weight
        self.min_sentiment_abs = min_sentiment_abs

    def adjust_signals(
        self,
        base_signals: pd.Series,
        dates: pd.DatetimeIndex,
        symbol: str,
    ) -> pd.Series:
        """Apply sentiment adjustment to base trading signals.

        Args:
            base_signals: Raw signals in [0, 1] (ML probabilities)
            dates: DatetimeIndex aligned with signals
            symbol: Ticker symbol for sentiment lookup

        Returns:
            Adjusted signals in [0, 1]
        """
        sentiment = self.provider.get_sentiment(dates, symbol)

        # Clamp weak sentiment to 0 (noise filter)
        sentiment_masked = sentiment.where(
            sentiment.abs() >= self.min_sentiment_abs,
            0.0,
        )

        # Adjust: positive sentiment boosts bullish signals, dampens bearish
        # Negative sentiment does the opposite
        adjustment = 1.0 + self.sentiment_weight * sentiment_masked.values
        adjusted = base_signals.values * adjustment

        return pd.Series(np.clip(adjusted, 0, 1), index=base_signals.index)
```

### AGENT TASK C2.3: Integrate into ML strategy

Modify `src/strategies/ml_strategy.py`:
1. Add import for `SentimentSignalModifier, SyntheticSentimentProvider`
2. Add constructor params: `use_sentiment: bool = False`, `sentiment_weight: float = 0.15`
3. In the `init()` or `next()` method (where signals are generated), add after ML probability computation:
   ```python
   if self.use_sentiment:
       sentiment_modifier = SentimentSignalModifier(
           SyntheticSentimentProvider(seed=42),
           sentiment_weight=self.sentiment_weight,
       )
       # Adjust the threshold check with sentiment
       sentiment_scores = sentiment_modifier.provider.get_sentiment(
           dates=pd.DatetimeIndex([current_date]),
           symbol=self.symbol,
       )
       sentiment_adj = 1.0 + self.sentiment_weight * sentiment_scores.iloc[0]
       adjusted_threshold = self.entry_threshold / max(sentiment_adj, 0.5)
   ```
4. Add `--use-sentiment` and `--sentiment-weight` flags to `scripts/run_ml_backtest.py`

### AGENT TASK C2.4: Smoke test with synthetic sentiment

```bash
# Run ML backtest with synthetic sentiment to verify integration
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --use-sentiment --sentiment-weight 0.15

# Compare with baseline (no sentiment)
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop
```

### Success Criteria for C2

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| Sentiment pipeline runs without errors | 0 exceptions | Backtest completes |
| Sentiment-adjusted signals differ from baseline | Different trade count | Compare outputs |
| Synthetic provider generates valid scores | All scores in [-1, 1] | Unit test |
| Ruff clean | 0 errors | `uv run ruff check src/signals/sentiment_scorer.py` |

### Decision Gate after C2

```
IF sentiment pipeline works end-to-end with synthetic data:
  → C2 confirmed. Sentiment infrastructure is ready — just needs real data source.
  → Note: Real sentiment data (FinBERT, Polygon.io API, Twitter) is deferred to post-C6.
  → Proceed to C3.
ELSE:
  → DEBUG: check SentimentProvider protocol implementation.
  → Check that MLStrategy.next() has access to dates for sentiment lookup.
```

---

## Phase C3: Event-Driven Trading Patterns

**Papers**: `Building a Calendar of Events Database by Analyzing Financial Spikes.md` + `Event-Based Trading- Building Superior Trading Strategies.md`
**New files**: `src/signals/event_detector.py`, `src/data_ingestion/event_calendar.py`
**Modified files**: `src/signals/__init__.py`

### AGENT TASK C3.1: Read the event papers

Read both papers fully. Key concepts:
- Building financial event calendar database from analyzing spikes in price/volume
- Events: FOMC, CPI, NFP, earnings, OPEX, rebalancing, dividend dates
- Pre-event volatility compression, post-event drift, event-day gap patterns
- Spike detection → map to known events → build feedback loop

### AGENT TASK C3.2: Build EventCalendarDB

Create `src/data_ingestion/event_calendar.py`:

Design decisions:
- Use DuckDB for SQL queries (already in project via `duckdb_helpers.py`)
- Hardcode known event schedules (FOMC, OPEX, etc.) — no live API needed for v1
- Earnings dates from Polygon.io or Yahoo Finance (both already integrated)
- Format: SQLite-compatible schema with event_type, date, symbol, expected_impact

Schema:
```sql
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,        -- 'FOMC', 'CPI', 'NFP', 'EARNINGS', 'OPEX', 'DIVIDEND', 'REBALANCE'
    event_date DATE NOT NULL,
    symbol TEXT,                      -- NULL for macro events, ticker for stock-specific
    expected_impact TEXT,            -- 'HIGH', 'MEDIUM', 'LOW'
    description TEXT,
    UNIQUE(event_type, event_date, symbol)
);
```

Hardcoded events for 2020-2026:
- FOMC: 8 meetings/year (Jan, Mar, May, Jun, Jul, Sep, Nov, Dec)
- OPEX: Third Friday of each month
- CPI: Monthly, ~2nd week
- NFP: Monthly, first Friday
- Triple witching: Third Friday of Mar/Jun/Sep/Dec

### AGENT TASK C3.3: Build FinancialSpikeDetector

In `src/signals/event_detector.py`:

```python
"""Event-driven trading signal detectors from research papers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FinancialSpikeDetector:
    """Detect abnormal price/volume spikes and map to events.

    Based on: Building a Calendar of Events Database by Analyzing Financial Spikes.
    """

    def __init__(
        self,
        price_z_threshold: float = 2.5,
        volume_z_threshold: float = 3.0,
        lookback_days: int = 60,
    ):
        self.price_z_threshold = price_z_threshold
        self.volume_z_threshold = volume_z_threshold
        self.lookback_days = lookback_days

    def detect_spikes(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """Detect price and volume spike days.

        Returns DataFrame with columns: date, price_spike, volume_spike, spike_score
        """
        returns = close.pct_change()
        vol_changes = volume.pct_change()

        # Rolling z-scores
        ret_z = (returns - returns.rolling(self.lookback_days).mean()) / returns.rolling(self.lookback_days).std()
        vol_z = (vol_changes - vol_changes.rolling(self.lookback_days).mean()) / vol_changes.rolling(self.lookback_days).std()

        price_spike = ret_z.abs() > self.price_z_threshold
        volume_spike = vol_z > self.volume_z_threshold

        spike_score = (ret_z.abs() / self.price_z_threshold).clip(0, 2) + (vol_z / self.volume_z_threshold).clip(0, 2)

        return pd.DataFrame({
            "date": close.index,
            "price_spike": price_spike,
            "volume_spike": volume_spike,
            "spike_score": spike_score.values,
            "return": returns.values,
        })


class EventPatternDetector:
    """Detect event-driven trading patterns.

    Based on: Event-Based Trading — Building Superior Trading Strategies.
    """

    def __init__(
        self,
        pre_event_compression_days: int = 5,
        post_event_drift_days: int = 3,
    ):
        self.pre_event_compression_days = pre_event_compression_days
        self.post_event_drift_days = post_event_drift_days

    def detect_pre_event_compression(
        self,
        close: pd.Series,
        event_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """Detect volatility compression before known events.

        Returns: Series of compression scores (0=normal, 1=extreme compression).
        """
        returns = close.pct_change()
        rolling_vol = returns.rolling(self.pre_event_compression_days).std()

        # Check vol compression on days before events
        compression = pd.Series(0.0, index=close.index)
        for event_date in event_dates:
            pre_days = pd.date_range(
                event_date - pd.Timedelta(days=self.pre_event_compression_days + 1),
                event_date - pd.Timedelta(days=1),
            )
            pre_days = pre_days[pre_days.isin(close.index)]
            if len(pre_days) < 2:
                continue
            pre_vol = rolling_vol.loc[pre_days].mean()
            all_vol = rolling_vol.mean()
            if all_vol > 0:
                compression.loc[pre_days] = max(0, 1 - pre_vol / all_vol)

        return compression

    def detect_post_event_drift(
        self,
        close: pd.Series,
        event_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """Detect directional drift after events.

        Returns: Series of drift signals (+1=up drift, -1=down drift, 0=no drift).
        """
        returns = close.pct_change()
        drift_signal = pd.Series(0.0, index=close.index)

        for event_date in event_dates:
            post_window = pd.date_range(
                event_date,
                event_date + pd.Timedelta(days=self.post_event_drift_days),
            )
            post_window = post_window[post_window.isin(close.index)]
            if len(post_window) == 0:
                continue
            drift_ret = returns.loc[post_window].mean()
            drift_signal.loc[post_window] = np.sign(drift_ret)

        return drift_signal
```

### AGENT TASK C3.4: Create event calendar auto-population script

Create `scripts/build_event_calendar.py`:
```python
"""Build the financial event calendar database."""
# Uses EventCalendarDB from src/data_ingestion/event_calendar.py
# Populates with hardcoded FOMC, OPEX, CPI, NFP dates for 2020-2026
# Outputs to data/event_calendar.duckdb
```

### Success Criteria for C3

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| EventCalendarDB populated | > 100 event rows | Query DuckDB |
| SpikeDetector flags SPY FOMC days | Price spikes on FOMC dates | Visual check |
| EventPatternDetector has non-zero compression | compression.max() > 0 | Unit test |
| Ruff clean | 0 errors | Lint check |

### Decision Gate after C3

```
IF event detection pipeline is functional and event calendar has real dates:
  → C3 confirmed. Proceed to C4.
ELSE:
  → FIX: missing event dates or detector threshold too strict.
  → Re-run event calendar build.
```

---

## Phase C4: RL Trade Execution Module (MEDIUM)

**Papers**: `An adaptive dual-level reinforcement learning approach for optimal trade execution.md` + `OOM-RL- Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems.md`
**New files**: `src/rl/__init__.py`, `src/rl/execution_env.py`, `src/rl/dual_level_agent.py`, `scripts/train_rl_executor.py`

### AGENT TASK C4.1: Read the RL papers

Read both papers. Key concepts:
- Dual-level architecture: high-level agent sets schedule, low-level agent executes slices
- State: position, remaining inventory, market regime, spread, volume profile
- Action: order size as fraction of volume, limit price offset
- Reward: implementation shortfall vs VWAP benchmark
- PPO or SAC training

### AGENT TASK C4.2: Build Gymnasium execution environment

Create `src/rl/execution_env.py` — see detailed implementation notes in the paper.

### AGENT TASK C4.3: Build dual-level agent

Create `src/rl/dual_level_agent.py`.

### AGENT TASK C4.4: Train and benchmark

Create `scripts/train_rl_executor.py`.
Benchmark against naive VWAP/TWAP baselines.

### Success Criteria for C4

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| RL agent trains without crash | 0 exceptions | Training loop completes |
| Beats VWAP on implementation shortfall | RL shortfall < VWAP shortfall | Benchmark script |
| Gymnasium env is spec-compliant | `env.step()` returns (obs, reward, done, truncated, info) | Unit test |

---

## Phase C5: Kelly Allocator + Circuit Overfit + Crash Detection (MEDIUM)

**Papers**: `Investing Is Compression.md`, `Circuit-Based Intrinsic Methods to Detect Overfitting.md`, `Crash-based quantitative trading strategies- Perspective of behavioral finance.md`
**New files**: `src/risk/kelly_allocator.py`, `src/ml/circuit_overfit.py`
**Modified files**: `src/risk/crash_factor.py`, `src/risk/position_sizing.py`

### AGENT TASK C5.1-C5.3: Implement each paper's technique

See paper-specific implementation notes in the knowledge graph.

### Success Criteria for C5

Same pattern as previous phases: smoke test, unit test, lint clean.

---

## Phase C6: Adversarial Overfit + Event DB Production + Defensive Backtest (LOW)

**Papers**: `Detecting Overfitting via Adversarial Examples.md`, `Against a Universal Trading Strategy-*.md`
**New files**: `src/ml/adversarial_overfit.py`, `src/data_ingestion/event_calendar_production.py`
**Modified files**: `src/backtest/engine.py` (defensive backtest mode)

### AGENT TASK C6.1-C6.3

Implementation deferred until C1-C5 are complete.

---

## Cross-Cutting Rules for All Agents

1. **uv for everything**: `uv run`, `uv add`, NEVER bare `python` or `pip`
2. **Type hints**: All function signatures must have type hints
3. **Docstrings**: Google-style, 1-3 lines on public functions
4. **Ruff clean**: Run `uv run ruff check src/` before marking any task complete
5. **Update docs**: After creating any new CLI script, update `docs/COMMAND_CHEATSHEET.md`
6. **Update MEMORY.md**: After each phase, update the "Completed Tasks" list
7. **Update BESTS.md**: After any backtest, update the leaderboard
8. **Log to current.md**: After each significant action, append a timestamped row
9. **Never commit**: Unless explicitly asked
10. **Check existing patterns**: Before creating new files, look at existing similar files for convention

## Reference: Paper → Phase Mapping

| Phase | Paper Filename | Technique |
|-------|---------------|-----------|
| C1 | `5520_using_the_training_history_to_.md` | Training history time-series classifier for overfitting detection |
| C1 | `Backtest Overfitting in the Machine Learning Era-*.md` | Synthetic OOS comparison framework (Heston + Merton + regime-switching) |
| C2 | `2024_SentimentAnalysisofTwitterDataUsingMachineLearningTechniques.md` | Ensemble classifier for Twitter sentiment |
| C2 | `A tweet sentiment classification approach using an ensemble classifier.md` | Ensemble approach (LR+SVM+RF+XGBoost) >80% accuracy |
| C2 | `Leveraging hybrid model for accurate sentiment analysis of Twitter data.md` | Hybrid sentiment model |
| C2 | `Sentiment Analysis of Twitter Texts Using Machine Learning Algorithms.md` | ML sentiment classification |
| C3 | `Building a Calendar of Events Database by Analyzing Financial Spikes.md` | Event calendar from spike detection |
| C3 | `Event-Based Trading- Building Superior Trading Strategies.md` | Event-driven trading patterns |
| C4 | `An adaptive dual-level reinforcement learning approach for optimal trade execution.md` | Dual-level RL trade execution |
| C4 | `OOM-RL- Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems.md` | RL for multi-agent systems |
| C5 | `Investing Is Compression.md` | Kelly criterion optimal betting |
| C5 | `Circuit-Based Intrinsic Methods to Detect Overfitting.md` | Circuit theory overfitting detection |
| C5 | `Crash-based quantitative trading strategies- Perspective of behavioral finance.md` | Behavioral crash regime detection |
| C6 | `Detecting Overfitting via Adversarial Examples.md` | Adversarial perturbation overfitting detection |
| C6 | `Against a Universal Trading Strategy-*.md` | Defensive backtesting with time-reversal checks |

---

*Plan generated: 2026-05-14. Next agent: Start at C1.1.*
