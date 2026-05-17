# Direction E: Foundation Model Integration — Agent Execution Plan

> **For AI agents inheriting this session**: This is the second track (after Direction C).
> Foundation models provide zero-shot forecasting without per-ticker retraining.
> Gated on: Direction C completion (or explicit decision to parallelize).

---

## Pre-Flight

```
□ Read MEMORY.md — check Direction C progress and any blockers
□ Read direction-c-research-signals.md — understand what was completed
□ Verify Chronos availability: `uv run python -c "import chronos; print(chronos.__version__)"`
□ Verify GPU: `uv run python -c "import torch; print(torch.cuda.is_available())"`
```

---

## Current State of Foundation Model Wrappers

| Model | Status | Wrapper File | Readiness |
|-------|--------|-------------|-----------|
| **Chronos-2** | Package NOT installed | `src/ml/models/chronos.py` (136 lines) | Wrapper exists, needs `uv add chronos-forecasting` |
| **FinCast** | Model NOT released | `src/ml/models/fincast.py` (89 lines) | Placeholder — raises NotImplementedError |
| **xLSTM** | Package NOT installed | `src/ml/models/xlstm.py` (201 lines) | Wrapper exists, needs training (not zero-shot) |

**Decision: Chronos is the primary target.** FinCast is unavailable. xLSTM requires training (defeats the "zero-shot" purpose). Chronos-2 provides zero-shot univariate/multivariate forecasting with multiple model sizes.

---

## Phase E1: Chronos Installation & Smoke Test

### AGENT TASK E1.1: Install Chronos

```bash
# Add chronos-forecasting package
uv add chronos-forecasting

# Verify installation
uv run python -c "
from chronos import ChronosPipeline, BaseChronosPipeline
print('Chronos installed successfully')
print('Available pipelines:', BaseChronosPipeline.__subclasses__())
"
```

**Pitfall:** Chronos requires PyTorch and transformers. If `uv add` fails with dependency conflicts:
1. Check `pyproject.toml` for existing torch version
2. Pin compatible versions: `uv add "chronos-forecasting>=1.0" "torch>=2.0"`
3. If GPU unavailable, use `device_map="cpu"` (slower but works)

### AGENT TASK E1.2: Smoke test Chronos zero-shot forecast

```bash
uv run python -c "
import pandas as pd
import numpy as np
from src.ml.models.chronos import ChronosForecaster

# Generate synthetic price data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=512, freq='D')
prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 512))
series = pd.Series(prices, index=dates)

# Try tiny model first (fastest, least RAM)
forecaster = ChronosForecaster(
    model_size='tiny',
    device='cpu',  # or 'cuda' if GPU available
    prediction_length=21,  # 1 month
    context_length=256,
)

# Generate forecast
result = forecaster.predict(series, num_samples=100)
print('Forecast shape:', result['mean'].shape)
print('Last 5 forecast values:', result['mean'].values[-5:])
print('SUCCESS: Chronos zero-shot forecast works')
"
```

**Expected output:** A 21-period forecast with mean, median, and quantile columns.

**Decision gate:**
```
IF forecast runs successfully (any output):
  → E1.1 PASS. Proceed to E1.3.
ELSE IF import error:
  → Fix dependency conflicts, re-run uv add.
ELSE IF OOM or CUDA error:
  → Switch to device='cpu' and model_size='tiny' (800MB RAM).
  → If still fails, Chronos is too heavy for current hardware — SKIP E1-E4, jump to E5 (cross-asset only).
```

### AGENT TASK E1.3: Verify the existing Chronos wrapper works

```bash
uv run python -c "
from src.ml.models.chronos import ChronosForecaster
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=512, freq='B')
returns = np.random.normal(0.0005, 0.01, 512)
prices = 100 * np.cumprod(1 + returns)
series = pd.Series(prices, index=dates)

fc = ChronosForecaster(model_size='tiny', device='cpu', prediction_length=5)
result = fc.predict(series)
assert 'mean' in result, 'Missing mean forecast'
assert 'median' in result, 'Missing median forecast'
print('ChronosForecaster wrapper verified')
"
```

---

## Phase E2: Chronos Signal Generator

### AGENT TASK E2.1: Create ChronosSignalGenerator

Create `src/signals/chronos_signal.py`:

```python
"""Chronos foundation model as signal source.

Zero-shot forecasting → directional signal → trade entry.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ChronosSignalGenerator:
    """Generate trading signals from Chronos zero-shot forecasts.

    Forecasts next N days of returns → compute expected direction → signal.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        forecast_horizon: int = 5,
        context_length: int = 252,
        signal_threshold: float = 0.005,  # minimum expected return for signal
    ):
        self.model_size = model_size
        self.device = device
        self.forecast_horizon = forecast_horizon
        self.context_length = context_length
        self.signal_threshold = signal_threshold
        self._forecaster = None

    def _get_forecaster(self):
        if self._forecaster is None:
            from src.ml.models.chronos import ChronosForecaster
            self._forecaster = ChronosForecaster(
                model_size=self.model_size,
                device=self.device,
                prediction_length=self.forecast_horizon,
                context_length=self.context_length,
            )
        return self._forecaster

    def generate_signal(
        self,
        prices: pd.Series,
    ) -> float:
        """Generate a trading signal from Chronos forecast.

        Args:
            prices: Historical OHLC close prices (needs at least context_length bars)

        Returns:
            Signal in [-1, 1]: positive = bullish, negative = bearish, 0 = neutral
        """
        if len(prices) < self.context_length:
            return 0.0

        recent = prices.iloc[-self.context_length:]

        try:
            forecaster = self._get_forecaster()
            result = forecaster.predict(recent, num_samples=50)
        except Exception as e:
            logger.warning(f"Chronos prediction failed: {e}")
            return 0.0

        # Compute expected return over forecast horizon
        forecast_mean = result["mean"]
        current_price = recent.iloc[-1]
        expected_return = (forecast_mean.iloc[-1] - current_price) / current_price

        # Convert to signal
        signal = np.clip(expected_return / self.signal_threshold, -1, 1)

        return float(signal)


class ChronosMultiHorizonSignal:
    """Multi-horizon Chronos signal with ensemble across horizons."""

    HORIZONS = [1, 5, 21]  # 1-day, 1-week, 1-month

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        weights: tuple[float, float, float] = (0.3, 0.4, 0.3),  # weights per horizon
    ):
        self.generators = {
            h: ChronosSignalGenerator(
                model_size=model_size,
                device=device,
                forecast_horizon=h,
            )
            for h in self.HORIZONS
        }
        self.weights = weights

    def generate_signal(self, prices: pd.Series) -> float:
        """Weighted ensemble across forecast horizons."""
        signals = []
        for h, gen in self.generators.items():
            sig = gen.generate_signal(prices)
            signals.append(sig)

        weighted = sum(w * s for w, s in zip(self.weights, signals))
        return float(np.clip(weighted, -1, 1))
```

### AGENT TASK E2.2: Integrate Chronos signal into MLStrategy

Modify `src/strategies/ml_strategy.py`:
1. Add import: `from src.signals.chronos_signal import ChronosSignalGenerator`
2. Add constructor params:
   - `use_chronos: bool = False`
   - `chronos_model_size: str = "tiny"`
   - `chronos_weight: float = 0.3`  # blend weight vs ML signal
3. In `next()`, after computing ML probability:
   ```python
   if self.use_chronos:
       chronos_signal = self._chronos_gen.generate_signal(self.data.Close)
       # Blend: chronos_weight * chronos_signal + (1-chronos_weight) * ml_signal
       blended = self.chronos_weight * chronos_signal + (1 - self.chronos_weight) * ml_prob
       ml_prob = blended
   ```
4. Add `--use-chronos`, `--chronos-weight` flags to `scripts/run_ml_backtest.py`

### AGENT TASK E2.3: Backtest Chronos-augmented strategy

```bash
# Quick smoke test on short period (Chronos is slow)
uv run scripts/run_ml_backtest.py SPY --start 2024-01-01 --end 2024-12-31 --use-chronos --chronos-weight 0.3

# Compare with baseline
uv run scripts/run_ml_backtest.py SPY --start 2024-01-01 --end 2024-12-31

# If too slow: add chronos caching
```

### Success Criteria for E2

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| Chronos signal generator runs | 0 exceptions | Backtest completes |
| Chronos signal is in [-1, 1] | All signals bounded | Unit test |
| Backtest completes within timeout | < 10 min for 1 year | Timed run |

### Decision Gate after E2

```
IF Chronos backtest runs and blends with ML signals:
  → E2 PASS. Foundation model is usable as signal source.
  → BUT: Chronos is slow (transformer inference each bar).
  → Consider: precompute forecasts nightly, cache to DuckDB.
  → Proceed to E3 (precomputation pipeline).
ELSE IF Chronos too slow for backtesting:
  → SKIP E3, jump to E4 (batch forecast + offline signal generation).
ELSE IF Chronos fails entirely:
  → This track is blocked on hardware. Proceed to Direction A+B.
```

---

## Phase E3: Chronos Precomputation Pipeline

### AGENT TASK E3.1: Build batch forecast script

Create `scripts/batch_chronos_forecast.py`:

```python
"""Precompute Chronos forecasts for all tickers and cache to DuckDB.

Runs daily: forecasts next 1/5/21 day returns for each ticker.
ML strategy reads cached forecasts at inference time.
"""
# Uses ChronosMultiHorizonSignal
# Outputs to data/chronos_forecasts.duckdb
# Schema: date, symbol, horizon_1d, horizon_5d, horizon_21d
```

### AGENT TASK E3.2: Create Chronos cache reader

Add `src/signals/chronos_cache.py`:
```python
"""Read precomputed Chronos forecasts from DuckDB cache.

Provides instant signal lookup without running transformer inference.
"""
```

### AGENT TASK E3.3: Update MLStrategy to use cache

Modify `src/strategies/ml_strategy.py`:
- If `chronos_cache_path` is set, read from DuckDB instead of running Chronos live
- This makes backtesting feasible (no transformer inference per bar)

---

## Phase E4: Cross-Asset Foundation Model Ensemble

### AGENT TASK E4.1: Cross-asset forecast aggregation

The 33-ticker experiment (C6b) showed that cross-asset features improve generalization.
Extend this to foundation models:

1. Run Chronos forecasts on SPY, QQQ, TLT, GLD, XLE simultaneously
2. Compute cross-asset expected return correlations
3. Weight forecasts by asset importance to target symbol

### AGENT TASK E4.2: Create CrossAssetChronosEnsemble

```python
class CrossAssetChronosEnsemble:
    """Ensemble Chronos forecasts across multiple related assets."""

    ASSETS = ["SPY", "QQQ", "TLT", "GLD", "XLE"]

    def generate_signal(self, target: str, prices: dict[str, pd.Series]) -> float:
        """Weighted forecast from cross-asset Chronos predictions."""
```

---

## Phase E5: Cross-Asset Feature Pipeline Upgrade (Fallback if Chronos fails)

### AGENT TASK E5.1: Enable cross-asset features at inference

**Problem identified in B6 (MEMORY.md):** MLStrategy uses FeatureExtractor per-ticker, so cross-asset features like `beta_QQQ_60d` are filled with 0.0 at inference.

**Fix:**
1. Modify `src/strategies/ml_strategy.py` to accept a basket of tickers for feature extraction
2. When `Data.basket` is available, pass all tickers to FeatureExtractor
3. Extract cross-asset features correctly (rel_ret, beta, corr for all pairs)
4. Gate: only enable for tickers in the training basket (33 tickers from C6)

### AGENT TASK E5.2: Test cross-asset inference

```bash
# Run backtest with cross-asset features enabled
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --cross-asset-inference

# Compare against baseline (no cross-asset)
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01
```

### Success Criteria for E5

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| Cross-asset features populated | No 0.0 fills for cross-asset cols | Debug print in MLStrategy |
| Performance difference from baseline | Sharpe delta > 0 | Compare backtests |
| No look-ahead | Verified with time check | `--strict-wf` flag |

---

## Reference: Model → Paper → Integration Path

| Foundation Model | Paper | Wrapper | Integration |
|-----------------|-------|---------|-------------|
| **Chronos-2** | arxiv.org/abs/2510.15821 | `src/ml/models/chronos.py` | PRIMARY — Zero-shot signal generator |
| **FinCast** | arxiv.org/abs/2508.19609 | `src/ml/models/fincast.py` | DEFERRED — Package not released |
| **xLSTM** | NeurIPS 2024 | `src/ml/models/xlstm.py` | SECONDARY — Requires training, not zero-shot |

---

*Plan generated: 2026-05-14. Gated on Direction C completion.*
