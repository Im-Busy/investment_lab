# Investment Trying Lab

Multi-pattern trading system with ML-powered regime detection, 34+ chart pattern detectors, and a custom event-driven backtesting framework.

## Overview

- **Pattern Detection**: 30+ detectors across 7 categories (basic, breakout, candlestick, classic, complex, continuation, harmonic)
- **ML Pipeline**: CatBoost/LightGBM with PurgedKFold cross-validation, SHAP explainability, triple-barrier labeling, meta-labeling
- **Backtesting**: Custom event-driven engine + backtesting.py strategy wrappers
- **Risk Management**: Position sizing, circuit breakers, Monte Carlo VaR, diversity constraints
- **Regime Detection**: HMM, GMM, HDBSCAN, CNN-based regime classifiers
- **Portfolio**: Black-Litterman allocation, multi-strategy engine, signal aggregation with confluence scoring

## Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Core source code (20 subpackages) |
| `tests/` | Test suite (60+ test files) |
| `scripts/` | CLI scripts for training, backtesting, analysis |
| `notebooks/` | Jupyter notebooks (.py paired files) |
| `reports/` | Analysis results, charts, backtest metrics |
| `experiments/` | ML experiment configs and results |
| `pipeline/` | Prefect pipeline definitions |
| `docs/` | Project documentation |
| `plans_misc/` | Planning documents |
| `progress_docs/` | Structured progress tracking |

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run a backtest
uv run scripts/run_backtest.py --symbol SPY --start 2020-01-01
```

## Requirements

- Python 3.13+
- Package management via [uv](https://github.com/astral-sh/uv)
