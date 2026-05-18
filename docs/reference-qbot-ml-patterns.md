# Qbot — ML/DL Strategy Patterns

> Source: [Qbot](https://github.com/UFund-Me/Qbot) (`C:\Dev\useful_repos\Qbot`)
> Studied: 2026-05-16 (R18d)
> Status: Study only — reference value for architecture patterns.

## Reality Check vs Initial Evaluation

Initial Phase 18 evaluation estimated "300+ ML/DL model implementations + 30+ indicator strategies." Actual codebase:

| Metric | Estimated | Actual |
|--------|-----------|--------|
| Total Python files | — | 423 |
| DL model strategies | "25+" | 2 (LSTM + RL, basic templates) |
| Classic indicator strategies | "30+" | ~10 (ADX, Bollinger, EMA, SMA, SSA, etc.) |
| Trading engines | — | 4 (vnpy.cn, easytrader, pyfunds, pytrader) |
| Plugins | — | 3 (auto_monitor, dagster, investool/quantstats) |

**Verdict: Over-estimated.** Qbot is primarily a trading infrastructure project (GUIs, engines, data providers), not a DL model catalog. The 2 DL strategies are basic Keras templates (150 lines each), not production-grade implementations.

## What Qbot Actually Provides

### Strategy Layer (`qbot/strategies/`, 18 files)

| File | Type | Value |
|------|------|-------|
| `lstm_strategy_bt.py` | LSTM price prediction (Keras Sequential) | Basic — 50-neuron LSTM, 20-bar lookback, MinMaxScaler |
| `rl_strategy_bt.py` | RL agent via rlkit (BacktraderEnv) | Minimal — incomplete integration, placeholder |
| `ssa_strategy_bt.py` | Singular Spectrum Analysis (SVD-based) | Interesting — custom bt.Indicator, SVD decomposition |
| `adx_strategy.py` | ADX trend following | Standard |
| `boll_strategy.py` | Bollinger Band mean reversion | Standard |
| `bigger_than_ema.py` | EMA crossover | Standard |
| `sma_cross_strategy_bt.py` | SMA crossover | Standard |
| `multi_strategy_bt.py` | Multi-strategy combo | Template |
| `undervalued_stock_picking_strategy.py` | Fundamental value | Interesting pattern |

### Engine Layer (`qbot/engine/`, ~50 files)

| Component | Purpose |
|-----------|---------|
| `engine/trade/` | Live trading via easytrader, vnpy.cn, pyfunds, pytrader |
| `engine/backtest/` | Backtrader-based backtesting |
| `engine/config.py` | Unified engine configuration |

### Plugin System (`qbot/plugins/`, ~30 files)

| Plugin | Purpose |
|--------|---------|
| `auto_monitor.py` | Real-time market monitoring |
| `dagster/` | Pipeline orchestration (Dagster) |
| `investool/` | Go-based data analytics tool |
| `quantstats/` | Performance analytics + visualization |

## Applicable Patterns for investment_trying

### Pattern 1: Multi-Engine Architecture

Qbot separates concerns cleanly:

```
GUI (qbot/gui/) → Strategy (qbot/strategies/) → Engine (qbot/engine/)
                                                     ├── backtest/  (offline)
                                                     └── trade/     (live)
```

**Application:** Our project already follows a similar pattern (src/strategies/ → scripts/backtest_*). Qbot's structure validates this approach.

### Pattern 2: Strategy-as-Plugin Registration

Qbot strategies are self-contained modules with `bt.Strategy` subclass. Registration happens via file discovery, not explicit imports.

**Application:** Consider a strategy registry pattern for our 5+ strategy variants (rules_first, combined, mean_reversion, pairs, regime_router) to enable `--strategy` CLI selection without if/elif chains.

### Pattern 3: SSA Indicator (Singular Spectrum Analysis)

`ssa_strategy_bt.py` implements a custom `bt.Indicator` that:
1. Builds a trajectory matrix from price window
2. Performs SVD decomposition
3. Extracts the dominant eigencomponent (trend)
4. Reconstructs the smoothed series

**Application:** SSA could be added as a signal smoother/filter in `src/indicators/` (~50 LOC). Complements existing HP filter (`src/ml/expected_returns.py`).

### Pattern 4: Quantstats Integration

Qbot uses `quantstats` for HTML report generation with:
- Tearsheet (returns, drawdown, monthly heatmap)
- Rolling Sharpe, Sortino, Calmar
- Distribution analysis
- Factor regression

**Application:** We already have basic analytics. Quantstats could replace manual plot generation in backtest scripts for professional reporting.

## Code Reference: LSTM Strategy Architecture

```python
# qbot/strategies/lstm_strategy_bt.py (149 lines)
class LSTMPredict(bt.Strategy):
    params = (('period', 10), ('neurons', 50), ('train_size', 0.8), ('lookback', 20))

    def __init__(self):
        # 1. Prepare data (train/test split, MinMax scaling)
        # 2. Build Keras Sequential(LSTM→Dropout→Dense)
        # 3. Fit on training data (MSE loss, Adam)

    def next(self):
        # 1. Get latest lookback window
        # 2. Predict next price via model.predict()
        # 3. If predicted > current: buy, else: sell
```

**Key weakness:** Trains on data that includes future bars relative to the backtest. The `_prepare_data()` method uses the full `self.dataclose` array, which is look-ahead biased. This is a common pitfall in ML backtesting that our project already avoids via PurgedKFold and OOS validation.

## Decision

**Study complete. No direct code integration.** Qbot's DL strategies are too basic to adopt. The architecture patterns (strategy registration, multi-engine separation) are already implemented in our project. SSA indicator may be worth a standalone implementation (~50 LOC) if a trend-smoothing indicator is needed.

**Key lesson:** The initial evaluation was overly optimistic. Qbot is a 2019-2023 Chinese retail trading platform effort — its value is in the GUI and infrastructure, not ML models. For modern DL in trading, reference `useful_resources/useful_repos/quant-resources/Deep-Learning-in-Quantitative-Trading/` instead.
