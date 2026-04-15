---
name: investment-trading-system
description: Domain knowledge and conventions for the investment_trying algorithmic trading system. Covers architecture, pattern detection framework, backtesting pipeline, indicator conventions, and project-specific search guidance. Use when working on trading strategies, pattern implementations, backtest analysis, or data pipeline modifications.
---

# Investment Trading System — Project Knowledge

## Project Overview

A Python-based algorithmic trading system focused on **pattern detection and multi-strategy backtesting**. Detects candlestick patterns, harmonic patterns, chart patterns, and continuation patterns across multiple timeframes, generates confluence-weighted signals, and backtests against historical OHLCV data.

## Architecture

```
src/
├── analysis/          # Pattern selection, contribution analysis, signal logging
├── backtest/          # Backtesting engine, benchmark, metrics, risk metrics
├── data_ingestion/    # Data fetching (yfinance, CSV), fetch_data.py
├── indicators/        # Technical indicators (Numba-optimized, cached)
├── patterns/          # Pattern detectors (all inherit from BasePattern)
│   ├── basic/         # MSL, matching lows, NR7ID, N-bar decline, floor pivot
│   ├── breakout/      # Donchian channels, gaps
│   ├── candlestick/   # Doji, engulfing, hammer, harami, dark cloud
│   ├── complex/       # Cup & handle, H&S, three hills, parabolic arc
│   └── harmonic/      # Gartley, ABC, Bollinger, symmetric triangle
├── risk/              # Position sizing, daily limits
├── signals/           # Signal generator, position manager
├── strategies/        # Standalone strategy implementations
│   └── backtest_py/   # Multi-pattern backtest runner
├── utils/             # Helpers, validators
└── visualization/     # Charts, tearsheets, pattern markers
```

## Key Conventions

### Pattern Implementation

All patterns inherit from `BasePattern` ([`src/patterns/base.py:134`](src/patterns/base.py:134)). Required methods:
- `detect(df, i, window_start)` → `PatternResult`
- `generate_signal(df, i)` → `Optional[TradeSignal]`

Optional optimizations:
- `detect_vectorized(df)` → `np.ndarray` — NumPy/Numba vectorized detection
- `detect_batch(df, start, end, window_start)` → `List[PatternResult]`
- `precompute_signals(df)` — Cache signals before backtest

### Data Format

OHLCV DataFrames use these exact column names: `Open`, `High`, `Low`, `Close`, `Volume` (capitalized).

### Configuration

All config lives in [`src/config.py`](src/config.py). Key dataclasses:
- `PatternConfig` — lookback, tolerance, confidence base
- `SignalConfig` — min confidence, confluence bonuses
- `RiskConfig` — equity, risk per trade, position limits
- `BacktestConfig` — commission, slippage, take profit levels

### Performance Patterns

- Use `IndicatorCache` for pre-computed indicators (avoid redundant calculations)
- Extract NumPy arrays via `_extract_arrays()` for hot loops
- Use `window_start` parameter to avoid DataFrame slicing
- Numba JIT for compute-heavy indicators in `indicators/*_numba.py`

## Search Guidance (Project-Specific)

### When to use each search tool for this project:

| Task | Primary Tool | Why |
|------|-------------|-----|
| pandas-ta indicator usage | Context7 → `pandas-ta` | Official docs first |
| backtesting.py API | Context7 → `backtesting` | Official docs first |
| New pattern algorithm | Exa (code) | Find implementations |
| Trading strategy research | Tavily (advanced) | Articles, comparisons |
| Market data APIs | Exa (code) | yfinance, Alpha Vantage examples |
| External repo understanding | Repomix | Pack the repo first |
| Quant/finance library | Context7 first, then Exa | `quantstats`, `scipy.stats` |
| Recent market research | Tavily (news) | Time-filtered results |

### Specific search patterns:

**Adding a new indicator:**
1. Context7: `resolve-library-id(libraryName="pandas-ta")` → check if built-in
2. If not in pandas-ta: Exa code search for implementation examples
3. Fallback: Tavily for academic/research context

**Adding a new pattern:**
1. Exa code search: `"<pattern name> pattern detection python implementation"`
2. Exa code search: `"<pattern name> algorithm fibonacci levels detection"`
3. Research the pattern's rules thoroughly before implementing

**Backtesting optimization:**
1. Context7: `query-docs(libraryId="/mementum/backtesting.py", query="...")`
2. Exa code: `"backtesting.py custom strategy multiple timeframes"`

**Data source research:**
1. Exa code: `"yfinance download historical data <asset>"`
2. Tavily: `"free alternative to yfinance <asset type> API"`

## Domain Knowledge

### Pattern Categories

| Category | Examples | Confidence Base | Lookback Range |
|----------|----------|----------------|----------------|
| Basic | MSL, matching lows, NR7ID | 0.45-0.55 | 1-7 bars |
| Candlestick | Doji, engulfing, hammer | 0.50-0.60 | 1-3 bars |
| Harmonic | Gartley, ABC, Bollinger | 0.45-0.60 | 20-50 bars |
| Complex | Cup & handle, H&S, three hills | 0.55-0.65 | 50-100 bars |
| Breakout | Donchian, gap | 0.50 | 20 bars |

### Signal Confluence System

Signals are boosted when multiple patterns align:
- 1 pattern: +0.00 bonus
- 2 patterns: +0.10 bonus
- 3 patterns: +0.20 bonus
- 4+ patterns: +0.30 bonus
- Trend alignment bonus: +0.05 (ADX-based)

### Backtest Metrics

Primary metrics tracked:
- Total return, annualized return
- Sharpe ratio, Sortino ratio
- Max drawdown, max drawdown duration
- Win rate, profit factor
- Calmar ratio

### Risk Parameters

- Initial equity: $100,000
- Risk per trade: 2%
- Max position size: 20%
- Max open positions: 5
- Max daily loss: 3%
- Max weekly loss: 6%
- Max monthly loss: 10%

## File Output Conventions

- Repomix outputs → `useful_resources/managed_by_AI/`
- Backtest reports → `reports/`
- Backtest plans → `plans/`
- Jupyter notebooks → `notebooks/`
- Notebook reports → `notebooks/reports/`
- Data files → `data/raw/` (CSV), `data/processed/` (cleaned)
- Logs → `logs/`
