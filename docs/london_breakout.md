# London Breakout Strategy

## Overview

The London Breakout Strategy is a time-of-day momentum strategy that uses Tokyo trading hour price action to predict London session breakouts. It targets the first 30 minutes of the London session (3:00-3:30 EST) for execution.

## Strategy Logic

### 1. Tokyo Hour Collection (2:00-3:00 EST)
- Collect high and low prices during Tokyo hour
- Calculate price range: `tokyo_high - tokyo_low`

### 2. London Open Setup (3:00 EST)
- Set upper threshold: `london_open + 0.5 * tokyo_range`
- Set lower threshold: `london_open - 0.5 * tokyo_range`

### 3. Trading Window (3:00-3:30 EST)
- Monitor for price breakouts above upper or below lower thresholds
- Execute buy on upper breakout, sell on lower breakout
- Apply 1% stop loss filter
- Allow only 1 position per session
- Support reverse position detection

### 4. London Close (12:00 EST)
- Clear all intraday positions
- Reset session state

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tokeyo_start_hour` | 2 | Tokyo hour start (EST) |
| `tokyo_end_hour` | 3 | Tokyo hour end (EST) |
| `london_trading_minutes` | 30 | London trading window duration |
| `london_close_hour` | 12 | London session end (EST) |
| `risky_stop` | 0.01 | Stop loss threshold (1%) |
| `param` | 0.5 | Range multiplier for thresholds |

## Data Requirements

- **Frequency:** Minute-level OHLCV data
- **Pairs:** FX pairs (GBP/USD recommended)
- **Time Period:** Minimum 2 years for robust validation
- **Coverage:** Must include Tokyo hour and London session times

## Expected Performance

- **Sharpe Ratio:** 0.5-1.5
- **Max Drawdown:** -10% to -25%
- **Win Rate:** 45-55%
- **Trade Frequency:** 50-100 trades per year on 2+ year data

## Validation Results

Run `python validate_london_breakout.py` to generate validation metrics and plots.

## Usage Example

```python
from src.strategies.london_breakout import LondonBreakout
from src.backtest.engine import BacktestEngine

engine = BacktestEngine(
    strategy=LondonBreakout,
    initial_cash=10000,
    commission=0.0001
)

results = engine.run(data)
```
