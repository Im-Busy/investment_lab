---
description: Run ML strategy backtests with correct configurations. Updates BESTS.md leaderboard automatically. Supports entry threshold sweeps, OOS validation, and ML-enhanced strategy comparison.
---

# Backtest — Run ML Strategy Backtests

Load the **backtest-runner** agent to execute backtests with proper configurations and leaderboard maintenance.

## Arguments

```
/backtest SYMBOL --entry-threshold 0.45 --trail-stop
/backtest SYMBOL --start 2025-01-01              # OOS test
/backtest sweep SYMBOL                           # Full threshold sweep
/backtest compare SYMBOL --strategy ema          # ML-enhanced vs baseline
```

## Quick Examples

```
# Best known config
/backtest SPY --entry-threshold 0.45 --trail-stop

# OOS validation (critical)
/backtest SPY --entry-threshold 0.45 --trail-stop --start 2025-01-01

# Sweep all thresholds
/backtest sweep SPY

# With vol gate
/backtest SPY --entry-threshold 0.45 --trail-stop --vol-gate 1.5
```
