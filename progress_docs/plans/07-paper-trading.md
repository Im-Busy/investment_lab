---
type: phase
phase: "07"
name: "Paper Trading & Live Readiness"
status: deferred
deferred_reason: "Optional phase — depends on Phase 06 completion and all prior phases stable"
deferred_since: 2026-04-19
revisit_when: "Phases 01-06 complete and stable, user requests live trading preparation"
---

# Phase 07: Paper Trading & Live Readiness

## Overview

Deploy paper trading pipeline using existing backtest engine with live data feed. Monitor signal quality, execution latency, and slippage for 2+ weeks. Establish go/no-go criteria for live trading.

## Planned Deliverables

| Item | File |
|------|------|
| Paper trading runner | `scripts/paper_trade.py` |
| Live signal log | `logs/paper_trade_signals.log` |
| Paper trading performance report | `reports/paper_trading_report.md` |
| Go/No-Go decision document | `reports/live_readiness.md` |

## Planned Workflow
1. Create `scripts/paper_trade.py` — polls live data, runs signal generation, logs signals
2. Simulate fills with realistic slippage (0.05% liquid, 0.2% illiquid)
3. Run paper trading for minimum 14 calendar days
4. Compare paper signals vs backtest predictions; flag discrepancies >5%
5. Calculate live Sharpe, win rate, profit factor
6. Write `reports/live_readiness.md` with go/no-go recommendation

## Go/No-Go Criteria

| Metric | Threshold |
|--------|-----------|
| Paper trading duration | >=14 calendar days |
| Signal match rate (vs. backtest) | >=90% |
| Live Sharpe ratio | >=0.5 |
| Live win rate | >=45% |
| Live profit factor | >=1.2 |
| Max drawdown | <=15% |
| Slippage within tolerance | <=0.2% average |
| Zero critical bugs | Yes |

**All criteria must pass for go-live decision.**

## Risk Management for Paper Trading
- Daily loss limit: 3% of simulated equity -> halt
- Weekly loss limit: 6% -> halt, review
- Monthly loss limit: 10% -> full system review
- Position sizing: Kelly capped at 2% risk per trade, max 5 open positions
- Circuit breaker: 3 consecutive losses >2% -> pause 24h
