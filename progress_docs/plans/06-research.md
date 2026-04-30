---
type: phase
phase: "06"
name: "Research-Based Enhancements"
status: active
started: 2026-04-26
completed: null
sub_phases:
  - name: "Tier 1: High Impact, Low Complexity"
    status: active
  - name: "Tier 2: High Impact, Medium Complexity"
    status: pending
source: "useful_resources/papers_md/research_synthesis_report.md"
---

# Phase 06: Research-Based Enhancements

## Overview

Implement research insights from 13 academic papers synthesized in `research_synthesis_report.md`.
Tier 1 items are high impact + low complexity. Tier 2 are high impact + medium complexity.

## Tier 1: Implement First

| # | Item | File | Status | Notes |
|---|------|------|--------|-------|
| R1 | Turnover Penalty | `src/risk/turnover_penalty.py` | ✅ Done | OOM-RL: 6700% turnover destroys alpha |
| R2 | Per-Position Risk Model | `src/risk/position_probability.py` | ⏳ Pending | Jorion BET: per-position probability > aggregate VaR |
| R3 | Circuit Breakers | `src/risk/circuit_breakers.py` | ✅ Done | Portfolio-level 20% max DD halt |
| R4 | Regime Declaration per Strategy | `src/patterns/base.py` | ⏳ Pending | Tag patterns with preferred/incompatible regimes |
| R5 | Dynamic Rebalancing | `src/backtest/engine.py` | ⏳ Pending | Signal decay vs transaction cost tradeoff |

### Completed Tier 1 Details

**R1: Turnover Penalty**
- `TurnoverPenalty` class with `calculate_penalty()` method
- Threshold: 2000% annualized turnover
- Integrates with engine.py signal confidence

**R3: Circuit Breakers**
- `CircuitBreaker` class: 20% max DD, 20-bar cooldown
- Portfolio-wide halt mechanism for cascade failure prevention

### Pending Tier 1 Tasks
1. R2: Create `src/risk/position_probability.py` — binomial outcome model per position
2. R4: Update `BasePattern` with `preferred_regimes` and `incompatible_regimes` fields
3. R5: Add `estimate_optimal_frequency()` to `BacktestEngine`

## Tier 2: Implement Second

| # | Item | File | Status |
|---|------|------|--------|
| R6 | Event-Weighted Signal Aggregation | `src/signals/event_weighting.py` | ✅ Done |
| R7 | Diversity Score | `src/risk/diversity_score.py` | ✅ Done |
| R8 | Epistemic Autopsy | `src/backtest/epistemic_autopsy.py` | ✅ Done |
| R9 | Failure Set Analyzers | `src/strategies/failure_analysis.py` | ✅ Done |
| R10 | Friction-Adjusted Scoring | `src/backtest/engine.py` | ✅ Done |

## Notes
- Tier 1 items R1 and R3 implemented during 2026-04-26 session
- Tier 2 items R6-R10 already existed or were created during same session
- R2, R4, R5 are the remaining gap — low complexity, can be done without dependencies
- Research insights source: 13 papers in `useful_resources/papers_md/`
- Architecture analysis at `useful_resources/useful_repos/ARCHITECTURE_ANALYSIS.md`
