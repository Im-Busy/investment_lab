---
type: phase
phase: "06"
name: "Research-Based Enhancements"
status: complete
started: 2026-04-26
completed: 2026-05-01
sub_phases:
  - name: "Tier 1: High Impact, Low Complexity"
    status: complete
  - name: "Tier 2: High Impact, Medium Complexity"
    status: complete
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
| R2 | Per-Position Risk Model | `src/risk/position_probability.py` | ✅ Done | Jorion BET: per-position probability > aggregate VaR |
| R3 | Circuit Breakers | `src/risk/circuit_breakers.py` | ✅ Done | Portfolio-level 20% max DD halt |
| R4 | Regime Declaration per Strategy | `src/patterns/base.py` + `engine.py` | ✅ Done | DEFAULT_REGIME_MAPPING + `_apply_regime_gating()` |
| R5 | Dynamic Rebalancing | `src/risk/dynamic_rebalancing.py` | ✅ Done | Signal decay vs transaction cost tradeoff |

### Completed Tier 1 Details

**R1: Turnover Penalty**
- `TurnoverPenalty` class with `calculate_penalty()` method
- Threshold: 2000% annualized turnover
- Integrates with engine.py signal confidence

**R2: Per-Position Risk Model**
- `PositionRiskModel` class using Jorion's binomial outcome model
- `estimate_success_prob()` with regime-adjusted base probability
- Kelly Criterion position sizing, batch risk estimation
- Integrated via `_apply_position_probability()` in BacktestEngine

**R3: Circuit Breakers**
- `CircuitBreaker` class: 20% max DD, 20-bar cooldown
- Portfolio-wide halt mechanism for cascade failure prevention

**R4: Regime Declaration per Strategy**
- `DEFAULT_REGIME_MAPPING` in `src/patterns/base.py` — auto-assigns preferred/incompatible regimes by PatternType
- `BasePattern.__post_init__()` auto-populates from type defaults if not explicitly set
- `is_regime_compatible()` and `get_regime_preference()` support cross-namespace enum compatibility
- `_apply_regime_gating()` in BacktestEngine filters signals by regime compatibility, adjusts confidence by preference score

**R5: Dynamic Rebalancing**
- `DynamicRebalancer` class with `estimate_optimal_frequency()` (signal_decay/tx_cost ratio)
- Integrated via `should_rebalance()` in BacktestEngine.run() main loop

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

---

## Study: Paper Analysis & Synthesis (2026-04-19)

*(Merged from study-paper-analysis.md)*

Analyzed 13 papers. Key insights → Phase 06 items:

| Paper | Key Insight | Applied To |
|-------|------------|------------|
| OOM-RL | 6700% turnover destroys alpha | Phase 06 R1: TurnoverPenalty |
| Jorion: Event-Driven Fund Risk | Per-position probability > aggregate VaR | Phase 06 R2: PositionRiskModel |
| Against Universal Trading | No single strategy works in all regimes | Phase 06 R4: Regime declaration |
| Event-Based Trading | Granular event-type signals beat aggregation | Phase 06 R6: EventWeighting |
| Fang et al.: Crash Factor | Take-profit beats RSI exits | Trading strategy design |
| BET Diversity Score | Effective number of independent bets | Phase 06 R7: DiversityScore |

**Output:** `useful_resources/papers_md/research_synthesis_report.md`

## Study: Repo Architecture Analysis (2026-04-26)

*(Merged from study-repo-architecture.md)*

Studied 8 reference repos. Key patterns extracted:
1. Durable State + Iterative Refinement
2. Versioned Resources (models, features, configs)
3. Findings Memory (persistent cross-run learning)
4. Bayesian Hypothesis Selection
5. Agent Tool-Use Patterns

**Decisions:** Adopted PurgedKFold + embargo (from ML4T), registry + experiment logger (from DLQT). 17 skills installed.
