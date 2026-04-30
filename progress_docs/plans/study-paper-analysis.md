---
type: study
name: "Paper Analysis & Synthesis"
status: complete
started: 2026-04-19
completed: 2026-04-22
references:
  - useful_resources/papers_md/*.md (13 papers)
key_findings: useful_resources/papers_md/research_synthesis_report.md
---

# Study: Paper Analysis & Synthesis

## Overview

Analyzed 13 academic papers on algorithmic trading, event-driven strategies, risk management, and ML-enhanced trading. Synthesized findings into `research_synthesis_report.md` to drive Phase 06 (Research-Based Enhancements).

## Key Papers & Insights

| Paper | Key Insight | Applied To |
|-------|------------|------------|
| OOM-RL: Friction-Aware Strategy Design | 6700% turnover destroys alpha | Phase 06 R1: TurnoverPenalty |
| Jorion: Event-Driven Fund Risk | Per-position probability > aggregate VaR | Phase 06 R2: PositionRiskModel |
| Against Universal Trading | No single strategy works in all regimes | Phase 06 R4: Regime declaration per strategy |
| Event-Based Trading: Granularity | Granular event-type signals beat aggregation | Phase 06 R6: EventWeighting |
| Fang et al.: Crash Factor | Take-profit beats RSI exits | Trading strategy design |
| BET Diversity Score | Effective number of independent bets | Phase 06 R7: DiversityScore |

## Knowledge Extracted

### Tier 1 (High Impact, Low Complexity)
1. Friction-aware design — the #1 under-addressed problem in systematic trading
2. Per-position probability modeling — necessary for proper capital allocation
3. Portfolio-level circuit breakers — prevent cascade failures
4. Regime declaration per strategy — not all patterns work in all conditions
5. Dynamic rebalancing — signal decay vs transaction cost tradeoff

### Tier 2 (High Impact, Medium Complexity)
6. Event-type weighted signals — granular beats aggregated
7. Diversity score — effective bets > simple position count
8. Epistemic autopsy — automated drawdown root cause analysis
9. Failure-set analyzers — per-strategy stress testing
10. Friction-adjusted scoring — TCA-aware performance metrics

## Output
- `useful_resources/papers_md/research_synthesis_report.md` — complete synthesis
- Phase 06 plan derived from these findings
