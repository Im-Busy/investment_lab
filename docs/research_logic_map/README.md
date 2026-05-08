# Research Logic Map: Navigation Guide

**Last Updated:** 2026-04-25
**Version:** 1.0
**Synthesized Papers:** 13
**Tracked Insights:** 42
**Hypotheses:** 12

---

## What Is This?

This is a **living knowledge registry** connecting research insights from 13 academic papers to implemented features in our multi-pattern trading system.

It solves:
- Research insights being documented but not actively maintained
- No visibility into which recommendations were implemented vs. deferred
- Session-to-session knowledge loss
- No structured backlog of testable hypotheses

---

## Document Overview

| Document | Purpose | Lines | Key Sections |
|----------|---------|-------|--------------|
| [`insight_registry.md`](insight_registry.md) | Catalog all research insights with tags | ~200 | 42 insights from 13 papers, tagged by topic/impact/status |
| [`implementation_status.md`](implementation_status.md) | Track implementation progress | ~150 | 29 recommendations, 52% implemented, priority queue |
| [`hypothesis_backlog.md`](hypothesis_backlog.md) | Testable hypotheses from research | ~180 | 12 hypotheses, 8 ready for testing |
| [`decision_log.md`](decision_log.md) | Architectural decisions and rationale | ~120 | 8 key decisions with context |
| [`open_questions.md`](open_questions.md) | Unresolved research questions | ~160 | 8 open questions, 4 high priority |

---

## How to Use

### For New Session Continuation

1. **Start here:** `open_questions.md` — see what's unresolved
2. **Check status:** `implementation_status.md` — see what's implemented vs. backlogged
3. **Pick a hypothesis:** `hypothesis_backlog.md` — choose one marked "Ready for Testing"
4. **Add new insights:** `insight_registry.md` — tag new insights from paper reading

### For Research Implementation

1. **Find relevant insights:** Search `insight_registry.md` by topic (Risk, Friction, Regime, etc.)
2. **Check implementation status:** Cross-reference with `implementation_status.md`
3. **Review decisions:** Check `decision_log.md` for architectural context
4. **Track open questions:** Note related questions in `open_questions.md`

### For Hypothesis Testing

1. **Browse backlog:** `hypothesis_backlog.md` — filter by "Ready for Testing"
2. **Check dependencies:** Each hypothesis lists prerequisites
3. **Run experiment:** Implement, backtest, measure against stated metric
4. **Update status:** Mark hypothesis ✅ Validated / ❌ Refuted / ⚠️ Inconclusive

---

## Quick Reference

### Top Priority Questions (Unassigned)

| # | Question | Impact | Target |
|---|----------|--------|--------|
| Q1 | Event-type weighting formula | 🔴 High | 2026-05-15 |
| Q2 | Portfolio diversity score (ρ) | 🟠 High | 2026-05-01 |
| Q4 | Failure-set analyzer efficacy | 🟠 High | 2026-07-31 |
| Q7 | 34 vs. 10 patterns ablation | 🟠 High | 2026-06-15 |

### Top Priority Hypotheses (Ready for Testing)

| # | Hypothesis | Metric | Priority |
|---|------------|--------|----------|
| H5 | Crash factor filter reduces tail loss | Δ99thLoss ≤ -25% | 🟠 High |
| H6 | Fixed take-profit > RSI exits | ΔWinRate ≥ 10% | 🟠 High |
| H7 | Friction scoring predicts failure | High-friction Sharpe < Low-friction by 1.0 | 🟡 Medium |
| H8 | Regime-adaptive strategies win | ΔSharpe ≥ 0.3 | 🟠 High |

### Implemented (Key Recommendations)

| Rec # | Recommendation | File | Status |
|-------|----------------|------|--------|
| R1 | Turnover penalty as hard constraint | `src/backtest/friction_scoring.py` | ✅ |
| R2 | Per-position risk modeling | `src/risk/position_sizing.py` | ✅ |
| R3 | Portfolio-level circuit breakers | `src/risk/daily_limits.py` | ✅ |
| R4 | Regime declaration per strategy | `src/indicators/regime_detector.py` | ✅ |
| R7 | Diversity score for portfolio | `src/risk/diversity_score.py` | ✅ |
| R8 | Adaptive parameter tuning | `src/strategies/adaptive_router.py` | ✅ |
| R10 | Friction-adjusted backtest scoring | `src/backtest/friction_scoring.py` | ✅ |
| R18 | Fixed take-profit exits | `src/signals/position_manager.py` | ✅ |

### Backlogged (High Priority)

| Rec # | Recommendation | Blocked By | Priority |
|-------|----------------|------------|----------|
| R6 | Event-type weighted aggregation | Event taxonomy | 🔴 High |
| R8 | Epistemic Autopsy module | — | 🟠 High |
| R9 | Failure-set analyzers | — | 🟠 High |
| R17 | Crash factor as risk filter | — | 🟠 High |

---

## Insights by Topic

### Risk (16 insights, 8 implemented)
**Implemented:** R1 (turnover penalty), R2 (per-position risk), R3 (circuit breakers), R4 (regime declaration), R10 (friction scoring)

**Backlogged:** R8 (epistemic autopsy), R9 (failure-set analyzers), R17 (crash factor), R23 (strategy decay), R24 (stress testing)

### Friction (4 insights, 3 implemented)
**Implemented:** R1 (turnover penalty), R10 (friction scoring), R18 (fixed take-profit)

**Backlogged:** R16 (liquidity loading filter)

### Regime (7 insights, 3 implemented)
**Implemented:** R4 (regime detection), R8 (adaptive routing)

**Backlogged:** R12 (holding-period alignment), R21 (volatility-adaptive thresholds), R22 (regime→break probability), R23 (strategy decay)

### Signal.Quality (11 insights, 4 implemented)
**Implemented:** R5 (dynamic rebalancing), R10 (friction scoring), R18 (fixed take-profit)

**Backlogged:** R6 (event-type weighting), R9 (failure-set analyzers), R11 (divergence-in-bits), R12 (holding-period alignment), R19 (CMRS scoring), R25 (spike detection), R28 (co-movement signals)

### Portfolio (6 insights, 2 implemented)
**Implemented:** R3 (circuit breakers), R7 (diversity score)

**Backlogged:** R13 (pairs trading), R14 (network factors), R15 (tensor fusion), R22 (regime→break prob), R27 (co-movement)

---

## Cross-Reference with Source Documents

### Primary Sources

| Document | Location | Key Content |
|----------|----------|-------------|
| Research Synthesis | `useful_resources/papers_md/research_synthesis_report.md` | 16 prioritized recommendations (R1-R16), 42 insights from 13 papers |
| Paper Summaries | `useful_resources/papers_md/SENTIMENT_ANALYSIS_SUMMARY.md` | 7 sentiment analysis papers with ML metrics |
| Progress Log | `plans/progress_log.md` | 13 phases of implementation, 34+ patterns |

### Related Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Phase Handovers | `.kilo/plans/*.md` | Session-specific implementation context |
| Test Results | `tests/` | Validation status for implemented features |
| Backtest Reports | `reports/` | Empirical results for hypothesis validation |

---

## Tag System

### Topics (12 tags)
`Risk` `Friction` `Regime` `Signal.Quality` `Event.Type` `Position.Sizing` `Portfolio` `Strategy.Lifecycle` `Pattern.Detection` `ML.Model` `Backtest` `Data.Infrastructure`

### Impact Levels (4 levels)
- 🔴 **Critical**: Must implement; alpha-destroying if ignored
- 🟠 **High**: Significant performance impact
- 🟡 **Medium**: Nice to have; incremental improvement
- 🟢 **Low**: Marginal benefit

### Status (4 states)
- ✅ **Implemented**: Code exists in `src/`
- 🔄 **In-Progress**: Currently being implemented
- ⏳ **Deferred**: Backlogged; not yet started
- ❌ **Rejected**: Explicitly decided against
- 🔬 **Research Phase**: Requires literature review / prototyping

---

## Maintenance

### Update Triggers

Update these documents when:
1. **New paper analyzed** → Add insights to `insight_registry.md`
2. **Feature implemented** → Update `implementation_status.md`
3. **Hypothesis tested** → Update `hypothesis_backlog.md` with results
4. **Decision made** → Add entry to `decision_log.md`
5. **Question resolved** → Move from `open_questions.md` to insights or decisions

### Review Cadence

- **Weekly:** Check hypothesis testing progress
- **Monthly:** Review implementation priority queue
- **Quarterly:** Revisit open questions, reprioritize backlog
- **Per Session:** Update progress log in `plans/progress_log.md`

---

## Search Tips

### Find Insights by Paper
```bash
grep "P1\|P2\|P3" docs/research_logic_map/insight_registry.md
```

### Find Implemented Features
```bash
grep "✅" docs/research_logic_map/implementation_status.md | head -20
```

### Find Backlog by Priority
```bash
grep -A3 "🔴 High\|🟠 High" docs/research_logic_map/open_questions.md
```

### Find Hypotheses Ready for Testing
```bash
grep -B2 "Ready for Testing" docs/research_logic_map/hypothesis_backlog.md
```

---

## Contributing

### Adding a New Insight

1. Read paper in `useful_resources/papers/`
2. Extract actionable insight (concrete recommendation)
3. Tag with:
   - Topic (from tag legend)
   - Impact (🔴/🟠/🟡/🟢)
   - Paper source (P1-P13)
4. Add to `insight_registry.md` table
5. Link to related recommendations/decisions/questions

### Adding a New Hypothesis

1. Identify testable claim from research
2. Format with:
   - Clear statement (falsifiable)
   - Source insight/paper
   - Experiment design
   - Success metric
   - Priority
3. Add to `hypothesis_backlog.md`
4. Link to related insights

### Recording a Decision

1. Identify architectural choice made
2. Document:
   - Context (why this came up)
   - Options considered (at least 2)
   - Decision + rationale
   - Consequences (expected/observed)
   - Review date
3. Add to `decision_log.md`

---

## Acknowledgments

This knowledge registry was created to address the problems identified in:
- OOM-RL (P1): Structured drawdown diagnostics
- Against Universal Trading (P4): Regime declaration, failure-set awareness
- Investing Is Compression (P2): Divergence-in-bits for strategy comparison

Design influenced by:
- Map-reduce pattern for insight extraction
- Information-theoretic documentation structure
- Modular architecture (separate docs for insights, implementations, hypotheses, decisions, questions)

---

*Generated from 13 research papers, 42 insights, 29 recommendations, 12 hypotheses, 8 decisions, 8 open questions.*

*Next review date: 2026-05-25 (monthly review)*
