# Handover: Research-to-Implementation Logic Map

**Created:** 2026-04-25
**Session Context:** Paper analysis confirmed working (DeepSeek V4 Flash via OpenRouter). User requested a "logic map" to track research insights, implementations, and idea generation.
**Estimated Effort:** 1-2 hours

---

## Objective

Create a **living knowledge registry** that connects research insights to implemented features, tracks what's missing, and surfaces opportunities for new experiments.

This solves the problem of:
- Research insights being documented but not actively maintained
- No visibility into which recommendations were implemented vs. deferred
- Session-to-session knowledge loss
- No structured backlog of testable hypotheses from research

---

## Deliverables

### 1. Directory Structure
```
docs/research_logic_map/
├── README.md                    # Overview + how to use
├── insight_registry.md          # All research insights tagged by topic
├── implementation_status.md     # What's implemented, in-progress, deferred
├── hypothesis_backlog.md        # Testable hypotheses from research
├── decision_log.md              # Why we made key architectural decisions
└── open_questions.md            # Unresolved questions for future research
```

### 2. Source Documents to Process

| Document | Location | Key Content |
|----------|----------|-------------|
| Research Synthesis | `useful_resources/papers_md/research_synthesis_report.md` | 16 prioritized recommendations (R1-R16) from 13 papers |
| Paper Summaries | `useful_resources/papers_md/SENTIMENT_ANALYSIS_SUMMARY.md` | 7 sentiment analysis papers with ML metrics |
| Progress Log | `plans/progress_log.md` | 13 phases of implementation, 34+ patterns |
| Phase Handovers | `.kilo/plans/*.md` | Session-specific context |

### 3. Document Templates

#### `insight_registry.md`
```markdown
# Insight Registry

## Tag Legend
- **Topic**: Risk, Friction, Regime, Signal.Quality, Event.Type, Position.Sizing, etc.
- **Impact**: 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low
- **Status**: ✅ Implemented | 🔄 In-Progress | ⏳ Deferred | ❌ Rejected

## Insights by Paper

### P1: OOM-RL (arXiv:2604.11477)
| ID | Insight | Topic | Impact | Status | Linked Feature |
|----|---------|-------|--------|--------|----------------|
| I1.1 | Turnover penalty as hard constraint | Friction | 🔴 Critical | ✅ | `src/risk/turnover_penalty.py` |
| I1.2 | Dynamic rebalancing frequency | Signal.Decay | 🟠 High | 🔄 | In PR review |
| I1.3 | Structured drawdown diagnostics | Risk | 🔴 Critical | ⏳ | Backlog #3 |

### P3: Risk Management for Event-Driven Funds (SSRN-1018281)
| ID | Insight | Topic | Impact | Status | Linked Feature |
|----|---------|-------|--------|--------|----------------|
| I3.1 | Per-position success/failure probability | Position.Sizing | 🔴 Critical | ✅ | `src/risk/position_sizing.py` |
| I3.2 | Diversity score for portfolio construction | Portfolio | 🟠 High | ⏳ | Backlog #7 |
```

#### `implementation_status.md`
```markdown
# Implementation Status Dashboard

## Recommendation Tracker

| Rec # | Recommendation | Source | Impact | Status | Implemented In | Validation |
|-------|----------------|--------|--------|--------|----------------|------------|
| R1 | Turnover penalty as hard constraint | P1 (OOM-RL) | 🔴 Critical | ✅ | `src/risk/turnover_penalty.py` | `tests/test_turnover.py` |
| R2 | Per-position risk modeling | P3 (Jorion) | 🔴 Critical | ✅ | `src/risk/position_sizing.py` | — |
| R3 | Portfolio-level circuit breakers | P4, P8 | 🔴 Critical | ✅ | `src/risk/daily_limits.py` | `tests/test_circuit_breaker.py` |
| R6 | Event-type weighted signal aggregation | P6 | 🟠 High | ⏳ | Backlog #2 | — |

## Implementation Gaps
- [ ] R6: Event-type weighting not implemented
- [ ] R11: Divergence-in-bits metric not implemented
- [ ] R14: Network-derived factor model (research phase)

## Priority Queue
1. **Next Up:** R6 (Event-type weighting) — High impact, medium complexity
2. **Blocked By:** Need event taxonomy before R6
3. **Deferred:** R14 (Network factors) — Requires significant research
```

#### `hypothesis_backlog.md`
```markdown
# Hypothesis Backlog

## Format
Each hypothesis includes:
- **Statement:** Testable claim
- **Source:** Paper/insight that motivated it
- **Experiment:** How to test
- **Metric:** Success criteria
- **Priority:** Based on impact × feasibility

---

## H1: Event-Type Weighting Improves Signal Quality
- **Statement:** Weighting signals by event-type (vs. equal weight) improves Sharpe by ≥15%
- **Source:** P6 (Event-Based Trading: IE Tools) — "Event-type signals outperform aggregated sentiment"
- **Experiment:**
  1. Tag all signals with event-type taxonomy
  2. Backtest equal-weight vs. event-type-weight aggregation
  3. Compare Sharpe, win rate, max drawdown
- **Metric:** ΔSharpe ≥ 0.15, p < 0.05
- **Priority:** 🔴 High
- **Status:** ⏳ Backlog

## H2: Dynamic Rebalancing Frequency Reduces Friction
- **Statement:** Adapting rebalance frequency to signal decay rate reduces turnover 30%+ without return loss
- **Source:** P1 (OOM-RL) — "Daily→weekly shift was most impactful change"
- **Experiment:** Compare fixed daily vs. adaptive (decay-based) rebalancing
- **Metric:** Turnover reduction ≥30%, ΔReturn ≤5%
- **Priority:** 🟠 Medium
- **Status:** ✅ Validated (Phase 11)

## H3: Failure-Set Analyzers Prevent Cascade Losses
- **Statement:** Pre-trade failure-set validation (time-reversal, counter-trend, fat-tail) reduces max drawdown by ≥20%
- **Source:** P4 (Against Universal Trading) — "Every strategy has a mapped failure set"
- **Experiment:**
  1. Build failure-set analyzers for top 5 strategies
  2. Backtest with/without validation
  3. Measure MDD, tail loss, recovery time
- **Metric:** ΔMDD ≤ -20%, ΔTail Loss ≤ -25%
- **Priority:** 🟠 Medium
- **Status:** ⏳ Backlog
```

#### `decision_log.md`
```markdown
# Decision Log

## Format
- **Date:** When decision was made
- **Context:** What situation prompted the decision
- **Options:** Alternatives considered
- **Decision:** What was chosen
- **Rationale:** Why this option
- **Consequences:** Expected or observed outcomes
- **Review Date:** When to revisit

---

## DEC-2026-04-25: Map-Reduce over Vector DB for Paper Insights
- **Date:** 2026-04-25
- **Context:** User asked about knowledge tracking for research insights
- **Options:**
  1. Vector database with embeddings (Chroma, Weaviate)
  2. Structured markdown with YAML frontmatter
  3. Notion/Airtable database
- **Decision:** Option 2 — Markdown files with consistent templates
- **Rationale:**
  - Git-versionable (unlike Notion)
  - No infrastructure overhead (unlike vector DB)
  - Editable in any text editor
  - Matches existing project documentation style
- **Consequences:** Query capabilities limited vs. vector DB; acceptable tradeoff
- **Review Date:** 2026-07-25 (or when docs exceed 50 pages)

## DEC-2026-04-20: Numba over CUDA for Pattern Detection
- **Date:** 2026-04-20
- **Context:** Phase 2 performance optimization
- **Options:**
  1. CUDA GPU acceleration
  2. Numba JIT CPU optimization
  3. Multiprocessing with Ray
- **Decision:** Option 2 — Numba JIT
- **Rationale:**
  - 50-100x speedup achieved (sufficient)
  - No GPU dependency (Windows-compatible)
  - Easier debugging than CUDA
- **Consequences:** 50-100x speedup on basic patterns; complex patterns still sequential
- **Review Date:** 2026-06-20 (if backtest time >30s)
```

#### `open_questions.md`
```markdown
# Open Questions

## Format
- **Question:** The unresolved question
- **Context:** Why this matters
- **Related Insights:** Links to insight registry
- **Research Needed:** What would answer this
- **Owner:** Who's responsible (or "Unassigned")
- **Target Date:** When to resolve by

---

## Q1: How Do We Model Event-Type Correlations?
- **Question:** What's the right mathematical framework for event-type signal weighting?
- **Context:** P6 shows event-type signals outperform aggregated sentiment, but no formula given
- **Related Insights:** I6.1, I6.2, R6
- **Research Needed:**
  - Review NLP event extraction literature (VIP platform, BERT-based)
  - Prototype Bayesian event-type weighting
  - Backtest vs. equal-weight baseline
- **Owner:** Unassigned
- **Target Date:** 2026-05-15

## Q2: What's the Optimal Diversity Score Threshold?
- **Question:** Jorion (P3) shows ρ=0.03 correlates to 6x capital reduction — what's our portfolio's effective ρ?
- **Context:** We have 34 patterns but unknown effective independent bets
- **Related Insights:** I3.2, R7
- **Research Needed:**
  - Calculate pattern signal correlation matrix
  - Compute diversity score using BET formula
  - Sensitivity analysis on ρ assumption
- **Owner:** Unassigned
- **Target Date:** 2026-05-01
```

---

## Step-by-Step Instructions

### Step 1: Create Directory Structure
```bash
mkdir -p docs/research_logic_map
```

### Step 2: Extract Insights from Research Synthesis
1. Open `useful_resources/papers_md/research_synthesis_report.md`
2. Extract all 16 recommendations (R1-R16) from Section 5
3. Extract all per-paper insights from Section 3 (P1-P13)
4. Tag each insight with:
   - Topic (Risk, Friction, Regime, Signal.Quality, Position.Sizing, Portfolio, etc.)
   - Impact level (from synthesis report)
   - Current implementation status (check `src/` directory)

### Step 3: Map Implementations to Insights
1. For each recommendation, search `src/` for matching implementations
2. Use `grep` or code search to verify:
   - Does `turnover_penalty.py` exist?
   - Does `daily_limits.py` implement circuit breakers?
   - Does `position_sizing.py` have per-position probability?
3. Document matches in `implementation_status.md`

### Step 4: Generate Hypotheses
1. Review "Cross-Cutting Themes & Contradictions" (Section 4 of synthesis)
2. Convert each theme into testable hypothesis format
3. Prioritize by stated impact in synthesis report

### Step 5: Populate Decision Log
1. Review `.kilo/plans/*.md` for major architectural decisions
2. Review `AGENTS.md`, `pyproject.toml`, `src/config.py` for configuration decisions
3. Document 3-5 key decisions with full context

### Step 6: Identify Open Questions
1. Note any "Future Work" or "Limitations" sections from synthesis report
2. Note any contradictions between papers that need resolution
3. Note any recommendations marked "Research & Plan" (Tier 3)

---

## Acceptance Criteria

- [ ] All 5 documents created with initial content
- [ ] All 16 recommendations from synthesis report are tracked
- [ ] At least 10 research insights catalogued with tags
- [ ] At least 5 hypotheses in backlog (testable format)
- [ ] At least 3 decisions documented with rationale
- [ ] At least 5 open questions identified
- [ ] README.md provides navigation guide
- [ ] Documents use consistent markdown format (tables, tags)

---

## Notes for Next Session

1. **Start with the synthesis report** — it's the single best source of structured insights
2. **Don't over-engineer** — Simple markdown tables are better than complex schemas
3. **Link to code** — Always include file paths in "Linked Feature" column
4. **Use emoji consistently** — 🔴🟠🟡🟢 for impact, ✅🔄⏳❌ for status
5. **Keep it living** — Add a "Last Updated" timestamp to each document header

---

## Quick Reference Commands

```bash
# Search for implementation status
grep -r "turnover" src/ --include="*.py" | head -20
grep -r "circuit_breaker" src/ --include="*.py" | head -20
grep -r "diversity" src/ --include="*.py" | head -20

# Find test files
fd test_.*\.py src/tests/

# Check recommendation mentions
grep -r "R1\|R2\|R3" docs/ plans/ --include="*.md"
```

---

## Completion

When done, update this handover:
1. ✅ Check all acceptance criteria
2. ✅ Add this file to `.kilo/plans/` as a reference
3. ✅ Note any follow-up needed in next session's opening

---

**End of Handover**
