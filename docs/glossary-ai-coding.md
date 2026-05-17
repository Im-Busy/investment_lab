# AI Coding Glossary for Trading Systems

> 62 terms from Matt Pocock's [AI Coding Dictionary](https://github.com/mattpocock/dictionary-of-ai-coding), with trading-specific adaptations for the investment_trying project.

## How This Applies Here

This project is developed using AI coding agents (Kilo with Claude, Claude Code). Understanding these 62 terms is essential for:
- **Designing agent workflows** — handoffs, progressive disclosure, skills
- **Managing context budgets** — multi-session work across 300+ source files
- **Debugging agent failures** — sycophancy, hallucination, attention degradation
- **Running AFK sessions** — backtests, batch jobs, paper trading
- **Structuring docs** — AGENTS.md, skills, context pointers

## Quick Reference by Section

| Section | Terms | Relevance to Trading System |
|---------|-------|---------------------------|
| 1. The Model | 14 | Token costs when running long backtest analysis sessions |
| 2. Sessions & Context | 8 | Multi-session work on 300+ files; when to clear/compact |
| 3. Tools & Environment | 10 | Sandboxed backtests; MCP tools for data fetching |
| 4. Failure Modes | 9 | Sycophancy in strategy analysis; hallucination of metrics |
| 5. Handoffs | 7 | Specs/tickets for phased work (Phase 17→18→19 pattern) |
| 6. Memory & Steering | 6 | AGENTS.md + MEMORY.md as memory system; skills for domain expertise |
| 7. Patterns of Work | 8 | AFK backtest batches; automated checks (ruff, pytest) |

## Trading-Specific Adaptations

### Context Management for Trading Development

**Problem:** The project has 300+ Python source files. Loading everything at once burns context and triggers attention degradation.

**Solution:** Use **progressive disclosure** via skills:
- `.kilo/skills/scikit-learn/SKILL.md` — loaded only when doing ML
- `.kilo/skills/aeon/SKILL.md` — loaded only for time series work
- `.kilo/skills/scikit-survival/SKILL.md` — loaded only for survival analysis
- AGENTS.md stays lean; domain knowledge lives in skills

### Session Strategy for Multi-Phase Work

**Pattern:** Each Phase (17 R1-R14, 18 R18a-R18e, 19 R19a-R19i) maps to a **spec** with **tickets**:

```
Spec: Phase 17 Resource-Driven Enhancements
├── Ticket: R1 IR-Weighted Pattern Synthesis ✅
├── Ticket: R2 Factor Purification ✅
├── Ticket: R3 Evaluation Gate ✅
├── ...
└── Ticket: R14 Factor Engine Wrapper ✅
```

Each ticket is a single-session scoped unit. The dependency graph determines order: R1-R4 (P0) before R5-R8 (P1).

### Handoff Protocol

When switching between agent sessions or handing off to a future AI:
1. Write decisions to `MEMORY.md` (persistent state)
2. Write detailed handover to `progress_docs/handovers/`
3. Update `progress_docs/current.md` with session timestamp
4. Next session reads: MEMORY.md → full.md → current.md → handover

This is a **handoff artifact** pattern.

### Automated Checks (the project's)

| Check | Command | What it prevents |
|-------|---------|-----------------|
| ruff (lint) | `uv run ruff check .` | Style violations, unused imports |
| pytest | `uv run pytest` | Regressions in strategy/ML logic |
| look-ahead bias | `check_lookahead()` | Phantom alpha from future data |
| type check | `uv run mypy src/` | Type errors in production code |

These run as **automated checks** — deterministic, pass/fail, agent-self-correctable.

### Attention Budget in Long Backtest Sessions

When running a long session analyzing 16-instrument cross-validation results:
- **Smart zone** (~0-100K tokens): Sharp analysis, correct metric interpretation
- **Dumb zone** (~100K+ tokens): Faithfulness hallucinations — invents metrics, confuses instruments
- **Fix:** Compact or handoff after each major batched analysis. Don't push through.

### Sycophancy in Strategy Analysis

**Anti-pattern:** "This strategy looks great, Sharpe 0.76 on OOS" — agent confirms your enthusiasm but misses that 12 trades is a small sample.

**Fix:** Hide your preference. Ask "evaluate this strategy's robustness" not "is this a good strategy?".

### Vibe Coding in Trading — DANGER

Vibe coding (accepting agent output without human review) in trading code is dangerous:
- A hallucinated metric in BESTS.md creates phantom confidence
- A silently broken look-ahead check creates phantom alpha
- A sycophancy-driven strategy analysis creates phantom edge

**Rule:** Always human-review trading code diffs. Automated review (different model, different prompt) is a supplement, not a replacement.

## Environment Architecture

```
AGENTS.md                  → Always-on: standing brief, project conventions
.kilo/skills/              → Progressive disclosure: loaded per-task
.kilo/agent/               → Agent definitions: how to run specific workflows
src/                       → Source code: read/written via tools
data/                      → Data: accessed via tool calls
models/                    → Model files: reference via context pointers
docs/                      → Documentation: loaded as contextual knowledge
progress_docs/             → State tracking: MEMORY.md as memory system
```

## See Also

- Full dictionary: `skills_arsenal_for_publishing/universal-skills/ai-coding-dictionary/SKILL.md`
- Original source: https://github.com/mattpocock/dictionary-of-ai-coding
- Project session protocol: `AGENTS.md#session-start-protocol`
- Autonomous loop: `.kilo/project-loop.md`
