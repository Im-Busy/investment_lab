---
type: eval
name: "Auto-Research Tools Evaluation"
status: complete
started: 2026-04-26
completed: 2026-04-26
criteria:
  - Runs on Windows without Docker
  - Has reusable skill/pattern
  - Active maintenance (>1 commit in 2025)
  - Relevant to quantitative finance or ML workflow
options:
  - name: DeepScientist
    decision: KEEP
    reason: CLI-first, local, installable on Windows, Findings Memory pattern
  - name: Idea2Paper
    decision: KEEP
    reason: Knowledge graph pipeline, anchored review system
  - name: GPT-Researcher
    decision: KEEP
    reason: Already cloned, web research aggregation
  - name: DeepResearchAgent
    decision: KEEP
    reason: Multi-step reasoning architecture reusable
  - name: MLE-Agent
    decision: KEEP
    reason: Agent tool-use patterns applicable
  - name: OpenScholar
    decision: KEEP
    reason: Reversed previous skip — retrieval patterns discovered
  - name: RD-Agent
    decision: SKIP
    reason: Requires Docker + Linux runtime
  - name: PaperQA2
    decision: SKIP
    reason: Redundant with paper2md workflow
  - name: AI-Scientist
    decision: SKIP
    reason: Academic paper generation, not finance-oriented
decision: "KEEP 6 repos for skill extraction, SKIP 3, install 17 skills"
rationale: "Focus on repos with extractable patterns or skills for investment_trying. Docker/Linux-only repos deferred."
---

# Evaluation: Auto-Research Tools

## Summary

Evaluated 9 handpicked auto-research repos for suitability in the investment_trying project. Kept 6 for skill/pattern extraction, skipped 3 due to Docker/Linux requirements or irrelevance.

## Decisions

| Repo | Decision | Rationale |
|------|----------|-----------|
| DeepScientist | KEEP | Findings Memory, Bayesian optimization, CLI-first |
| Idea2Paper | KEEP | KG pipeline, anchored multi-agent review |
| GPT-Researcher | KEEP | Web research aggregation, already cloned |
| DeepResearchAgent | KEEP | Multi-step reasoning architecture |
| MLE-Agent | KEEP | Agent tool-use patterns |
| OpenScholar | KEEP | Reversed skip — retrieval architecture |
| RD-Agent | SKIP | Docker + Linux required |
| PaperQA2 | SKIP | Redundant with existing paper2md |
| AI-Scientist | SKIP | Academic focus, not finance |

## Artifacts
- 17 new skills installed to `.kilo/skills/`
- Architecture analysis at `useful_resources/useful_repos/ARCHITECTURE_ANALYSIS.md`
- RD-Agent and OpenScholar blocked until Linux/Docker available (see Phase 05)
