---
type: study
name: "Repo Architecture Study"
status: complete
started: 2026-04-26
completed: 2026-04-26
references:
  - useful_resources/useful_repos/ (ML4T, DLQT, and 5 cloned repos)
key_findings: useful_resources/useful_repos/ARCHITECTURE_ANALYSIS.md
---

# Study: Repo Architecture Analysis

## Overview

Studied reference repos to extract architectural patterns, best practices, and reusable code patterns for the investment_trying project. Informed Phase 04 (ML Foundation) design.

## Repos Studied

| Repo | Purpose | Key Extraction |
|------|---------|---------------|
| Machine-Learning-for-Algorithmic-Trading (Jansen) | ML4T textbook code | PurgedKFold, embargo periods, financial metrics |
| Deep-Learning-for-Quantitative-Trading (DLQT) | DL for quant finance | Model registry pattern, experiment tracking |
| DeepScientist | Autonomous research | Findings Memory, Bayesian hypothesis selection |
| Idea2Paper | Research agent | Knowledge graph pipeline, anchored review |
| OpenScholar | Paper retrieval | Retrieval architecture, embeddings |
| DeepResearchAgent | Research framework | Multi-step reasoning architecture |
| MLE-Agent | ML automation | Agent tool-use patterns |
| GPT-Researcher | Web research | Search aggregation patterns |

## Key Patterns Extracted

1. **Durable State + Iterative Refinement** — experiments accumulate, never discard
2. **Versioned Resources** — models, features, configs all carry version identifiers
3. **Findings Memory** — persistent cross-run learning (what worked, what failed)
4. **Bayesian Hypothesis Selection** — explore/exploit with uncertainty quantification
5. **Agent Tool-Use Patterns** — structured tool interfaces with validation

## Decisions

- Adopted PurgedKFold + embargo for Phase 04 (from ML4T)
- Adopted registry + experiment logger pattern (from DLQT)
- Installed 17 skills to `.kilo/skills/` for agent tooling
- Reversed OpenScholar skip decision — architecture showed reusable patterns

## Output
- `useful_resources/useful_repos/ARCHITECTURE_ANALYSIS.md` — full analysis
- 17 skills installed in `.kilo/skills/`
