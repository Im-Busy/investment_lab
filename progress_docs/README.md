# Progress Documentation

**Project:** investment_trying — Rule-Based Multi-Pattern Trading System
**Last Updated:** 2026-05-01

---

## Quick Start for Agents

1. **Read `current.md`** — know what was happening when the last session ended
2. **Read `plans/full.md`** — see all phases, studies, and deferred items
3. **Resume** from the last incomplete action

---

## Phase Map

| # | Phase | Plan | Log | Status |
|---|-------|------|-----|--------|
| 01 | Pattern Detection & Strategies | [plan](plans/01-patterns.md) | [log](logs/01-patterns.md) | ✅ Complete |
| 02 | Performance Optimization | [plan](plans/02-optimization.md) | [log](logs/02-optimization.md) | 🔄 Partially Done |
| 03 | Regime Detection & Adaptation | [plan](plans/03-regime.md) | [log](logs/03-regime.md) | ✅ Complete |
| 04 | ML Foundation | [plan](plans/04-ml-foundation.md) | [log](logs/04-ml-foundation.md) | ✅ Complete |
| 05 | ML Advanced | [plan](plans/05-ml-advanced.md) | [log](logs/05-ml-advanced.md) | ⏸️ Deferred |
| 06 | Research-Based Enhancements | [plan](plans/06-research.md) | [log](logs/06-research.md) | 🔄 Partially Done |
| 07 | Paper Trading & Live Readiness | [plan](plans/07-paper-trading.md) | [log](logs/07-paper-trading.md) | ⏸️ Deferred |
| 08 | Contribution & Attribution | [plan](plans/08-attribution.md) | [log](logs/08-attribution.md) | ⏳ Pending |

## Studies

| Name | Plan | Log | Status | Key Output |
|------|------|-----|--------|------------|
| Paper Analysis & Synthesis | [plan](plans/study-paper-analysis.md) | [log](logs/study-paper-analysis.md) | ✅ Done | `research_synthesis_report.md` |
| Repo Architecture Study | [plan](plans/study-repo-architecture.md) | [log](logs/study-repo-architecture.md) | ✅ Done | `ARCHITECTURE_ANALYSIS.md` |

## Evaluations

| Name | Plan | Log | Status | Decision |
|------|------|-----|--------|----------|
| Auto-Research Tools | [plan](plans/eval-auto-research-tools.md) | [log](logs/eval-auto-research-tools.md) | ✅ Done | KEEP 5, SKIP 4 |

## Setups

| Name | Plan | Log | Status |
|------|------|-----|--------|
| Pixi to UV Migration | [plan](plans/setup-pixi-migration.md) | [log](logs/setup-pixi-migration.md) | ✅ Done |
| SearXNG MCP Setup | [plan](plans/setup-searxng-mcp.md) | [log](logs/setup-searxng-mcp.md) | 🔄 Partial |
| Jupyter + VS Code Setup | [plan](plans/setup-jupyter-vscode.md) | [log](logs/setup-jupyter-vscode.md) | ✅ Done |
| Tooling Audit & Adoption | [plan](plans/setup-tooling.md) | [log](logs/setup-tooling.md) | 🔴 Active — See T1-T3 |

## Enhancements

| Name | Plan | Status |
|------|------|--------|
| ML Capability Enhancements (GWO, InterpretML, AutoGluon) | [plan](plans/enhance-ml-capabilities.md) | ✅ Done (Phases 1-4+) |
| Pioneer Research — Experimental Features | [plan](plans/enhance-pioneer-research.md) | 🔴 Planned |

---

## Known Plan Types

Plans are categorized by a `type` field in YAML frontmatter. Each type has different structure and expectations.

| Type | Prefix | Frontmatter Required | Frontmatter Optional | When To Use |
|------|--------|---------------------|---------------------|-------------|
| `phase` | `NN-` | phase, name, status | tasks, sub_phases, deferred_items | Multi-step implementation with deliverables and checkpoints |
| `study` | `study-` | name, status, references | key_findings | Reading papers, studying repos, knowledge acquisition |
| `eval` | `eval-` | name, status, criteria, decision | rationale | Comparing tools/approaches, making a go/no-go choice |
| `setup` | `setup-` | name, status | tools_installed, configs_changed | Environment, tooling, infrastructure changes |
| `migration` | `migration-` | name, status, from, to | rollback | One-off system/data migrations |
| `enhancement` | `enhance-` | name, status, depends_on, blocks | rationale, tasks | Cross-cutting capability improvements that touch multiple existing modules |

**Agent Self-Extension:** When encountering a new activity type not in the catalog above:
1. Determine a short `type` name (lowercase, no spaces)
2. Create the plan file with `{type}-{descriptor}.md` naming
3. Add the new type to this catalog table
4. Add a new section in `plans/full.md` for the type

---

## File Conventions

| Convention | Rule |
|------------|------|
| **Folder** | `progress_docs/` — single entry point for all progress tracking |
| **Naming** | Phases: `NN-short-name.md`. Non-phases: `{type}-{descriptor}.md`. All snake_case. |
| **Plan format** | Markdown + YAML frontmatter for metadata + Markdown tables for tasks |
| **Log format** | Markdown tables — append-only, chronological |
| **Session recovery** | `current.md` — read first, reset per phase |
| **Completion marker** | YAML `status: complete` — never rename files |
| **Deferral marker** | YAML `status: deferred` + `deferred_reason` + `revisit_when` |
| **Aggregation** | `plans/full.md` has Deferred + Pending sections pulled from all files |
| **Full timeline** | `logs/full.md` is the single chronological log — all types interleaved |

## Related Files

- `AGENTS.md` — project-wide agent instructions
- `COMMAND_CHEATSHEET.md` — all CLI commands
- `.kilo/global-rules.md` — coding standards
- `.kilo/project-rules.md` — trading-specific rules
- `plans/` (root) — one-off plans not tied to phases
