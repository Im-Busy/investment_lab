---
project: investment_trying
last_updated: 2026-05-01
summary: |
  Rule-based multi-pattern trading system with 34+ chart pattern detectors,
  ML-enhanced regime detection, Numba-accelerated indicators, and event-driven
  backtesting engine. 8 phases spanning pattern detection through paper trading.
phases_total: 8
phases_complete: 4
phases_active: 2
phases_deferred: 2
---

# Master Plan

## Phase Status

| # | Phase | Plan | Log | Status |
|---|-------|------|-----|--------|
| 01 | Pattern Detection & Strategies | [plan](01-patterns.md) | [log](../logs/01-patterns.md) | ✅ Complete |
| 02 | Performance Optimization | [plan](02-optimization.md) | [log](../logs/02-optimization.md) | 🔄 Numba done, vectorbt deferred (Windows) |
| 03 | Regime Detection & Adaptation | [plan](03-regime.md) | [log](../logs/03-regime.md) | ✅ Complete |
| 04 | ML Foundation | [plan](04-ml-foundation.md) | [log](../logs/04-ml-foundation.md) | ✅ Complete |
| 05 | ML Advanced | [plan](05-ml-advanced.md) | [log](../logs/05-ml-advanced.md) | ⏸️ Deferred |
| 06 | Research-Based Enhancements | [plan](06-research.md) | [log](../logs/06-research.md) | 🔄 Tier 1 partially done |
| 07 | Paper Trading & Live Readiness | [plan](07-paper-trading.md) | [log](../logs/07-paper-trading.md) | ⏸️ Deferred |
| 08 | Contribution & Attribution | [plan](08-attribution.md) | [log](../logs/08-attribution.md) | ⏳ Files exist, verify completeness |

---

## Pending (Ready to Start)

| Priority | # | Phase | Depends On | Notes |
|----------|---|-------|------------|-------|
| P1 | 06 | Research-Based Enhancements (remaining Tier 1 + Tier 2) | Phase 04 complete | R2, R4, R5 pending |
| P2 | 08 | Contribution & Attribution | Phase 01 complete | Verify existing files, integrate |

---

## Deferred

| # | Phase/Item | Reason | Since | Revisit When |
|---|-----------|--------|-------|-------------|
| 05 | ML Advanced (full phase) | GPU >=16GB required for CNN training, autoencoder sweeps | 2026-04-30 | WSL2 with GPU passthrough or cloud GPU |
| 02 | vectorbt integration (sub-phase) | C++ compilation fails on Windows | 2026-04-20 | Linux/macOS environment |
| 07 | Paper Trading (full phase) | Optional — depends on Phase 06 completion | 2026-04-19 | Stages 01-06 stable |

---

## Studies

| Name | Plan | Log | Status | Key Output |
|------|------|-----|--------|------------|
| Paper Analysis & Synthesis | [plan](study-paper-analysis.md) | [log](../logs/study-paper-analysis.md) | ✅ Done | `useful_resources/papers_md/research_synthesis_report.md` |
| Repo Architecture Study | [plan](study-repo-architecture.md) | [log](../logs/study-repo-architecture.md) | ✅ Done | `useful_resources/useful_repos/ARCHITECTURE_ANALYSIS.md` |

## Evaluations

| Name | Plan | Log | Status | Decision |
|------|------|-----|--------|----------|
| Auto-Research Tools | [plan](eval-auto-research-tools.md) | [log](../logs/eval-auto-research-tools.md) | ✅ Done | KEEP 5, SKIP 4, 17 skills installed |

## Setups

| Name | Plan | Log | Status |
|------|------|-----|--------|
| Pixi to UV Migration | [plan](setup-pixi-migration.md) | [log](../logs/setup-pixi-migration.md) | ✅ Done |
| SearXNG MCP Setup | [plan](setup-searxng-mcp.md) | [log](../logs/setup-searxng-mcp.md) | 🔄 Partial |
| Jupyter + VS Code | [plan](setup-jupyter-vscode.md) | [log](../logs/setup-jupyter-vscode.md) | ✅ Done |

---

## Execution Order

```
Phase 01 (Patterns) ✅
  → Phase 02 (Perf) 🔄
  → Phase 03 (Regime) ✅
  → Phase 04 (ML Foundation) ✅
      → Phase 06 (Research Enhancements) 🔄 — ACTIVE
          → Phase 07 (Paper Trading) ⏸️
  → Phase 08 (Attribution) ⏳
  Phase 05 (ML Advanced) ⏸️ — wait for GPU
```
